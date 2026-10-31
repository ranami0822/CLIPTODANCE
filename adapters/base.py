"""判断側(Clef / Clairic)の抽象インターフェース。

docs/design.md「ClefとClairicの使い方」参照。画素処理(SAM/OpenCV)とは分離し、
判断はこのインターフェース越しに呼ぶ。

設計上の要点:
- 質問は「まとめて」渡す(`Judge.ask`)。Clef は1リクエストで最大64問なので、1問ずつの呼び出しにしない。
- 上限(質問数・画像数)は実装ごとの `capabilities` が持つ。抽象側に Cloudflare 固有の定数を置かない。
- 結果には、どのモデルか・キャッシュか・消費量(分かれば)を付ける。ログと予算管理に使う。
- 失敗は例外で返す。規則だけの縮退モードは設けない(設計書 R10)ので、握りつぶして代わりの値を返さない。
- キャッシュキーは (画像のハッシュ, 正規化した質問, model_id, context)。質問の `id` はキーに入れない。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence, Union, runtime_checkable

# 画像の渡し方: ファイルのパス、またはメモリ上のバイト列(PNG/JPEG のエンコード済みデータ)。
# 透明部分を持つ切り抜きを、何色で埋めて渡すかは呼び出し側が決める(設計書の未決事項)。
ImageInput = Union[Path, str, bytes]


# ---------------------------------------------------------------- 例外

class JudgeError(Exception):
    """判断側の呼び出しの失敗の基底クラス。"""


class InvalidRequest(JudgeError):
    """質問や画像の数・形が上限や規則に合わない(呼び出す前に検出できるもの)。"""


class BudgetExceeded(JudgeError):
    """無料枠などの予算を超えた。縮退モードは設けないので、呼び出し側が止める。"""


class RateLimited(JudgeError):
    """呼び出し頻度の上限に当たった。`retry_after_s` が分かれば入る。"""

    def __init__(self, message: str = "", retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ApiError(JudgeError):
    """サーバー側のエラーや、想定外の応答。"""

    def __init__(self, message: str = "", status: int | None = None):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- 使用量

@dataclass(frozen=True)
class Usage:
    """呼び出しの使用量。`neurons` は分かる範囲の合計(未実測のときは None)。"""
    api_calls: int = 0
    cache_hits: int = 0
    neurons: float | None = None
    latency_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        if self.neurons is None and other.neurons is None:
            neurons = None
        else:
            neurons = (self.neurons or 0.0) + (other.neurons or 0.0)
        return Usage(
            api_calls=self.api_calls + other.api_calls,
            cache_hits=self.cache_hits + other.cache_hits,
            neurons=neurons,
            latency_s=self.latency_s + other.latency_s,
        )


# ---------------------------------------------------------------- 質問(Clef)

class QuestionKind(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


@dataclass(frozen=True)
class Question:
    """Clef に尋ねる1問。`id` は1回の `ask` の中で重複しない呼び出し側の名前(キャッシュキーには入らない)。"""
    id: str
    kind: QuestionKind
    text: str
    options: tuple[str, ...] = ()  # CHOICE の選択肢
    scale: tuple[str, ...] = ()    # SCORE の評価基準(低い順)

    def __post_init__(self) -> None:
        if not self.id:
            raise InvalidRequest("質問の id が空")
        if not self.text.strip():
            raise InvalidRequest(f"質問 {self.id!r} の text が空")
        if self.kind is QuestionKind.CHOICE:
            if len(self.options) < 2 or len(set(self.options)) != len(self.options):
                raise InvalidRequest(f"質問 {self.id!r}: choice の選択肢は重複なしで2つ以上必要")
        elif self.options:
            raise InvalidRequest(f"質問 {self.id!r}: options は choice のときだけ指定できる")
        if self.kind is QuestionKind.SCORE:
            if len(self.scale) < 2 or len(set(self.scale)) != len(self.scale):
                raise InvalidRequest(f"質問 {self.id!r}: score の評価基準は重複なしで2段階以上必要")
        elif self.scale:
            raise InvalidRequest(f"質問 {self.id!r}: scale は score のときだけ指定できる")

    def canonical(self) -> str:
        """キャッシュキー用の正規化した文字列。`id` は含めない。"""
        return json.dumps(
            {"kind": self.kind.value, "text": self.text, "options": list(self.options), "scale": list(self.scale)},
            ensure_ascii=False, sort_keys=True,
        )


def noul(id: str, text: str) -> Question:
    return Question(id=id, kind=QuestionKind.NOUL, text=text)


def choice(id: str, text: str, options: Sequence[str]) -> Question:
    return Question(id=id, kind=QuestionKind.CHOICE, text=text, options=tuple(options))


def score(id: str, text: str, scale: Sequence[str]) -> Question:
    return Question(id=id, kind=QuestionKind.SCORE, text=text, scale=tuple(scale))


# ---------------------------------------------------------------- 結果(Clef)

@dataclass(frozen=True)
class NoulResult:
    """はい/いいえ。p_yes は「はい」の確率。"""
    question_id: str
    p_yes: float
    cached: bool = False


@dataclass(frozen=True)
class ChoiceResult:
    """選択肢ごとの確率と確信度。"""
    question_id: str
    probs: dict[str, float]
    confidence: float | None = None
    cached: bool = False


@dataclass(frozen=True)
class ScoreResult:
    """順序付き評価基準での確率重み付きスコアと各段階の確率。"""
    question_id: str
    score: float
    level_probs: dict[str, float]
    cached: bool = False


JudgeResult = Union[NoulResult, ChoiceResult, ScoreResult]

RESULT_CLASS: dict[QuestionKind, type] = {
    QuestionKind.NOUL: NoulResult,
    QuestionKind.CHOICE: ChoiceResult,
    QuestionKind.SCORE: ScoreResult,
}


@dataclass(frozen=True)
class JudgeCapabilities:
    """実装ごとの上限。None は上限なし。"""
    max_questions: int | None
    max_images: int | None


@dataclass(frozen=True)
class JudgeResponse:
    """`results` は渡した質問と同じ順序。"""
    results: tuple[JudgeResult, ...]
    usage: Usage
    model_id: str


@runtime_checkable
class Judge(Protocol):
    """選択・採点(Clef 相当)。文章は生成せず、確率や型付きの値だけを返す。

    実装は `capabilities` を超える質問・画像を渡されたら `InvalidRequest` を投げる。
    上限を超える質問数を扱いたいときは `ask_in_chunks` を使う。
    """

    model_id: str
    capabilities: JudgeCapabilities

    def ask(
        self, questions: Sequence[Question], images: Sequence[ImageInput] = (), context: str = ""
    ) -> JudgeResponse: ...


# ---------------------------------------------------------------- 補助

def validate_batch(questions: Sequence[Question]) -> None:
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise InvalidRequest("1回の ask の中で質問の id が重複している")


def check_capabilities(caps: JudgeCapabilities, n_questions: int, n_images: int) -> None:
    if caps.max_questions is not None and n_questions > caps.max_questions:
        raise InvalidRequest(f"質問が多すぎる: {n_questions} > {caps.max_questions}")
    if caps.max_images is not None and n_images > caps.max_images:
        raise InvalidRequest(f"画像が多すぎる: {n_images} > {caps.max_images}")


def image_digest(image: ImageInput) -> str:
    """画像の内容のハッシュ(sha256)。パスならファイルの中身を読む。"""
    data = image if isinstance(image, bytes) else Path(image).read_bytes()
    return hashlib.sha256(data).hexdigest()


def ask_in_chunks(
    judge: Judge, questions: Sequence[Question], images: Sequence[ImageInput] = (), context: str = ""
) -> JudgeResponse:
    """質問数が `capabilities.max_questions` を超えるとき、分けて送って結果と使用量を合わせる。"""
    validate_batch(questions)
    limit = judge.capabilities.max_questions
    if not questions:
        return JudgeResponse(results=(), usage=Usage(), model_id=judge.model_id)
    size = limit if limit is not None else len(questions)
    results: list[JudgeResult] = []
    usage = Usage()
    for i in range(0, len(questions), size):
        resp = judge.ask(questions[i:i + size], images, context)
        results.extend(resp.results)
        usage = usage + resp.usage
    return JudgeResponse(results=tuple(results), usage=usage, model_id=judge.model_id)


# ---------------------------------------------------------------- 推定(Clairic)

class FieldKind(str, Enum):
    NUMBER = "number"
    DATE = "date"
    LIST = "list"
    ESTIMATE = "estimate"  # 値と範囲を返す推定値(Clairic の型名は実装時に公式文書で確認)


@dataclass(frozen=True)
class EstimateField:
    """Estimator に埋めさせる項目。単位と範囲を書いて、座標の規約などを伝える。"""
    name: str
    kind: FieldKind
    description: str = ""
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class EstimateValue:
    name: str
    value: object
    confidence: float | None = None
    range80: tuple[float, float] | None = None


@dataclass(frozen=True)
class EstimateResponse:
    values: dict[str, EstimateValue]
    usage: Usage
    model_id: str


@runtime_checkable
class Estimator(Protocol):
    """数値・項目の推定(Clairic 相当)。位置・倍率・回転など(設計書 R10)。"""

    model_id: str
    capabilities: JudgeCapabilities

    def estimate(
        self,
        text: str,
        fields: Sequence[EstimateField],
        images: Sequence[ImageInput] = (),
        context: str = "",
    ) -> EstimateResponse: ...
