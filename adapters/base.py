"""判断側(Clef / Clairic)の抽象インターフェース。

docs/design.md「ClefとClairicの使い方」参照。画素処理(SAM/OpenCV)とは分離し、
判断はこのインターフェース越しに呼ぶ。実装は差し替え可能(Cloudflare REST → ローカル)。
質問の型・上限は設計書の「確認済みの仕様」に基づく。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

ImageRef = str | Path  # 画像ファイルのパス(Clef は状態と一緒に最大4枚)

MAX_QUESTIONS_PER_REQUEST = 64
MAX_IMAGES_PER_REQUEST = 4


@dataclass(frozen=True)
class NoulResult:
    """はい/いいえ。p_yes は「はい」の確率。"""
    p_yes: float


@dataclass(frozen=True)
class ChoiceResult:
    """選択肢ごとの確率と確信度。"""
    probs: dict[str, float]
    confidence: float | None = None


@dataclass(frozen=True)
class ScoreResult:
    """順序付き評価基準での確率重み付きスコアと各段階の確率。"""
    score: float
    level_probs: dict[str, float]


@dataclass(frozen=True)
class EstimateField:
    """Estimator に埋めさせる項目の定義(名前と型)。"""
    name: str
    kind: str  # "number" | "date" | "list" | "estimate" (Clairic の型に合わせて実装時に確定)


@dataclass(frozen=True)
class EstimateResult:
    """値と、確信度または80%範囲。"""
    value: object
    confidence: float | None = None
    range80: tuple[float, float] | None = None


@runtime_checkable
class Judge(Protocol):
    """選択・採点(Clef 相当)。文章は生成せず確率/型付き値のみ返す。"""

    def noul(self, question: str, images: Sequence[ImageRef], context: str = "") -> NoulResult: ...

    def choice(
        self, question: str, options: Sequence[str], images: Sequence[ImageRef], context: str = ""
    ) -> ChoiceResult: ...

    def score(
        self, question: str, scale: Sequence[str], images: Sequence[ImageRef], context: str = ""
    ) -> ScoreResult: ...


@runtime_checkable
class Estimator(Protocol):
    """数値・項目の推定(Clairic 相当)。位置・大きさ・回転など。"""

    def estimate(
        self, text: str, fields: Sequence[EstimateField], images: Sequence[ImageRef] = ()
    ) -> dict[str, EstimateResult]: ...
