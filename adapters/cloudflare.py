"""Cloudflare Workers AI の REST 実装(Clef / Clairic)。

仕様の出典(2026-10-31 に公式ドキュメントで確認し、同日に合成データで実際に呼んで裏取りした):
  - POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/{clef|clef-flash|clairic}
  - 認証: `Authorization: Bearer <token>`(環境変数 CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID)
  - 応答は {"result": {...}, "success": bool, "errors": [{"code", "message"}], "messages": []} に包まれる。
  - 消費ニューロンは応答ヘッダ `cf-ai-neurons` に入る(例: Clef 326 入力トークン → 7.11)。本文の usage にはない。
  - Clef: body = {model, state(必須。空文字は可・省略は 400), questions(1〜64問), images(最大4枚)}。
    answers は noul → {noul}、choice → {choice, probabilities, confidence}、
    score → {score(0 始まり), legend, probabilities(キーは段階の番号の文字列), confidence}。
  - Clairic: body = {model, state, fields, images, ...}。estimate 型は min / max が必須で、
    {value, mean, median, p10, p90, p_below, p_above} を返す(p10〜p90 が 80% 範囲)。
  - 画像: PNG / JPEG / WebP のみ。1枚 4 MiB・16 メガピクセルまで、合計 8 MiB まで。URL は不可。
  - エラー表: 3036 (429) = 1日の無料枠 10,000 ニューロンを使い切った / 3040 (429) = 一時的な容量不足。

未検証(実際には試していない):
  - 無料枠を使い切ったときの応答の実物(ドキュメントの表のコードに従って割り当てている)。
  - 16 メガピクセルを超える画像のときのエラー。
  - 透明部分のある PNG を渡したときの Clef の見え方。
"""
from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Callable, Sequence

from .base import (
    ApiError,
    BudgetExceeded,
    ChoiceResult,
    EstimateField,
    EstimateResponse,
    EstimateValue,
    FieldKind,
    ImageInput,
    InvalidRequest,
    JudgeCapabilities,
    JudgeResponse,
    JudgeResult,
    NoulResult,
    Question,
    QuestionKind,
    RateLimited,
    ScoreResult,
    State,
    Usage,
    check_capabilities,
    validate_batch,
)

BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"

CLEF_CAPABILITIES = JudgeCapabilities(max_questions=64, max_images=4)
CLAIRIC_CAPABILITIES = JudgeCapabilities(max_questions=None, max_images=4)  # 項目数に上限なし

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 8 * 1024 * 1024

_QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,100}$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")

# 公式ドキュメント「Errors」の表
CODE_BUDGET = 3036
CODE_CAPACITY = 3040
INVALID_REQUEST_CODES = {3003, 3006, 5004, 5006, 5007}  # 5006 は実測(state 省略時)

# 通信の差し替え口: (url, headers, json_body, timeout) -> (HTTP status, 小文字化したヘッダ, JSON 本文)
Transport = Callable[[str, dict, dict, float], "tuple[int, dict, object]"]


def requests_transport(url: str, headers: dict, body: dict, timeout: float):
    import requests

    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        data = r.json()
    except ValueError:
        data = {"non_json": r.text[:500]}
    return r.status_code, {k.lower(): v for k, v in r.headers.items()}, data


# ---------------------------------------------------------------- 画像

def sniff_content_type(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise InvalidRequest("画像は PNG / JPEG / WebP のみ対応")


def encode_images(images: Sequence[ImageInput]) -> list[dict]:
    out: list[dict] = []
    total = 0
    for img in images:
        data = img if isinstance(img, bytes) else Path(img).read_bytes()
        if len(data) > MAX_IMAGE_BYTES:
            raise InvalidRequest(f"画像が大きすぎる: {len(data)} B > {MAX_IMAGE_BYTES} B(1枚)")
        total += len(data)
        out.append({"content_type": sniff_content_type(data), "base64": base64.b64encode(data).decode("ascii")})
    if total > MAX_TOTAL_IMAGE_BYTES:
        raise InvalidRequest(f"画像の合計が大きすぎる: {total} B > {MAX_TOTAL_IMAGE_BYTES} B")
    return out


# ---------------------------------------------------------------- 通信とエラー

class CloudflareClient:
    def __init__(
        self,
        account_id: str | None = None,
        token: str | None = None,
        transport: Transport | None = None,
        timeout_s: float = 60.0,
    ):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self._token = token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not self.account_id or not self._token:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN が未設定")
        self._transport = transport or requests_transport
        self.timeout_s = timeout_s

    def run(self, model: str, body: dict) -> "tuple[dict, Usage]":
        """`result` と、ヘッダから作った Usage(トークン数は呼び出し側が足す)を返す。"""
        url = BASE_URL.format(account=self.account_id, model=model)
        headers = {"Authorization": f"Bearer {self._token}"}
        t0 = time.monotonic()
        status, resp_headers, data = self._transport(url, headers, body, self.timeout_s)
        latency = time.monotonic() - t0
        raise_for_error(status, resp_headers, data)
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict):
            raise ApiError(f"想定外の応答の形: {str(data)[:200]}", status=status)
        neurons = resp_headers.get("cf-ai-neurons")
        return result, Usage(
            api_calls=1, neurons=float(neurons) if neurons is not None else None, latency_s=latency
        )


def raise_for_error(status: int, headers: dict, data: object) -> None:
    ok = isinstance(data, dict) and data.get("success") is True and 200 <= status < 300
    if ok:
        return
    errors = data.get("errors") if isinstance(data, dict) else None
    first = errors[0] if isinstance(errors, list) and errors and isinstance(errors[0], dict) else {}
    code, message = first.get("code"), str(first.get("message", ""))[:300]
    text = f"HTTP {status} code={code}: {message}"
    if code == CODE_BUDGET:
        raise BudgetExceeded(text)
    if status == 429 or code == CODE_CAPACITY:
        retry = headers.get("retry-after")
        raise RateLimited(text, retry_after_s=float(retry) if retry and retry.replace(".", "", 1).isdigit() else None)
    if code in INVALID_REQUEST_CODES or status in (400, 413):
        raise InvalidRequest(text)
    raise ApiError(text, status=status)


# ---------------------------------------------------------------- Clef

def clef_question_spec(q: Question) -> dict:
    if not _QUESTION_ID_RE.match(q.id):
        raise InvalidRequest(f"質問の id は英数字と _ . - の100文字まで: {q.id!r}")
    if q.kind is QuestionKind.NOUL:
        return {"type": "noul", "instructions": q.text}
    if q.kind is QuestionKind.CHOICE:
        if not 2 <= len(q.options) <= 255:
            raise InvalidRequest(f"質問 {q.id!r}: choice の選択肢は 2〜255 個")
        notes = q.notes or ("",) * len(q.options)
        return {
            "type": "choice",
            "instructions": q.text,
            "criteria": {opt: (note or None) for opt, note in zip(q.options, notes)},
        }
    if not 2 <= len(q.scale) <= 10:
        raise InvalidRequest(f"質問 {q.id!r}: score の段階は 2〜10 個")
    return {"type": "score", "instructions": q.text, "criteria": list(q.scale)}


def build_clef_body(model: str, questions: Sequence[Question], images: Sequence[ImageInput], state: State) -> dict:
    body: dict = {
        "model": model,
        "state": state,  # 必須。空文字は可(実測)
        "questions": {q.id: clef_question_spec(q) for q in questions},
    }
    if images:
        body["images"] = encode_images(images)
    return body


def parse_clef_answers(questions: Sequence[Question], answers: dict) -> list[JudgeResult]:
    results: list[JudgeResult] = []
    for q in questions:
        ans = answers.get(q.id)
        if not isinstance(ans, dict) or ans.get("type") != q.kind.value:
            raise ApiError(f"質問 {q.id!r} の答えが想定外: {str(ans)[:150]}")
        try:
            if q.kind is QuestionKind.NOUL:
                results.append(NoulResult(question_id=q.id, p_yes=float(ans["noul"])))
            elif q.kind is QuestionKind.CHOICE:
                probs = {str(k): float(v) for k, v in ans["probabilities"].items()}
                if set(probs) != set(q.options):
                    raise ApiError(f"質問 {q.id!r}: 選択肢が一致しない: {sorted(probs)}")
                results.append(ChoiceResult(question_id=q.id, probs=probs, confidence=float(ans["confidence"])))
            else:
                results.append(
                    ScoreResult(
                        question_id=q.id,
                        score=float(ans["score"]),
                        level_probs=_level_probs(q, ans["probabilities"]),
                        confidence=float(ans["confidence"]),
                    )
                )
        except (KeyError, TypeError, ValueError) as e:
            raise ApiError(f"質問 {q.id!r} の答えを読めない: {e!r}") from e
    return results


def _level_probs(q: Question, probs: dict) -> dict[str, float]:
    """応答のキーは段階の番号の文字列("0", "1", ...。実測)。質問の scale の説明文に直す。"""
    out: dict[str, float] = {}
    for key, value in probs.items():
        if str(key).isdigit() and int(key) < len(q.scale):
            out[q.scale[int(key)]] = float(value)
        elif key in q.scale:
            out[str(key)] = float(value)
        else:
            raise ApiError(f"質問 {q.id!r}: 段階のキーが想定外: {key!r}")
    if set(out) != set(q.scale):
        raise ApiError(f"質問 {q.id!r}: 段階が一致しない")
    return out


class CloudflareJudge:
    """Clef(または clef-flash)。`model` は "clef" か "clef-flash"。"""

    capabilities = CLEF_CAPABILITIES

    def __init__(self, client: CloudflareClient | None = None, model: str = "clef"):
        if model not in ("clef", "clef-flash"):
            raise ValueError("model は 'clef' か 'clef-flash'")
        self._client = client or CloudflareClient()
        self._model = model
        self.model_id = f"@cf/cloudflare/{model}"

    def ask(
        self, questions: Sequence[Question], images: Sequence[ImageInput] = (), state: State = ""
    ) -> JudgeResponse:
        validate_batch(questions)
        if not questions:
            return JudgeResponse(results=(), usage=Usage(), model_id=self.model_id)
        check_capabilities(self.capabilities, len(questions), len(images))
        body = build_clef_body(self._model, questions, images, state)
        result, usage = self._client.run(self._model, body)
        answers = result.get("answers")
        if not isinstance(answers, dict):
            raise ApiError("応答に answers がない")
        raw = result.get("usage") or {}
        usage = usage + Usage(input_tokens=int(raw.get("input_tokens", 0)), output_tokens=int(raw.get("output_tokens", 0)))
        return JudgeResponse(results=tuple(parse_clef_answers(questions, answers)), usage=usage, model_id=self.model_id)


# ---------------------------------------------------------------- Clairic

def clairic_field_spec(f: EstimateField) -> dict:
    if not _FIELD_NAME_RE.match(f.name):
        raise InvalidRequest(f"項目名は英字か _ で始まり、英数字と _ . - の64文字まで: {f.name!r}")
    spec: dict = {"type": f.kind.value}
    if f.instructions:
        spec["instructions"] = f.instructions
    if f.unit:
        spec["unit"] = f.unit
    if f.kind is FieldKind.ESTIMATE:
        if f.minimum is None or f.maximum is None:
            raise InvalidRequest(f"項目 {f.name!r}: estimate は minimum と maximum が必須")
        if f.minimum >= f.maximum:
            raise InvalidRequest(f"項目 {f.name!r}: minimum < maximum にする")
        spec["min"], spec["max"] = f.minimum, f.maximum
        if f.step is not None:
            spec["step"] = f.step
        if f.log_scale:
            spec["scale"] = "log"
    if f.integer:
        spec["integer"] = True
    if f.decimals is not None:
        spec["decimals"] = f.decimals
    if f.kind is FieldKind.LIST:
        if f.items not in ("text", "number", "date"):
            raise InvalidRequest(f"項目 {f.name!r}: list は items に text / number / date のどれかが必須")
        spec["items"] = f.items
    return spec


def parse_clairic_answers(fields: Sequence[EstimateField], answers: dict) -> dict[str, EstimateValue]:
    out: dict[str, EstimateValue] = {}
    for f in fields:
        ans = answers.get(f.name)
        if not isinstance(ans, dict) or "value" not in ans:
            raise ApiError(f"項目 {f.name!r} の答えが想定外: {str(ans)[:150]}")
        p10, p90 = ans.get("p10"), ans.get("p90")
        out[f.name] = EstimateValue(
            name=f.name,
            value=ans["value"],
            confidence=ans.get("confidence"),
            median=ans.get("median"),
            mean=ans.get("mean"),
            range80=(float(p10), float(p90)) if p10 is not None and p90 is not None else None,
            p_null=ans.get("p_null"),
        )
    return out


class CloudflareEstimator:
    """Clairic(プレビュー)。モデルは予告なく変わる・止まる可能性がある(公式の利用条件)。"""

    capabilities = CLAIRIC_CAPABILITIES
    model_id = "@cf/cloudflare/clairic"

    def __init__(self, client: CloudflareClient | None = None):
        self._client = client or CloudflareClient()

    def estimate(
        self, text: str, fields: Sequence[EstimateField], images: Sequence[ImageInput] = ()
    ) -> EstimateResponse:
        if not fields:
            raise InvalidRequest("項目が1つも指定されていない")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise InvalidRequest("項目名が重複している")
        check_capabilities(self.capabilities, len(fields), len(images))
        if not text.strip() and not images:
            raise InvalidRequest("text が空のときは画像が必要")
        body: dict = {
            "model": "clairic",
            "state": text,
            "fields": {f.name: clairic_field_spec(f) for f in fields},
        }
        if images:
            body["images"] = encode_images(images)
        result, usage = self._client.run("clairic", body)
        answers = result.get("answers")
        if not isinstance(answers, dict):
            raise ApiError("応答に answers がない")
        if result.get("truncated"):
            raise ApiError("入力が長すぎて切り詰められた(truncated=true)")
        raw = result.get("usage") or {}
        usage = usage + Usage(input_tokens=int(raw.get("input_tokens", 0)), output_tokens=int(raw.get("output_tokens", 0)))
        return EstimateResponse(values=parse_clairic_answers(fields, answers), usage=usage, model_id=self.model_id)
