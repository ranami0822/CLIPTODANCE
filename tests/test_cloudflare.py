"""Cloudflare アダプタのテスト。通信は差し替え、応答は 2026-10-31 に実際に得た応答(合成データ)を使う。"""
import base64
import json
import struct
import zlib
from pathlib import Path

import pytest

from adapters import (
    ApiError,
    BudgetExceeded,
    Estimator,
    EstimateField,
    FieldKind,
    InvalidRequest,
    Judge,
    RateLimited,
    choice,
    noul,
    score,
)
from adapters.cloudflare import (
    CloudflareClient,
    CloudflareEstimator,
    CloudflareJudge,
    encode_images,
    sniff_content_type,
)

FIX = Path(__file__).parent / "fixtures" / "cloudflare"


def fixture(name):
    return json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))


def png(w=8, h=8):
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Recorder:
    """通信の代わり。渡された要求を覚えて、決めた応答を返す。"""

    def __init__(self, status=200, headers=None, body=None):
        self.status, self.headers, self.body = status, headers or {}, body
        self.requests = []

    def __call__(self, url, headers, body, timeout):
        self.requests.append((url, headers, body))
        return self.status, self.headers, self.body


def client_with(rec):
    return CloudflareClient(account_id="acct", token="tok", transport=rec)


def judge_with_fixture(name, model="clef"):
    fx = fixture(name)
    rec = Recorder(fx["status"], fx["headers"], fx["body"])
    return CloudflareJudge(client_with(rec), model=model), rec


# ---------------------------------------------------------------- Clef: 要求

def test_clef_request_matches_documented_shape():
    judge, rec = judge_with_fixture("clef_text")
    qs = [
        noul("has_tree", "Is there a tree in the scene?"),
        choice("tree_side", "Which side is the tree on?", ["left", "right", "center"]),
        score("sunny", "How sunny is the scene?", ["dark", "cloudy", "bright", "very bright"]),
    ]
    judge.ask(qs, state="A sunny meadow with one large oak tree on the right.")
    url, headers, body = rec.requests[0]
    assert url.endswith("/accounts/acct/ai/run/@cf/cloudflare/clef")
    assert headers == {"Authorization": "Bearer tok"}
    assert body["model"] == "clef"
    assert body["state"] == "A sunny meadow with one large oak tree on the right."
    assert body["questions"]["has_tree"] == {"type": "noul", "instructions": "Is there a tree in the scene?"}
    assert body["questions"]["tree_side"]["criteria"] == {"left": None, "right": None, "center": None}
    assert body["questions"]["sunny"]["criteria"] == ["dark", "cloudy", "bright", "very bright"]
    assert "images" not in body


def test_clef_state_is_always_sent_even_when_empty():
    judge, rec = judge_with_fixture("clef_image")
    judge.ask([noul("is_red", "Is the image mostly red?")], images=[png()])
    body = rec.requests[0][2]
    assert body["state"] == ""  # 省略すると 400 になることを実測で確認済み
    assert body["images"][0]["content_type"] == "image/png"
    assert base64.b64decode(body["images"][0]["base64"]) == png()


def test_clef_flash_selects_other_model():
    judge, rec = judge_with_fixture("clef_text", model="clef-flash")
    judge.ask([noul("has_tree", "tree?")], state="x")
    assert rec.requests[0][0].endswith("/clef-flash") and rec.requests[0][2]["model"] == "clef-flash"
    assert judge.model_id == "@cf/cloudflare/clef-flash"


def test_note_descriptions_are_sent_for_choice_options():
    judge, rec = judge_with_fixture("clef_text")
    q = choice("tree_side", "side?", ["left", "right", "center"], notes=["左端", "", "中央"])
    judge.ask([q], state="x")
    assert rec.requests[0][2]["questions"]["tree_side"]["criteria"] == {"left": "左端", "right": None, "center": "中央"}


# ---------------------------------------------------------------- Clef: 応答(実測)

def test_clef_parses_real_text_response_including_zero_based_score_and_neurons():
    judge, _ = judge_with_fixture("clef_text")
    qs = [
        noul("has_tree", "tree?"),
        choice("tree_side", "side?", ["left", "right", "center"]),
        score("sunny", "sunny?", ["dark", "cloudy", "bright", "very bright"]),
    ]
    resp = judge.ask(qs, state="x")
    n, c, s = resp.results
    assert n.p_yes == pytest.approx(0.9918)
    assert c.top == "right" and c.probs["right"] == pytest.approx(0.9892) and c.confidence == pytest.approx(0.9679)
    assert s.score == pytest.approx(2.5942)  # 0 始まり: 段階 2 と 3 の間
    assert s.level_probs["very bright"] == pytest.approx(0.6258) and s.confidence == pytest.approx(0.354)
    assert resp.usage.neurons == pytest.approx(7.11)  # ヘッダ cf-ai-neurons
    assert resp.usage.input_tokens == 326 and resp.usage.output_tokens == 0 and resp.usage.api_calls == 1


def test_clef_parses_real_image_response():
    judge, _ = judge_with_fixture("clef_image")
    resp = judge.ask([noul("is_red", "red?")], images=[png()])
    assert resp.results[0].p_yes == pytest.approx(0.9934)
    assert resp.usage.neurons == pytest.approx(4.58) and resp.usage.input_tokens == 210


def test_clef_rejects_answers_that_do_not_match_the_questions():
    judge, _ = judge_with_fixture("clef_text")
    with pytest.raises(ApiError):  # 質問の id が応答にない
        judge.ask([noul("missing", "?")], state="x")
    with pytest.raises(ApiError):  # 選択肢が応答と合わない
        judge.ask([noul("has_tree", "?"), choice("tree_side", "?", ["a", "b"])], state="x")


def test_empty_question_list_makes_no_call():
    judge, rec = judge_with_fixture("clef_text")
    assert judge.ask([]).results == () and rec.requests == []


# ---------------------------------------------------------------- 入力の検査(通信の前)

@pytest.mark.parametrize("bad_id", ["日本語", "has space", "a" * 101])
def test_question_ids_outside_documented_charset_are_rejected_before_sending(bad_id):
    judge, rec = judge_with_fixture("clef_text")
    with pytest.raises(InvalidRequest):
        judge.ask([noul(bad_id, "?")])
    assert rec.requests == []


def test_choice_and_score_size_limits_are_checked_before_sending():
    judge, rec = judge_with_fixture("clef_text")
    with pytest.raises(InvalidRequest):
        judge.ask([score("s", "?", [str(i) for i in range(11)])])
    with pytest.raises(InvalidRequest):
        judge.ask([choice("c", "?", [str(i) for i in range(256)])])
    assert rec.requests == []


def test_more_than_64_questions_or_4_images_are_rejected_before_sending():
    judge, rec = judge_with_fixture("clef_text")
    with pytest.raises(InvalidRequest):
        judge.ask([noul(f"q{i}", "?") for i in range(65)])
    with pytest.raises(InvalidRequest):
        judge.ask([noul("q", "?")], images=[png()] * 5)
    assert rec.requests == []


def test_image_encoding_sniffs_type_and_enforces_size_limits(tmp_path):
    assert sniff_content_type(png()) == "image/png"
    assert sniff_content_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert sniff_content_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    with pytest.raises(InvalidRequest):
        sniff_content_type(b"GIF89a....")
    f = tmp_path / "a.png"
    f.write_bytes(png())
    assert encode_images([f])[0]["content_type"] == "image/png"  # パスでも読める
    big = png() + b"\x00" * (4 * 1024 * 1024)
    with pytest.raises(InvalidRequest):
        encode_images([big])
    with pytest.raises(InvalidRequest):  # 1枚は 4 MiB 以下でも合計が 8 MiB を超える
        encode_images([png() + b"\x00" * (3 * 1024 * 1024)] * 3)


# ---------------------------------------------------------------- エラーの対応

def err(status, code, message="m", headers=None):
    return Recorder(status, headers or {}, {"success": False, "errors": [{"code": code, "message": message}], "result": {}})


def test_error_codes_map_to_exceptions():
    q = [noul("q", "?")]
    with pytest.raises(BudgetExceeded):  # 3036: 1日の無料枠を使い切った(ドキュメントの表。実物は未検証)
        CloudflareJudge(client_with(err(429, 3036))).ask(q)
    with pytest.raises(RateLimited) as e:  # 3040: 一時的な容量不足
        CloudflareJudge(client_with(err(429, 3040, headers={"retry-after": "3"}))).ask(q)
    assert e.value.retry_after_s == 3.0
    with pytest.raises(RateLimited):
        CloudflareJudge(client_with(err(429, 9999))).ask(q)
    with pytest.raises(InvalidRequest):  # 5006: state を省略したときに実測したコード
        CloudflareJudge(client_with(err(400, 5006, "AiError: Bad input"))).ask(q)
    with pytest.raises(InvalidRequest):
        CloudflareJudge(client_with(err(413, 3006))).ask(q)
    with pytest.raises(ApiError) as e2:
        CloudflareJudge(client_with(err(500, 1000))).ask(q)
    assert e2.value.status == 500


def test_real_400_response_for_missing_state_is_invalid_request():
    fx = fixture("clef_image_nostate")
    rec = Recorder(fx["status"], fx["headers"], fx["body"])
    with pytest.raises(InvalidRequest):
        CloudflareJudge(client_with(rec)).ask([noul("is_red", "?")], images=[png()])


def test_success_flag_false_with_200_is_still_an_error():
    rec = Recorder(200, {}, {"success": False, "errors": [], "result": {}})
    with pytest.raises(ApiError):
        CloudflareJudge(client_with(rec)).ask([noul("q", "?")])


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    with pytest.raises(ValueError):
        CloudflareClient()


# ---------------------------------------------------------------- Clairic

PLACEMENT_FIELDS = [
    EstimateField("x", FieldKind.ESTIMATE, "Horizontal position, 0 = left edge", minimum=0, maximum=1, step=0.01),
    EstimateField("scale_mult", FieldKind.ESTIMATE, "Size multiplier", minimum=0.5, maximum=2.0, step=0.05, log_scale=True),
    EstimateField("height_m", FieldKind.ESTIMATE, "Typical height", unit="m", minimum=0, maximum=40, step=0.5),
]


def estimator_with_fixture(name):
    fx = fixture(name)
    rec = Recorder(fx["status"], fx["headers"], fx["body"])
    return CloudflareEstimator(client_with(rec)), rec


def test_clairic_request_matches_documented_shape():
    est, rec = estimator_with_fixture("clairic_text")
    est.estimate("A sunny meadow with one large oak tree.", PLACEMENT_FIELDS)
    url, _, body = rec.requests[0]
    assert url.endswith("/ai/run/@cf/cloudflare/clairic") and body["model"] == "clairic"
    assert body["fields"]["x"] == {"type": "estimate", "instructions": "Horizontal position, 0 = left edge", "min": 0, "max": 1, "step": 0.01}
    assert body["fields"]["scale_mult"]["scale"] == "log"
    assert body["fields"]["height_m"]["unit"] == "m"


def test_clairic_parses_real_response_with_80_percent_range():
    est, _ = estimator_with_fixture("clairic_text")
    resp = est.estimate("A sunny meadow with one large oak tree.", PLACEMENT_FIELDS)
    x = resp.values["x"]
    assert x.value == pytest.approx(0.81) and x.median == pytest.approx(0.83)
    assert x.range80 == (pytest.approx(0.71), pytest.approx(0.88))
    assert resp.values["height_m"].range80 == (4.0, 9.5)
    assert resp.usage.neurons == pytest.approx(2.95) and resp.usage.input_tokens == 325


def test_clairic_accepts_image_with_empty_text():
    est, rec = estimator_with_fixture("clairic_image")
    resp = est.estimate("", [EstimateField("red_fraction", FieldKind.ESTIMATE, minimum=0, maximum=1, step=0.05)], images=[png()])
    assert rec.requests[0][2]["state"] == "" and rec.requests[0][2]["images"]
    assert resp.values["red_fraction"].value == 1


def test_clairic_input_validation_before_sending():
    est, rec = estimator_with_fixture("clairic_text")
    with pytest.raises(InvalidRequest):  # estimate は min / max が必須
        est.estimate("t", [EstimateField("x", FieldKind.ESTIMATE)])
    with pytest.raises(InvalidRequest):
        est.estimate("t", [EstimateField("x", FieldKind.ESTIMATE, minimum=1, maximum=1)])
    with pytest.raises(InvalidRequest):  # 項目名の文字種
        est.estimate("t", [EstimateField("1x", FieldKind.NUMBER)])
    with pytest.raises(InvalidRequest):  # テキストも画像もない
        est.estimate("  ", [EstimateField("x", FieldKind.NUMBER)])
    with pytest.raises(InvalidRequest):  # 重複
        est.estimate("t", [EstimateField("x", FieldKind.NUMBER), EstimateField("x", FieldKind.NUMBER)])
    with pytest.raises(InvalidRequest):
        est.estimate("t", [])
    assert rec.requests == []


def test_clairic_truncated_input_is_an_error():
    body = {"success": True, "errors": [], "result": {"model": "clairic", "answers": {}, "truncated": True, "usage": {"input_tokens": 1, "output_tokens": 0}}}
    est = CloudflareEstimator(client_with(Recorder(200, {}, body)))
    with pytest.raises(ApiError):
        est.estimate("t", [EstimateField("x", FieldKind.NUMBER)])


def test_adapters_satisfy_protocols_and_documented_limits():
    judge, est = CloudflareJudge(client_with(Recorder())), CloudflareEstimator(client_with(Recorder()))
    assert isinstance(judge, Judge) and isinstance(est, Estimator)
    assert judge.capabilities.max_questions == 64 and judge.capabilities.max_images == 4
    assert est.capabilities.max_images == 4
