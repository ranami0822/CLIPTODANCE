import base64

import pytest

from server import arrange as arr
from server import imaging
from scripts import make_samples as ms

# サンプルのスケッチの塊: 1 = 丸(右上) 2 = 三角(左下) 3 = 四角(右寄り)
TRUTH = {"A": 2, "B": 1, "C": 3}     # 木→2、太陽→1、家→3


def objs3():
    names = [("A", "木", ms.tree()), ("B", "太陽", ms.sun()), ("C", "家", ms.house().convert("RGBA"))]
    return [arr.Obj(i, n, im) for i, n, im in names]


def est(v, lo=None, hi=None):
    if v is None:
        return {"type": "estimate", "value": None}
    return {"type": "estimate", "value": v, "median": v, "mean": v,
            "p10": v - 0.05 if lo is None else lo, "p90": v + 0.05 if hi is None else hi}


def clairic_answers(objs, **override):
    ans = {}
    for i, o in enumerate(objs):
        base = {"x": 0.2 + 0.3 * i, "y": 0.6, "z": 0.3 + 0.2 * i, "scale": 0.25, "rot": 5 * i}
        base.update(override.get(o.id, {}))
        for k, v in base.items():
            ans[arr.field_name(o.id, k)] = est(v)
    return ans


def clef_answers(objs, picks, p=0.7):
    """picks = {素材ID: 塊番号}。選んだ塊に p、残りに均等に配る。"""
    ans = {}
    for o in objs:
        keys = ["0", "1", "2", "3"]
        rest = (1 - p) / (len(keys) - 1)
        ans[arr.field_name(o.id, "shape")] = {
            "type": "choice", "choice": str(picks[o.id]), "confidence": 0.5,
            "probabilities": {k: (p if k == str(picks[o.id]) else rest) for k in keys},
        }
    return ans


class Fake:
    """URL の末尾(clef / clairic)で応答を切り替える通信の差し替え。呼ばれた中身を calls に残す。"""

    def __init__(self, objs, picks=TRUTH, clairic=None, neurons=("10", "5")):
        self.calls = []
        self.objs, self.picks, self.clairic, self.neurons = objs, picks, clairic, neurons

    def __call__(self, url, headers, body, timeout):
        model = url.rsplit("/", 1)[1]
        self.calls.append((model, url, headers, body))
        answers = clef_answers(self.objs, self.picks) if model == "clef" else (self.clairic or clairic_answers(self.objs))
        n = self.neurons[0] if model == "clef" else self.neurons[1]
        return 200, {"cf-ai-neurons": n}, {"success": True, "errors": [], "result": {"answers": answers, "truncated": False}}


def sketch():
    return ms.sketch().convert("RGBA")


def test_fields_have_required_estimate_shape():
    f = arr.build_fields(objs3(), 4 / 3)
    assert len(f) == 15
    spec = f["B_scale"]
    assert spec["type"] == "estimate" and spec["min"] < spec["max"]
    assert "Object B (太陽)" in spec["instructions"]
    assert f["A_rot"]["step"] == 1
    import re
    assert all(re.match(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$", k) for k in f)


def test_state_tells_clairic_what_is_already_decided():
    regions = imaging.sketch_regions(sketch())
    s = arr.build_state("夏の朝", objs3(), True, 1.33, regions, {"A": 2, "B": 1, "C": 0})
    assert "夏の朝" in s and "A = 木" in s and "3 shapes" in s
    assert "A replaces sketch shape 2" in s and "B replaces sketch shape 1" in s
    assert "Not matched to any sketch shape" in s and "empty areas): C." in s
    s2 = arr.build_state("", objs3(), False, 1.33)
    assert "no sketch" in s2 and "Image 1 is a contact sheet" in s2 and "Already decided" not in s2


def test_assign_body_is_a_choice_per_object_with_zero_option():
    regions = imaging.sketch_regions(sketch())
    body = arr.build_assign_body(objs3(), "夏", regions, [b"x"])
    q = body["questions"]["A_shape"]
    assert body["model"] == "clef" and q["type"] == "choice"
    assert list(q["criteria"]) == ["0", "1", "2", "3"] and len(body["questions"]) == 3


def test_solve_assignment_resolves_conflicts_by_total_probability():
    # A も B も塊 1 が第一候補。B のほうが強く 1 を欲しがるので、A は 2 に回る
    probs = {"A": {0: 0.1, 1: 0.5, 2: 0.4}, "B": {0: 0.05, 1: 0.9, 2: 0.05}}
    assert arr.solve_assignment(probs, ["A", "B"]) == {"A": 2, "B": 1}
    # 対応なし(0)も、全体の確率が高ければ選ばれる
    probs = {"A": {0: 0.8, 1: 0.2}, "B": {0: 0.1, 1: 0.9}}
    assert arr.solve_assignment(probs, ["A", "B"]) == {"A": 0, "B": 1}
    assert arr.solve_assignment({"A": {0: 1.0}}, ["A"]) == {"A": 0}


def test_arrange_two_stage_with_fake_transport():
    objs = objs3()
    fake = Fake(objs)
    client = arr.CloudflareClient("acct", "tok", transport=fake)
    out = arr.arrange("夏", objs, sketch(), 960, 720, client)

    assert [c[0] for c in fake.calls] == ["clef", "clairic"]                   # Clef(対応づけ)→ Clairic(z・rot)
    clef_call, clairic_call = fake.calls
    assert clef_call[1].endswith("/accounts/acct/ai/run/@cf/cloudflare/clef")
    assert clef_call[2]["Authorization"] == "Bearer tok"
    assert len(clef_call[3]["images"]) == 2 and len(clef_call[3]["questions"]) == 3
    assert len(clairic_call[3]["fields"]) == 15 and len(clairic_call[3]["images"]) == 2
    assert "A replaces sketch shape 2" in clairic_call[3]["state"]            # Clef の結果が Clairic に渡る
    for im in clairic_call[3]["images"]:
        assert base64.b64decode(im["base64"])[:8] == b"\x89PNG\r\n\x1a\n"

    assert out["source"] == "clairic" and out["sketch_used"] and out["neurons"] == 15.0
    ps = {p["id"]: p for p in out["placements"]}
    assert [ps[i]["shape"] for i in "ABC"] == [2, 1, 3] and all(p["geom"] == "sketch" for p in ps.values())
    assert ps["A"]["shape_p"] == pytest.approx(0.7)
    # x・y は塊の中心から決まる(Clairic が出した x=0.2 などは使わない)
    assert (ps["A"]["x"], ps["A"]["y"]) == pytest.approx((0.24, 0.62), abs=0.03)
    assert (ps["B"]["x"], ps["B"]["y"]) == pytest.approx((0.86, 0.17), abs=0.03)
    assert (ps["C"]["x"], ps["C"]["y"]) == pytest.approx((0.67, 0.65), abs=0.03)
    # 木は 幅:高さ = 2:3。塊は 0.17 × 0.24 なので長辺は高さで決まり ≒ 0.24
    assert ps["A"]["scale"] == pytest.approx(0.24, abs=0.03)
    # z・rot は Clairic の値、order は z の小さい順(奥から)
    assert ps["B"]["z"] == pytest.approx(0.5) and ps["C"]["rot"] == pytest.approx(10)
    assert sorted(p["order"] for p in out["placements"]) == [0, 1, 2]
    assert ps["A"]["range80"]["x"][0] < ps["A"]["x"] < ps["A"]["range80"]["x"][1]


def test_unmatched_object_uses_clairic_geometry():
    objs = objs3()
    out = arr.arrange("", objs, sketch(), 960, 720,
                      arr.CloudflareClient("a", "t", transport=Fake(objs, picks={"A": 2, "B": 1, "C": 0})))
    ps = {p["id"]: p for p in out["placements"]}
    assert ps["C"]["geom"] == "estimate" and ps["C"]["shape"] == 0
    assert ps["C"]["x"] == pytest.approx(0.8)         # clairic_answers の C: 0.2 + 0.3*2
    assert ps["A"]["geom"] == "sketch"


def test_no_sketch_skips_clef_entirely():
    objs = objs3()
    fake = Fake(objs)
    out = arr.arrange("夏", objs, None, 960, 720, arr.CloudflareClient("a", "t", transport=fake))
    assert [c[0] for c in fake.calls] == ["clairic"] and len(fake.calls[0][3]["images"]) == 1
    assert not out["sketch_used"] and all(p["geom"] == "estimate" for p in out["placements"])


def test_blank_sketch_is_treated_as_no_sketch():
    from PIL import Image
    objs = objs3()
    fake = Fake(objs)
    out = arr.arrange("", objs, Image.new("RGBA", (200, 150), (255, 255, 255, 255)), 960, 720,
                      arr.CloudflareClient("a", "t", transport=fake))
    assert [c[0] for c in fake.calls] == ["clairic"] and not out["sketch_used"]


def test_postprocess_clamps_and_logs():
    objs = objs3()
    ps = arr.parse_answers(objs, clairic_answers(objs, A={"x": 1.4, "scale": 3.0, "rot": 400}))
    log = arr.postprocess(ps)
    assert ps[0].x == 0.98 and ps[0].scale == 1.0 and ps[0].rot == 180.0
    assert any("A(木) x" in l for l in log)


def test_uncertain_rotation_is_reset_to_zero():
    objs = objs3()
    ans = clairic_answers(objs)
    ans["A_rot"] = est(15, -11, 89)          # 実測に近い: 値は 15° だが範囲が 100° 幅
    ans["B_rot"] = est(10, 0, 20)            # 範囲が狭ければそのまま
    ps = arr.parse_answers(objs, ans)
    log = arr.postprocess(ps)
    assert ps[0].rot == 0.0 and ps[1].rot == 10
    assert any("A(木) rot" in l and "0°" in l for l in log)


def test_missing_values_fall_back_and_are_reported():
    objs = objs3()
    ps = arr.parse_answers(objs, clairic_answers(objs, B={"rot": None}))
    log = arr.postprocess(ps)
    assert ps[1].missing == ["rot"] and ps[1].rot == 0.0
    assert any("B(太陽)" in l and "既定値" in l for l in log)


def test_errors_are_mapped(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    def make(status, errors):
        return arr.CloudflareClient("a", "t", transport=lambda *a: (status, {}, {"success": False, "errors": errors}))

    with pytest.raises(arr.ArrangeError) as e:
        arr.arrange("", objs3(), None, 960, 720, make(429, [{"code": 3036, "message": "daily limit"}]))
    assert e.value.status == 429 and "無料枠" in str(e.value)
    with pytest.raises(arr.ArrangeError) as e:
        arr.arrange("", objs3(), None, 960, 720, make(400, [{"code": 5006, "message": "bad"}]))
    assert e.value.status == 400
    with pytest.raises(arr.ArrangeError) as e:      # 資格情報なし
        arr.arrange("", objs3(), None, 960, 720, arr.CloudflareClient("", ""))
    assert e.value.status == 503


def test_clef_error_is_not_swallowed():
    def transport(url, *a):
        if url.endswith("/clef"):
            return 429, {}, {"success": False, "errors": [{"code": 3040, "message": "capacity"}]}
        raise AssertionError("Clef が失敗したら Clairic は呼ばない")

    with pytest.raises(arr.ArrangeError) as e:
        arr.arrange("", objs3(), sketch(), 960, 720, arr.CloudflareClient("a", "t", transport=transport))
    assert e.value.status == 429


def test_truncated_is_an_error():
    c = arr.CloudflareClient("a", "t", transport=lambda *a: (200, {}, {"success": True, "result": {"answers": {}, "truncated": True}}))
    with pytest.raises(arr.ArrangeError):
        arr.arrange("", objs3(), None, 960, 720, c)


def test_too_many_objects():
    many = [arr.Obj(chr(65 + i), "x", ms.sun()) for i in range(arr.MAX_OBJECTS + 1)]
    with pytest.raises(arr.ArrangeError):
        arr.arrange("", many, None, 960, 720, None, mock=True)


def test_mock_puts_objects_on_sketch_shapes():
    out = arr.arrange("", objs3(), sketch(), 960, 720, None, mock=True)
    assert out["source"] == "mock" and len(out["regions"]) == 3
    spots = [(0.86, 0.17), (0.24, 0.62), (0.67, 0.65)]
    for p in out["placements"]:
        assert p["geom"] == "sketch"
        assert min(((p["x"] - sx) ** 2 + (p["y"] - sy) ** 2) ** 0.5 for sx, sy in spots) < 0.05
    assert len({p["shape"] for p in out["placements"]}) == 3      # 同じ塊を2つの素材が使わない


def test_mock_without_sketch_is_deterministic():
    a = arr.arrange("", objs3(), None, 960, 720, None, mock=True)["placements"]
    b = arr.arrange("", objs3(), None, 960, 720, None, mock=True)["placements"]
    assert a == b
    assert all(0 < p["x"] < 1 and 0 < p["y"] < 1 and p["geom"] == "estimate" for p in a)
