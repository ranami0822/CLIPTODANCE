"""水と陸の地図・habitat 補正・逐次配置(pipeline/compose/water.py, habitat.py, placer.py)。"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adapters.base import JudgeResponse, NoulResult, Usage
from adapters.fake import FakeEstimator, FakeJudge
from pipeline.compose.heights import HeightTable
from pipeline.compose.habitat import apply_habitat, foot_ok_for
from pipeline.compose.placer import PlaceContext, SequentialClairicPlacer
from pipeline.compose.render import RenderConfig, render_collage
from pipeline.compose.run import ComposeRequest, compose
from pipeline.compose.types import CanvasInfo, ElementSpec, Instance, Placement
from pipeline.compose.water import WaterMap, estimate_water_map

from pool_helpers import FakeEmbedder
from test_compose_pipeline import HEIGHTS, add_image, build_pool

HORIZON = 0.5


class RectJudge:
    """送られた画像から赤い枠の位置を読み、x < 0.5 かつ y > 0.75 のマスだけを「水」と答える。尋ねられた回数と中身を記録する。"""
    model_id = "rect"

    def __init__(self):
        self.calls = []
        from adapters.base import JudgeCapabilities
        self.capabilities = JudgeCapabilities(max_questions=64, max_images=4)

    def ask(self, questions, images=(), state=""):
        assert len(questions) == 1 and len(images) == 1
        arr = np.asarray(Image.open(io.BytesIO(images[0])).convert("RGB")).astype(int)
        red = (arr[..., 0] > 240) & (arr[..., 1] < 30) & (arr[..., 2] < 30)
        ys, xs = np.nonzero(red)
        cx, cy = xs.mean() / arr.shape[1], ys.mean() / arr.shape[0]
        self.calls.append((round(cx, 2), round(cy, 2)))
        p = 1.0 if (cx < 0.5 and cy > 0.75) else 0.0
        return JudgeResponse(results=(NoulResult(question_id=questions[0].id, p_yes=p),), usage=Usage(api_calls=1, neurons=1.0), model_id=self.model_id)


def wm(rows):
    """文字の絵(W = 水)から WaterMap を作る。"""
    p = np.array([[1.0 if ch == "W" else 0.0 for ch in r] for r in rows])
    return WaterMap(p=p, horizon_y=HORIZON)


# 4x4: 下の 2 行の左半分が水
MAP = wm(["....", "....", "WW..", "WW.."])


def inst(id, kind="boat", habitat=None, airborne=False, role="prop", intent=None):
    return Instance(id=id, spec=ElementSpec(name=kind, query=kind, kind=kind, role=role, intent=intent), cutout_id="c", height_m=None if airborne else 1.6,
                    height_frac=0.1 if airborne else None, airborne=airborne, aspect=1.0, habitat=habitat)


def canvas_info():
    return CanvasInfo(width=600, height=400, horizon_y=HORIZON, source_image_id="x", crop_box=(0, 0, 10, 10), horizon_deviation=0.0, horizon_source="human")


# ---------------------------------------------------------------- 水の地図

def test_estimate_water_map_asks_one_cell_per_call_and_skips_cells_above_the_horizon():
    j = RectJudge()
    water, info = estimate_water_map(j, Image.new("RGB", (300, 200), (90, 120, 90)), horizon_y=0.5, nx=4, ny=4, workers=1)
    assert water.p.shape == (4, 4)
    assert info["cells_asked"] == len(j.calls) == 8 and info["calls"] == 8       # 地平線(0.5)より下の 2 行 x 4 列だけ
    assert all(cy > 0.5 for _, cy in j.calls)
    assert (water.p[:2] == 0).all()                                                # 尋ねなかったマスは 0
    assert water.p[3].tolist() == [1, 1, 0, 0] and water.p[2].tolist() == [0, 0, 0, 0]   # y=0.625 のマスは水でない、y=0.875 は左半分が水


def test_estimate_water_map_draws_a_red_frame_on_exactly_the_asked_cell():
    j = RectJudge()
    estimate_water_map(j, Image.new("RGB", (400, 400), (10, 10, 10)), horizon_y=0.0, nx=4, ny=2, workers=1)
    want = sorted((round((i + .5) / 4, 2), round((jj + .5) / 2, 2)) for i in range(4) for jj in range(2))
    assert len(j.calls) == 8 and all(abs(c[0] - w[0]) < 0.02 and abs(c[1] - w[1]) < 0.02 for c, w in zip(sorted(j.calls), want))   # 枠の重心(画素の丸めの誤差は許す)


def test_water_map_queries():
    assert MAP.is_water(0.1, 0.9) and MAP.is_water(0.4, 0.6) and not MAP.is_water(0.9, 0.9) and not MAP.is_water(0.1, 0.1)
    assert MAP.bbox() == (0.0, 0.5, 0.5, 1.0)
    assert wm(["...."] * 4).bbox() is None and not wm(["...."] * 4).any_water()
    txt = MAP.describe()
    assert "y 0.50-0.75: x 0.00-0.50" in txt and "y 0.75-1.00: x 0.00-0.50" in txt
    assert "no water" in wm(["...."] * 4).describe()


def test_nearest_returns_a_point_inside_the_nearest_matching_cell():
    x, y = MAP.nearest(0.9, 0.9, water=True)
    assert MAP.is_water(x, y) and x < 0.5 and 0.5 < y <= 1.0
    x, y = MAP.nearest(0.1, 0.9, water=False, min_y=0.52)             # 陸で、y が 0.52 以上のマス
    assert not MAP.is_water(x, y) and y >= 0.52 and x >= 0.5
    assert MAP.nearest(0.5, 0.9, water=True, min_y=0.9) is not None      # マスの一部が min_y 以上ならよい(下の行)
    assert wm(["...."] * 4).nearest(0.5, 0.9, water=True) is None


# ---------------------------------------------------------------- habitat の補正

def test_apply_habitat_moves_boats_into_water_and_land_things_out_of_it_and_logs_it():
    insts = [inst("boat_land", "boat", "water"), inst("boat_ok", "boat", "water"), inst("horse_wet", "horse", "land"),
             inst("horse_ok", "horse", "land"), inst("free", "dog", None), inst("cloud", "cloud", "water", airborne=True)]
    pls = [Placement(0.9, 0.9, 1.3, 5), Placement(0.2, 0.9), Placement(0.2, 0.9), Placement(0.9, 0.9), Placement(0.2, 0.9), Placement(0.2, 0.1)]
    out, log = apply_habitat(insts, pls, MAP)
    assert MAP.is_water(out[0].x, out[0].y) and (out[0].scale_mult, out[0].rotation_deg) == (1.3, 5)    # 位置だけ直す
    assert out[1] == pls[1] and out[3] == pls[3] and out[4] == pls[4] and out[5] == pls[5]
    assert not MAP.is_water(out[2].x, out[2].y)
    assert [e["instance"] for e in log] == ["boat_land", "horse_wet"]
    assert log[0]["moved"][0] == [0.9, 0.9]


def test_apply_habitat_keeps_feet_below_the_horizon_and_reports_when_no_place_exists():
    out, _ = apply_habitat([inst("b", "boat", "water")], [Placement(0.2, 0.3)], MAP)      # 地平線より上 → 水の中の、地平線より下へ
    assert out[0].y > HORIZON and MAP.is_water(out[0].x, out[0].y) and out[0].x == 0.2          # x は水の中のままなので変えず、y だけ地平線の下へ
    dry = wm(["...."] * 4)
    out, log = apply_habitat([inst("b", "boat", "water")], [Placement(0.2, 0.9)], dry)
    assert out[0] == Placement(0.2, 0.9) and log == [{"instance": "b", "habitat": "water", "status": "no_place_found"}]


def test_foot_ok_for_allows_everything_without_a_map_and_follows_habitat_with_one():
    assert foot_ok_for(None) is None
    ok = foot_ok_for(MAP)
    assert ok(inst("b", "boat", "water"), 0.1, 0.9) and not ok(inst("b", "boat", "water"), 0.9, 0.9)
    assert ok(inst("h", "horse", "land"), 0.9, 0.9) and not ok(inst("h", "horse", "land"), 0.1, 0.9)
    assert ok(inst("d", "dog", None), 0.1, 0.9) and ok(inst("c", "cloud", "water", airborne=True), 0.9, 0.1)


def test_render_does_not_nudge_a_foot_to_a_place_the_habitat_forbids():
    def sprite(color):
        a = np.zeros((80, 40, 4), np.uint8); a[..., :3] = color; a[..., 3] = 255
        return a
    base = Image.new("RGB", (600, 400), (128, 128, 128))
    a, b = inst("a", "horse"), inst("b", "horse")
    objs = [(a, sprite((0, 0, 255)), Placement(0.5, 0.85)), (b, sprite((255, 0, 0)), Placement(0.5, 0.86))]
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0)
    _, free = render_collage(base, 2 / 3, objs, cfg)
    assert any("nudged_x" in c for r in free for c in r["corrections"])                      # 制約がなければ横にずらす
    _, fixed = render_collage(base, 2 / 3, objs, cfg, foot_ok=lambda i, x, y: False)
    assert not any("nudged_x" in c for r in fixed for c in r["corrections"])
    assert any("overlap_exceeded" in c for r in fixed for c in r["corrections"])             # ずらせないので、超過をログに残す


# ---------------------------------------------------------------- 逐次配置

def sprites_for(insts):
    out = {}
    for n, i in enumerate(insts):
        a = np.zeros((60, 60, 4), np.uint8); a[..., :3] = (40 * (n + 1), 80, 160); a[..., 3] = 255
        out[i.id] = a
    return out


def context(insts, water=MAP):
    return PlaceContext(base=Image.new("RGB", (600, 400), (120, 130, 110)), sprites=sprites_for(insts), cfg=RenderConfig(color_strength=0.0), water=water)


def test_sequential_placer_asks_one_element_per_call_in_the_order_main_water_ground_sky():
    insts = [inst("cloud", "cloud", airborne=True), inst("tree", "tree", "land"), inst("boat", "boat", "water"), inst("horse", "horse", "land", role="main")]
    est = FakeEstimator(values={"horse_x": 0.7, "horse_y": 0.9})
    pl = SequentialClairicPlacer(est)
    out = pl.place("a quiet scene", insts, canvas_info(), b"", context(insts))
    assert [[f.name.rsplit("_", 1)[0] for f in c["fields"]][0] for c in est.calls] == ["horse", "boat", "tree", "cloud"]
    assert all(len(c["fields"]) == 4 and c["n_images"] == 1 for c in est.calls)
    assert len(out) == 4 and (out[3].x, out[3].y) == (0.7, 0.9)                 # 返りは元の個体の順


def test_sequential_placer_tells_each_call_what_is_already_placed_and_the_intent():
    insts = [inst("horse", "horse", "land", role="main"), inst("dog", "dog", "land", intent="it chases the horse")]
    est = FakeEstimator(values={"horse_x": 0.7, "horse_y": 0.9})
    SequentialClairicPlacer(est).place("req", insts, canvas_info(), b"", context(insts, water=None))
    first, second = est.calls[0]["text"], est.calls[1]["text"]
    assert "Nothing is placed yet" in first and "main subject" in first and "0.1 coordinate grid" in first
    assert "Already placed" in second and "horse (horse): foot at x=0.70, y=0.90" in second and "covers x" in second
    assert "it chases the horse" in second                                       # 要望に書かれた意図をそのまま渡す


def test_sequential_placer_shows_the_composite_so_far_not_the_empty_canvas():
    seen = []

    class SpyEstimator(FakeEstimator):
        def estimate(self, text, fields, images=()):
            seen.append(np.asarray(Image.open(io.BytesIO(images[0])).convert("RGB")).astype(int))
            return super().estimate(text, fields, images)
    insts = [inst("horse", "horse", "land", role="main"), inst("dog", "dog", "land")]
    SequentialClairicPlacer(SpyEstimator(values={"horse_x": 0.5, "horse_y": 0.9}), use_grid=False).place("r", insts, canvas_info(), b"", context(insts, water=None))
    assert np.abs(seen[0] - seen[0][0, 0]).sum() == 0                            # 1 回目: 何も置いていない一色の土台
    assert np.abs(seen[1] - seen[1][0, 0]).sum() > 0                             # 2 回目: 馬が描かれている


def test_sequential_placer_limits_a_boat_to_the_water_and_corrects_the_answer():
    insts = [inst("boat", "boat", "water")]
    est = FakeEstimator(values={"boat_x": 0.9, "boat_y": 0.6})                   # 水のない場所を答えたとする
    out = SequentialClairicPlacer(est).place("r", insts, canvas_info(), b"", context(insts))
    by = {f.name: f for f in est.calls[0]["fields"]}
    assert by["boat_x"].maximum <= 0.5 and by["boat_y"].minimum >= 0.5            # 項目の範囲を水の箱(x 0-0.5, y 0.5-1)に絞る
    assert "Water is found at" in est.calls[0]["text"] and "inside it" in est.calls[0]["text"]
    assert MAP.is_water(out[0].x, out[0].y)                                       # 答えが水の外でも、足元は水に直る


def test_sequential_placer_switches_can_remove_the_image_the_history_and_the_water_hint():
    insts = [inst("horse", "horse", "land", role="main"), inst("boat", "boat", "water")]
    est = FakeEstimator()
    SequentialClairicPlacer(est, use_image=False, use_history=False, use_water_hint=False).place("r", insts, canvas_info(), b"", context(insts))
    assert all(c["n_images"] == 0 for c in est.calls)
    assert all("Already placed" not in c["text"] and "Water is found" not in c["text"] for c in est.calls)
    by = {f.name: f for f in est.calls[1]["fields"]}
    assert by["boat_x"].maximum == pytest.approx(0.97)                            # 範囲を水に絞らない


def test_sequential_placer_needs_the_context():
    with pytest.raises(ValueError):
        SequentialClairicPlacer(FakeEstimator()).place("r", [inst("a")], canvas_info(), b"")


# ---------------------------------------------------------------- 全体の流れ

def test_compose_uses_the_table_habitat_and_keeps_boats_on_water(tmp_path):
    db = build_pool(tmp_path)
    req = ComposeRequest(text="t", base_query="green", canvas_width=600, elements=[ElementSpec(name="b", query="red", kind="boat", role="main")])
    judge = RectJudge()
    from pipeline.compose.placer import RuleBasedPlacer
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), req, HEIGHTS, RenderConfig(color_strength=0.0), judge=judge)
    wmap = np.array(res.log["water_map"]["p"])
    r = res.log["render"][0]
    assert wmap.shape == (6, 8) and res.log["water_map"]["calls"] == len(judge.calls) > 0
    assert wmap[min(5, int(r["applied"]["y"] * 6)), min(7, int(r["applied"]["x"] * 8))] >= 0.5      # 船の足元は水のマス
    assert res.log["habitat_corrections"]                                                           # 規則の配置は中央の手前 → 水へ動かした


def test_compose_without_a_judge_does_not_look_at_water_and_says_so(tmp_path):
    from pipeline.compose.placer import RuleBasedPlacer
    db = build_pool(tmp_path)
    req = ComposeRequest(text="t", base_query="green", canvas_width=600, elements=[ElementSpec(name="b", query="red", kind="boat", role="main")])
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), req, HEIGHTS, RenderConfig(color_strength=0.0))
    assert "water_map" not in res.log and any("水と陸は見ていない" in w for w in res.log["warnings"])


def test_compose_warns_when_boats_are_requested_but_the_base_has_no_water(tmp_path):
    from pipeline.compose.placer import RuleBasedPlacer
    db = build_pool(tmp_path)
    req = ComposeRequest(text="t", base_query="green", canvas_width=600, elements=[ElementSpec(name="b", query="red", kind="boat", role="main")])
    dry = FakeJudge()
    dry.ask = lambda questions, images=(), state="": JudgeResponse(results=(NoulResult(question_id=questions[0].id, p_yes=0.0),), usage=Usage(api_calls=1), model_id="d")
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), req, HEIGHTS, RenderConfig(color_strength=0.0), judge=dry)
    assert any("水が見つからなかった" in w for w in res.log["warnings"])


def test_height_table_declares_boats_on_water_and_ground_animals_on_land():
    assert HEIGHTS.get("boat").habitat == "water"
    assert all(HEIGHTS.get(k).habitat == "land" for k in ("horse", "dog", "tree", "tower"))
    assert HEIGHTS.get("cloud").habitat is None


def test_render_shrinks_instead_of_moving_toward_the_horizon_when_that_would_leave_the_habitat():
    boat = Instance(id="b", spec=ElementSpec(name="boat", query="boat", kind="boat"), cutout_id="c", height_m=None, height_frac=None,
                    airborne=False, aspect=1.0, width_m=30.0, habitat="water")       # 大きすぎる船(幅 30 m)
    a = np.zeros((60, 60, 4), np.uint8); a[..., :3] = (0, 0, 255); a[..., 3] = 255
    base = Image.new("RGB", (600, 400), (128, 128, 128))
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0)
    _, free = render_collage(base, 2 / 3, [(boat, a, Placement(0.5, 0.95))], cfg)
    assert any("moved_toward_horizon" in c for c in free[0]["corrections"])
    _, kept = render_collage(base, 2 / 3, [(boat, a, Placement(0.5, 0.95))], cfg, foot_ok=lambda i, x, y: y > 0.9)
    r = kept[0]
    assert r["applied"]["y"] == 0.95 and r["applied"]["scale_mult"] < 1.0                # 足元は動かさず、縮めた
    assert any("shrunk_for_habitat" in c for c in r["corrections"]) and not any("moved_toward_horizon" in c for c in r["corrections"])
    assert r["height_px"] < free[0]["height_px"] * 3 and r["bbox_px"][2] - r["bbox_px"][0] <= 0.5 * 600 + 2     # max_width_frac に収まる


def test_sequential_placer_shows_the_next_call_the_boat_where_it_really_ended_up_on_water():
    insts = [inst("boatA", "boat", "water", role="main"), inst("boatB", "boat", "water"), inst("horse", "horse", "land")]
    est = FakeEstimator(values={"boatA_x": 0.40, "boatA_y": 0.8, "boatB_x": 0.40, "boatB_y": 0.8})     # 同じ場所に重ねて答える → ずらす必要がある
    SequentialClairicPlacer(est).place("r", insts, canvas_info(), b"", context(insts))
    hist = est.calls[2]["text"]                                                   # 3 回目(馬)の呼び出しに載る、船 B の位置
    line = next(l for l in hist.splitlines() if "boatB" in l)
    x = float(line.split("foot at x=")[1].split(",")[0])
    assert x < 0.5, line                                                          # ずらした先も水の上(x < 0.5)


def test_cached_judge_can_be_used_from_several_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from adapters.base import noul
    from adapters.cache import CachedJudge
    cj = CachedJudge(FakeJudge(), tmp_path / "c.sqlite")
    def go(n):
        return cj.ask([noul("q", f"question {n % 5}")], images=[], state="").results[0].p_yes
    with ThreadPoolExecutor(6) as ex:
        assert list(ex.map(go, range(40))) == [0.5] * 40
