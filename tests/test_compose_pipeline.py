from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from adapters.base import ApiError
from adapters.fake import FakeEstimator
from pipeline.compose.heights import HeightTable
from pipeline.compose.placer import ClairicPlacer, RuleBasedPlacer
from pipeline.compose.render import RenderConfig
from pipeline.compose.run import ComposeRequest, compose
from pipeline.compose.select import approved_single, select_base, select_cutouts
from pipeline.compose.types import CanvasInfo, ElementSpec, Instance
from pipeline.pool.db import PoolDB, vec_to_blob
from pipeline.pool.ingest import IngestConfig, ingest_image

from pool_helpers import FakeEmbedder, FakeSegmenter, jpeg_bytes, landscape_array, make_item

HEIGHTS = HeightTable.load(Path(__file__).parent.parent / "configs" / "height_table.yaml")


def build_pool(tmp_path, label="human"):
    """合成の風景 1 枚(円=赤、四角=黄の切り抜きが 2 つできる)を取り込み、印を付ける。"""
    db = PoolDB(tmp_path / "pool.sqlite")
    img, _, _ = landscape_array()
    ingest_image(db, tmp_path, "test", make_item("1"), jpeg_bytes(img), FakeSegmenter(), FakeEmbedder(), IngestConfig())
    for c in db.rows("cutouts"):
        db.update("cutouts", c["id"], **({"usable": 1} if label == "human" else {"usable_claude": 1}))
    db.commit()
    return db


def add_image(db, id, width, height, horizon_y, checked=0, color=(0, 1, 0)):
    db.upsert("images", {"id": id, "pool": "test", "source": "s", "source_id": id, "width": width, "height": height, "work_path": "-",
                         "horizon_y": horizon_y, "horizon_auto_y": horizon_y, "horizon_conf": 0.5, "horizon_checked": checked,
                         "embedding": vec_to_blob(np.array(color, np.float32)), "embedding_model": "fake"})
    db.commit()


def canvas(horizon=2 / 3):
    return CanvasInfo(width=600, height=400, horizon_y=horizon, source_image_id="x", crop_box=(0, 0, 10, 10), horizon_deviation=0.0, horizon_source="human")


def inst(id, kind="horse", role="prop", airborne=False):
    return Instance(id=id, spec=ElementSpec(name=kind, query=kind, kind=kind, role=role), cutout_id="c", height_m=None if airborne else 1.6,
                    height_frac=0.1 if airborne else None, airborne=airborne, aspect=1.0)


# ---------------------------------------------------------------- 素材の選択

def test_approved_single_prefers_human_over_claude_and_ignores_unlabeled():
    assert approved_single({"usable": 1, "usable_claude": 0}) == (True, "human")
    assert approved_single({"usable": 0, "usable_claude": 1}) == (False, "human")     # 人が「使えない」なら、Claude が使えるとしても使わない
    assert approved_single({"usable": None, "usable_claude": 1}) == (True, "claude")
    assert approved_single({"usable": None, "usable_claude": None}) == (False, "none")


def test_select_cutouts_ranks_by_similarity_and_uses_only_approved(tmp_path):
    db = build_pool(tmp_path)
    emb = FakeEmbedder()
    red = select_cutouts(db, "test", emb, "red", k=2)
    assert [c.row["id"] for c in red][0] == "test-1-c00" and red[0].similarity > red[1].similarity   # 円(赤)が先
    assert select_cutouts(db, "test", emb, "yellow", k=1)[0].row["id"] == "test-1-c01"
    assert [c.row["id"] for c in select_cutouts(db, "test", emb, "red", k=2, exclude={"test-1-c00"})] == ["test-1-c01"]
    db.update("cutouts", "test-1-c00", usable=0); db.commit()                                         # 人が使えないと印したものは出ない
    assert "test-1-c00" not in [c.row["id"] for c in select_cutouts(db, "test", emb, "red", k=5)]


def test_select_cutouts_accepts_claude_labels_when_no_human_label(tmp_path):
    db = build_pool(tmp_path, label="claude")
    got = select_cutouts(db, "test", FakeEmbedder(), "red", k=2)
    assert {c.label_source for c in got} == {"claude"}


def test_effective_horizon_priority_human_then_claude_then_auto():
    from pipeline.compose.select import effective_horizon
    base = {"horizon_y": 0.5, "horizon_checked": 0, "horizon_claude_y": None, "horizon_claude_na": None}
    assert effective_horizon(base) == (0.5, "auto")
    assert effective_horizon({**base, "horizon_claude_y": 0.6}) == (0.6, "claude")
    assert effective_horizon({**base, "horizon_claude_na": 1}) is None                      # Claude が「地平線なし」とした
    assert effective_horizon({**base, "horizon_checked": 1, "horizon_y": 0.4, "horizon_claude_y": 0.6}) == (0.4, "human")   # 人の印が最優先
    assert effective_horizon({**base, "horizon_checked": 2, "horizon_claude_y": 0.6}) is None   # 人が「地平線なし」とした
    assert effective_horizon({**base, "horizon_checked": 1, "horizon_y": 0.4, "horizon_claude_na": 1}) == (0.4, "human")   # 人が値を付けたなら Claude の NA より優先


def test_select_base_filters_by_horizon_feasibility_and_ranks_by_deviation(tmp_path):
    db = build_pool(tmp_path)                                  # 360x240、地平線 ≈ 0.6(自動)。3:2 で、窓を詰めれば 2/3 に近づける
    add_image(db, "near", 3000, 2000, 0.5, checked=1)           # 窓を詰めれば 2/3 に届く
    add_image(db, "portrait", 1000, 2000, 0.5)                  # 縦長: 3:2 の窓が作れない
    add_image(db, "low", 3000, 2000, 0.95)                      # 地平線が低すぎて、ずれが大きい
    add_image(db, "nohorizon", 3000, 2000, 0.5, checked=2)      # 地平線のない絵
    ids = [c.row["id"] for c in select_base(db, "test", FakeEmbedder(), "green")]
    assert "portrait" not in ids and "low" not in ids and "nohorizon" not in ids and "near" in ids
    by = {c.row["id"]: c for c in select_base(db, "test", FakeEmbedder(), "green")}
    assert by["near"].horizon_source == "human" and by["near"].plan.deviation < 0.03


# ---------------------------------------------------------------- 配置担当

def test_rule_placer_is_deterministic_and_puts_main_first_near_the_center():
    insts = [inst("b_prop"), inst("a_main", role="main"), inst("c_bird", "bird", airborne=True)]
    a = RuleBasedPlacer().place("t", insts, canvas(), b"")
    b = RuleBasedPlacer().place("t", insts, canvas(), b"")
    assert a == b
    assert a[1].x == 0.5 and a[1].y == 0.90                    # 主役は中央・手前
    assert all(0 < p.x < 1 for p in a) and a[2].y < 2 / 3      # 空中の物は空に置く


def test_clairic_placer_builds_typed_fields_with_ranges_and_maps_values_back():
    est = FakeEstimator(values={"e0_horse_0_x": 0.3, "e0_horse_0_y": 0.88, "e0_horse_0_scale": 1.2, "e0_horse_0_rot": -4})
    insts = [inst("e0_horse_0", role="main"), inst("e1_bird_0", "bird", airborne=True)]
    out = ClairicPlacer(est).place("a horse under a bird", insts, canvas(), jpeg_bytes(landscape_array()[0]))
    call = est.calls[0]
    names = [f.name for f in call["fields"]]
    assert names == ["e0_horse_0_x", "e0_horse_0_y", "e0_horse_0_scale", "e0_horse_0_rot", "e1_bird_0_x", "e1_bird_0_y", "e1_bird_0_scale", "e1_bird_0_rot"]
    by = {f.name: f for f in call["fields"]}
    assert by["e0_horse_0_y"].minimum == pytest.approx(2 / 3 + 0.02) and by["e0_horse_0_y"].maximum == 1.0     # 地上: 地平線の下
    assert by["e1_bird_0_y"].maximum == pytest.approx(2 / 3 - 0.04)                                              # 空中: 空の中
    assert by["e0_horse_0_scale"].log_scale and (by["e0_horse_0_scale"].minimum, by["e0_horse_0_scale"].maximum) == (0.5, 2.0)
    assert (by["e0_horse_0_rot"].minimum, by["e0_horse_0_rot"].maximum) == (-15.0, 15.0)
    assert call["n_images"] == 1 and "a horse under a bird" in call["text"] and "0.67" in call["text"]
    assert (out[0].x, out[0].y, out[0].scale_mult, out[0].rotation_deg) == (0.3, 0.88, 1.2, -4)
    assert out[0].x_range80 is not None and out[0].y_range80 is not None


def test_clairic_placer_raises_when_a_value_is_missing_instead_of_inventing_one():
    class NullEstimator(FakeEstimator):
        def estimate(self, text, fields, images=()):
            resp = super().estimate(text, fields, images)
            first = fields[0].name
            resp.values[first] = type(resp.values[first])(name=first, value=None)
            return resp
    with pytest.raises(ApiError):
        ClairicPlacer(NullEstimator()).place("t", [inst("e0_horse_0")], canvas(), jpeg_bytes(landscape_array()[0]))


# ---------------------------------------------------------------- 全体の流れ

def request(count=1):
    return ComposeRequest(text="a red thing and a yellow thing in a meadow", base_query="green", canvas_width=600, elements=[
        ElementSpec(name="red", query="red", kind="horse", role="main", count=count),
        ElementSpec(name="yellow", query="yellow", kind="dog", role="prop")])


def test_compose_end_to_end_places_every_chosen_cutout_inside_the_canvas(tmp_path):
    db = build_pool(tmp_path)
    add_image(db, "alt", 3000, 2000, 0.5, checked=1)           # 候補は増やすが、work_path がない画像(上位に来ないよう色を遠くする)
    db.update("images", "alt", embedding=vec_to_blob(np.array([1, 0, 0], np.float32))); db.commit()
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), request(), HEIGHTS, RenderConfig(color_strength=0.0))
    assert res.image.size == (600, 400) and res.base.size == (600, 400)
    log = res.log
    assert log["base"]["image_id"] == "test-1" and log["placer"] == "rules"
    assert [c["cutout_id"] for c in log["cutouts"]] == ["test-1-c00", "test-1-c01"]
    assert len(log["render"]) == 2
    for r in log["render"]:
        x0, y0, x1, y1 = r["bbox_px"]
        assert 0 <= x0 < x1 <= 600 and 0 <= y0 < y1 <= 400
    horizon = log["base"]["horizon_in_canvas"]
    assert all(p["y"] > horizon for p in log["placements"])    # 地上の物は地平線の下
    assert any("目視の確認がない" in w for w in log["warnings"])  # 土台の地平線は人の確認がない


def test_compose_warns_when_fewer_candidates_than_requested(tmp_path):
    db = build_pool(tmp_path)
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), request(count=5), HEIGHTS, RenderConfig(color_strength=0.0))
    assert any("候補が" in w and "red" in w for w in res.log["warnings"])


def test_compose_fails_loudly_when_no_base_is_usable(tmp_path):
    db = build_pool(tmp_path)
    db.update("images", "test-1", horizon_checked=2); db.commit()   # 唯一の風景が「地平線なし」
    with pytest.raises(RuntimeError):
        compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), request(), HEIGHTS)


def test_unknown_kind_is_an_error_not_a_guess(tmp_path):
    db = build_pool(tmp_path)
    bad = ComposeRequest(text="t", base_query="green", canvas_width=600, elements=[ElementSpec(name="x", query="red", kind="unicorn")])
    with pytest.raises(KeyError):
        compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), bad, HEIGHTS)


def test_compose_sizes_each_object_from_the_height_table_and_perspective(tmp_path):
    from pipeline.compose.geometry import ground_height_px
    db = build_pool(tmp_path)
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), request(), HEIGHTS, RenderConfig(color_strength=0.0))
    horizon = res.log["base"]["horizon_in_canvas"]
    for r in res.log["render"]:
        mean = HEIGHTS.get(r["kind"]).mean                      # 馬 1.6m、犬 0.5m(表の値)
        expected = ground_height_px(mean, r["applied"]["y"], horizon, 400, r["applied"]["scale_mult"])
        assert abs(r["height_px"] - expected) <= 2, (r["kind"], r["height_px"], expected)


def test_explicit_height_overrides_the_table(tmp_path):
    from pipeline.compose.geometry import ground_height_px
    db = build_pool(tmp_path)
    req = ComposeRequest(text="t", base_query="green", canvas_width=600, elements=[ElementSpec(name="r", query="red", kind="horse", height_m=3.0, role="main")])
    res = compose(db, "test", FakeEmbedder(), RuleBasedPlacer(), req, HEIGHTS, RenderConfig(color_strength=0.0))
    r = res.log["render"][0]
    assert abs(r["height_px"] - ground_height_px(3.0, r["applied"]["y"], res.log["base"]["horizon_in_canvas"], 400, r["applied"]["scale_mult"])) <= 2


def test_rule_placer_puts_tall_objects_farther_so_they_fit():
    from pipeline.compose.geometry import ground_height_px
    tall = Instance(id="e0_tower_0", spec=ElementSpec(name="tower", query="tower", kind="tower"), cutout_id="c", height_m=15.0,
                    height_frac=None, airborne=False, aspect=0.5)
    short = inst("e1_dog_0", "dog")
    out = RuleBasedPlacer().place("t", [tall, short], canvas(), b"")
    assert out[0].y < out[1].y
    assert ground_height_px(15.0, out[0].y, 2 / 3, 400) <= 0.36 * 400


def test_select_base_rejects_windows_that_still_contain_backdrop(tmp_path):
    db = build_pool(tmp_path)
    clean = Image.new("RGB", (900, 600), (120, 160, 200)); clean.save(tmp_path / "clean.png")
    arch = np.full((600, 900, 3), 120, np.uint8); arch[:, :] = (120, 160, 200)
    yy, xx = np.mgrid[:600, :900]
    arch[(yy < 120 - xx * 0.4)] = 0                      # 左上の黒い四隅
    Image.fromarray(arch).save(tmp_path / "arch.png")
    for name, path in (("clean", "clean.png"), ("arch", "arch.png")):
        add_image(db, name, 900, 600, 0.5, checked=1)
        db.update("images", name, work_path=str(tmp_path / path)); db.commit()
    ids = [c.row["id"] for c in select_base(db, "test", FakeEmbedder(), "green")]
    assert "clean" in ids and "arch" not in ids
