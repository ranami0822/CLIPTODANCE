import numpy as np
import pytest
from PIL import Image

from pipeline.compose.canvas import ASPECT, plan_crop, render_base
from pipeline.compose.geometry import Limits, ground_height_px
from pipeline.compose.render import RenderConfig, render_collage
from pipeline.compose.types import ElementSpec, Instance, Placement

W, H = 600, 400
HORIZON = 2 / 3


def base_image(color=(128, 128, 128)):
    return Image.new("RGB", (W, H), color)


def sprite(w=40, h=80, color=(255, 0, 0)):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., :3] = color
    a[..., 3] = 255
    return a


def inst(id="e0_person_0", kind="person", height_m=1.7, airborne=False, height_frac=None, aspect=0.5):
    return Instance(id=id, spec=ElementSpec(name=kind, query=kind, kind=kind), cutout_id=f"c-{id}", height_m=None if airborne else height_m,
                    height_frac=height_frac, airborne=airborne, aspect=aspect)


def draw(objs, cfg=None, base=None):
    cfg = cfg or RenderConfig(color_strength=0.0, feather_sigma=0.0)
    return render_collage(base or base_image(), HORIZON, objs, cfg)


def mask_of(img, color, tol=40):
    a = np.asarray(img).astype(int)
    return (np.abs(a - np.array(color)).sum(axis=2) < tol)


# ---------------------------------------------------------------- 土台の切り出し(R2 / R2a)

def test_plan_crop_puts_the_horizon_at_the_lower_third_with_a_3_2_window():
    p = plan_crop(3000, 2000, horizon_y=0.5)
    x0, y0, x1, y1 = p.box
    assert p.deviation < 0.02 and p.window_ratio <= 0.76
    assert abs((x1 - x0) / (y1 - y0) - ASPECT) < 0.01 and 0 <= x0 and x1 <= 3000 and 0 <= y0 and y1 <= 2000
    assert abs((0.5 * 2000 - y0) / (y1 - y0) - p.horizon_y) < 0.01      # 窓の中での地平線の高さと一致する


def test_plan_crop_prefers_the_largest_window_when_the_horizon_is_already_right():
    p = plan_crop(3000, 2000, horizon_y=2 / 3)
    assert p.window_ratio > 0.99 and p.deviation < 0.01


def test_plan_crop_rejects_images_too_narrow_for_a_3_2_window_and_reports_large_deviation():
    assert plan_crop(1000, 2000, 0.5) is None                           # 縦長: 幅が足りない
    low = plan_crop(3000, 2000, horizon_y=0.9)                           # 地平線が低すぎて、下の三分の一には届かない
    assert low is not None and low.deviation > 0.1


def test_render_base_outputs_the_requested_canvas_size():
    img = Image.new("RGB", (3000, 2000), (10, 20, 30))
    out = render_base(img, plan_crop(3000, 2000, 0.5), canvas_width=1800)
    assert out.size == (1800, 1200)


# ---------------------------------------------------------------- サイズ(R5)

def test_ground_height_follows_the_perspective_formula():
    # 人 1.7m、カメラ高さ 1.6m、足元 y=0.9、地平線 2/3、画面の高さ 1200 → 1.0625 × 0.2333 × 1200
    assert ground_height_px(1.7, 0.9, 2 / 3, 1200) == pytest.approx(297.5, abs=0.1)
    assert ground_height_px(1.7, 0.9, 2 / 3, 1200, scale_mult=1.5) == pytest.approx(446.25, abs=0.1)
    assert ground_height_px(1.7, 0.5, 2 / 3, 1200) == 0.0                # 地平線より上には地上の物は立てない
    assert ground_height_px(1.7, 0.8, 2 / 3, 1200) < ground_height_px(1.7, 0.9, 2 / 3, 1200)   # 奥ほど小さい


# ---------------------------------------------------------------- 描画

def test_object_stands_with_its_feet_at_the_requested_point_and_has_the_perspective_height():
    img, log = draw([(inst(), sprite(), Placement(x=0.5, y=0.9))])
    ys, xs = np.nonzero(mask_of(img, (255, 0, 0)))
    expected_h = ground_height_px(1.7, 0.9, HORIZON, H)
    assert abs((ys.max() + 1) - 0.9 * H) <= 1.5 and abs((xs.min() + xs.max()) / 2 - 0.5 * W) <= 1.5
    assert abs((ys.max() - ys.min() + 1) - expected_h) <= 2
    assert log[0]["applied"]["y"] == pytest.approx(0.9, abs=0.002) and log[0]["corrections"] == []


def test_nearer_objects_are_drawn_over_farther_ones_regardless_of_input_order():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0, max_cover=1.0)       # 重なり補正は切る(重ね順だけを見る)
    far = (inst("far", height_m=3.0), sprite(60, 60, (0, 0, 255)), Placement(x=0.5, y=0.78))
    near = (inst("near", height_m=3.0), sprite(60, 60, (255, 0, 0)), Placement(x=0.5, y=0.95))
    for objs in ([far, near], [near, far]):
        img, _ = draw(objs, cfg)
        a = np.asarray(img)
        overlap = mask_of(img, (255, 0, 0))
        assert overlap.any()
        # 手前(赤)の足元の真上は赤、奥(青)は赤の後ろに隠れている
        fy = int(0.95 * H) - 5
        assert tuple(a[fy, W // 2]) == (255, 0, 0)


def test_airborne_objects_go_behind_ground_objects():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0, max_cover=1.0)
    sky_obj = (inst("cloud", "cloud", airborne=True, height_frac=0.6, aspect=1.0), sprite(100, 100, (0, 0, 255)), Placement(x=0.5, y=0.3))
    ground = (inst("tree", "tree", height_m=40.0), sprite(100, 100, (255, 0, 0)), Placement(x=0.5, y=0.7))
    img, log = draw([ground, sky_obj], cfg)
    assert [r["instance"] for r in log] == ["cloud", "tree"]                         # 空中の物が先(奥)に描かれる


def test_airborne_object_is_centered_and_kept_in_the_sky():
    img, log = draw([(inst("bird", "bird", airborne=True, height_frac=0.1, aspect=1.0), sprite(40, 40, (0, 0, 255)), Placement(x=0.4, y=0.99))])
    assert log[0]["applied"]["y"] <= HORIZON - 0.04 + 1e-3 and any("y" in c for c in log[0]["corrections"])   # ログは4桁に丸める


def test_corrections_clamp_scale_rotation_and_ground_position():
    img, log = draw([(inst(), sprite(), Placement(x=0.5, y=0.3, scale_mult=5.0, rotation_deg=40))])
    ap = log[0]["applied"]
    assert ap["scale_mult"] == 2.0 and ap["rotation_deg"] == 15.0 and ap["y"] == pytest.approx(HORIZON + 0.02, abs=0.002)
    keys = {k for c in log[0]["corrections"] for k in c}
    assert {"scale_mult", "rotation_deg", "y"} <= keys


def test_objects_are_kept_inside_the_canvas():
    img, log = draw([(inst(height_m=6.0), sprite(80, 80), Placement(x=0.995, y=0.95))])
    x0, y0, x1, y1 = log[0]["bbox_px"]
    assert 0 <= x0 and x1 <= W and 0 <= y0 and y1 <= H


def test_overlapping_objects_are_nudged_apart_and_the_overlap_limit_is_reported():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0, max_cover=0.3)
    a = (inst("a", height_m=3.0), sprite(60, 60, (0, 0, 255)), Placement(x=0.5, y=0.78))
    b = (inst("b", height_m=3.0), sprite(60, 60, (255, 0, 0)), Placement(x=0.5, y=0.95))
    img, log = draw([a, b], cfg)
    nb = next(r for r in log if r["instance"] == "b")
    assert any("nudged_x" in c for c in nb["corrections"]) and nb["covered_max"] <= 0.3
    assert mask_of(img, (0, 0, 255)).sum() > 0     # 奥の物は(ほとんど)見えている


def test_rotation_keeps_the_feet_anchored():
    img, log = draw([(inst(), sprite(), Placement(x=0.5, y=0.9, rotation_deg=15))])
    m = mask_of(img, (255, 0, 0))
    assert m[int(0.9 * H) - 2, W // 2 - 1: W // 2 + 2].any()          # 足元中央のすぐ上は、回転しても物の内側


def test_flip_mirrors_the_sprite():
    sp = np.zeros((40, 40, 4), np.uint8)
    sp[..., 3] = 255
    sp[:, :20, :3] = (255, 0, 0)    # 左 = 赤
    sp[:, 20:, :3] = (0, 0, 255)    # 右 = 青
    for flip, left in ((False, (255, 0, 0)), (True, (0, 0, 255))):
        img, log = draw([(inst(height_m=3.0, aspect=1.0), sp, Placement(x=0.5, y=0.9, flip=flip))])
        x0, y0, x1, y1 = log[0]["bbox_px"]
        assert tuple(np.asarray(img)[(y0 + y1) // 2, x0 + 3]) == left


def test_feather_leaves_no_dark_fringe():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=1.5)
    img, _ = draw([(inst(), sprite(), Placement(x=0.5, y=0.9))], cfg)
    a = np.asarray(img).astype(int)
    assert a[..., 0].min() >= 127 and a[..., 1].max() <= 129 and a[..., 2].max() <= 129   # 灰色と赤の混合の範囲を出ない(黒い縁がない)


def test_color_transfer_pulls_the_object_toward_its_surroundings_by_strength():
    green = base_image((40, 160, 60))
    objs = lambda: [(inst(height_m=3.0, aspect=1.0), sprite(80, 80, (255, 0, 0)), Placement(x=0.5, y=0.9))]
    center = lambda img: np.asarray(img).astype(int)[int(0.9 * H) - 20, W // 2]
    c0 = center(draw(objs(), RenderConfig(color_strength=0.0, feather_sigma=0.0), green)[0])
    c1 = center(draw(objs(), RenderConfig(color_strength=1.0, feather_sigma=0.0), green)[0])
    cmid = center(draw(objs(), RenderConfig(color_strength=0.5, feather_sigma=0.0), green)[0])
    assert tuple(c0) == (255, 0, 0)
    assert c1[1] > c0[1] + 50 and c1[0] < c0[0] - 50           # 緑の土台に寄る
    assert min(c0[1], c1[1]) <= cmid[1] <= max(c0[1], c1[1])   # 強さが中間なら、結果も中間


def test_positive_rotation_is_clockwise_so_the_top_leans_right():
    img, _ = draw([(inst(height_m=4.0, aspect=0.25), sprite(20, 80), Placement(x=0.5, y=0.95, rotation_deg=15))])
    ys, xs = np.nonzero(mask_of(img, (255, 0, 0)))
    mid = (ys.min() + ys.max()) / 2
    top_x, bottom_x = xs[ys < mid].mean(), xs[ys >= mid].mean()
    assert top_x > bottom_x + 3          # 時計回りに回すと、足元を軸に、頭は右へ倒れる


def test_color_transfer_also_matches_the_contrast_of_the_surroundings():
    sp = np.zeros((80, 80, 4), np.uint8)
    sp[..., 3] = 255
    sp[:, :40, :3] = (60, 60, 60)       # 暗い半分
    sp[:, 40:, :3] = (200, 200, 200)    # 明るい半分(コントラストが大きい)
    flat = base_image((120, 120, 120))  # 周囲は一様(コントラストなし)
    contrast = lambda img: int(np.asarray(img).astype(int)[int(0.9 * H) - 40, W // 2 + 15].sum() - np.asarray(img).astype(int)[int(0.9 * H) - 40, W // 2 - 15].sum())
    objs = lambda: [(inst(height_m=3.0, aspect=1.0), sp, Placement(x=0.5, y=0.9))]
    c0 = contrast(draw(objs(), RenderConfig(color_strength=0.0, feather_sigma=0.0), flat)[0])
    c1 = contrast(draw(objs(), RenderConfig(color_strength=1.0, feather_sigma=0.0), flat)[0])
    assert c0 > 300 and c1 < c0 * 0.7     # 周囲が平らなら、物のコントラストも下がる(標準偏差の比は 0.5 までで切る)


def test_too_tall_objects_move_toward_the_horizon_keeping_the_perspective_formula():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0, max_height_frac=0.5)
    tower = inst("tower", "tower", height_m=15.0, aspect=0.4)
    img, log = draw([(tower, sprite(40, 100), Placement(x=0.5, y=0.95))], cfg)
    r = log[0]
    moved = next(c["moved_toward_horizon"] for c in r["corrections"] if "moved_toward_horizon" in c)
    assert moved[1] < moved[0] == 0.95 and r["applied"]["y"] == pytest.approx(moved[1], abs=0.002)   # 足元が奥へ動いた
    assert r["height_px"] <= 0.5 * H + 1                                                              # 高さが上限に収まる
    assert r["height_px"] == pytest.approx(ground_height_px(15.0, r["applied"]["y"], HORIZON, H), abs=2)   # 動かしたあとも遠近の式のとおり
    assert not any("shrunk_to_fit" in c for c in r["corrections"])


def test_objects_that_fit_are_not_moved():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0, max_height_frac=0.6)
    _, log = draw([(inst(), sprite(), Placement(x=0.5, y=0.9))], cfg)
    assert not any("moved_toward_horizon" in c for c in log[0]["corrections"])


def test_top_overflow_shrinks_around_the_feet_instead_of_pushing_the_feet_down():
    cfg = RenderConfig(color_strength=0.0, feather_sigma=0.0, max_height_frac=5.0)    # 上限を外して、上へのはみ出しだけを見る
    img, log = draw([(inst(height_m=15.0, aspect=0.5), sprite(40, 80), Placement(x=0.5, y=0.8))], cfg)
    r = log[0]
    assert any("shrunk_to_fit" in c for c in r["corrections"]) and r["applied"]["y"] == pytest.approx(0.8, abs=0.002)   # 足元は動かない
    assert r["bbox_px"][1] >= 0
