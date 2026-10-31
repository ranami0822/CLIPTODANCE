import numpy as np
import pytest

from pipeline.pool import prefilter as pf


def row(area, thick=0.1, clutter=20, aspect=1.5, edge=0.85, bt=0, cen=0.3, q=0.9, src="sam", det=None, usable=None, claude=None, image="i"):
    return {"area_ratio": area, "thickness": thick, "clutter": clutter, "aspect": aspect, "edge_score": edge, "border_touch": bt, "centrality": cen,
            "mask_quality": q, "source": src, "det_score": det, "usable": usable, "usable_claude": claude, "image_id": image}


def test_effective_label_prefers_human_then_claude_then_none():
    assert pf.effective_label(row(0.01, usable=1, claude=0)) == 1
    assert pf.effective_label(row(0.01, usable=0, claude=1)) == 0
    assert pf.effective_label(row(0.01, claude=1)) == 1
    assert pf.effective_label(row(0.01)) is None


def test_features_separate_detector_and_sam_quality_and_have_fixed_length():
    sam = pf.features(row(0.02, q=0.8))
    det = pf.features(row(0.02, q=0.95, src="maskrcnn", det=0.95))
    assert len(sam) == len(det) == len(pf.FEATURES)
    i = {n: k for k, n in enumerate(pf.FEATURES)}
    assert sam[i["sam_quality"]] == pytest.approx(0.8) and sam[i["from_detector"]] == 0 and sam[i["det_score"]] == 0
    assert det[i["sam_quality"]] == 0 and det[i["from_detector"]] == 1 and det[i["det_score"]] == pytest.approx(0.95)   # 検出のスコアを SAM の品質と混ぜない


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    area = 10 ** rng.uniform(-3, -0.7, n)
    y = (np.log10(area) + rng.normal(0, 0.35, n) > -1.8).astype(float)       # 面積が大きいほど「使える」(ノイズつき)
    rows = [row(a, image=f"img{k % 8}") for k, a in enumerate(area)]
    return rows, y


def test_fit_learns_the_area_signal_and_predicts_probabilities():
    rows, y = synthetic()
    X = np.array([pf.features(r) for r in rows])
    m = pf.fit(X, y)
    p = m.predict(X)
    assert pf.auc(p, y) > 0.85 and 0 <= p.min() and p.max() <= 1
    assert m.w[pf.FEATURES.index("log_area")] > 0.5                           # 面積の係数が正


def test_auc_perfect_reverse_and_degenerate():
    assert pf.auc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1.0])) == 1.0
    assert pf.auc(np.array([4, 3, 2, 1.0]), np.array([0, 0, 1, 1.0])) == 0.0
    assert pf.auc(np.array([1, 2.0]), np.array([1, 1.0])) is None


def test_cross_validation_leaves_each_group_out_and_still_generalizes():
    rows, y = synthetic()
    X = np.array([pf.features(r) for r in rows])
    groups = [r["image_id"] for r in rows]
    pred = pf.cross_validate_by_group(X, y, groups)
    assert pf.auc(pred, y) > 0.8


def test_cross_validation_does_not_leak_the_left_out_group():
    # 各グループの正解が、その特徴量と無関係(グループごとに反転)なら、グループ単位の交差検証の AUC は 0.5 付近になる
    rng = np.random.default_rng(1)
    rows, ys = [], []
    for g in range(10):
        for _ in range(30):
            a = 10 ** rng.uniform(-3, -1)
            rows.append(row(a, image=f"g{g}"))
            ys.append(float((np.log10(a) > -2) ^ (g % 2 == 0)))     # 偶数グループは向きが逆
    X = np.array([pf.features(r) for r in rows]); y = np.array(ys)
    assert pf.auc(pf.cross_validate_by_group(X, y, [r["image_id"] for r in rows]), y) < 0.7


def test_pruning_curve_reports_how_much_can_be_dropped_at_each_recall():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0.0])
    curve = {c["keep_recall"]: c for c in pf.pruning_curve(scores, y)}
    assert curve[1.0]["kept"] == 4 and curve[1.0]["pruned_fraction"] == pytest.approx(0.6) and curve[1.0]["precision_of_kept"] == 0.75
    assert curve[0.8]["kept"] == 2 or curve[0.8]["kept"] == 4          # 3 個中 3 個(ceil(0.8×3)=3)を残すには 4 個


def test_model_roundtrips_through_json():
    rows, y = synthetic(100)
    X = np.array([pf.features(r) for r in rows])
    m = pf.fit(X, y)
    m2 = pf.Model.from_json(m.to_json())
    assert np.allclose(m.predict(X), m2.predict(X))
