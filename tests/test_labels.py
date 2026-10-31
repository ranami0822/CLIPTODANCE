import numpy as np
import pytest

from pipeline.pool import labels
from pipeline.pool.db import PoolDB
from pipeline.pool.sheet import numbered_cutout_sheets, numbered_horizon_sheet

from PIL import Image


def test_parse_numbers_accepts_ranges_and_fullwidth_marks():
    assert labels.parse_numbers("1,3,5-7") == [1, 3, 5, 6, 7]
    assert labels.parse_numbers(" 2、4 〜 6 ,9") == [2, 4, 5, 6, 9]
    assert labels.parse_numbers("") == []
    with pytest.raises(ValueError):
        labels.parse_numbers("7-3")


def test_parse_horizon_values_checks_range():
    assert labels.parse_horizon_values("3=0.45, 5=0.6") == {3: 0.45, 5: 0.6}
    with pytest.raises(ValueError):
        labels.parse_horizon_values("3=1.5")


def test_auc_perfect_reverse_tie_and_degenerate():
    assert labels.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert labels.auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert labels.auc([1, 1, 1, 1], [False, False, True, True]) == 0.5
    assert labels.auc([1, 2], [True, True]) is None


def make_db(tmp_path, n=6):
    db = PoolDB(tmp_path / "p.sqlite")
    db.upsert("images", {"id": "i1", "pool": "p", "source": "s", "source_id": "1", "horizon_y": 0.6, "horizon_auto_y": 0.6, "horizon_conf": 0.8, "horizon_checked": 0, "work_path": str(tmp_path / "w.png")})
    Image.new("RGB", (60, 40), (90, 140, 200)).save(tmp_path / "w.png")
    for k in range(n):
        path = tmp_path / f"c{k}.png"
        Image.new("RGBA", (30, 30), (200, 50, 50, 255)).save(path)
        db.upsert("cutouts", {"id": f"i1-c{k:02d}", "image_id": "i1", "pool": "p", "cutout_path": str(path), "area_ratio": 0.01 * (k + 1),
                              "edge_score": 0.5 + 0.05 * k, "clutter": 10 * (n - k), "border_touch": 0, "aspect": 1.2, "thickness": 0.1,
                              "mask_quality": 0.9, "centrality": 0.2})
    db.commit()
    return db


def test_numbering_is_stable_and_import_marks_unlisted_as_not_usable(tmp_path):
    db = make_db(tmp_path)
    nums = labels.numbering(db, "p")
    assert nums == {i + 1: f"i1-c{i:02d}" for i in range(6)}
    out = labels.import_usable(db, "p", [1, 3, 4])
    assert out == {"labeled": 6, "usable": 3, "not_usable": 3}
    assert [r["usable"] for r in db.rows("cutouts")] == [1, 0, 1, 1, 0, 0]


def test_import_usable_upto_leaves_the_rest_unlabeled_and_rejects_bad_numbers(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [2], upto=3)
    assert [r["usable"] for r in db.rows("cutouts")] == [0, 1, 0, None, None, None]
    with pytest.raises(ValueError):
        labels.import_usable(db, "p", [5], upto=3)   # upto の外
    with pytest.raises(ValueError):
        labels.import_usable(db, "p", [99])


def test_features_that_track_the_human_label_get_high_auc(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [4, 5, 6])  # 面積比・縁は k とともに増え、clutter は k とともに減る
    res = labels.evaluate_cutouts(db, "p")
    by = {f["feature"]: f for f in res["features"]}
    assert res["n_labeled"] == 6 and res["n_usable"] == 3
    assert by["area_ratio"]["auc"] == 1.0 and by["area_ratio"]["direction"] == "高いほど使える"
    assert by["clutter"]["auc"] == 0.0 and by["clutter"]["direction"] == "低いほど使える"
    assert res["features"][0]["strength"] == 1.0   # 強い順に並ぶ


def test_clutter_direction_is_reported_when_low_is_better(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1, 2, 3])  # k が小さい = clutter が大きい = 使える
    by = {f["feature"]: f for f in labels.evaluate_cutouts(db, "p")["features"]}
    assert by["clutter"]["auc"] == 1.0 and by["area_ratio"]["auc"] == 0.0
    assert by["area_ratio"]["direction"] == "低いほど使える" and by["area_ratio"]["strength"] == 1.0


def test_clef_rule_agreement_counts(tmp_path):
    db = make_db(tmp_path)
    for k in range(6):
        single = 0.9 if k < 3 else 0.1
        db.update("cutouts", f"i1-c{k:02d}", attrs={"single_object": single, "cut_off": 0.1, "flip_unnatural": 0.1,
                  "edge_quality": {"normalized": 0.5}, "facing": {"top": "left", "probs": {"left": 0.7, "right": 0.3}},
                  "light_dir": {"top": "left", "probs": {"left": 0.5, "right": 0.5}}})
    db.commit()
    labels.import_usable(db, "p", [1, 2, 4])  # 3 個のうち 2 個は Clef の予想どおり、1 個(4)は外れ、3 は逆に外れ
    r = labels.evaluate_cutouts(db, "p")["clef_rule_single_object>0.5"]
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2) and r["accuracy"] == pytest.approx(4 / 6, abs=1e-3)


def test_horizon_import_and_error_are_measured_against_the_auto_value(tmp_path):
    db = make_db(tmp_path)
    db.upsert("images", {"id": "i2", "pool": "p", "source": "s", "source_id": "2", "horizon_y": 0.3, "horizon_auto_y": 0.3, "horizon_conf": 0.5, "horizon_checked": 0})
    db.upsert("images", {"id": "i3", "pool": "p", "source": "s", "source_id": "3", "horizon_y": 0.5, "horizon_auto_y": 0.5, "horizon_conf": 0.9, "horizon_checked": 0})
    db.commit()
    assert labels.import_horizons(db, "p", {1: 0.62, 2: 0.5}, [3]) == {"set": 2, "not_applicable": 1, "accepted_auto": 0}
    i1 = db.get("images", "i1")
    assert i1["horizon_y"] == 0.62 and i1["horizon_auto_y"] == 0.6 and i1["horizon_checked"] == 1  # 自動の値は残る
    res = labels.evaluate_horizons(db, "p")
    assert res["n_labeled"] == 2 and res["n_not_applicable"] == 1  # 対象外の 1 枚は誤差に入れない
    assert res["mean_abs_error"] == pytest.approx(0.11, abs=1e-3) and res["max_error"] == 0.2
    assert res["within_0.05"] == 1 and res["within_0.10"] == 1
    with pytest.raises(ValueError):
        labels.import_horizons(db, "p", {9: 0.5}, [])


def test_accept_rest_marks_unmentioned_images_as_checked_with_the_auto_value(tmp_path):
    db = make_db(tmp_path)
    for k in (2, 3):
        db.upsert("images", {"id": f"i{k}", "pool": "p", "source": "s", "source_id": str(k), "horizon_y": 0.4, "horizon_auto_y": 0.4, "horizon_conf": 0.5, "horizon_checked": 0})
    db.commit()
    out = labels.import_horizons(db, "p", {1: 0.7}, [3], accept_rest=True)
    assert out == {"set": 1, "not_applicable": 1, "accepted_auto": 1}
    assert [(r["horizon_checked"], r["horizon_y"]) for r in db.rows("images")] == [(1, 0.7), (1, 0.4), (2, 0.4)]
    res = labels.evaluate_horizons(db, "p")
    assert res["n_labeled"] == 2 and res["max_error"] == pytest.approx(0.1, abs=1e-3)  # 受け入れた分は誤差 0


def test_numbered_sheets_are_written(tmp_path):
    db = make_db(tmp_path, n=7)
    cuts = [(n, db.get("cutouts", cid)) for n, cid in labels.numbering(db, "p").items()]
    paths = numbered_cutout_sheets(cuts, tmp_path / "out", per_sheet=3, cols=3, thumb=80)
    assert [p.name for p in paths] == ["cutouts_001-003.png", "cutouts_004-006.png", "cutouts_007-007.png"]
    assert all(Image.open(p).size[0] > 0 for p in paths)
    h = numbered_horizon_sheet([(1, db.get("images", "i1"))], tmp_path / "out" / "h.png", thumb=120, cols=1)
    assert h.exists()


# ---------------------------------------------------------------- AI の印(人の印とは別枠)

def test_ai_labels_are_stored_separately_and_never_touch_human_labels(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1, 2], source="human")
    labels.import_usable(db, "p", [5, 6], source="ai")
    rows = db.rows("cutouts")
    assert [r["usable"] for r in rows] == [1, 1, 0, 0, 0, 0]
    assert [r["usable_ai"] for r in rows] == [0, 0, 0, 0, 1, 1]
    db.upsert("images", {"id": "i2", "pool": "p", "source": "s", "source_id": "2", "horizon_y": 0.4, "horizon_auto_y": 0.4, "horizon_conf": 0.5, "horizon_checked": 0})
    db.commit()
    labels.import_horizons(db, "p", {1: 0.7}, [2], source="ai")
    i1, i2 = db.get("images", "i1"), db.get("images", "i2")
    assert i1["horizon_ai_y"] == 0.7 and i1["horizon_checked"] == 0 and i1["horizon_y"] == 0.6   # 人の印・値は動かない
    assert i2["horizon_ai_na"] == 1 and i2["horizon_checked"] == 0


def test_evaluate_uses_the_requested_source(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [4, 5, 6], source="ai")
    assert labels.evaluate_cutouts(db, "p", "human")["n_labeled"] == 0
    ai = labels.evaluate_cutouts(db, "p", "ai")
    assert ai["source"] == "ai" and ai["n_labeled"] == 6 and ai["n_usable"] == 3
    labels.import_horizons(db, "p", {1: 0.7}, [], source="ai")
    h = labels.evaluate_horizons(db, "p", "ai")
    assert h["n_labeled"] == 1 and h["mean_abs_error"] == pytest.approx(0.1, abs=1e-3)
    assert labels.evaluate_horizons(db, "p", "human")["n_labeled"] == 0


def test_compare_labelers_computes_agreement_and_kappa(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1, 2, 3], source="human")     # 人: 1,2,3 が使える
    labels.import_usable(db, "p", [2, 3, 4], source="ai")         # AI: 2,3,4 が使える
    c = labels.compare_labelers(db, "p")["cutouts"]
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 1, 2) and c["n"] == 6
    assert c["agreement"] == pytest.approx(4 / 6, abs=1e-3)
    assert c["kappa"] == pytest.approx(1 / 3, abs=1e-3)          # po=4/6, pe=0.5 → (0.667-0.5)/0.5
    assert c["ai_precision"] == pytest.approx(2 / 3, abs=1e-3) and c["ai_recall"] == pytest.approx(2 / 3, abs=1e-3)


def test_compare_with_only_a_subset_labeled_by_the_human(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1], upto=2, source="human")    # 人は 1〜2 番だけ
    labels.import_usable(db, "p", [1, 2], source="ai")            # AI は全部
    c = labels.compare_labelers(db, "p")["cutouts"]
    assert c["n"] == 2 and c["tp"] == 1 and c["fp"] == 1          # 比べるのは両方が付いている 2 個だけ
    assert labels.compare_labelers(PoolDB(tmp_path / "empty.sqlite"), "p")["cutouts"] == {"n": 0}


def test_compare_horizon_values_and_na(tmp_path):
    db = make_db(tmp_path)
    db.upsert("images", {"id": "i2", "pool": "p", "source": "s", "source_id": "2", "horizon_y": 0.4, "horizon_auto_y": 0.4, "horizon_conf": 0.5, "horizon_checked": 0})
    db.commit()
    labels.import_horizons(db, "p", {1: 0.50}, [2], source="human")
    labels.import_horizons(db, "p", {1: 0.58}, [2], source="ai")
    h = labels.compare_labelers(db, "p")["horizon"]
    assert h["n_values"] == 1 and h["mean_abs_error"] == pytest.approx(0.08, abs=1e-3) and h["within_0.10"] == 1 and h["within_0.05"] == 0
    assert h["na_agreement"] == 1.0


# ---------------------------------------------------------------- 独立した2つの印(単体で使える / 反復して使える)

def test_repeat_is_independent_of_single_and_roles_are_derived_from_both(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1, 2], upto=6)          # 単体: 1, 2
    out = labels.import_repeat(db, "p", [2, 3, 5], upto=6)  # 反復: 2, 3, 5(2 は単体でもある = 馬を並べて群れにする)
    assert out == {"covered": 6, "repeat": 3}
    assert [labels.role_of(r) for r in db.rows("cutouts")] == ["single", "both", "repeat", "none", "repeat", "none"]
    assert [r["repeat"] for r in db.rows("cutouts")] == [0, 1, 1, 0, 1, 0]


def test_role_is_unknown_until_both_marks_exist(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1], upto=3)
    assert [labels.role_of(r) for r in db.rows("cutouts")[:3]] == [None, None, None]   # 反復の印がまだ
    labels.import_repeat(db, "p", [], upto=2)
    assert [labels.role_of(r) for r in db.rows("cutouts")[:3]] == ["single", "none", None]  # 3 番はまだ


def test_import_repeat_rejects_bad_numbers_and_never_touches_the_single_mark(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1], upto=3)
    with pytest.raises(ValueError):
        labels.import_repeat(db, "p", [5], upto=3)
    labels.import_repeat(db, "p", [1, 2], upto=3)
    assert [r["usable"] for r in db.rows("cutouts")[:3]] == [1, 0, 0]


def test_repeat_source_ai_is_separate_from_human(tmp_path):
    db = make_db(tmp_path)
    labels.import_repeat(db, "p", [1], upto=2, source="ai")
    assert [r["repeat_ai"] for r in db.rows("cutouts")[:3]] == [1, 0, None] and all(r["repeat"] is None for r in db.rows("cutouts"))


def test_evaluate_reports_roles_and_repeat_features(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [5, 6])           # 単体: 5, 6
    labels.import_repeat(db, "p", [3, 4, 6])        # 反復: 3, 4, 6(6 は両方)
    res = labels.evaluate_cutouts(db, "p")
    assert res["roles"] == {"both": 1, "single": 1, "repeat": 2, "none": 2, "repeat_unlabeled": 0}
    rep = {f["feature"]: f for f in res["features_repeat"]}
    assert rep["area_ratio"]["auc"] == pytest.approx(7 / 9, abs=1e-3)   # 反復 3,4,6 と非反復 1,2,5 の面積比(番号に比例)の 9 組のうち 7 組で反復が大きい
    any_use = {f["feature"]: f for f in res["features_single_or_repeat"]}
    assert any_use["area_ratio"]["auc"] == 1.0                           # 単体か反復: k=3,4,5,6 / 使えない: 1,2
    assert {f["feature"]: f for f in res["features"]}["area_ratio"]["auc"] == 1.0


def test_compare_repeat_marks_agreement(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [], upto=6, source="human"); labels.import_repeat(db, "p", [1, 2, 3], source="human")   # 人: 反復 1,2,3
    labels.import_usable(db, "p", [], upto=6, source="ai");    labels.import_repeat(db, "p", [2, 3, 4], source="ai")      # AI: 反復 2,3,4
    r = labels.compare_labelers(db, "p")["repeat"]
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2) and r["kappa"] == pytest.approx(1 / 3, abs=1e-3)
    assert r["human_repeat_rate"] == 0.5


# ---------------------------------------------------------------- 出どころは3つ(human / ai = Haiku / claude)

def test_claude_labels_are_separate_from_human_and_haiku(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1], source="human"); labels.import_usable(db, "p", [2], source="ai"); labels.import_usable(db, "p", [3], source="claude")
    labels.import_repeat(db, "p", [4], source="claude")
    rows = db.rows("cutouts")
    assert [r["usable"] for r in rows][:4] == [1, 0, 0, 0] and [r["usable_ai"] for r in rows][:4] == [0, 1, 0, 0]
    assert [r["usable_claude"] for r in rows][:4] == [0, 0, 1, 0] and [r["repeat_claude"] for r in rows][:4] == [0, 0, 0, 1]
    assert all(r["repeat"] is None and r["repeat_ai"] is None for r in rows)


def test_compare_and_horizon_work_for_the_claude_labeler(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [1, 2, 3], source="human")
    labels.import_usable(db, "p", [2, 3, 4], source="claude")
    c = labels.compare_labelers(db, "p", "claude")
    assert c["labeler"] == "claude" and (c["cutouts"]["tp"], c["cutouts"]["fp"], c["cutouts"]["fn"], c["cutouts"]["tn"]) == (2, 1, 1, 2)
    assert labels.compare_labelers(db, "p", "ai")["cutouts"] == {"n": 0}   # Haiku の印はまだない
    labels.import_horizons(db, "p", {1: 0.7}, [], source="claude")
    i1 = db.get("images", "i1")
    assert i1["horizon_claude_y"] == 0.7 and i1["horizon_ai_y"] is None and i1["horizon_checked"] == 0
    assert labels.evaluate_horizons(db, "p", "claude")["n_labeled"] == 1


def test_start_leaves_earlier_numbers_untouched(tmp_path):
    db = make_db(tmp_path)
    labels.import_usable(db, "p", [4], start=3, upto=6, source="claude")
    labels.import_repeat(db, "p", [5], start=3, upto=6, source="claude")
    rows = db.rows("cutouts")
    assert [r["usable_claude"] for r in rows] == [None, None, 0, 1, 0, 0]
    assert [r["repeat_claude"] for r in rows] == [None, None, 0, 0, 1, 0]
    with pytest.raises(ValueError):
        labels.import_usable(db, "p", [2], start=3, source="claude")   # start より前の番号は指定できない


def test_import_marks_labels_only_the_given_numbers_with_both_flags(tmp_path):
    db = make_db(tmp_path)
    out = labels.import_marks(db, "p", [2, 4, 5], single=[2], repeat=[4, 5], source="human")
    assert out == {"labeled": 3, "single": 1, "repeat": 2}
    assert [(r["usable"], r["repeat"]) for r in db.rows("cutouts")] == [(None, None), (1, 0), (None, None), (0, 1), (0, 1), (None, None)]
    with pytest.raises(ValueError):
        labels.import_marks(db, "p", [2], single=[3], repeat=[], source="human")   # universe の外
    with pytest.raises(ValueError):
        labels.import_marks(db, "p", [99], single=[], repeat=[])


def test_import_marks_by_ids_labels_only_the_given_cutouts(tmp_path):
    db = make_db(tmp_path)
    ids = [r["id"] for r in db.rows("cutouts")]
    out = labels.import_marks_by_ids(db, ids[:3], single=[ids[1]], repeat=[ids[2]], source="claude")
    assert out == {"labeled": 3, "single": 1, "repeat": 1}
    rows = db.rows("cutouts")
    assert [(r["usable_claude"], r["repeat_claude"]) for r in rows][:4] == [(0, 0), (1, 0), (0, 1), (None, None)]
    with pytest.raises(ValueError):
        labels.import_marks_by_ids(db, ids[:2], single=[ids[4]], repeat=[], source="claude")      # universe の外
    with pytest.raises(ValueError):
        labels.import_marks_by_ids(db, ["no-such-id"], single=[], repeat=[], source="claude")
