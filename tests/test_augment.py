import numpy as np
import pytest

from pipeline.pool.augment import AugmentConfig, augment_image
from pipeline.pool.db import PoolDB
from pipeline.pool.detect import Detection
from pipeline.pool.ingest import IngestConfig, ingest_image

from pool_helpers import FakeEmbedder, FakeSegmenter, H, W, jpeg_bytes, landscape_array, make_item


class FakeDetector:
    name = "fake-detector"

    def __init__(self, dets):
        self.dets, self.calls = dets, 0

    def detect(self, rgb):
        self.calls += 1
        return list(self.dets)


def pool(tmp_path):
    db = PoolDB(tmp_path / "pool.sqlite")
    img, circle, square = landscape_array()
    ingest_image(db, tmp_path, "test", make_item("1"), jpeg_bytes(img), FakeSegmenter(), FakeEmbedder(), IngestConfig())
    return db, circle, square


def rect(y0, y1, x0, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def run(db, tmp_path, dets, cfg=None):
    return augment_image(db, tmp_path, "test", "test-1", FakeDetector(dets), FakeEmbedder(), cfg or AugmentConfig(min_height_px=10))


def test_detection_overlapping_a_sam_cutout_annotates_it_instead_of_adding_a_new_one(tmp_path):
    db, circle, _ = pool(tmp_path)
    before = db.count("cutouts")
    res = run(db, tmp_path, [Detection(circle, "horse", 0.9)])
    assert (res.detections, res.matched, res.added) == (1, 1, 0) and db.count("cutouts") == before
    c = db.get("cutouts", "test-1-c00")
    assert c["det_label"] == "horse" and c["det_score"] == pytest.approx(0.9) and c["source"] == "sam"


def test_new_detection_is_added_as_a_maskrcnn_cutout_with_metrics_and_embedding(tmp_path):
    db, _, _ = pool(tmp_path)
    box = rect(30, 100, 20, 70)       # 既存の切り抜き(円・四角)とは重ならない
    res = run(db, tmp_path, [Detection(box, "person", 0.8)])
    assert (res.matched, res.added) == (0, 1) and res.labels == {"person": 1}
    c = db.get("cutouts", "test-1-d00")
    assert c["source"] == "maskrcnn" and c["det_label"] == "person" and c["mask_quality"] == pytest.approx(0.8)
    assert c["embedding"] is not None and c["embedding_model"] == "fake-color" and 1.0 <= c["aspect"] < 3.0
    assert (c["x0"], c["y0"], c["x1"], c["y1"]) == (18, 28, 72, 102)   # 余白 2 画素
    assert db.get("cutouts", "test-1-c00")["source"] == "sam"          # 既存の切り抜きは SAM のまま


def test_rerunning_does_not_duplicate_and_ids_keep_counting_up(tmp_path):
    db, _, _ = pool(tmp_path)
    first = run(db, tmp_path, [Detection(rect(30, 100, 20, 70), "person", 0.8)])
    again = run(db, tmp_path, [Detection(rect(30, 100, 20, 70), "person", 0.8)])
    assert first.added == 1 and again.added == 0 and again.skipped_dup == 1
    other = run(db, tmp_path, [Detection(rect(20, 90, 200, 260), "boat", 0.7)])
    assert other.added == 1 and sorted(r["id"] for r in db.rows("cutouts") if "-d" in r["id"]) == ["test-1-d00", "test-1-d01"]


def test_too_small_or_thin_detections_are_skipped_and_counted(tmp_path):
    db, _, _ = pool(tmp_path)
    res = run(db, tmp_path, [Detection(rect(10, 14, 10, 14), "person", 0.9),           # 小さすぎる(面積・高さ)
                              Detection(rect(150, 153, 20, 300), "boat", 0.9)],          # 細長い帯
               AugmentConfig(min_height_px=10))
    assert res.added == 0 and res.skipped_small == 2


def test_overlapping_detections_in_one_run_are_added_once(tmp_path):
    db, _, _ = pool(tmp_path)
    a, b = rect(30, 100, 20, 70), rect(31, 101, 21, 71)
    res = run(db, tmp_path, [Detection(a, "person", 0.9), Detection(b, "person", 0.6)])
    assert res.added == 1 and res.skipped_dup + res.matched == 1


def test_old_database_gets_source_backfilled_as_sam(tmp_path):
    db, _, _ = pool(tmp_path)
    assert {r["source"] for r in db.rows("cutouts")} == {"sam"}


def test_reopening_a_database_backfills_a_null_source_as_sam(tmp_path):
    db, _, _ = pool(tmp_path)
    db._db.execute("UPDATE cutouts SET source = NULL"); db.commit(); db.close()
    assert {r["source"] for r in PoolDB(tmp_path / "pool.sqlite").rows("cutouts")} == {"sam"}


def test_height_and_area_limits_are_each_enforced(tmp_path):
    db, _, _ = pool(tmp_path)
    short = Detection(rect(100, 108, 20, 50), "person", 0.9)        # 高さ 8 画素: 面積比・細さ・縦横比は通るが、低すぎる
    res = run(db, tmp_path, [short], AugmentConfig(min_height_px=10))
    assert res.added == 0 and res.skipped_small == 1
    assert run(db, tmp_path, [short], AugmentConfig(min_height_px=5)).added == 1       # 高さの基準を下げれば入る(高さの足切りが原因だった)
    second = tmp_path / "second"
    second.mkdir()
    db2, _, _ = pool(second)
    mid = Detection(rect(20, 50, 200, 230), "person", 0.9)           # 30x30 = 面積比 0.0104
    res2 = augment_image(db2, second, "test", "test-1", FakeDetector([mid]), FakeEmbedder(), AugmentConfig(min_height_px=10, min_area_ratio=0.02))
    assert res2.added == 0 and res2.skipped_small == 1
