import json

import numpy as np
import pytest
from PIL import Image

from adapters.fake import FakeJudge
from pipeline.pool import attrs as attrs_mod
from pipeline.pool.db import PoolDB, blob_to_vec, vec_to_blob
from pipeline.pool.horizon import estimate_horizon
from pipeline.pool.imageops import cutout_rgba, fit_long_side, on_gray
from pipeline.pool.index import VectorIndex
from pipeline.pool.ingest import IngestConfig, ingest_image, judge_pool
from pipeline.pool.metrics import compute_metrics, edge_smoothness, segment_map
from pipeline.pool.segment import Candidate, select_masks
from pipeline.pool.sources.met import MetSource

from pool_helpers import H, HORIZON, W, FakeEmbedder, FakeSegmenter, jpeg_bytes, landscape_array, make_item


# ---------------------------------------------------------------- 指標

def test_metrics_for_a_centered_disc():
    img, _, _ = landscape_array()
    yy, xx = np.mgrid[:H, :W]
    disc = (yy - H // 2) ** 2 + (xx - W // 2) ** 2 <= 40 ** 2
    m = compute_metrics(disc, segment_map(img))
    assert m.area_ratio == pytest.approx(np.pi * 40 ** 2 / (H * W), rel=0.05)
    assert m.centrality < 0.02 and m.border_touch == 0
    assert m.edge_score > 0.95  # 円は滑らか


def test_border_touch_counts_edges():
    img, _, _ = landscape_array()
    corner = np.zeros((H, W), bool)
    corner[:30, :30] = True
    assert compute_metrics(corner, segment_map(img)).border_touch == 2  # 上と左
    assert compute_metrics(corner, segment_map(img)).centrality > 0.8


def test_jagged_edge_scores_lower_than_smooth_edge():
    rng = np.random.default_rng(0)
    smooth = np.zeros((H, W), bool)
    smooth[60:180, 80:280] = True
    jagged = smooth.copy()
    noise = rng.random((H, W)) < 0.5
    ring = (jagged ^ np.pad(jagged, 1)[1:-1, 1:-1]) | True  # 全体
    edge = smooth & ~np.roll(smooth, 3, 0) | smooth & ~np.roll(smooth, -3, 0) | smooth & ~np.roll(smooth, 3, 1) | smooth & ~np.roll(smooth, -3, 1)
    jagged[edge & noise] = False
    assert edge_smoothness(jagged) < edge_smoothness(smooth) - 0.1


def test_clutter_counts_segments_in_band():
    plain = np.full((H, W, 3), 120, np.uint8)
    noisy = np.random.default_rng(1).integers(0, 255, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), bool)
    mask[80:160, 120:240] = True
    assert compute_metrics(mask, segment_map(plain)).clutter <= 2
    assert compute_metrics(mask, segment_map(noisy)).clutter > 10


# ---------------------------------------------------------------- 地平線・マスク選別

def test_horizon_estimate_on_synthetic_landscape():
    img, _, _ = landscape_array()
    y, conf = estimate_horizon(img)
    assert abs(y - HORIZON) < 0.05 and conf > 0.5


def test_horizon_on_flat_image_has_no_confidence():
    y, conf = estimate_horizon(np.full((H, W, 3), 128, np.uint8))
    assert conf == 0.0


def test_select_masks_filters_by_area_and_dedupes():
    img, circle, square = landscape_array()
    big = np.zeros((H, W), bool); big[: int(H * 0.8)] = True
    tiny = np.zeros((H, W), bool); tiny[0:2, 0:2] = True
    dup = circle.copy(); dup[170, 90] = False
    kept = select_masks([Candidate(circle, 0.9), Candidate(square, 0.8), Candidate(big, 0.99), Candidate(tiny, 0.99), Candidate(dup, 0.85)])
    assert [c.quality for c in kept] == [0.9, 0.8]
    assert len(select_masks([Candidate(circle, 0.9), Candidate(square, 0.8)], max_cutouts=1)) == 1


def test_cutout_rgba_has_alpha_and_bbox():
    img, circle, _ = landscape_array()
    rgba, (x0, y0, x1, y1) = cutout_rgba(img, circle)
    a = np.asarray(rgba)[..., 3]
    assert rgba.mode == "RGBA" and (x1 - x0, y1 - y0) == rgba.size
    assert a.max() == 255 and a.min() == 0
    assert np.asarray(on_gray(rgba))[0, 0].tolist() == [127, 127, 127]  # 角は透明 → 灰色


def test_fit_long_side_never_upscales():
    im = Image.new("RGB", (100, 50))
    assert fit_long_side(im, 200).size == (100, 50)
    assert fit_long_side(im, 50).size == (50, 25)


# ---------------------------------------------------------------- DB

def test_db_roundtrips_json_and_vectors(tmp_path):
    db = PoolDB(tmp_path / "p.sqlite")
    db.upsert("images", {"id": "a", "pool": "p", "source": "s", "source_id": "1", "attrs": {"x": 1}, "meta": {"k": "日本語"},
                         "embedding": vec_to_blob(np.array([1, 2, 3], np.float32))})
    db.commit()
    row = db.get("images", "a")
    assert row["attrs"] == {"x": 1} and row["meta"] == {"k": "日本語"}
    assert blob_to_vec(row["embedding"]).tolist() == [1, 2, 3]
    db.update("images", "a", attrs={"y": 2}, horizon_checked=1)
    assert db.get("images", "a")["attrs"] == {"y": 2} and db.get("images", "a")["horizon_checked"] == 1


# ---------------------------------------------------------------- 取り込み

def run_ingest(tmp_path, source_id="1", force=False, segmenter=None):
    db = PoolDB(tmp_path / "pool.sqlite")
    seg = segmenter or FakeSegmenter()
    img, _, _ = landscape_array()
    res = ingest_image(db, tmp_path, "test", make_item(source_id), jpeg_bytes(img), seg, FakeEmbedder(), IngestConfig(), force=force)
    return db, seg, res


def test_ingest_creates_image_and_cutout_records(tmp_path):
    db, _, res = run_ingest(tmp_path)
    assert (res.n_candidates, res.n_cutouts) == (5, 2)  # 大きすぎ・小さすぎ・重複が落ちる
    img = db.get("images", "test-1")
    assert img["license"] == "CC0 (test)" and img["source_url"] == "http://example/x" and img["pool"] == "test"
    assert abs(img["horizon_y"] - HORIZON) < 0.05 and img["horizon_checked"] == 0
    assert img["embedding_model"] == "fake-color" and len(blob_to_vec(img["embedding"])) == 3
    cuts = db.rows("cutouts")
    assert [c["id"] for c in cuts] == ["test-1-c00", "test-1-c01"]
    for c in cuts:
        im = Image.open(c["cutout_path"])
        assert im.mode == "RGBA" and im.size == (c["x1"] - c["x0"], c["y1"] - c["y0"])
        assert 0 < c["area_ratio"] < 0.6 and c["attrs"] is None and c["judge_version"] is None


def test_ingest_is_idempotent_and_force_replaces(tmp_path):
    db, seg, _ = run_ingest(tmp_path)
    img, _, _ = landscape_array()
    again = ingest_image(db, tmp_path, "test", make_item("1"), jpeg_bytes(img), seg, FakeEmbedder())
    assert again.skipped and seg.calls == 1
    old = [c["cutout_path"] for c in db.rows("cutouts")]
    db.update("images", "test-1", horizon_checked=1)  # 目視確認済みの印は、--force でも消えることを確かめる(下)
    ingest_image(db, tmp_path, "test", make_item("1"), jpeg_bytes(img), seg, FakeEmbedder(), force=True)
    assert seg.calls == 2 and db.count("cutouts") == 2
    assert db.get("images", "test-1")["horizon_checked"] == 0  # 再取り込みで初期値に戻る
    assert all(__import__("pathlib").Path(p).exists() for p in old)  # 同じ id なので書き直されている


def test_search_finds_the_red_cutout_for_red(tmp_path):
    db, _, _ = run_ingest(tmp_path)
    emb = FakeEmbedder()
    idx = VectorIndex.from_db(db, "cutouts")
    top = idx.search(emb.encode_texts(["red"])[0], k=2)
    assert top[0][0] == "test-1-c00" and top[0][1] > top[1][1]  # c00 = 円(赤)
    assert idx.search(emb.encode_texts(["yellow"])[0], k=1)[0][0] == "test-1-c01"
    base = VectorIndex.from_db(db, "images")
    assert len(base) == 1


# ---------------------------------------------------------------- 判定(Clef)

def test_judge_pool_writes_attrs_and_version(tmp_path):
    db, _, _ = run_ingest(tmp_path)
    judge = FakeJudge(neurons_per_call=2.0)
    out = judge_pool(db, judge, "test")
    assert out["judged"] == 2 and out["remaining"] == 0 and out["neurons"] == 4.0 and out["stopped"] is None
    c = db.get("cutouts", "test-1-c00")
    assert c["judge_version"] == attrs_mod.judge_version(judge)
    assert set(c["attrs"]) == {"watermark_text", "cut_off", "single_object", "flip_unnatural", "facing", "light_dir", "edge_quality"}
    assert c["attrs"]["facing"]["top"] in attrs_mod.FACING and 0 <= c["attrs"]["edge_quality"]["normalized"] <= 1
    assert len(judge.calls) == 2 and len(judge.calls[0]) == 7  # 1 切り抜き = 1 回の呼び出しに 7 問をまとめる


def test_judge_pool_stops_at_budget_and_resumes(tmp_path):
    db, _, _ = run_ingest(tmp_path)
    judge = FakeJudge(neurons_per_call=5.0)
    first = judge_pool(db, judge, "test", max_neurons=5.0)
    assert first["judged"] == 1 and first["stopped"] == "budget" and first["remaining"] == 1
    second = judge_pool(db, judge, "test", max_neurons=100.0)
    assert second["judged"] == 1 and second["remaining"] == 0  # 続きから(判定済みは再判定しない)
    assert len(judge.calls) == 2


def test_judge_pool_rejudges_when_model_changes(tmp_path):
    db, _, _ = run_ingest(tmp_path)
    judge_pool(db, FakeJudge(model_id="model-a"), "test")
    out = judge_pool(db, FakeJudge(model_id="model-b"), "test")
    assert out["judged"] == 2  # model_id が変われば版が変わる


def test_questions_version_changes_with_questions(monkeypatch):
    v1 = attrs_mod.questions_version()
    monkeypatch.setattr(attrs_mod, "FACING", ["front", "left"])
    assert attrs_mod.questions_version() != v1


# ---------------------------------------------------------------- Met

class FakeSession:
    def __init__(self, payload=None):
        self.calls, self.payload = [], payload or {"total": 3, "objectIDs": [1, 2, 3]}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))

        class R:
            status_code = 200
            def raise_for_status(s): pass
            def json(s): return self.payload
        return R()


def test_met_search_uses_paginated_v1_1_endpoint():
    sess = FakeSession()
    total, ids = MetSource(min_interval_s=0, session=sess).search("landscape", department_id=11, medium="Paintings", offset=100, limit=999)
    url, params = sess.calls[0]
    assert url.endswith("/v1.1/search")  # v1/search は 2026-10-01 に廃止
    assert params == {"q": "landscape", "hasImages": "true", "offset": 100, "limit": 500, "departmentId": 11, "medium": "Paintings"}
    assert (total, ids) == (3, [1, 2, 3])


def test_met_to_item_keeps_only_public_domain_paintings_with_images():
    base = {"objectID": 7, "isPublicDomain": True, "primaryImage": "http://img/7.jpg", "classification": "Paintings",
            "title": "T", "artistDisplayName": "A", "objectDate": "1889", "objectURL": "http://met/7"}
    item = MetSource.to_item(base)
    assert item.source_id == "7" and item.license.startswith("CC0") and item.source_url == "http://met/7"
    assert MetSource.to_item({**base, "isPublicDomain": False}) is None
    assert MetSource.to_item({**base, "primaryImage": ""}) is None
    assert MetSource.to_item({**base, "classification": "Vases"}) is None
    assert MetSource.to_item({**base, "classification": "Vases"}, classification=None) is not None
