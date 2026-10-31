import sqlite3

import numpy as np
import pytest
from PIL import Image

from pipeline.pool.db import PoolDB
from pipeline.pool.frame import MIN_CONTENT_FRAC, edge_profile, frame_depth, remove_frame
from pipeline.pool.horizon import estimate_horizon
from pipeline.pool.ingest import IngestConfig, ingest_image
from pipeline.pool.metrics import compute_metrics, segment_map
from pipeline.pool.segment import Candidate, select_masks

from pool_helpers import H, HORIZON, W, FakeEmbedder, FakeSegmenter, jpeg_bytes, landscape_array, make_item

GOLD = (190, 150, 60)


def with_backdrop(painting, margin, color):
    h, w = painting.shape[:2]
    out = np.zeros((h + 2 * margin, w + 2 * margin, 3), np.uint8)
    out[:] = color
    out[margin:margin + h, margin:margin + w] = painting
    return out


def with_frame(painting, thickness, ridge=2):
    """金色の額縁(ridge 画素ごとに明暗が交互になる縞つき)を絵の周りにつける。"""
    h, w = painting.shape[:2]
    out = np.zeros((h + 2 * thickness, w + 2 * thickness, 3), np.uint8)
    for d in range(thickness):  # 外側の輪から内側の輪へ
        color = (225, 185, 90) if (d // ridge) % 2 else GOLD
        out[d:out.shape[0] - d, d:out.shape[1] - d] = color  # 内側は次の輪が塗り直す
    out[thickness:thickness + h, thickness:thickness + w] = painting
    return out


class EmptySegmenter:
    """切り抜きを作らない(額縁の切り出しだけを確かめるテスト用。マスクの大きさを合わせなくてよい)。"""
    name = "empty"

    def segment(self, rgb):
        return []


def box_error(box, expected):
    return max(abs(a - b) for a, b in zip(box, expected))


def test_black_backdrop_is_cropped_to_the_painting():
    painting, _, _ = landscape_array()
    r = remove_frame(with_backdrop(painting, 30, (3, 3, 3)))
    assert box_error(r.box, (30, 30, 30 + W, 30 + H)) <= 2 and not r.fallback


def test_white_margin_is_cropped():
    painting, _, _ = landscape_array()
    r = remove_frame(with_backdrop(painting, 25, (250, 250, 250)))
    assert box_error(r.box, (25, 25, 25 + W, 25 + H)) <= 2


def test_flat_dark_gray_backdrop_is_cropped():
    painting, _, _ = landscape_array()
    r = remove_frame(with_backdrop(painting, 30, (42, 42, 42)))  # 黒ではなく濃い灰色(met-436451 の型)
    assert box_error(r.box, (30, 30, 30 + W, 30 + H)) <= 2


def test_gold_frame_inside_black_backdrop_is_removed_to_the_inner_edge():
    painting, _, _ = landscape_array()
    framed = with_backdrop(with_frame(painting, 14), 20, (3, 3, 3))
    r = remove_frame(framed)
    off = 20 + 14
    assert box_error(r.box, (off, off, off + W, off + H)) <= 3
    assert r.trimmed["top"] > 0  # 余白の外ではなく、額縁の分を削った


def test_arched_painting_keeps_black_corners_as_backdrop_mask():
    painting, _, _ = landscape_array()
    arched = painting.copy()
    yy, xx = np.mgrid[:H, :W]
    corner = ((xx < 50) & (yy < 40 - xx * 0.8)) | ((xx > W - 50) & (yy < 40 - (W - xx) * 0.8))
    arched[corner] = (0, 0, 0)
    r = remove_frame(with_backdrop(arched, 20, (0, 0, 0)))
    assert box_error(r.box, (20, 20, 20 + W, 20 + H)) <= 3
    assert 0.005 < r.backdrop.mean() < 0.2  # 四隅が余白として残る
    assert r.backdrop[0, 0] and not r.backdrop[H // 2, W // 2]


def test_frameless_busy_image_is_left_untouched():
    rng = np.random.default_rng(3)
    base = rng.integers(40, 220, (200, 300, 3), dtype=np.uint8)
    r = remove_frame(base)
    assert r.box == (0, 0, 300, 200) and not r.fallback


def test_low_horizon_in_a_frameless_painting_is_not_mistaken_for_a_frame():
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (70, 130, 60)
    img[:14] = (120, 170, 235)  # 上端から 4.7% のところに強い水平線
    img += np.random.default_rng(1).integers(0, 6, img.shape, dtype=np.uint8)
    assert remove_frame(img).box == (0, 0, 400, 300)


def test_nearly_all_dark_image_falls_back_to_the_whole_image():
    dark = np.full((200, 300, 3), 12, np.uint8)
    dark[90:110, 140:160] = (200, 200, 200)  # 余白と誤認すると内容が極端に小さくなる
    r = remove_frame(dark)
    assert r.fallback and r.box == (0, 0, 300, 200)
    assert (r.box[2] - r.box[0]) * (r.box[3] - r.box[1]) >= MIN_CONTENT_FRAC * 300 * 200


def test_frame_depth_follows_multiple_ridges_but_stops_at_a_long_gap():
    profile = np.array([30, 25, 4, 3, 20, 22, 3, 2, 18, 3] + [2] * 30 + [40, 40], np.float32)
    assert frame_depth(profile, max_gap=4) == 8 - 1      # 三重の縁取りの最も内側(8 番目の層)の手前まで。その先の長い隙間の向こうの線は数えない
    isolated = np.array([30, 25] + [2] * 20 + [40, 40], np.float32)
    assert frame_depth(isolated, max_gap=4) == 0         # 絵の縁そのものの染み出し(短い層)と、離れた強い線は額縁に数えない
    inner_peak = np.array([5, 8, 20, 30, 22, 9, 3, 2, 2, 2], np.float32)  # 境界の山が内側(3 番目)にある
    assert frame_depth(inner_peak, max_gap=4) == 5 - 1   # 層の端(5)の 1 つ手前まで
    assert frame_depth(np.full(30, 2.0, np.float32), max_gap=4) == 0


def test_edge_profile_is_high_across_a_full_length_boundary_and_low_for_noise():
    img = np.zeros((100, 200), np.float32)
    img[:20] = 200.0  # 上 20 行に明るい帯 → 20 行目に全幅の境界
    p = edge_profile(img, np.ones_like(img, bool), 40)
    assert p[19] > 50 and p[30] < 1
    noisy = np.random.default_rng(0).normal(100, 3, (100, 200)).astype(np.float32)
    assert edge_profile(noisy, np.ones_like(noisy, bool), 40).max() < 8


# ---------------------------------------------------------------- 細長いマスクと余白

def mask_of(rows, cols):
    m = np.zeros((H, W), bool)
    m[rows, cols] = True
    return m


def test_thin_masks_are_dropped_but_compact_ones_stay():
    img, circle, square = landscape_array()
    labels = segment_map(img)
    bar = mask_of(slice(100, 103), slice(10, 340))   # 額縁の帯・水平線のような細長い帯
    post = mask_of(slice(20, 200), slice(180, 184))  # 柱
    cands = [Candidate(m, 0.9, compute_metrics(m, labels)) for m in (circle, square, bar, post)]
    assert cands[2].metrics.aspect > 50 and cands[2].metrics.thickness < 0.02
    kept = select_masks(cands, min_area_ratio=0.0005)
    assert len(kept) == 2 and all(c.metrics.aspect < 2 for c in kept)


def test_masks_mostly_on_the_backdrop_are_dropped():
    img, circle, square = landscape_array()
    backdrop = np.zeros((H, W), bool)
    backdrop[:, :150] = True  # 左側が余白
    labels = segment_map(img)
    c_in = Candidate(circle, 0.9, compute_metrics(circle, labels, backdrop=backdrop))      # 円は x≈90〜150 の大半が余白
    c_out = Candidate(square, 0.9, compute_metrics(square, labels, backdrop=backdrop))
    assert c_in.metrics.backdrop_frac > 0.9 and c_out.metrics.backdrop_frac == 0
    assert select_masks([c_in, c_out]) == [c_out]


def test_horizon_ignores_black_corners_of_arched_paintings():
    painting, _, _ = landscape_array()
    arched = painting.copy()
    backdrop = np.zeros((H, W), bool)
    backdrop[:60, :120] = True
    backdrop[:60, -120:] = True
    arched[backdrop] = 0
    y, _ = estimate_horizon(arched, backdrop=backdrop)
    assert abs(y - HORIZON) < 0.05


# ---------------------------------------------------------------- 取り込みとの結合

def test_ingest_crops_the_frame_and_records_the_box(tmp_path):
    painting, _, _ = landscape_array()
    framed = with_backdrop(with_frame(painting, 14), 20, (3, 3, 3))
    db = PoolDB(tmp_path / "pool.sqlite")
    ingest_image(db, tmp_path, "test", make_item("f"), jpeg_bytes(framed), EmptySegmenter(), FakeEmbedder(), IngestConfig())
    img = db.get("images", "test-f")
    off = 34
    assert box_error(img["crop_box"], (off, off, off + W, off + H)) <= 3
    assert abs(img["width"] - W) <= 6 and abs(img["height"] - H) <= 6  # 作業用画像は絵の部分だけ
    assert abs(img["horizon_y"] - HORIZON) < 0.06                      # 地平線は、額縁を除いた絵の中での高さ
    assert Image.open(img["orig_path"]).size == framed.shape[1::-1]    # 元画像は額縁つきのまま保存


def test_frame_removal_can_be_turned_off(tmp_path):
    painting, _, _ = landscape_array()
    framed = with_backdrop(painting, 20, (3, 3, 3))
    db = PoolDB(tmp_path / "pool.sqlite")
    ingest_image(db, tmp_path, "test", make_item("g"), jpeg_bytes(framed), EmptySegmenter(), FakeEmbedder(), IngestConfig(remove_frame=False))
    img = db.get("images", "test-g")
    assert img["crop_box"] == [0, 0, framed.shape[1], framed.shape[0]] and img["width"] == framed.shape[1]


def test_cutouts_store_aspect_and_thickness(tmp_path):
    painting, _, _ = landscape_array()
    db = PoolDB(tmp_path / "pool.sqlite")
    ingest_image(db, tmp_path, "test", make_item("h"), jpeg_bytes(painting), FakeSegmenter(), FakeEmbedder(), IngestConfig())
    for c in db.rows("cutouts"):
        assert 1.0 <= c["aspect"] < 2.0 and c["thickness"] > 0.05


def test_old_database_without_new_columns_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE images (id TEXT PRIMARY KEY, pool TEXT NOT NULL, source TEXT NOT NULL, source_id TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE cutouts (id TEXT PRIMARY KEY, image_id TEXT NOT NULL, pool TEXT NOT NULL, cutout_path TEXT NOT NULL, created_at REAL NOT NULL);
        INSERT INTO images VALUES ('a','p','s','1',0);
    """)
    con.commit(); con.close()
    db = PoolDB(path)
    db.update("images", "a", crop_box=[1, 2, 3, 4])
    db.update("cutouts", "none", aspect=1.0)  # 列があれば例外にならない(0 行の更新)
    assert db.get("images", "a")["crop_box"] == [1, 2, 3, 4]


def test_find_black_corners_finds_arch_corners_but_not_dark_flat_paintings():
    from pipeline.pool.frame import find_black_corners
    arch = np.full((200, 300, 3), 120, np.uint8)
    yy, xx = np.mgrid[:200, :300]
    arch[yy < 60 - xx * 0.5] = 0
    assert 0.02 < find_black_corners(arch).mean() < 0.2 and find_black_corners(arch)[0, 0]
    dark_painting = np.full((200, 300, 3), 35, np.uint8)         # 暗いが純黒ではない平らな絵(find_backdrop は余白と誤認する型)
    assert find_black_corners(dark_painting).mean() == 0.0
    black_inside = np.full((200, 300, 3), 120, np.uint8); black_inside[80:120, 100:160] = 0   # 端につながらない黒は拾わない
    assert find_black_corners(black_inside).mean() == 0.0
