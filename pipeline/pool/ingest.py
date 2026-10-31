"""取り込みの本体: 1 枚の元画像 → 作業用画像 → 地平線 → 切り抜き → 指標 → 埋め込み → DB。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import attrs as attrs_mod
from .db import PoolDB, vec_to_blob
from .embed import Embedder
from .frame import remove_frame
from .horizon import estimate_horizon
from .imageops import cutout_rgba, fit_long_side, load_rgb, on_gray, png_bytes, sha256_bytes
from .metrics import compute_metrics, segment_map
from .segment import Segmenter, select_masks
from .sources.met import SourceItem


@dataclass
class IngestConfig:
    work_max_side: int = 2048       # 作業用画像の長辺。SAM と切り抜きはこの大きさで行う
    band_frac: float = 0.03         # clutter の帯の幅(長辺に対する比)
    min_area_ratio: float = 0.004
    max_area_ratio: float = 0.6
    dedupe_iou: float = 0.85
    max_cutouts: int = 12
    max_aspect: float = 8.0          # 細長いマスク(額縁の帯・柱・水平線)を落とす
    min_thickness: float = 0.02
    max_backdrop_frac: float = 0.2   # 余白(アーチの四隅など)にかかるマスクを落とす
    remove_frame: bool = True


@dataclass
class IngestResult:
    image_id: str
    n_candidates: int = 0
    n_cutouts: int = 0
    seconds: dict = field(default_factory=dict)
    skipped: bool = False


def ingest_image(
    db: PoolDB,
    root: Path,
    pool: str,
    item: SourceItem,
    image_bytes: bytes,
    segmenter: Segmenter,
    embedder: Embedder,
    cfg: IngestConfig = IngestConfig(),
    force: bool = False,
) -> IngestResult:
    image_id = f"{item.source}-{item.source_id}"
    if db.get("images", image_id) is not None and not force:
        return IngestResult(image_id, skipped=True)

    t = {}
    clock = time.monotonic
    root = Path(root)
    (root / "originals").mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(exist_ok=True)
    (root / "cutouts").mkdir(exist_ok=True)

    orig_path = root / "originals" / f"{image_id}.jpg"
    orig_path.write_bytes(image_bytes)
    t0 = clock()
    full = load_rgb(image_bytes)
    frame = remove_frame(np.asarray(full)) if cfg.remove_frame else None
    if frame is not None and not frame.fallback:
        full = full.crop(frame.box)
    crop_box = list(frame.box) if frame is not None else [0, 0, *full.size]
    work = fit_long_side(full, cfg.work_max_side)
    work_path = root / "work" / f"{image_id}.png"
    work.save(work_path)
    rgb = np.asarray(work)
    t["load"] = clock() - t0

    t0 = clock()
    backdrop = None
    if frame is not None and frame.backdrop.any() and not frame.fallback:
        backdrop = np.asarray(Image.fromarray(frame.backdrop.astype(np.uint8) * 255).resize(work.size, Image.NEAREST)) > 0
    horizon_y, horizon_conf = estimate_horizon(rgb, backdrop=backdrop)
    image_vec = embedder.encode_images([work])[0]
    t["horizon+embed"] = clock() - t0

    for p in db.delete_cutouts_of(image_id):  # 再取り込み(--force)のときは古い切り抜きを消す
        Path(p).unlink(missing_ok=True)

    db.upsert("images", {
        "id": image_id, "pool": pool, "source": item.source, "source_id": item.source_id,
        "source_url": item.source_url, "license": item.license,
        "title": item.title, "artist": item.artist, "date_text": item.date_text,
        "width": work.width, "height": work.height,
        "orig_path": str(orig_path), "work_path": str(work_path), "sha256": sha256_bytes(image_bytes),
        "horizon_y": horizon_y, "horizon_auto_y": horizon_y, "horizon_conf": horizon_conf, "horizon_checked": 0,
        "crop_box": crop_box,
        "embedding": vec_to_blob(image_vec), "embedding_model": embedder.name,
        "attrs": None, "judge_version": None, "meta": item.meta,
    })

    t0 = clock()
    candidates = segmenter.segment(rgb)
    t["segment"] = clock() - t0

    t0 = clock()
    labels = segment_map(rgb)
    for c in candidates:
        c.metrics = compute_metrics(c.mask, labels, cfg.band_frac, backdrop)
    kept = select_masks(candidates, cfg.min_area_ratio, cfg.max_area_ratio, cfg.dedupe_iou, cfg.max_cutouts,
                        cfg.max_aspect, cfg.min_thickness, cfg.max_backdrop_frac)
    t["metrics"] = clock() - t0

    t0 = clock()
    cut_images: list[Image.Image] = []
    rows = []
    for i, c in enumerate(kept):
        rgba, (x0, y0, x1, y1) = cutout_rgba(rgb, c.mask)
        cut_id = f"{image_id}-c{i:02d}"
        path = root / "cutouts" / f"{cut_id}.png"
        rgba.save(path)
        cut_images.append(on_gray(rgba))
        m = c.metrics
        rows.append({
            "id": cut_id, "image_id": image_id, "pool": pool, "cutout_path": str(path),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "area_ratio": m.area_ratio, "centrality": m.centrality, "clutter": m.clutter,
            "edge_score": m.edge_score, "border_touch": m.border_touch, "aspect": m.aspect, "thickness": m.thickness,
            "mask_quality": c.quality,
            "attrs": None, "judge_version": None,
        })
    vecs = embedder.encode_images(cut_images) if cut_images else np.zeros((0, embedder.dim), np.float32)
    for row, v in zip(rows, vecs):
        row["embedding"], row["embedding_model"] = vec_to_blob(v), embedder.name
        db.upsert("cutouts", row)
    db.commit()
    t["cutouts"] = clock() - t0
    return IngestResult(image_id, n_candidates=len(candidates), n_cutouts=len(kept), seconds=t)


def judge_pool(db: PoolDB, judge, pool: str, max_neurons: float | None = None, limit: int | None = None,
               max_side: int = 512) -> dict:
    """まだ判定していない(または質問の版が古い)切り抜きを Clef で判定する。

    `max_neurons` を超えそうになったら止める(次回、続きから再開できる)。
    判定に渡す画像は、灰色の背景に載せて長辺 max_side 以下に縮めたもの(トークン数と消費を抑える)。
    """
    version = attrs_mod.judge_version(judge)
    todo = [r for r in db.rows("cutouts", "pool = ?", (pool,)) if r["judge_version"] != version]
    if limit is not None:
        todo = todo[:limit]
    spent, done, stopped = 0.0, 0, None
    for r in todo:
        if max_neurons is not None and spent >= max_neurons:
            stopped = "budget"
            break
        img = fit_long_side(on_gray(Image.open(r["cutout_path"]).convert("RGBA")), max_side)
        a, usage = attrs_mod.judge_image(judge, png_bytes(img))
        spent += usage.neurons or 0.0
        db.update("cutouts", r["id"], attrs=a, judge_version=version)
        db.commit()
        done += 1
    return {"judged": done, "remaining": len(todo) - done, "neurons": spent, "stopped": stopped, "judge_version": version}
