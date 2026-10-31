"""取り込み済みの画像に、検出器の候補を足す(SAM の結果は作り直さない)。

- 検出が、すでにある切り抜きと重なる(IoU ≥ match_iou)なら、新しい切り抜きにせず、既存の切り抜きにクラス名とスコアを付ける。
- 重ならない検出は、新しい切り抜き(source = 'maskrcnn'、id = <画像 id>-dNN)として足す。小さすぎるもの(低解像度で貼ると粗い)は足さない。
- 同じ検出を二度足さない(再実行しても増えない)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .db import PoolDB, vec_to_blob
from .detect import Detector
from .embed import Embedder
from .imageops import cutout_rgba, on_gray
from .metrics import compute_metrics, segment_map


@dataclass
class AugmentConfig:
    match_iou: float = 0.5
    min_area_ratio: float = 0.0005    # 小さな人物(風景のスケール感に要る)も拾う
    min_height_px: int = 40           # 作業用画像で、これより低いものは足さない(拡大すると粗い)
    max_aspect: float = 8.0
    min_thickness: float = 0.01


@dataclass
class AugmentResult:
    image_id: str
    detections: int = 0
    matched: int = 0
    added: int = 0
    skipped_small: int = 0
    skipped_dup: int = 0
    labels: dict = field(default_factory=dict)


def cutout_mask(row: dict, shape: tuple[int, int]) -> np.ndarray:
    """既存の切り抜き(PNG のアルファと外接矩形)から、作業用画像上のマスクを作る。"""
    a = np.asarray(Image.open(row["cutout_path"]).convert("RGBA"))[..., 3] > 127
    m = np.zeros(shape, bool)
    m[row["y0"]:row["y1"], row["x0"]:row["x1"]] = a
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def augment_image(db: PoolDB, root: Path, pool: str, image_id: str, detector: Detector, embedder: Embedder,
                  cfg: AugmentConfig = AugmentConfig()) -> AugmentResult:
    img = db.get("images", image_id)
    rgb = np.asarray(Image.open(img["work_path"]).convert("RGB"))
    shape = rgb.shape[:2]
    res = AugmentResult(image_id)
    existing = db.rows("cutouts", "image_id = ?", (image_id,))
    masks = [(r, cutout_mask(r, shape)) for r in existing]
    dets = detector.detect(rgb)
    res.detections = len(dets)
    labels_map = None
    used = [int(m.group(1)) for r in existing if (m := re.search(r"-d(\d+)$", r["id"]))]
    next_idx = max(used, default=-1) + 1
    new_images, new_rows = [], []
    for d in sorted(dets, key=lambda d: -d.score):
        best = max(((_iou(d.mask, m), r) for r, m in masks), key=lambda t: t[0], default=(0.0, None))
        if best[0] >= cfg.match_iou:
            r = best[1]
            if r["source"] == "maskrcnn" and r["det_label"] == d.label:
                res.skipped_dup += 1
                continue
            if r["det_score"] is None or d.score > r["det_score"]:
                db.update("cutouts", r["id"], det_label=d.label, det_score=d.score)
            res.matched += 1
            res.labels[d.label] = res.labels.get(d.label, 0) + 1
            continue
        ys, xs = np.nonzero(d.mask)
        if xs.size == 0 or d.mask.mean() < cfg.min_area_ratio or (ys.max() - ys.min() + 1) < cfg.min_height_px:
            res.skipped_small += 1
            continue
        if labels_map is None:
            labels_map = segment_map(rgb)
        m = compute_metrics(d.mask, labels_map)
        if m.aspect > cfg.max_aspect or m.thickness < cfg.min_thickness:
            res.skipped_small += 1
            continue
        rgba, (x0, y0, x1, y1) = cutout_rgba(rgb, d.mask)
        cid = f"{image_id}-d{next_idx:02d}"
        next_idx += 1
        path = Path(root) / "cutouts" / f"{cid}.png"
        rgba.save(path)
        new_images.append(on_gray(rgba))
        new_rows.append({"id": cid, "image_id": image_id, "pool": pool, "cutout_path": str(path), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "area_ratio": m.area_ratio, "centrality": m.centrality, "clutter": m.clutter, "edge_score": m.edge_score,
                         "border_touch": m.border_touch, "aspect": m.aspect, "thickness": m.thickness, "mask_quality": d.score,
                         "source": "maskrcnn", "det_label": d.label, "det_score": d.score, "attrs": None, "judge_version": None})
        masks.append((new_rows[-1], d.mask))   # 同じ画像の、あとの検出との重複を見るため
        res.added += 1
        res.labels[d.label] = res.labels.get(d.label, 0) + 1
    if new_images:
        vecs = embedder.encode_images(new_images)
        for row, v in zip(new_rows, vecs):
            row["embedding"], row["embedding_model"] = vec_to_blob(v), embedder.name
            db.upsert("cutouts", row)
    db.commit()
    return res
