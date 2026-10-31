"""切り抜き(SAM)と、マスクの選別。

`Segmenter` は画像から候補マスクを返すだけ。どれを採るかは `select_masks` で決める。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .metrics import MaskMetrics


@dataclass
class Candidate:
    mask: np.ndarray            # bool (H, W)
    quality: float              # 大きいほど良い(SAM では 予測IoU × 安定度)
    metrics: MaskMetrics | None = None


class Segmenter(Protocol):
    name: str

    def segment(self, rgb: np.ndarray) -> list[Candidate]: ...


class SamSegmenter:
    """Segment Anything(Apache 2.0)の自動マスク生成。CPU では 16×16 の格子で 1 枚約 30 秒(ViT-B、2026-10-31 実測)。"""

    def __init__(
        self,
        checkpoint: str | Path = "data/models/sam_vit_b_01ec64.pth",
        model_type: str = "vit_b",
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 200,
        device: str = "cpu",
    ):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device)
        sam.eval()
        self._gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        self.name = f"sam-{model_type}-pps{points_per_side}"

    def segment(self, rgb: np.ndarray) -> list[Candidate]:
        return [
            Candidate(mask=m["segmentation"].astype(bool), quality=float(m["predicted_iou"] * m["stability_score"]))
            for m in self._gen.generate(rgb)
        ]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def select_masks(
    candidates: list[Candidate],
    min_area_ratio: float = 0.004,
    max_area_ratio: float = 0.6,
    dedupe_iou: float = 0.85,
    max_cutouts: int = 12,
    max_aspect: float = 8.0,
    min_thickness: float = 0.02,
    max_backdrop_frac: float = 0.2,
) -> list[Candidate]:
    """足切りし、重なりの大きい重複を落として、品質の高い順に上位だけ残す。数値はすべて初期案。

    足切り: 面積比 / 細長さ(`aspect` が大きい、または `thickness` が小さい = 額縁の帯・柱・水平線のような帯)/
    余白にかかる割合(`backdrop_frac`)。指標(`metrics`)が付いていないマスクは、面積だけで判定する。
    """
    pool = []
    for c in candidates:
        m = c.metrics
        ratio = m.area_ratio if m is not None else float(c.mask.mean())
        if not (min_area_ratio <= ratio <= max_area_ratio):
            continue
        if m is not None and (m.aspect > max_aspect or m.thickness < min_thickness or m.backdrop_frac > max_backdrop_frac):
            continue
        pool.append(c)
    pool.sort(key=lambda c: c.quality, reverse=True)
    kept: list[Candidate] = []
    for c in pool:
        if all(iou(c.mask, k.mask) < dedupe_iou for k in kept):
            kept.append(c)
        if len(kept) >= max_cutouts:
            break
    return kept
