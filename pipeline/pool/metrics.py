"""切り抜きの機械的な指標(設計書「素材プールと事前処理」3、「パッチ選択」3)。

すべて初期案。数値の意味は docstring に書き、足切りの閾値は M2 で素材を見て決める。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb


@dataclass(frozen=True)
class MaskMetrics:
    area_ratio: float   # マスクの面積 / 画像の面積
    centrality: float   # マスクの重心と画像の中心の距離 / 画像の半対角線(0 = 中心、1 = 隅)
    clutter: int        # マスクの周囲の帯と交わるセグメントの数(Sketch2Photo の基準。多いほど背景が複雑)
    edge_score: float   # 縁の滑らかさ(0〜1、1 = 滑らか)。生の輪郭の長さに対する、平滑化した輪郭の長さの比
    border_touch: int   # 外接矩形が画像の端に接している辺の数(0〜4。見切れの目安)


def segment_map(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    """帯の中のセグメント数を数えるための領域分割(Felzenszwalb)。画像ごとに1回作って使い回す。"""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    labels = felzenszwalb(small, scale=100, sigma=0.8, min_size=20)
    if scale < 1:
        labels = cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    return labels


def _perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(c, True) for c in contours))


def edge_smoothness(mask: np.ndarray) -> float:
    raw = _perimeter(mask)
    if raw <= 0:
        return 0.0
    sigma = max(1.0, 0.01 * max(mask.shape))
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) > 0.5
    smooth = _perimeter(blurred)
    return float(min(1.0, smooth / raw))


def compute_metrics(
    mask: np.ndarray, labels: np.ndarray, band_frac: float = 0.03
) -> MaskMetrics:
    """`mask` は bool の (H, W)。`labels` は `segment_map` の結果。`band_frac` は帯の幅(長辺に対する比)。"""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    area_ratio = float(mask.sum() / (h * w))
    cy, cx = ys.mean(), xs.mean()
    centrality = float(np.hypot(cy - h / 2, cx - w / 2) / np.hypot(h / 2, w / 2))

    band_px = max(1, round(band_frac * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    band = dilated & ~mask
    clutter = int(len(np.unique(labels[band]))) if band.any() else 0

    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    border_touch = int(x0 == 0) + int(y0 == 0) + int(x1 == w) + int(y1 == h)
    return MaskMetrics(area_ratio, centrality, clutter, edge_smoothness(mask), border_touch)
