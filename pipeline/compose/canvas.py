"""土台の切り出し(R2 / R2a): 3:2 に切り、地平線を下の三分の一に置く。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

ASPECT = 1.5   # 3:2


@dataclass(frozen=True)
class CropPlan:
    box: tuple[int, int, int, int]    # (x0, y0, x1, y1) 元画像上
    horizon_y: float                  # 切り出し後の地平線(0-1)
    deviation: float                  # |horizon_y − target|
    window_ratio: float               # 窓の高さ / 元画像の高さ


def plan_crop(width: int, height: int, horizon_y: float, target: float = 2 / 3, min_crop_ratio: float = 0.6,
              steps: int = 41) -> CropPlan | None:
    """3:2 の窓で、地平線が target に最も近くなる切り出しを選ぶ。窓が大きいほど(解像度を保てるので)好ましい。
    窓の高さは元画像の高さの c 倍(min_crop_ratio ≤ c ≤ 上限)。上限は幅で決まる(窓の幅 = 1.5·c·H ≤ W)。
    条件を満たす窓が作れない(幅が狭すぎるなど)ときは None。"""
    c_max = min(1.0, width / (ASPECT * height))
    if c_max < min_crop_ratio:
        return None
    best = None
    for c in np.linspace(c_max, min_crop_ratio, steps):
        win_h = c * height
        # 窓の上端 top(0 ≤ top ≤ H − win_h)での地平線は (y0·H − top) / win_h
        lo = (horizon_y * height - (height - win_h)) / win_h   # top が最大のとき
        hi = horizon_y * height / win_h                         # top = 0 のとき
        y_best = min(max(target, lo), hi)                       # 範囲の中で target に最も近い値
        dev = abs(y_best - target)
        if not (0.0 <= y_best <= 1.0):
            continue
        cost = max(0.0, dev - 0.01) * 10 + (1 - c) * 0.5
        if best is None or cost < best[0]:
            top = horizon_y * height - y_best * win_h
            best = (cost, c, y_best, dev, top)
    if best is None:
        return None
    _, c, y_best, dev, top = best
    win_h = int(round(c * height))
    win_w = int(round(win_h * ASPECT))
    win_w = min(win_w, width)
    x0 = (width - win_w) // 2
    y0 = int(round(min(max(top, 0), height - win_h)))
    return CropPlan(box=(x0, y0, x0 + win_w, y0 + win_h), horizon_y=float(y_best), deviation=float(dev), window_ratio=float(c))


def render_base(image: Image.Image, plan: CropPlan, canvas_width: int = 1800) -> Image.Image:
    cropped = image.crop(plan.box)
    h = round(canvas_width / ASPECT)
    return cropped.resize((canvas_width, h), Image.LANCZOS)
