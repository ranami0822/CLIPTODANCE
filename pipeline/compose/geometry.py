"""遠近とサイズ(R5)と、補正の範囲(R10)。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    camera_height_m: float = 1.6
    scale_mult_range: tuple[float, float] = (0.5, 2.0)
    rotation_range_deg: tuple[float, float] = (-15.0, 15.0)
    min_ground_gap: float = 0.02      # 足元は地平線よりこれ以上下(0-1)
    margin_x: float = 0.03
    sky_margin: float = 0.04


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ground_height_px(height_m: float, y_foot: float, horizon_y: float, canvas_h: int, scale_mult: float = 1.0,
                     camera_height_m: float = 1.6) -> float:
    """R5: 地上の物の画面上の高さ(画素)= (実寸 / カメラ高さ) × (足元の y − 地平線の y) × 画面の高さ × 倍率。
    足元が地平線に近いほど小さく、下にあるほど大きい(Photo Clip Art の遠近の式と同じ形)。"""
    return (height_m / camera_height_m) * max(y_foot - horizon_y, 0.0) * canvas_h * scale_mult
