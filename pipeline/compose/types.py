from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ElementSpec:
    """要望に含まれる要素。`kind` は実寸の表(configs/height_table.yaml)のキー。"""
    name: str                 # 要望での呼び名(ログ用)
    query: str                # 切り抜きの検索語(例: "a horse")
    kind: str                 # 実寸の表のキー(例: "horse")
    count: int = 1
    role: str = "prop"        # "main" | "prop"(配置の初期位置と、構図の目安に使う)
    height_m: float | None = None   # 実寸を直接指定するとき(表より優先)


@dataclass(frozen=True)
class Instance:
    """配置する1個。ElementSpec を count 個に展開したもの。"""
    id: str                   # 例 "e0_horse_0"(英数字と _ のみ)
    spec: ElementSpec
    cutout_id: str
    height_m: float | None    # 地上の物の実寸(m)。空中の物は None
    height_frac: float | None  # 空中の物の高さ(画面の高さに対する比)
    airborne: bool
    aspect: float             # 切り抜きの 幅 / 高さ
    width_m: float | None = None   # 幅(長さ)で大きさを決める種類(船など)の実寸(m)。あれば、height_m でなくこちらで決める


@dataclass(frozen=True)
class Placement:
    """1個の配置。x, y は画面の幅・高さに対する 0-1(上が 0)。地上物は足元中央、空中物は中心。"""
    x: float
    y: float
    scale_mult: float = 1.0
    rotation_deg: float = 0.0   # 時計回り
    flip: bool = False
    x_range80: tuple[float, float] | None = None   # Clairic の提案の 80% 範囲(ログ用)
    y_range80: tuple[float, float] | None = None


@dataclass
class CanvasInfo:
    width: int
    height: int
    horizon_y: float          # 切り出し後の地平線(0-1、上が 0)
    source_image_id: str
    crop_box: tuple[int, int, int, int]   # 作業用画像上の窓 (x0, y0, x1, y1)
    horizon_deviation: float  # 目標(下の三分の一)との差
    horizon_source: str       # "human" | "auto"
    notes: list[str] = field(default_factory=list)
