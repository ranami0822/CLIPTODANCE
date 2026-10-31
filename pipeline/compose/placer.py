"""配置の担当。Clairic が提案する(R10)。規則だけの Placer は、テストと比較の基準用(縮退モードとしては使わない)。"""
from __future__ import annotations

import io
import re
from typing import Protocol

from PIL import Image

from adapters.base import ApiError, EstimateField, Estimator, FieldKind

from .geometry import Limits
from .types import CanvasInfo, Instance, Placement


class Placer(Protocol):
    name: str

    def place(self, text: str, instances: list[Instance], canvas: CanvasInfo, base_png: bytes) -> list[Placement]: ...


class RuleBasedPlacer:
    """決まった順に、画面の幅を使い分けるだけの配置。テストと、Clairic の提案との比較の基準用。"""
    name = "rules"
    camera_height_m = 1.6
    SLOTS = (0.5, 0.28, 0.72, 0.14, 0.86, 0.4, 0.6, 0.2, 0.8)

    def place(self, text, instances, canvas, base_png):
        out: list[Placement] = []
        # 主役を先に(中央の枠から)。同じ役割の中では id の順
        order = {id(inst): n for n, inst in enumerate(sorted(instances, key=lambda i: (i.spec.role != "main", i.id)))}
        for inst in instances:
            n = order[id(inst)]
            if inst.airborne:
                out.append(Placement(x=self.SLOTS[(n + 2) % len(self.SLOTS)], y=0.16 + 0.07 * (n % 3)))
            else:
                y = 0.90 if inst.spec.role == "main" else 0.80 + 0.03 * (n % 3)
                if inst.height_m and inst.height_m > 0:
                    # 背の高い物(木・塔)は、最初から奥に置く。足元が地平線から離れるほど大きく写る(R5)ので、高さが画面の約 35% に収まる位置まで
                    y = min(y, canvas.horizon_y + 0.35 * self.camera_height_m / inst.height_m)
                out.append(Placement(x=self.SLOTS[n % len(self.SLOTS)], y=y))
        return out


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _downscale_png(png: bytes, max_side: int = 768) -> bytes:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class ClairicPlacer:
    """Clairic に、要素ごとの x・足元の y・倍率・回転を estimate 型で尋ねる(値と 80% 範囲が返る)。
    画像(土台)と文章(要望・座標の規約)を渡す。Clairic が画像から自然な位置を出せるかは未検証(設計書 M3 の実験)。"""
    name = "clairic"

    def __init__(self, estimator: Estimator, limits: Limits | None = None):
        self.estimator = estimator
        self.limits = limits or Limits()

    def place(self, text, instances, canvas, base_png):
        lim, hy = self.limits, canvas.horizon_y
        fields: list[EstimateField] = []
        lines = []
        for inst in instances:
            sid = _safe(inst.id)
            kind = "floating in the sky (centre point)" if inst.airborne else "standing on the ground (point where its feet touch the ground)"
            lines.append(f"- {sid}: a {inst.spec.name}, {kind}, role={inst.spec.role}")
            y_lo, y_hi = (lim.sky_margin, max(lim.sky_margin + 0.05, hy - lim.sky_margin)) if inst.airborne else (min(hy + lim.min_ground_gap, 0.99), 1.0)
            fields += [
                EstimateField(f"{sid}_x", FieldKind.ESTIMATE, f"Horizontal position of the {inst.spec.name} as a fraction of the image width (0 = left edge, 1 = right edge).",
                              minimum=lim.margin_x, maximum=1 - lim.margin_x, step=0.01),
                EstimateField(f"{sid}_y", FieldKind.ESTIMATE, f"Vertical position of the {inst.spec.name} as a fraction of the image height (0 = top edge, 1 = bottom edge). "
                              + ("It must stay in the sky, above the horizon." if inst.airborne else "It must stay on the ground, below the horizon; lower in the image means closer to the viewer."),
                              minimum=y_lo, maximum=y_hi, step=0.01),
                EstimateField(f"{sid}_scale", FieldKind.ESTIMATE, f"Size multiplier of the {inst.spec.name} relative to a realistic size at that position (1 = realistic).",
                              minimum=lim.scale_mult_range[0], maximum=lim.scale_mult_range[1], step=0.05, log_scale=True),
                EstimateField(f"{sid}_rot", FieldKind.ESTIMATE, f"Tilt of the {inst.spec.name} in degrees (clockwise positive, 0 = upright).",
                              minimum=lim.rotation_range_deg[0], maximum=lim.rotation_range_deg[1], step=1.0),
            ]
        state = (
            f"Task: place cut-out objects into a landscape collage. Request: {text}\n"
            f"The attached image is the empty landscape canvas (landscape 3:2). The horizon is at y = {hy:.2f} (0 = top, 1 = bottom): "
            f"the sky is above it and the ground is below it.\nObjects to place:\n" + "\n".join(lines)
            + "\nChoose natural, balanced positions; avoid stacking objects on top of each other."
        )
        resp = self.estimator.estimate(state, fields, images=[_downscale_png(base_png)])
        def v(name: str) -> float:
            val = resp.values[name].value
            if val is None:
                raise ApiError(f"Clairic が値を返さなかった: {name}")
            return float(val)
        out = []
        for inst in instances:
            sid = _safe(inst.id)
            vx, vy = resp.values[f"{sid}_x"], resp.values[f"{sid}_y"]
            out.append(Placement(x=v(f"{sid}_x"), y=v(f"{sid}_y"), scale_mult=v(f"{sid}_scale"), rotation_deg=v(f"{sid}_rot"),
                                 x_range80=vx.range80, y_range80=vy.range80))
        return out
