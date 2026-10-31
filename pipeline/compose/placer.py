"""配置の担当。Clairic が提案する(R10)。規則だけの Placer は、テストと比較の基準用(縮退モードとしては使わない)。"""
from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from adapters.base import ApiError, EstimateField, Estimator, FieldKind

from .geometry import Limits
from .habitat import apply_habitat, foot_ok_for
from .render import RenderConfig, render_collage
from .types import CanvasInfo, Instance, Placement
from .water import WaterMap


@dataclass
class PlaceContext:
    """配置の担当に渡す、画像と水の地図など。逐次配置(合成途中の画像を見せる)が使う。"""
    base: Image.Image
    sprites: dict[str, np.ndarray]
    cfg: RenderConfig
    water: WaterMap | None = None


class Placer(Protocol):
    name: str

    def place(self, text: str, instances: list[Instance], canvas: CanvasInfo, base_png: bytes, context: PlaceContext | None = None) -> list[Placement]: ...


class RuleBasedPlacer:
    """決まった順に、画面の幅を使い分けるだけの配置。テストと、Clairic の提案との比較の基準用。"""
    name = "rules"
    camera_height_m = 1.6
    SLOTS = (0.5, 0.28, 0.72, 0.14, 0.86, 0.4, 0.6, 0.2, 0.8)

    def place(self, text, instances, canvas, base_png, context=None):
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
                if inst.width_m and inst.width_m > 0:
                    # 幅(長さ)の大きい物(船)も奥に置く。幅が画面の約 25% に収まる位置まで(画面は 3:2 なので、幅の比を高さの比に直す)
                    y = min(y, canvas.horizon_y + 0.25 * 1.5 * self.camera_height_m / inst.width_m)
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

    def place(self, text, instances, canvas, base_png, context=None):
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


def _grid_overlay(img: Image.Image) -> Image.Image:
    """0.1 ごとの白線と、黄色の座標の数字を重ねる。画像の座標を読み取らせる助け(効くかは実験で確かめる)。"""
    g = img.convert("RGB").copy()
    d = ImageDraw.Draw(g)
    W, H = g.size
    try:
        f = ImageFont.load_default(size=max(10, W // 55))
    except TypeError:           # 古い Pillow
        f = ImageFont.load_default()
    for k in range(1, 10):
        d.line([(k / 10 * W, 0), (k / 10 * W, H)], fill=(255, 255, 255), width=1)
        d.line([(0, k / 10 * H), (W, k / 10 * H)], fill=(255, 255, 255), width=1)
        d.text((k / 10 * W + 2, 2), f"{k / 10:.1f}", fill=(255, 255, 0), font=f)
        d.text((2, k / 10 * H + 2), f"{k / 10:.1f}", fill=(255, 255, 0), font=f)
    return g


def _png_of(img: Image.Image, max_side: int = 768) -> bytes:
    im = img.convert("RGB").copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def default_intent(inst: Instance) -> str:
    """要望に意図が書かれていないときの、置く意図の文。"""
    spec = inst.spec
    if inst.airborne:
        return f"the {spec.name} floats in the sky; keep it away from the other objects and from the horizon"
    parts = []
    if spec.role == "main":
        parts.append("this is the main subject, so make it prominent in the foreground (low in the image), near a third of the width rather than dead centre")
    else:
        parts.append("a supporting element; place it so that it does not compete with the main subject")
    if inst.habitat == "water":
        parts.append("it floats on water, so its foot point must be inside the water area")
    elif inst.habitat == "land":
        parts.append("it stands on dry land, so its foot point must not be in water")
    if inst.width_m or (inst.height_m and inst.height_m > 5):
        parts.append("it is large in real life, so put it far away (close to the horizon) where it looks small")
    return "; ".join(parts)


class SequentialClairicPlacer:
    """Clairic に 1 要素ずつ尋ねる。毎回、合成途中の画像と、置いた物の位置・大きさ・意図を渡す。
    一括で尋ねた場合(ClairicPlacer)に位置が固まり、水と陸も分かっていなかったので、その対策の試作(設計書 M3)。
    `use_image` / `use_grid` / `use_history` / `use_water_hint` は、どれが効くかを実験で切り分けるための入れ替え口。"""
    name = "clairic-seq"

    def __init__(self, estimator: Estimator, limits: Limits | None = None, use_image: bool = True, use_grid: bool = True,
                 use_history: bool = True, use_water_hint: bool = True, order: str = "main_first", seed: int = 0,
                 water_hint_parts: tuple[str, ...] = ("text", "range")):
        if order not in self.ORDERS:
            raise ValueError(f"order は {self.ORDERS} のどれか")
        self.order_mode, self.seed = order, seed
        if not set(water_hint_parts) <= {"text", "range"}:
            raise ValueError("water_hint_parts は 'text'(水の領域の文章)と 'range'(項目の範囲を水域の箱に絞る)の組")
        self.water_hint_parts = tuple(water_hint_parts) if use_water_hint else ()
        self.estimator = estimator
        self.limits = limits or Limits()
        self.use_image, self.use_grid, self.use_history, self.use_water_hint = use_image, use_grid, use_history, use_water_hint
        self.trace: list[dict] = []     # 呼び出しごとの記録(実験の確認用)

    ORDERS = ("main_first", "reverse", "back_to_front", "front_to_back", "random")

    def order(self, instances: list[Instance], canvas: CanvasInfo | None = None) -> list[int]:
        """尋ねる順。どれが良いかは未確認なので、`order` で入れ替えて実験する(設計書 M3)。
        main_first: 主役 → 水の上の物 → 地上の物 → 空中の物(既定。最初に私が決め打ちにした順)。
        reverse: main_first の逆。
        back_to_front: 空中の物、次に地上の物を奥から手前へ(規則の配置の y の小さい順 = 実寸が大きい物や船が先、主役が最後)。画家が奥から描く順。
        front_to_back: back_to_front の逆(手前から)。
        random: 乱数(seed で固定)。"""
        n_ = len(instances)
        if self.order_mode == "random":
            idx = list(range(n_))
            random.Random(self.seed).shuffle(idx)
            return idx
        if self.order_mode in ("back_to_front", "front_to_back"):
            if canvas is None:
                raise ValueError("back_to_front / front_to_back には canvas が要る")
            prior = RuleBasedPlacer().place("", instances, canvas, b"")
            idx = sorted(range(n_), key=lambda n: (not instances[n].airborne, prior[n].y, n))
            return idx if self.order_mode == "back_to_front" else idx[::-1]

        def key(n: int):
            i = instances[n]
            return (i.airborne, i.spec.role != "main", i.habitat != "water", n)
        base = sorted(range(n_), key=key)
        return base[::-1] if self.order_mode == "reverse" else base

    def place(self, text, instances, canvas, base_png, context=None):
        if context is None:
            raise ValueError("SequentialClairicPlacer には context(土台の画像・切り抜き・水の地図)が要る")
        lim, hy = self.limits, canvas.horizon_y
        placed: dict[int, Placement] = {}
        self.trace = []
        for step, n in enumerate(self.order(instances, canvas)):
            inst = instances[n]
            sid = _safe(inst.id)
            water = context.water
            done_idx = [m for m in placed]
            # 合成途中の画像(これまでの配置を、補正のあとの姿で)
            if done_idx:
                partial, rlog = render_collage(context.base, hy, [(instances[m], context.sprites[instances[m].id], placed[m]) for m in done_idx], context.cfg, foot_ok=foot_ok_for(water))
            else:
                partial, rlog = context.base, []
            applied = {r["instance"]: r for r in rlog}
            hist = []
            for m in done_idx:
                r = applied.get(instances[m].id)
                if r and r.get("bbox_px"):
                    Wc, Hc = partial.size
                    bx0, by0, bx1, by1 = r["bbox_px"]
                    ap = r["applied"]
                    hist.append(f"- {instances[m].spec.name} ({_safe(instances[m].id)}): foot at x={ap['x']:.2f}, y={ap['y']:.2f}; "
                                f"covers x {bx0 / Wc:.2f}-{bx1 / Wc:.2f}, y {by0 / Hc:.2f}-{by1 / Hc:.2f}")
            # 範囲(水の上の物は、水のある範囲に絞る)
            x_lo, x_hi = lim.margin_x, 1 - lim.margin_x
            y_lo, y_hi = (lim.sky_margin, max(lim.sky_margin + 0.05, hy - lim.sky_margin)) if inst.airborne else (min(hy + lim.min_ground_gap, 0.99), 1.0)
            water_note = ""
            if self.water_hint_parts and water is not None and not inst.airborne:
                bb = water.bbox()
                if inst.habitat == "water" and bb is not None and "range" in self.water_hint_parts:
                    x_lo, x_hi = max(x_lo, bb[0] + 0.02), min(x_hi, bb[1] - 0.02)
                    y_lo, y_hi = max(y_lo, bb[2] + 0.02), min(y_hi, bb[3] - 0.02)
                    if x_hi <= x_lo + 0.02 or y_hi <= y_lo + 0.02:
                        x_lo, x_hi = lim.margin_x, 1 - lim.margin_x
                        y_lo, y_hi = (min(hy + lim.min_ground_gap, 0.99), 1.0)
                water_note = "" if "text" not in self.water_hint_parts else water.describe() + (" The foot point must be inside it." if inst.habitat == "water" else " The foot point must not be in the water." if inst.habitat == "land" else "")
            fields = [
                EstimateField(f"{sid}_x", FieldKind.ESTIMATE, f"Horizontal position of the {inst.spec.name} as a fraction of the image width (0 = left edge, 1 = right edge).",
                              minimum=x_lo, maximum=x_hi, step=0.01),
                EstimateField(f"{sid}_y", FieldKind.ESTIMATE, f"Vertical position of the {inst.spec.name} as a fraction of the image height (0 = top edge, 1 = bottom edge). "
                              + ("It must stay in the sky, above the horizon." if inst.airborne else "It must stay on the ground, below the horizon; lower in the image means closer to the viewer."),
                              minimum=y_lo, maximum=y_hi, step=0.01),
                EstimateField(f"{sid}_scale", FieldKind.ESTIMATE, f"Size multiplier of the {inst.spec.name} relative to a realistic size at that position (1 = realistic).",
                              minimum=lim.scale_mult_range[0], maximum=lim.scale_mult_range[1], step=0.05, log_scale=True),
                EstimateField(f"{sid}_rot", FieldKind.ESTIMATE, f"Tilt of the {inst.spec.name} in degrees (clockwise positive, 0 = upright).",
                              minimum=lim.rotation_range_deg[0], maximum=lim.rotation_range_deg[1], step=1.0),
            ]
            kind = "floating in the sky (centre point)" if inst.airborne else "standing on the ground (point where its feet touch the ground)"
            intent = inst.spec.intent or default_intent(inst)
            lines = [f"Task: build a landscape collage one object at a time. Overall request: {text}",
                     f"The image is the canvas so far" + (" with a 0.1 coordinate grid (white lines, yellow numbers)" if self.use_grid else "") + f" (landscape 3:2). The horizon is at y = {hy:.2f} (0 = top, 1 = bottom); sky above, ground below."]
            if not self.use_image:
                lines[1] = f"The horizon is at y = {hy:.2f} (0 = top, 1 = bottom); sky above, ground below. Positions are fractions of the image width and height."
            if self.use_history:
                lines.append(("Already placed:\n" + "\n".join(hist)) if hist else "Nothing is placed yet.")
            if water_note:
                lines.append(water_note)
            lines.append(f"Now place: {sid}, a {inst.spec.name}, {kind}, role={inst.spec.role}. Intent: {intent}.")
            lines.append("Choose a spot that keeps the whole collage balanced and does not overlap objects already placed.")
            state = "\n".join(lines)
            imgs = []
            if self.use_image:
                imgs = [_png_of(_grid_overlay(partial) if self.use_grid else partial)]
            resp = self.estimator.estimate(state, fields, images=imgs)
            vals = {}
            for suffix in ("x", "y", "scale", "rot"):
                val = resp.values[f"{sid}_{suffix}"].value
                if val is None:
                    raise ApiError(f"Clairic が値を返さなかった: {sid}_{suffix}")
                vals[suffix] = float(val)
            pl = Placement(x=vals["x"], y=vals["y"], scale_mult=vals["scale"], rotation_deg=vals["rot"],
                           x_range80=resp.values[f"{sid}_x"].range80, y_range80=resp.values[f"{sid}_y"].range80)
            proposal = pl
            if water is not None:   # 補正は、次の要素に見せる画像に反映するため、その場で行う
                fixed, _ = apply_habitat([inst], [pl], water, self.limits)
                pl = fixed[0]
            placed[n] = pl
            self.trace.append({"step": step, "instance": inst.id, "state": state, "n_images": len(imgs),
                               "proposal": {"x": proposal.x, "y": proposal.y, "scale_mult": proposal.scale_mult, "rotation_deg": proposal.rotation_deg}})
        return [placed[n] for n in range(len(instances))]
