"""切り抜きを土台に載せる(R5 サイズ、R6 重ね順、R8 範囲と重なり、R9 なじませ、R10 補正)。"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .geometry import Limits, clamp, ground_height_px
from .types import Instance, Placement


@dataclass
class RenderConfig:
    limits: Limits = field(default_factory=Limits)
    feather_sigma: float = 1.0          # 縁のぼかし(画素)。アルファと色を同じだけぼかす
    color_strength: float = 0.2         # Reinhard 方式の色の転送の強さ(0〜1)
    max_cover: float = 0.30             # 先に置いた(奥の)物のうち、手前の物に隠れてよい面積の割合(R8)
    allow_out_of_canvas: bool = False
    nudge_steps: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)   # 重なりを避ける横ずれ(画面の幅に対する比)
    min_height_px: float = 6.0
    max_height_frac: float = 0.6        # 地上の物の高さの上限(画面の高さに対する比)。超えるなら、足元を地平線の方へ動かして収める(遠近を保つ)
    max_width_frac: float = 0.5         # 地上の物の幅の上限(画面の幅に対する比)。同じく、足元を地平線の方へ動かして収める


def _premult_sprite(rgba: np.ndarray, height: int, flip: bool) -> np.ndarray:
    """RGBA(uint8)を、高さ height に縮尺し、(必要なら左右反転して)乗算済みアルファの float32 にする。"""
    h, w = rgba.shape[:2]
    width = max(1, int(round(w * height / h)))
    f = rgba.astype(np.float32) / 255.0
    f[..., :3] *= f[..., 3:4]                      # 乗算済み: 縮尺で縁に色の滲みが出ない
    f = cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA if height < h else cv2.INTER_LINEAR)
    return f[:, ::-1].copy() if flip else f


def _transform(sprite_w: int, sprite_h: int, anchor: tuple[float, float], angle_cw: float, target: tuple[float, float]) -> np.ndarray:
    """スプライト座標 → 画面座標の 2×3 行列。anchor を中心に時計回りに angle_cw 度回してから、anchor を target に移す。"""
    m = cv2.getRotationMatrix2D(anchor, -angle_cw, 1.0)   # OpenCV は反時計回りが正
    m[:, 2] += (target[0] - anchor[0], target[1] - anchor[1])
    return m


def _bbox(m: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    pts = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ m.T
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _warp(sprite: np.ndarray, m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(sprite, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _transfer_colors(canvas_u8: np.ndarray, color: np.ndarray, alpha: np.ndarray, strength: float) -> np.ndarray:
    """Reinhard 2001: Lab の各軸の平均・標準偏差を、物の周囲の土台に合わせる。strength で弱める。color は 0-1 の RGB。"""
    if strength <= 0:
        return color
    mask = alpha > 0.5
    if mask.sum() < 200:
        return color
    ys, xs = np.nonzero(mask)
    h, w = alpha.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad_y, pad_x = int(0.3 * (y1 - y0)), int(0.3 * (x1 - x0))
    ry0, ry1, rx0, rx1 = max(0, y0 - pad_y), min(h, y1 + pad_y), max(0, x0 - pad_x), min(w, x1 + pad_x)
    ring = ~(alpha[ry0:ry1, rx0:rx1] > 0.05)
    if ring.sum() < 500:
        return color
    lab_obj = cv2.cvtColor((np.clip(color, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_ref = cv2.cvtColor(canvas_u8[ry0:ry1, rx0:rx1], cv2.COLOR_RGB2LAB).astype(np.float32)[ring]
    obj = lab_obj[mask]
    mu_o, sd_o = obj.mean(0), obj.std(0) + 1e-3
    mu_r, sd_r = lab_ref.mean(0), lab_ref.std(0) + 1e-3
    ratio = np.clip(sd_r / sd_o, 0.5, 1.5)
    transferred = (lab_obj - mu_o) * ratio + mu_r
    out = lab_obj * (1 - strength) + transferred * strength
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.where(alpha[..., None] > 0.02, rgb, color)


def render_collage(base: Image.Image, horizon_y: float, objects: list[tuple[Instance, np.ndarray, Placement]],
                   cfg: RenderConfig | None = None) -> tuple[Image.Image, list[dict]]:
    """objects = (配置する個体, 切り抜き RGBA(uint8)、配置)。奥から手前へ重ねる(R6)。"""
    cfg = cfg or RenderConfig()
    lim = cfg.limits
    canvas = np.asarray(base.convert("RGB")).copy()
    H, W = canvas.shape[:2]
    # 重ね順: 空中の物(雲・月・鳥など)は最奥、地上の物は足元の y が小さい(奥の)ものから
    order = sorted(objects, key=lambda o: (0, o[2].y) if o[0].airborne else (1, o[2].y))
    drawn_masks: list[np.ndarray] = []
    log: list[dict] = []

    for inst, rgba, pl in order:
        rec: dict = {"instance": inst.id, "cutout_id": inst.cutout_id, "kind": inst.spec.kind, "airborne": inst.airborne,
                     "requested": {"x": pl.x, "y": pl.y, "scale_mult": pl.scale_mult, "rotation_deg": pl.rotation_deg, "flip": pl.flip},
                     "corrections": []}
        sm = clamp(pl.scale_mult, *lim.scale_mult_range)
        rot = clamp(pl.rotation_deg, *lim.rotation_range_deg)
        x = clamp(pl.x, lim.margin_x, 1 - lim.margin_x)
        if inst.airborne:
            y = clamp(pl.y, lim.sky_margin, max(lim.sky_margin, horizon_y - lim.sky_margin))
        else:
            y = clamp(pl.y, horizon_y + lim.min_ground_gap, 1.0)
        for name, a, b in (("scale_mult", pl.scale_mult, sm), ("rotation_deg", pl.rotation_deg, rot), ("x", pl.x, x), ("y", pl.y, y)):
            if abs(a - b) > 1e-9:
                rec["corrections"].append({name: [round(a, 4), round(b, 4)]})

        if inst.airborne:
            h_px = (inst.height_frac or 0.08) * H * sm
        else:
            def size_at(yy: float) -> float:
                """足元が yy のときの高さ(画素)。幅で決める種類は、幅から高さを出す(同じ遠近の式、縦横の縮尺は同じ)。"""
                if inst.width_m:
                    return ground_height_px(inst.width_m, yy, horizon_y, H, sm, lim.camera_height_m) / inst.aspect
                return ground_height_px(inst.height_m or 1.0, yy, horizon_y, H, sm, lim.camera_height_m)
            h_px = size_at(y)
            # 大きすぎる(高さが画面の max_height_frac、または幅が max_width_frac を超える): 足元を奥(地平線の方)へ動かす。
            # 大きさは (足元の y − 地平線の y) に比例するので、比で動かせば、遠近の式のまま上限に収まる
            ratio = min(1.0, cfg.max_height_frac * H / h_px, cfg.max_width_frac * W / (h_px * inst.aspect))
            if ratio < 1.0:
                y_new = max(horizon_y + (y - horizon_y) * ratio, horizon_y + lim.min_ground_gap)
                rec["corrections"].append({"moved_toward_horizon": [round(y, 4), round(y_new, 4)]})
                y = y_new
                h_px = size_at(y)
        h_px = max(h_px, cfg.min_height_px)
        sprite = _premult_sprite(rgba, int(round(h_px)), pl.flip)
        sh, sw = sprite.shape[:2]
        anchor = (sw / 2, sh / 2) if inst.airborne else (sw / 2, float(sh))   # 空中: 中心、地上: 足元中央

        def bbox_at(px: float, py: float):
            m = _transform(sw, sh, anchor, rot, (px, py))
            return m, _bbox(m, sw, sh)

        px, py = x * W, y * H
        m, (bx0, by0, bx1, by1) = bbox_at(px, py)
        if not cfg.allow_out_of_canvas:
            if bx1 - bx0 > W or by1 - by0 > H:           # 画面より大きい: 縮める
                k = min(W / (bx1 - bx0), H / (by1 - by0)) * 0.98
                sprite = _premult_sprite(rgba, max(2, int(sprite.shape[0] * k)), pl.flip)
                sh, sw = sprite.shape[:2]
                anchor = (sw / 2, sh / 2) if inst.airborne else (sw / 2, float(sh))
                rec["corrections"].append({"shrunk_to_fit": round(k, 3)})
                m, (bx0, by0, bx1, by1) = bbox_at(px, py)
            if by0 < 0 and not inst.airborne:                    # 上にはみ出す(背が高い): 足元を固定して縮める
                k = max(0.1, py / (py - by0)) * 0.99
                sprite = _premult_sprite(rgba, max(2, int(sprite.shape[0] * k)), pl.flip)
                sh, sw = sprite.shape[:2]
                anchor = (sw / 2, float(sh))
                rec["corrections"].append({"shrunk_to_fit": round(k, 3)})
                m, (bx0, by0, bx1, by1) = bbox_at(px, py)
            dx = -bx0 if bx0 < 0 else (W - bx1 if bx1 > W else 0.0)
            dy = -by0 if by0 < 0 else (H - by1 if by1 > H else 0.0)
            if dx or dy:
                px, py = px + dx, py + dy
                rec["corrections"].append({"shifted_into_canvas": [round(dx, 1), round(dy, 1)]})
                m, _ = bbox_at(px, py)

        def covered(mask: np.ndarray) -> float:
            return max((float((mask & prev).sum()) / max(prev.sum(), 1) for prev in drawn_masks), default=0.0)

        warped = _warp(sprite, m, (W, H))
        mask = warped[..., 3] > 0.5
        worst = covered(mask)
        if worst > cfg.max_cover and not inst.airborne:
            best = (worst, px, py, m, warped, mask)
            for step in cfg.nudge_steps:
                for sign in (1, -1):
                    cx = float(np.clip(px + sign * step * W, lim.margin_x * W, (1 - lim.margin_x) * W))
                    m2 = _transform(sw, sh, anchor, rot, (cx, py))
                    wp2 = _warp(sprite, m2, (W, H))
                    mk2 = wp2[..., 3] > 0.5
                    c2 = covered(mk2)
                    if c2 < best[0]:
                        best = (c2, cx, py, m2, wp2, mk2)
                if best[0] <= cfg.max_cover:
                    break
            if best[1] != px:
                rec["corrections"].append({"nudged_x": [round(px / W, 3), round(best[1] / W, 3)]})
            worst, px, py, m, warped, mask = best
            if worst > cfg.max_cover:
                rec["corrections"].append({"overlap_exceeded": round(worst, 3)})
        rec["covered_max"] = round(worst, 3)

        # 縁のぼかし: 乗算済みの色とアルファを同じだけぼかす(縁に黒い滲みが出ない)
        if cfg.feather_sigma > 0:
            warped = cv2.GaussianBlur(warped, (0, 0), cfg.feather_sigma)
        a = np.clip(warped[..., 3], 0, 1)
        color = np.where(a[..., None] > 1e-4, warped[..., :3] / np.maximum(a[..., None], 1e-4), 0.0)
        color = _transfer_colors(canvas, color, a, cfg.color_strength)
        canvas = np.clip(canvas.astype(np.float32) * (1 - a[..., None]) + color * 255 * a[..., None], 0, 255).astype(np.uint8)
        drawn_masks.append(mask)

        ys, xs = np.nonzero(mask)
        rec.update({"applied": {"x": round(px / W, 4), "y": round(py / H, 4), "scale_mult": round(sm, 3), "rotation_deg": round(rot, 2), "flip": pl.flip},
                    "height_px": round(float(sh), 1), "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if xs.size else None})
        log.append(rec)
    return Image.fromarray(canvas), log
