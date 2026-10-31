"""画素まわりの処理(切り抜き・コンタクトシート・スケッチの整形)。

依存は Pillow と numpy だけ。重いモデル(SAM / rembg など)は使わない。
「透過 PNG はそのまま使う」「背景が単色に近い写真は縁からの塗りつぶしで切り抜く」の2段で足りる範囲を狙う。
うまくいかない画像は ok=False を返して、呼び出し側(UI)に「透過 PNG を使ってください」と伝えさせる。
"""
from __future__ import annotations

import base64
import io
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

MAX_OBJ_SIDE = 1024      # 取り込む素材の長辺の上限
MAX_SKETCH_SIDE = 768    # Clairic に渡すスケッチの長辺
TILE = 256               # コンタクトシート1マスの一辺

_DATA_URL = re.compile(r"^data:image/(?:png|jpeg|jpg|webp);base64,(.*)$", re.S)


class ImageError(ValueError):
    """画像として読めない・形式が合わない。"""


# ---------------------------------------------------------------- 入出力

def decode_data_url(s: str) -> bytes:
    m = _DATA_URL.match(s or "")
    if not m:
        raise ImageError("画像は data URL(png / jpeg / webp)で渡してください")
    try:
        return base64.b64decode(m.group(1))
    except Exception as e:  # noqa: BLE001
        raise ImageError(f"base64 を読めません: {e}") from e


def open_rgba(data: bytes) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im)
        return im.convert("RGBA")
    except Exception as e:  # noqa: BLE001
        raise ImageError(f"画像を開けません: {e}") from e


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_data_url(im: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(im)).decode("ascii")


def fit_max_side(im: Image.Image, side: int) -> Image.Image:
    w, h = im.size
    m = max(w, h)
    if m <= side:
        return im
    k = side / m
    return im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)


# ---------------------------------------------------------------- 透過・トリミング

def transparent_fraction(im: Image.Image, thr: int = 250) -> float:
    a = np.asarray(im.getchannel("A"))
    return float((a < thr).mean())


def trim_to_alpha(im: Image.Image, pad: int = 2, thr: int = 8) -> Image.Image:
    """不透明な部分の外接矩形に切り詰める(まったく不透明がなければそのまま返す)。"""
    a = np.asarray(im.getchannel("A")) > thr
    if not a.any():
        return im
    ys, xs = np.where(a)
    x0, x1 = max(0, xs.min() - pad), min(im.width, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(im.height, ys.max() + 1 + pad)
    return im.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------- 切り抜き

def cutout_border_flood(im: Image.Image, tol: float | None = None, feather: float = 1.2) -> tuple[Image.Image, dict]:
    """縁と同じ色の領域を、縁からつながる範囲だけ背景として抜く。

    背景が単色に近い写真・イラスト向け。縁の色が一様でないとき(spread が大きいとき)は
    うまくいかないので ok=False にする。返す info:
      ok, note, bg(縁の代表色), tol, spread(縁の色のばらつき), fg_fraction
    """
    im = fit_max_side(im.convert("RGBA"), MAX_OBJ_SIDE)
    rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape
    b = max(2, min(h, w) // 100)
    border = np.concatenate([
        rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    spread = float(np.percentile(np.linalg.norm(border - bg, axis=1), 90))
    if tol is None:
        tol = float(np.clip(spread * 1.5 + 12, 16, 80))
    info = {"ok": False, "note": "", "bg": [int(v) for v in bg], "tol": round(tol, 1),
            "spread": round(spread, 1), "fg_fraction": None}
    if spread > 60:
        info["note"] = "縁の色が一様ではありません。背景が単色の画像か、透過 PNG を使ってください"
        return im, info

    dist = np.linalg.norm(rgb - bg, axis=2)
    cand = dist < tol

    # 縮小した画像で「縁からつながる背景」を求め、元の解像度の色判定で縁を詰める
    k = 512 / max(h, w) if max(h, w) > 512 else 1.0
    sw, sh = max(1, round(w * k)), max(1, round(h * k))
    small = Image.fromarray((cand * 255).astype(np.uint8)).resize((sw, sh), Image.NEAREST)
    pad = Image.new("L", (sw + 2, sh + 2), 255)
    pad.paste(small, (1, 1))
    ImageDraw.floodfill(pad, (0, 0), 128)
    flooded = np.asarray(pad.crop((1, 1, sw + 1, sh + 1))) == 128
    bg_full = Image.fromarray((flooded * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    bg_full = bg_full.filter(ImageFilter.MaxFilter(5))          # 縮小で失った縁の細かさを、色判定と掛けて戻す
    bg_mask = (np.asarray(bg_full) > 0) & cand

    alpha = Image.fromarray(np.where(bg_mask, 0, 255).astype(np.uint8))
    alpha = alpha.filter(ImageFilter.MedianFilter(5))
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    fg = float((np.asarray(alpha) > 127).mean())
    info["fg_fraction"] = round(fg, 3)
    if fg < 0.02 or fg > 0.97:
        info["note"] = "うまく分離できませんでした(前景が小さすぎる/ほぼ全面)。許容値を変えるか、透過 PNG を使ってください"
        return im, info

    out = im.copy()
    out.putalpha(alpha)
    info["ok"] = True
    return trim_to_alpha(out), info


def prepare_object(im: Image.Image, tol: float | None = None) -> tuple[Image.Image, dict]:
    """素材を使える形にする。すでに透過があればそのまま(トリミングだけ)、なければ切り抜く。"""
    im = fit_max_side(im.convert("RGBA"), MAX_OBJ_SIDE)
    if transparent_fraction(im) > 0.01:
        return trim_to_alpha(im), {"ok": True, "note": "透過 PNG をそのまま使いました", "method": "alpha"}
    out, info = cutout_border_flood(im, tol=tol)
    info["method"] = "border_flood"
    return out, info


# ---------------------------------------------------------------- Clairic に渡す画像

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # 古い Pillow
        return ImageFont.load_default()


def contact_sheet(items: list[tuple[str, Image.Image]], tile: int = TILE) -> bytes:
    """素材を A, B, C… の札つきで1枚に並べる(Clairic の画像は最大4枚なので、素材は1枚にまとめる)。"""
    n = len(items)
    if n == 0:
        raise ImageError("素材がありません")
    cols = min(n, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * tile, rows * tile), (205, 205, 205))
    d = ImageDraw.Draw(sheet)
    f = _font(28)
    for i, (label, im) in enumerate(items):
        cx, cy = (i % cols) * tile, (i // cols) * tile
        t = fit_max_side(im.convert("RGBA"), tile - 40)
        if max(t.size) < tile - 40:  # 小さい素材は拡大して見分けやすくする
            k = (tile - 40) / max(t.size)
            t = t.resize((max(1, round(t.width * k)), max(1, round(t.height * k))), Image.LANCZOS)
        sheet.paste(t, (cx + (tile - t.width) // 2, cy + (tile - t.height) // 2 + 8), t)
        d.rectangle((cx + 4, cy + 4, cx + 40, cy + 40), fill=(255, 255, 255), outline=(60, 60, 60))
        d.text((cx + 12, cy + 4), label, fill=(0, 0, 0), font=f)
        d.rectangle((cx, cy, cx + tile - 1, cy + tile - 1), outline=(150, 150, 150))
    return png_bytes(sheet)


def sketch_regions(im: Image.Image, max_regions: int = 12, grid_long: int = 128, merge: int = 3) -> list[dict]:
    """スケッチの線を「つながった塊」に分け、塊ごとの外接矩形を返す(読む順 = 上から下・左から右で 1,2,3…)。

    近い線は1つの塊にまとめる(merge は粗い格子での膨張幅)。小さすぎる点は捨てる。座標はキャンバスに対する 0–1。
    w_long / h_long は、矩形の幅・高さをキャンバスの長辺で割った値(素材の scale と同じ単位)。
    """
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im.convert("RGBA"))
    w, h = bg.size
    ink = (np.asarray(bg.convert("L")) < 200).astype(np.uint8) * 255
    k = min(1.0, grid_long / max(w, h))
    gw, gh = max(1, round(w * k)), max(1, round(h * k))
    coarse = np.asarray(Image.fromarray(ink).resize((gw, gh), Image.BOX)) > 8       # 少しでも線があれば 1
    if not coarse.any():
        return []
    grown = np.asarray(Image.fromarray(coarse.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * merge + 1))) > 0
    seen = np.zeros_like(grown, dtype=bool)
    comps = []
    for sy in range(gh):
        for sx in range(gw):
            if not grown[sy, sx] or seen[sy, sx]:
                continue
            stack, cells = [(sy, sx)], []
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                if coarse[y, x]:
                    cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < gh and 0 <= nx < gw and grown[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) >= 3:
                ys, xs = zip(*cells)
                comps.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(cells)))
    comps.sort(key=lambda c: -(c[2] - c[0]) * (c[3] - c[1]))
    comps = comps[:max_regions]
    comps.sort(key=lambda c: ((c[1] + c[3]) / 2, (c[0] + c[2]) / 2))
    long_side = max(w, h)
    out = []
    for i, (x0, y0, x1, y1, _n) in enumerate(comps, start=1):
        nx0, nx1, ny0, ny1 = x0 / gw, x1 / gw, y0 / gh, y1 / gh
        out.append({"id": i, "x0": nx0, "y0": ny0, "x1": nx1, "y1": ny1,
                    "cx": (nx0 + nx1) / 2, "cy": (ny0 + ny1) / 2,
                    "w_long": (nx1 - nx0) * w / long_side, "h_long": (ny1 - ny0) * h / long_side})
    return out


def flatten_sketch(im: Image.Image, regions: list[dict] = ()) -> tuple[bytes, float]:
    """スケッチを白地に重ねて縮小する。(PNG のバイト列, 線のある画素の割合)を返す。

    regions を渡すと、各塊を薄い青の枠と番号で示す(Clairic が「どの素材がどの塊か」を答えるため)。
    線の割合は枠を描く前に測る。
    """
    im = fit_max_side(im.convert("RGBA"), MAX_SKETCH_SIDE)
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    rgb = bg.convert("RGB")
    ink = float((np.asarray(rgb.convert("L")) < 200).mean())
    if regions:
        d = ImageDraw.Draw(rgb)
        f = _font(22)
        w, h = rgb.size
        col = (40, 110, 230)
        for r in regions:
            x0, y0, x1, y1 = r["x0"] * w, r["y0"] * h, r["x1"] * w, r["y1"] * h
            d.rectangle((x0, y0, x1, y1), outline=col, width=2)
            d.rectangle((x0, y0, x0 + 26, y0 + 26), fill=col)
            d.text((x0 + 6, y0), str(r["id"]), fill=(255, 255, 255), font=f)
    return png_bytes(rgb), ink


