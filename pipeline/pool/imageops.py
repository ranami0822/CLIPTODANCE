"""画像の読み込み・縮小・切り抜きの書き出し。"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000  # 美術館の高解像度画像を許す(圧縮爆弾の検査は別途 PIL の警告に任せる)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_rgb(path_or_bytes: str | Path | bytes) -> Image.Image:
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    im = Image.open(src)
    im.load()
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def fit_long_side(im: Image.Image, max_side: int) -> Image.Image:
    """長辺を max_side 以下に縮小する(拡大はしない)。"""
    w, h = im.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return im
    return im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def cutout_rgba(rgb: np.ndarray, mask: np.ndarray, pad: int = 2) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """マスクの外接矩形(余白 pad)で切り出した RGBA 画像と、その矩形 (x0,y0,x1,y1) を返す。"""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    rgba = np.dstack([rgb[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA"), (int(x0), int(y0), int(x1), int(y1))


def on_gray(rgba: Image.Image, gray: int = 127) -> Image.Image:
    """透明部分を灰色で埋めた RGB 画像(埋め込みや Clef に渡すとき用)。"""
    bg = Image.new("RGB", rgba.size, (gray, gray, gray))
    bg.paste(rgba, mask=rgba.getchannel("A"))
    return bg


def png_bytes(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
