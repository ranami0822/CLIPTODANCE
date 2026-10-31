"""水と陸の地図。土台の絵を格子に区切り、Clef に「この四角の中は水か」を 1 マスずつ尋ねる。

Clairic に水域の座標(箱)を直接答えさせたところ大きくずれた(設計書 M3: 目視の箱との IoU 0.00〜0.39)ので、
こちらの聞き方に分けた。目視の箱との IoU は 0.22〜0.65 だった(4 枚だけ。基準の箱も私の目測)。
"""
from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from adapters.base import Judge, noul

WATER_QUESTION = ("Is the area inside the red rectangle water (sea, lake, river or pond)? "
                  "Answer yes only if most of the area inside the rectangle is water.")


@dataclass
class WaterMap:
    p: np.ndarray            # (ny, nx) 各マスが水である確率(地平線より上のマスは 0)
    horizon_y: float
    threshold: float = 0.5

    @property
    def ny(self) -> int:
        return self.p.shape[0]

    @property
    def nx(self) -> int:
        return self.p.shape[1]

    def mask(self) -> np.ndarray:
        return self.p >= self.threshold

    def is_water(self, x: float, y: float) -> bool:
        i = min(self.nx - 1, max(0, int(x * self.nx)))
        j = min(self.ny - 1, max(0, int(y * self.ny)))
        return bool(self.p[j, i] >= self.threshold)

    def any_water(self) -> bool:
        return bool(self.mask().any())

    def bbox(self) -> tuple[float, float, float, float] | None:
        """水のマスを囲む箱 (x0, x1, y0, y1)。水がなければ None。"""
        m = self.mask()
        if not m.any():
            return None
        js, is_ = np.nonzero(m)
        return (float(is_.min()) / self.nx, float(is_.max() + 1) / self.nx, float(js.min()) / self.ny, float(js.max() + 1) / self.ny)

    def nearest(self, x: float, y: float, water: bool, min_y: float = 0.0) -> tuple[float, float] | None:
        """(x, y) に最も近い、水(water=True)または陸(False)のマスの中の点。y が min_y 以上のマスだけを見る。なければ None。"""
        best = None
        for j in range(self.ny):
            for i in range(self.nx):
                if (self.p[j, i] >= self.threshold) != water:
                    continue
                x0, x1, y0, y1 = i / self.nx, (i + 1) / self.nx, j / self.ny, (j + 1) / self.ny
                if y1 <= min_y:
                    continue
                y0 = max(y0, min_y)
                m = 0.5 / max(self.nx, self.ny) * 0.4        # マスの縁ぎりぎりを避ける
                px = min(max(x, x0 + m), x1 - m)
                py = min(max(y, y0 + m), y1 - m)
                d = (px - x) ** 2 + (py - y) ** 2
                if best is None or d < best[0]:
                    best = (d, px, py)
        return None if best is None else (best[1], best[2])

    def describe(self) -> str:
        bb = self.bbox()
        if bb is None:
            return "There is no water in this landscape."
        rows = []
        for j in range(self.ny):
            runs, i = [], 0
            while i < self.nx:
                if self.p[j, i] >= self.threshold:
                    k = i
                    while k + 1 < self.nx and self.p[j, k + 1] >= self.threshold:
                        k += 1
                    runs.append(f"x {i / self.nx:.2f}-{(k + 1) / self.nx:.2f}")
                    i = k + 1
                else:
                    i += 1
            if runs:
                rows.append(f"y {j / self.ny:.2f}-{(j + 1) / self.ny:.2f}: " + ", ".join(runs))
        return "Water is found at: " + "; ".join(rows) + "."


def estimate_water_map(judge: Judge, canvas: Image.Image, horizon_y: float, nx: int = 8, ny: int = 6, workers: int = 4,
                       max_side: int = 640) -> tuple[WaterMap, dict]:
    """格子の各マスについて、そのマスに赤い枠を描いた画像と一緒に Clef に尋ねる(1 マス 1 呼び出し、地平線より上のマスは尋ねない)。"""
    base = canvas.convert("RGB").copy()
    base.thumbnail((max_side, max_side))
    W, H = base.size

    def png_of(i: int, j: int) -> bytes:
        im = base.copy()
        ImageDraw.Draw(im).rectangle([i * W / nx, j * H / ny, (i + 1) * W / nx - 1, (j + 1) * H / ny - 1], outline=(255, 0, 0), width=4)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    cells = [(i, j) for j in range(ny) for i in range(nx) if (j + 1) / ny > horizon_y]

    def ask(c):
        r = judge.ask([noul("w", WATER_QUESTION)], images=[png_of(*c)], state="")
        return c, float(r.results[0].p_yes), r.usage

    p = np.zeros((ny, nx))
    neurons, calls = 0.0, 0
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for (i, j), pv, usage in ex.map(ask, cells):
            p[j, i] = pv
            neurons += usage.neurons or 0.0
            calls += usage.api_calls
    return WaterMap(p=p, horizon_y=horizon_y), {"grid": [nx, ny], "calls": calls, "neurons": round(neurons, 1), "cells_asked": len(cells)}
