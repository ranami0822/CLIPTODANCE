"""地平線の高さの初期値(OpenCV の軽い処理。設計書 R3a)。

出すのは「空と地面の境界らしい行」(スカイライン寄り)で、視線の高さの地平線(消失線)とは
一致しないことがある(山・森・建物)。ここは初期値で、コンタクトシートで目視確認して直す前提。
方法: 画像を縮小 → 輝度の縦方向の勾配を列ごとに出す → 行ごとに、列をまたいだ勾配の符号がそろっているほど高く評価
(横に長い境界ほど強く出る)→ 最も強い行を採る。
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_horizon(
    rgb: np.ndarray, width: int = 256, margin: float = 0.1, backdrop: np.ndarray | None = None
) -> tuple[float, float]:
    """(y, 確信度)を返す。y は 0〜1(上が0)。確信度は、最も強い行が全体の中でどれだけ突出しているか(0〜1)。

    `backdrop`(余白のマスク)があれば、余白にかからない列だけを使う(アーチ形の絵の黒い四隅を避けるため)。
    確信度は目安で、実際に当たっているかとは別(額縁の縁に引っ張られても高く出る)。"""
    h, w = rgb.shape[:2]
    scale = width / w
    small = cv2.resize(rgb, (width, max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    blurred = cv2.GaussianBlur(lab, (0, 0), sigmaX=2.0)
    gy = np.stack([cv2.Sobel(blurred[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)], axis=-1)
    if backdrop is not None:
        bd = cv2.resize(backdrop.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ok = bd.mean(axis=0) < 0.02
        if ok.sum() >= max(8, 0.2 * ok.size):
            gy = gy[:, ok]
    # 行ごとに、列方向に平均した勾配(符号つき)の大きさ。同じ向きの境界が横に長いほど大きい。
    row_energy = np.linalg.norm(gy.mean(axis=1), axis=-1)
    n = len(row_energy)
    lo, hi = int(margin * n), int((1 - margin) * n)
    window = row_energy[lo:hi]
    if window.size == 0 or window.max() <= 1e-6:
        return 0.5, 0.0
    peak = int(window.argmax())
    y = (lo + peak + 0.5) / n
    confidence = float(np.clip(1.0 - window.mean() / window.max(), 0.0, 1.0))
    return float(y), confidence
