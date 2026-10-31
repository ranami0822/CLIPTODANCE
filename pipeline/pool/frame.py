"""額縁・撮影の余白の除去(取り込みの最初に行う)。

Met の絵画画像には、撮影の背景(黒や白の余白)、アーチ形の絵の四隅の黒、細い金の額縁が写っている。
これらが残ると、地平線の推定が縁に引っ張られ、SAM が額縁を「物体」として切り抜く(2026-10-31 の試験取り込みで確認)。

2 段で除く:
  1. 余白: 画像の端につながった「ほぼ黒」「ほぼ白」の領域を余白とし、残りの外接矩形に切る。
     アーチ形の絵の四隅のように矩形の内側に残る余白は `backdrop` マスクとして返す。
  2. 細い額縁: 余白を除いた矩形の端から浅い範囲で、辺全体にわたってそろった強い境界(額縁の内側の縁)があれば、そこまで削る。
すべて初期案。数値は実際の画像で確かめながら決めた(`tests/` と設計書を参照)。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MIN_CONTENT_FRAC = 0.25  # 内容部分が画像の面積のこれ未満なら、検出を信用しない


@dataclass(frozen=True)
class FrameResult:
    box: tuple[int, int, int, int]   # 入力画像上の内容部分 (x0, y0, x1, y1)。x1, y1 は含まない
    backdrop: np.ndarray             # box で切った後の (h, w) bool。True = 余白(アーチの四隅など)
    trimmed: dict                    # どの辺を解析用の縮小画像で何画素削ったか(記録用)
    fallback: bool = False           # 検出結果が不自然(内容が小さすぎる)で、画像全体を返したとき True


def _analysis_scale(shape, max_side: int) -> float:
    return min(1.0, max_side / max(shape[:2]))


def find_backdrop(rgb: np.ndarray, dark: int = 22, light: int = 238, sat_max: int = 25,
                  flat_grad: float = 1.5, flat_dark: int = 70, flat_light: int = 205) -> np.ndarray:
    """画像の端につながった余白の画素の集合(bool)。次のどちらかを余白の候補にする:
      - ほぼ黒(V < dark)、またはほぼ白(全チャンネル > light で彩度が低い)
      - 平ら(縮小した輝度の勾配が flat_grad 未満)で、暗い(V < flat_dark)か明るい(V > flat_light、彩度が低い)
        … 濃い灰色の撮影背景や、影のある白い台紙のため。絵の暗い部分は筆致やキャンバスの目でわずかに勾配があり、
        平らの基準には入らない。
    候補のうち、画像の端に接している連結成分だけを余白とする。
    """
    v = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    s = v.astype(np.int16) - mn
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    flat = grad < flat_grad
    is_dark = v < flat_dark
    is_light = (mn > flat_light) & (s < sat_max)
    candidate = (v < dark) | ((mn > light) & (s < sat_max)) | (flat & (is_dark | is_light))
    n, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros(candidate.shape, bool)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    edge_labels = edge_labels[edge_labels != 0]
    backdrop = np.isin(labels, edge_labels)
    # 勾配はぼかして求めるので、余白と絵の境の数画素は「平ら」でなくなる。連結成分を求めた「後」で、余白の縁から
    # 2 画素だけ、同じように暗い(明るい)画素を余白に足す。(求める「前」に足すと、余白と暗い絵がつながって、絵を食う)
    fringe = (cv2.dilate(backdrop.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (is_dark | is_light) & ~backdrop
    return backdrop | fringe


def _content_box(backdrop: np.ndarray, min_run: float = 0.05) -> tuple[int, int, int, int]:
    """余白でない画素の外接矩形。ごく少数の画素(ノイズ)で広がらないよう、各行・各列の割合で決める。"""
    h, w = backdrop.shape
    content = ~backdrop
    rows = np.nonzero(content.mean(axis=1) > min_run)[0]
    cols = np.nonzero(content.mean(axis=0) > min_run)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def edge_profile(img: np.ndarray, valid: np.ndarray, depth: int) -> np.ndarray:
    """辺から d 行目(d = 1..depth)の、辺に沿った勾配の大きさの中央値(辺に垂直な向きの中心差分)。

    `img` は (H, W) の輝度か、(H, W, C) の色(Lab など。色の差は C 個の差のノルム)。
    額縁は辺の全体にわたって境界が続くので、この値が高い。絵の内側は低い値に落ち着く。
    余白の画素(`valid` が False)は除く。
    """
    if img.ndim == 2:
        img = img[..., None]
    g = np.linalg.norm(img[2:depth + 2] - img[0:depth], axis=-1)
    v = valid[1:depth + 1]
    out = np.zeros(g.shape[0], np.float32)
    for r in range(g.shape[0]):
        row = g[r][v[r]]
        out[r] = float(np.median(row)) if row.size > 50 else 0.0
    return out


def frame_depth(profile: np.ndarray, max_gap: int, min_tau: float = 4.0, tau_factor: float = 2.5, min_trim: int = 4) -> int:
    """辺の外側から続く「勾配の高い層」の最も内側の深さ(0 = 額縁なし)。

    しきい値は、窓の中の低い方(20 パーセンタイル)の勾配を基準にする(絵が元々細かいときは、しきい値も上がる)。
    高い層は、間の隙間が max_gap 以下なら同じ額縁の続きとして数える(多重の縁取りの額縁のため)。
    辺のすぐそば(最初の max_gap 以内)に高い層がなければ、額縁はないとみなす。

    削る位置は、高い層の最も内側の画素から 1 つ手前。境界の勾配は、ぼかしと中心差分のために境界の外へ 2〜3 画素
    尾を引くので、層の端まで削ると絵を削ってしまう。結果が min_trim 未満なら、余白と絵の境そのものの染み出しと
    区別できないので 0 とする(合成画像と実画像で確かめた値)。
    """
    if profile.size == 0:
        return 0
    tau = max(min_tau, tau_factor * float(np.percentile(profile, 20)))
    high = np.nonzero(profile > tau)[0]
    if high.size == 0 or high[0] > max_gap:
        return 0
    end = int(high[0])
    for idx in high[1:]:
        if idx - end - 1 <= max_gap:
            end = int(idx)
        else:
            break
    trim = end - 1
    return trim if trim >= min_trim else 0


def _trim_thin_frame(img: np.ndarray, valid: np.ndarray, max_frac: float = 0.10) -> dict:
    """`img` は (H, W, 3) の色(Lab)。"""
    h, w = img.shape[:2]
    depth = max(8, int(max_frac * min(h, w)))
    max_gap = max(3, round(0.01 * max(h, w)))
    t = np.transpose(img, (1, 0, 2))
    out = {}
    for side, (g, v) in {
        "top": (img, valid), "bottom": (img[::-1], valid[::-1]),
        "left": (t, valid.T), "right": (t[::-1], valid.T[::-1]),
    }.items():
        out[side] = frame_depth(edge_profile(g, v, depth), max_gap)
    return out


def remove_frame(rgb: np.ndarray, analysis_side: int = 800) -> FrameResult:
    """`rgb`(H, W, 3, uint8)から余白と額縁を除いた内容の矩形と、その中に残る余白のマスクを返す。"""
    H, W = rgb.shape[:2]
    k = _analysis_scale(rgb.shape, analysis_side)
    small = cv2.resize(rgb, (max(1, round(W * k)), max(1, round(H * k))), interpolation=cv2.INTER_AREA) if k < 1 else rgb

    backdrop_s = find_backdrop(small)
    x0, y0, x1, y1 = _content_box(backdrop_s)
    sub = small[y0:y1, x0:x1]
    sub_backdrop = backdrop_s[y0:y1, x0:x1]
    lab = cv2.GaussianBlur(cv2.cvtColor(sub, cv2.COLOR_RGB2LAB).astype(np.float32), (0, 0), 1.0)
    trim = _trim_thin_frame(lab, ~sub_backdrop)
    x0, y0 = x0 + trim["left"], y0 + trim["top"]
    x1, y1 = x1 - trim["right"], y1 - trim["bottom"]

    # 解析用の縮小画像の座標を、入力画像の座標に戻す
    box = (int(round(x0 / k)), int(round(y0 / k)), min(W, int(round(x1 / k))), min(H, int(round(y1 / k))))
    # 余白マスクは縮小画像で求めたものを、箱の大きさに拡大して返す(原寸で求めると平らさの基準が変わるため)
    inner = backdrop_s[y0:y1, x0:x1].astype(np.uint8)
    bw, bh = box[2] - box[0], box[3] - box[1]
    backdrop = cv2.resize(inner, (bw, bh), interpolation=cv2.INTER_NEAREST).astype(bool) if inner.size else np.zeros((bh, bw), bool)
    if (box[2] - box[0]) * (box[3] - box[1]) < MIN_CONTENT_FRAC * W * H:
        # 暗い絵や白い絵を丸ごと余白と誤認したときの安全弁。何も削らない
        return FrameResult(box=(0, 0, W, H), backdrop=np.zeros((H, W), bool), trimmed={}, fallback=True)
    return FrameResult(box=box, backdrop=backdrop, trimmed=trim)
