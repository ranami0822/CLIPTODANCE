"""素材の選択(M2 の試作): 土台(風景)と、要素ごとの切り抜きを、CLIP の検索で選ぶ。

切り抜きは、人または Claude が「単体で使える」と印したものだけを候補にする(足切りの手がかりが
まだ固まっていないので、印のある素材を使う)。人の印があれば人の印を優先する。"""
from __future__ import annotations

from dataclasses import dataclass

from pipeline.pool.db import PoolDB
from pipeline.pool.embed import Embedder
from pipeline.pool.index import VectorIndex

from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.pool.frame import find_black_corners

from .canvas import CropPlan, plan_crop


def approved_single(row: dict) -> tuple[bool, str]:
    """(単体で使えるか, 印の出どころ)。人の印があれば人、なければ Claude の印。どちらもなければ使わない。"""
    if row.get("usable") is not None:
        return bool(row["usable"]), "human"
    if row.get("usable_claude") is not None:
        return bool(row["usable_claude"]), "claude"
    return False, "none"


def effective_horizon(row: dict) -> tuple[float, str] | None:
    """土台の地平線の値と出どころ。人の印(確認済み)> Claude の印 > 自動の初期値。地平線のない絵(人または Claude の印)は None。"""
    if row["horizon_checked"] == 2:
        return None
    if row["horizon_checked"] == 1:
        return row["horizon_y"], "human"
    if row.get("horizon_claude_na") == 1:
        return None
    if row.get("horizon_claude_y") is not None:
        return row["horizon_claude_y"], "claude"
    if row["horizon_y"] is None:
        return None
    return row["horizon_y"], "auto"


@dataclass
class CutoutChoice:
    row: dict
    similarity: float
    label_source: str


@dataclass
class BaseChoice:
    row: dict
    plan: CropPlan
    similarity: float
    horizon_source: str       # "human"(目視で確認済み)、"claude"(Claude の印)、"auto"(自動の初期値)
    score: float


def select_cutouts(db: PoolDB, pool: str, embedder: Embedder, query: str, k: int = 10, exclude: set[str] | None = None) -> list[CutoutChoice]:
    where = "pool = ? AND ((usable = 1) OR (usable IS NULL AND usable_claude = 1))"
    index = VectorIndex.from_db(db, "cutouts", where, (pool,))
    vec = embedder.encode_texts([f"a painting of {query}"])[0]
    out = []
    for cid, sim in index.search(vec, k + len(exclude or ())):
        if exclude and cid in exclude:
            continue
        row = db.get("cutouts", cid)
        ok, src = approved_single(row)
        if ok:
            out.append(CutoutChoice(row, sim, src))
        if len(out) >= k:
            break
    return out


def backdrop_in_window(row: dict, plan: CropPlan, max_side: int = 400) -> float:
    """切り出す窓の中に、黒い余白(アーチ形の絵の四隅)がどれだけ残るか(0〜1)。画像ファイルがなければ 0。"""
    path = Path(row.get("work_path") or "")
    if not path.is_file():
        return 0.0
    im = Image.open(path).convert("RGB")
    k = min(1.0, max_side / max(im.size))
    small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
    bd = find_black_corners(np.asarray(small))
    x0, y0, x1, y1 = [int(round(v * k)) for v in plan.box]
    win = bd[y0:y1, x0:x1]
    return float(win.mean()) if win.size else 0.0


def select_base(db: PoolDB, pool: str, embedder: Embedder, query: str, canvas_target: float = 2 / 3, min_crop_ratio: float = 0.6,
                tolerance: float = 0.08, top: int = 30, dev_weight: float = 2.0, max_backdrop: float = 0.002) -> list[BaseChoice]:
    """土台の候補。CLIP の類似度から、切り出し後の地平線が目標(下の三分の一)からずれる分を引いて並べる。
    地平線のない絵(horizon_checked = 2)と、3:2 の窓が作れない絵、ずれが許容を超える絵は除く。"""
    index = VectorIndex.from_db(db, "images", "pool = ?", (pool,))
    vec = embedder.encode_texts([f"a landscape painting, {query}"])[0]
    out = []
    for iid, sim in index.search(vec, top):
        row = db.get("images", iid)
        eh = effective_horizon(row)
        if eh is None:
            continue
        plan = plan_crop(row["width"], row["height"], eh[0], canvas_target, min_crop_ratio)
        if plan is None or plan.deviation > tolerance:
            continue
        if backdrop_in_window(row, plan) > max_backdrop:      # 窓の中に余白が残る土台は使わない
            continue
        out.append(BaseChoice(row, plan, sim, eh[1], sim - dev_weight * plan.deviation))
    return sorted(out, key=lambda c: -c.score)
