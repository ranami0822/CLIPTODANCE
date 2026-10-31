"""要望 → 土台の選択 → 素材の選択 → 配置 → 合成 → 出力(PNG と JSON のログ)。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.pool.db import PoolDB
from pipeline.pool.embed import Embedder
from pipeline.pool.imageops import png_bytes

from .canvas import render_base
from .heights import HeightTable
from .habitat import apply_habitat, foot_ok_for
from .placer import PlaceContext, Placer
from .render import RenderConfig, render_collage
from .select import select_base, select_cutouts
from .types import CanvasInfo, ElementSpec, Instance
from .water import WaterMap, estimate_water_map


@dataclass
class ComposeRequest:
    text: str
    base_query: str
    elements: list[ElementSpec]
    canvas_width: int = 1800

    @classmethod
    def from_json(cls, path: str | Path) -> "ComposeRequest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(text=d["text"], base_query=d.get("base_query", "landscape"),
                   elements=[ElementSpec(**e) for e in d["elements"]], canvas_width=d.get("canvas_width", 1800))


@dataclass
class ComposeResult:
    image: Image.Image
    base: Image.Image
    log: dict = field(default_factory=dict)


def compose(db: PoolDB, pool: str, embedder: Embedder, placer: Placer, request: ComposeRequest, heights: HeightTable,
            cfg: RenderConfig | None = None, base_rank: int = 0, judge=None, base_db: PoolDB | None = None, base_pool: str | None = None,
            base_id: str | None = None) -> ComposeResult:
    """judge(Clef 相当)を渡すと、水と陸の地図を作り、habitat(水・陸)の指定を守るよう足元を直す。渡さなければ、水と陸は見ない。"""
    t0 = time.monotonic()
    cfg = cfg or RenderConfig()
    log: dict = {"request": {"text": request.text, "base_query": request.base_query, "elements": [asdict(e) for e in request.elements]},
                 "placer": placer.name, "warnings": []}

    # 土台は別のプール(base_db / base_pool)から選べる(開発用と評価用の土台を分けるため)。base_id があれば、その土台を使う(使えなければ失敗)。
    bases = select_base(base_db or db, base_pool or pool, embedder, request.base_query, top=200 if base_id else 30)
    if not bases:
        raise RuntimeError("土台にできる風景がない(地平線を下の三分の一に置けるものがない)")
    if base_id is not None:
        matches = [b for b in bases if b.row["id"] == base_id]
        if not matches:
            raise RuntimeError(f"指定した土台 {base_id} は使えない(地平線がない・窓が作れない・余白が残る、のどれか)")
        base = matches[0]
    else:
        base = bases[min(base_rank, len(bases) - 1)]
    src = Image.open(base.row["work_path"]).convert("RGB")
    canvas_img = render_base(src, base.plan, request.canvas_width)
    canvas = CanvasInfo(width=canvas_img.width, height=canvas_img.height, horizon_y=base.plan.horizon_y, source_image_id=base.row["id"],
                        crop_box=base.plan.box, horizon_deviation=base.plan.deviation, horizon_source=base.horizon_source)
    if base.horizon_source != "human":
        log["warnings"].append(f"土台の地平線は {base.horizon_source} の値で、あなたの目視の確認がない")
    log["base"] = {"image_id": base.row["id"], "title": base.row.get("title"), "similarity": round(base.similarity, 4), "crop_box": list(base.plan.box),
                   "horizon_in_canvas": round(base.plan.horizon_y, 4), "horizon_source": base.horizon_source, "window_ratio": round(base.plan.window_ratio, 3),
                   "alternatives": [b.row["id"] for b in bases[:5]]}

    instances: list[Instance] = []
    sprites: dict[str, np.ndarray] = {}
    used: set[str] = set()
    chosen_log = []
    for ei, spec in enumerate(request.elements):
        entry = heights.get(spec.kind)
        cands = select_cutouts(db, pool, embedder, spec.query, k=spec.count, exclude=used)
        if len(cands) < spec.count:
            log["warnings"].append(f"{spec.name}: 候補が {len(cands)} 個しかない(要求 {spec.count} 個)")
        for n, c in enumerate(cands):
            rgba = np.asarray(Image.open(c.row["cutout_path"]).convert("RGBA"))
            iid = f"e{ei}_{spec.name}_{n}".replace(" ", "_")
            by_width = (not entry.airborne) and spec.height_m is None and entry.width_m is not None   # 明示した高さがなければ、表の幅を使う
            inst = Instance(id=iid, spec=spec, cutout_id=c.row["id"],
                            height_m=None if (entry.airborne or by_width) else (spec.height_m or entry.mean),
                            height_frac=entry.height_frac if entry.airborne else None, airborne=entry.airborne,
                            aspect=rgba.shape[1] / rgba.shape[0], width_m=entry.width_m if by_width else None,
                            habitat=None if entry.airborne else (spec.habitat or entry.habitat))
            instances.append(inst); sprites[iid] = rgba; used.add(c.row["id"])
            chosen_log.append({"instance": iid, "cutout_id": c.row["id"], "similarity": round(c.similarity, 4), "label_source": c.label_source,
                               "cutout_area_ratio": round(c.row["area_ratio"], 4)})
    log["cutouts"] = chosen_log
    if not instances:
        raise RuntimeError("配置する切り抜きが 1 つもない")

    water: WaterMap | None = None
    if judge is not None and any(i.habitat for i in instances):
        water, wlog = estimate_water_map(judge, canvas_img, canvas.horizon_y)
        log["water_map"] = {**wlog, "p": water.p.round(2).tolist(), "bbox": water.bbox()}
        if any(i.habitat == "water" for i in instances) and not water.any_water():
            log["warnings"].append("水の上に置く物があるが、土台に水が見つからなかった(足元は補正されない)")
    elif any(i.habitat for i in instances):
        log["warnings"].append("水・陸の指定がある物があるが、judge がないので水と陸は見ていない")
    ctx = PlaceContext(base=canvas_img, sprites=sprites, cfg=cfg, water=water)
    placements = placer.place(request.text, instances, canvas, png_bytes(canvas_img), context=ctx)
    if getattr(placer, "trace", None):
        log["placer_trace"] = placer.trace
    log["placements_raw"] = [{"instance": i.id, "x": round(p.x, 4), "y": round(p.y, 4), "scale_mult": round(p.scale_mult, 3), "rotation_deg": round(p.rotation_deg, 2)}
                             for i, p in zip(instances, placements)]       # 補正の前(Clairic などの提案そのまま。逐次配置では、要素ごとの補正のあと)
    if water is not None:
        placements, hlog = apply_habitat(instances, placements, water, cfg.limits)
        log["habitat_corrections"] = hlog
    image, render_log = render_collage(canvas_img, canvas.horizon_y, [(i, sprites[i.id], p) for i, p in zip(instances, placements)], cfg, foot_ok=foot_ok_for(water))
    log["placements"] = [{"instance": i.id, "x": round(p.x, 4), "y": round(p.y, 4), "scale_mult": round(p.scale_mult, 3), "rotation_deg": round(p.rotation_deg, 2),
                          "x_range80": [round(v, 3) for v in p.x_range80] if p.x_range80 else None,
                          "y_range80": [round(v, 3) for v in p.y_range80] if p.y_range80 else None} for i, p in zip(instances, placements)]
    log["render"] = render_log
    log["render_size"] = [image.width, image.height]
    log["seconds"] = round(time.monotonic() - t0, 1)
    return ComposeResult(image=image, base=canvas_img, log=log)
