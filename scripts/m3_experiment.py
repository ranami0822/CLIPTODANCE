"""逐次配置の切り分け実験(設計書 M3)。土台ごと・構成ごとに複数回実行し、結果を JSONL に追記する(途中で止まっても、再実行で続きから)。

  python scripts/m3_experiment.py --pool meiga --request requests/quiet_landscape.json --base-rank 0 --reps 3 --out data/out/m3exp

構成(variant)は VARIANTS を参照。指標は、補正の前の提案が生息場所(水・陸)を守れたか、重なり・散らばり、Clef(clef-flash)による
「自然さ」の採点(**検証していない目安**)など。1 回の呼び出しは数ニューロンで、全部で 1 日の無料枠(10,000)の半分弱の見込み。
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.base import BudgetExceeded, RateLimited, score  # noqa: E402
from pipeline.compose.heights import HeightTable  # noqa: E402
from pipeline.compose.placer import ClairicPlacer, RuleBasedPlacer, SequentialClairicPlacer  # noqa: E402
from pipeline.compose.render import RenderConfig  # noqa: E402
from pipeline.compose.run import ComposeRequest, compose  # noqa: E402

# 名前 → (配置の作り方, 水の地図と habitat の補正を使うか, 1 回だけ実行するか)。Clairic は同じ入力に同じ出力を返す(22/24 が一致。2026-10-31 の実測)ので、乱数の順(seq_random)以外は 1 回でよい
VARIANTS = {
    "rules":               ("rules", True, True),
    "allatonce":           ("allatonce", True, True),
    "allatonce_nocorr":    ("allatonce", False, True),
    "seq_full":            ({}, True, True),
    "seq_noimage":         ({"use_image": False, "use_grid": False}, True, True),
    "seq_nogrid":          ({"use_grid": False}, True, True),
    "seq_nohistory":       ({"use_history": False}, True, True),
    "seq_nowaterhint":     ({"use_water_hint": False}, True, True),
    "seq_nowater_nocorr":  ({"use_water_hint": False}, False, True),
    "seq_hint_text":       ({"water_hint_parts": ("text",)}, True, True),
    "seq_hint_range":      ({"water_hint_parts": ("range",)}, True, True),
    "seq_nointent":        ({"use_intent": False}, True, True),
    "seq_bare":            ({"use_intent": False, "use_image": False, "use_grid": False, "use_history": False}, True, True),   # 1 要素ずつ尋ねる以外を全部外す
    "seq_reverse":         ({"order": "reverse"}, True, True),
    "seq_back_to_front":   ({"order": "back_to_front"}, True, True),
    "seq_front_to_back":   ({"order": "front_to_back"}, True, True),
    "seq_random":          ({"order": "random"}, True, False),
}

NAT_SCALE = ["very unnatural: objects clash, float in the wrong place or overlap badly", "unnatural", "acceptable", "natural",
             "very natural: it reads as one coherent landscape"]


class CountingEstimator:
    """Estimator を包み、消費ニューロンと呼び出し数を数える。"""
    def __init__(self, inner):
        self.inner, self.neurons, self.calls = inner, 0.0, 0
        self.model_id, self.capabilities = inner.model_id, inner.capabilities

    def estimate(self, text, fields, images=()):
        r = self.inner.estimate(text, fields, images)
        self.neurons += r.usage.neurons or 0.0
        self.calls += 1
        return r


def cell_of(water, x, y):
    ny, nx = water.p.shape
    return min(ny - 1, max(0, int(y * ny))), min(nx - 1, max(0, int(x * nx)))


def in_habitat(water, inst_habitat, x, y):
    if inst_habitat not in ("water", "land"):
        return None
    j, i = cell_of(water, x, y)
    return bool((water.p[j, i] >= water.threshold) == (inst_habitat == "water"))


def metrics(log, water, habitats, nat):
    rend = log["render"]
    raw = {}
    for t in log.get("placer_trace", []):
        raw[t["instance"]] = (t["proposal"]["x"], t["proposal"]["y"])
    if not raw:
        raw = {r["instance"]: (r["x"], r["y"]) for r in log["placements_raw"]}
    ground = [i for i, h in habitats.items() if h in ("water", "land")]
    raw_ok = [in_habitat(water, habitats[i], *raw[i]) for i in ground]
    fin = {r["instance"]: (r["applied"]["x"], r["applied"]["y"]) for r in rend}
    fin_ok = [in_habitat(water, habitats[i], *fin[i]) for i in ground]
    boats = [i for i in ground if habitats[i] == "water"]
    pts = np.array(list(fin.values()))
    pair = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(pts, 2)]
    corr = [c for r in rend for c in r["corrections"]]
    names = lambda k: sum(1 for c in corr if k in c)
    sc = [r["applied"]["scale_mult"] for r in rend]
    return {
        "habitat_raw_ok": sum(raw_ok) / max(len(raw_ok), 1), "habitat_final_ok": sum(fin_ok) / max(len(fin_ok), 1),
        "boats_raw_ok": (sum(in_habitat(water, "water", *raw[i]) for i in boats) / len(boats)) if boats else None,
        "boats_final_ok": (sum(in_habitat(water, "water", *fin[i]) for i in boats) / len(boats)) if boats else None,
        "min_dist": min(pair), "mean_dist": float(np.mean(pair)), "std_x": float(pts[:, 0].std()),
        "max_cover": max(r["covered_max"] for r in rend), "overlap_exceeded": names("overlap_exceeded"), "nudged": names("nudged_x"),
        "moved_toward_horizon": names("moved_toward_horizon"), "shrunk_for_habitat": names("shrunk_for_habitat"),
        "scale_mean": float(np.mean(sc)), "scale_min": min(sc), "scale_max": max(sc),
        "habitat_moves": len(log.get("habitat_corrections", [])), "nat_score": nat,
        "boat_width_frac_max": max([(r["bbox_px"][2] - r["bbox_px"][0]) / log["render_size"][0] for r in rend if r["kind"] == "boat" and r["bbox_px"]], default=None),
        "height_frac_max": max((r["bbox_px"][3] - r["bbox_px"][1]) / log["render_size"][1] for r in rend if r["bbox_px"] and not r["airborne"]),
        "final": {k: [round(v[0], 3), round(v[1], 3)] for k, v in fin.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--base-rank", type=int, default=0)
    ap.add_argument("--base-pool", default=None, help="土台を別のプールから選ぶ(評価用の土台)")
    ap.add_argument("--base-id", default=None, help="使う土台の画像 ID(--base-pool と一緒に)")
    ap.add_argument("--no-nat", action="store_true", help="Clef の自然さの採点をしない(使えないと分かったので、評価では省く)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default="data/out/m3exp")
    a = ap.parse_args()

    from adapters.cache import CachedJudge
    from adapters.cloudflare import CloudflareEstimator, CloudflareJudge
    from pipeline.compose.water import estimate_water_map
    from pipeline.pool.db import PoolDB
    from pipeline.pool.embed import ClipEmbedder
    from pipeline.pool.imageops import png_bytes

    out_dir = Path(a.out) / (f"base_{a.base_id}" if a.base_id else f"base{a.base_rank}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "results.jsonl"
    done = {(r["variant"], r["rep"]) for r in map(json.loads, jsonl.read_text().splitlines())} if jsonl.exists() else set()

    db = PoolDB(Path("data/pool") / a.pool / "pool.sqlite")
    embedder, heights, request = ClipEmbedder(), HeightTable.load(), ComposeRequest.from_json(a.request)
    cached = CachedJudge(CloudflareJudge(model="clef-flash"), Path("data/cache/judge.sqlite"))
    nat_judge = CloudflareJudge(model="clef-flash")
    cfg = RenderConfig(color_strength=0.2)

    # 指標用の水の地図(compose の中と同じ土台・同じ質問なので、キャッシュに当たる)
    base_db = PoolDB(Path("data/pool") / a.base_pool / "pool.sqlite") if a.base_pool else None
    kw = dict(base_rank=a.base_rank, base_db=base_db, base_pool=a.base_pool, base_id=a.base_id)
    try:
        probe = compose(db, a.pool, embedder, RuleBasedPlacer(), request, heights, cfg, judge=cached, **kw)
    except RuntimeError as e:
        print("土台として使えない:", e, flush=True)
        return
    except (BudgetExceeded, RateLimited) as e:
        print("中断(予算かレート制限):", type(e).__name__, e, flush=True)
        raise SystemExit(3)
    from pipeline.compose.water import WaterMap
    water = WaterMap(p=np.array(probe.log["water_map"]["p"]), horizon_y=probe.log["base"]["horizon_in_canvas"])
    info = {"base": probe.log["base"]["image_id"], "water_bbox": water.bbox(), "water_cells": int(water.mask().sum())}
    print("base", a.base_rank, info, flush=True)
    (out_dir / "base.json").write_text(json.dumps(info, ensure_ascii=False))
    # habitat は compose のログに入っていないので、要望と表から作る(compose と同じ規則)
    habitats = {}
    for ei, spec in enumerate(request.elements):
        e = heights.get(spec.kind)
        for n in range(spec.count):
            habitats[f"e{ei}_{spec.name}_{n}".replace(" ", "_")] = None if e.airborne else (spec.habitat or e.habitat)

    t_start = time.monotonic()
    for rep in range(a.reps):
        for name in a.variants:
            kind, use_water, single = VARIANTS[name]
            if (name, rep) in done or (single and rep > 0):
                continue
            est = CountingEstimator(CloudflareEstimator())
            if kind == "rules":
                placer = RuleBasedPlacer()
            elif kind == "allatonce":
                placer = ClairicPlacer(est)
            else:
                placer = SequentialClairicPlacer(est, seed=rep, **kind)
            try:
                res = compose(db, a.pool, embedder, placer, request, heights, cfg, judge=cached if use_water else None, **kw)
                nat = None
                if not a.no_nat:
                    buf = io.BytesIO(); small = res.image.convert("RGB"); small.thumbnail((768, 768)); small.save(buf, "PNG")
                    r = nat_judge.ask([score("nat", "How natural and coherent does this collage look as one landscape painting?", NAT_SCALE)], images=[buf.getvalue()], state="")
                    nat = float(r.results[0].score)
            except (BudgetExceeded, RateLimited) as e:
                print("中断(予算かレート制限):", type(e).__name__, e, flush=True)
                return
            m = metrics(res.log, water, habitats, nat)
            img_path = out_dir / f"{name}_r{rep}.png"
            res.image.save(img_path)
            rec = {"base_rank": a.base_rank, "base": info["base"], "variant": name, "rep": rep, "neurons": round(est.neurons, 1), "calls": est.calls,
                   "seconds": res.log["seconds"], "image": str(img_path), **m}
            with jsonl.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{time.monotonic() - t_start:5.0f}s] rep{rep} {name:20s} 水陸(提案){m['habitat_raw_ok']:.2f} 重なり超過{m['overlap_exceeded']} 最小距離{m['min_dist']:.2f} ({est.neurons:.0f}n)", flush=True)


if __name__ == "__main__":
    main()
