"""m3_experiment.py の結果(data/out/m3exp/base*/results.jsonl)を集計する。有意性の検定はしない(回数が少ない)。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METRICS = [("habitat_raw_ok", "水陸を守った提案の割合(補正前)"), ("boats_raw_ok", "船が水の上(補正前)"), ("overlap_exceeded", "重なり超過の数"),
           ("min_dist", "点どうしの最小距離"), ("max_cover", "最大被覆"), ("scale_mean", "倍率の平均"), ("boat_width_frac_max", "船の幅の最大比"), ("height_frac_max", "地上の物の高さの最大比")]


def load(root: Path):
    rows = []
    for f in sorted(root.glob("base*/results.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines()]
    return rows


def collapse(rows):
    """同じ土台・同じ構成の繰り返し(乱数の順のとき以外は同じ結果)を平均して、1 土台 1 行にする。独立な観測は土台の数だけ。"""
    keys = [k for k, _ in METRICS]
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["base"], r["variant"]), []).append(r)
    for (b, v), rs in groups.items():
        row = {"base": b, "variant": v, "rep": 0, "n_runs": len(rs), "final": rs[0]["final"]}
        for k in keys:
            xs = [r.get(k) for r in rs if r.get(k) is not None]
            row[k] = float(np.mean(xs)) if xs else None
        out.append(row)
    return out


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/out/m3exp")
    ap.add_argument("--ref", default="seq_full")
    a = ap.parse_args()
    raw_rows = load(Path(a.root))
    rows = collapse(raw_rows)
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    print(f"実行数 {len(raw_rows)}、土台 {len(sorted({r['base'] for r in rows}))} 個 {sorted({r['base'] for r in rows})}(構成ごとに土台ごとの平均 → 土台をまたいで集計。n は土台の数)\n")
    print("== 構成ごとの平均(範囲)")
    print(f"{'構成':20s} {'n':>3s} " + " ".join(f"{k:>16s}" for k, _ in METRICS))
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        cells = []
        for k, _ in METRICS:
            xs = [r.get(k) for r in rs if r.get(k) is not None]
            cells.append(f"{np.mean(xs):.2f}({min(xs):.2f}-{max(xs):.2f})" if xs else "-")
        print(f"{v:20s} {len(rs):3d} " + " ".join(f"{c:>16s}" for c in cells))
    print(f"\n== {a.ref} との差(同じ土台・同じ回どうし。平均差 [増えた回数/減った回数/同じ] )")
    ref = {(r["base"], r["rep"]): r for r in rows if r["variant"] == a.ref}
    for v in variants:
        if v == a.ref:
            continue
        line = []
        for k, _ in METRICS:
            d = [r[k] - ref[(r["base"], r["rep"])][k] for r in rows if r["variant"] == v and (r["base"], r["rep"]) in ref and r.get(k) is not None and ref[(r["base"], r["rep"])].get(k) is not None]
            line.append(f"{np.mean(d):+.2f} [{sum(x > 1e-9 for x in d)}/{sum(x < -1e-9 for x in d)}/{sum(abs(x) <= 1e-9 for x in d)}]" if d else "-")
        print(f"{v:20s} n={sum(1 for r in rows if r['variant']==v and (r['base'], r['rep']) in ref):2d} " + " ".join(f"{c:>20s}" for c in line))
    rows_for_var = raw_rows
    print("\n== 繰り返しのぶれ(同じ土台・同じ構成で、各要素の最終位置の標準偏差の平均。小さいほど再現する)")
    for v in variants:
        sds = []
        for b in {r["base"] for r in raw_rows}:
            rs = [r for r in raw_rows if r["variant"] == v and r["base"] == b]
            if len(rs) < 2:
                continue
            for inst in rs[0]["final"]:
                xy = np.array([r["final"][inst] for r in rs if inst in r["final"]])
                sds.append(float(np.sqrt(xy.var(axis=0).sum())))
        print(f"{v:20s} {np.mean(sds):.3f}" if sds else f"{v:20s} (1 回のみ)")
    print("\n== 土台ごと(seq_full / allatonce_nocorr / rules の habitat 守り(補正前)と自然さ)")
    for b in sorted({r["base"] for r in rows}):
        for v in ("rules", "allatonce_nocorr", "seq_full", "seq_nowater_nocorr"):
            rs = [r for r in rows if r["base"] == b and r["variant"] == v]
            if rs:
                print(f"{b:12s} {v:20s} 水陸{np.mean([r['habitat_raw_ok'] for r in rs]):.2f} 最小距離{np.mean([r['min_dist'] for r in rs]):.2f}")


if __name__ == "__main__":
    main()
