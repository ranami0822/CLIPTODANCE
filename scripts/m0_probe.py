"""M0 のゲート確認: Clef と Clairic を実際に呼び、結果・使用量・消費ニューロンを JSON ログに残す。

合成データ(文字列と自作の小さな画像)だけを送る。素材は送らない。無料枠の消費は数ニューロン程度。
使い方:  python scripts/m0_probe.py [--out data/logs]
必要な環境変数: CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters import CachedJudge, EstimateField, FieldKind, choice, noul, score  # noqa: E402
from adapters.cloudflare import CloudflareEstimator, CloudflareJudge  # noqa: E402


def solid_png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/logs")
    ap.add_argument("--cache", default="data/judge_cache/m0_probe.sqlite")
    args = ap.parse_args()

    red = solid_png(64, 64, (200, 30, 30))
    judge = CachedJudge(CloudflareJudge(), args.cache)
    estimator = CloudflareEstimator()
    log: dict = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "calls": []}

    qs = [
        noul("has_tree", "Is there a tree in the scene?"),
        choice("tree_side", "Which side is the tree on?", ["left", "right", "center"]),
        score("sunny", "How sunny is the scene?", ["dark", "cloudy", "bright", "very bright"]),
    ]
    for label, kwargs in [
        ("clef_text", dict(questions=qs, state="A sunny meadow with one large oak tree on the right.")),
        ("clef_image", dict(questions=[noul("is_red", "Is the image mostly red?")], images=[red])),
    ]:
        resp = judge.ask(**kwargs)
        log["calls"].append({"label": label, "model": resp.model_id,
                             "usage": dataclasses.asdict(resp.usage),
                             "results": [dataclasses.asdict(r) for r in resp.results]})

    fields = [
        EstimateField("x", FieldKind.ESTIMATE, "Horizontal position of the tree centre, 0 = left edge",
                      minimum=0, maximum=1, step=0.01),
        EstimateField("height_m", FieldKind.ESTIMATE, "Typical real-world height", unit="m",
                      minimum=0, maximum=40, step=0.5),
    ]
    est = estimator.estimate("A sunny meadow with one large oak tree on the right side.", fields)
    log["calls"].append({"label": "clairic_text", "model": est.model_id,
                         "usage": dataclasses.asdict(est.usage),
                         "values": {k: dataclasses.asdict(v) for k, v in est.values.items()}})

    total = sum((c["usage"]["neurons"] or 0.0) for c in log["calls"])
    log["total_neurons_this_run"] = total
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"m0_probe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"log: {path}")
    for c in log["calls"]:
        u = c["usage"]
        print(f"{c['label']:14s} api_calls={u['api_calls']} cache_hits={u['cache_hits']} "
              f"neurons={u['neurons']} input_tokens={u['input_tokens']} latency={u['latency_s']:.2f}s")
    print(f"total neurons this run: {total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
