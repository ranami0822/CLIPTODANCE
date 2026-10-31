"""要望(JSON)から、風景コラージュを作る。

  python scripts/compose.py --pool meiga --request requests/meadow.json [--placer clairic|rules] [--base-rank 0] [--out data/out]

要望の JSON 例:
  {"text": "a horse and a dog on a quiet shore", "base_query": "calm landscape with sky",
   "elements": [{"name": "horse", "query": "a horse", "kind": "horse", "role": "main"},
                {"name": "dog", "query": "a dog", "kind": "dog"}]}
出力: <out>/<時刻>/collage.png, base.png, log.json。配置は、既定では Clairic に尋ねる(要 CLOUDFLARE_API_TOKEN / ACCOUNT_ID)。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.compose.heights import HeightTable  # noqa: E402
from pipeline.compose.placer import ClairicPlacer, RuleBasedPlacer  # noqa: E402
from pipeline.compose.render import RenderConfig  # noqa: E402
from pipeline.compose.run import ComposeRequest, compose  # noqa: E402
from pipeline.pool.db import PoolDB  # noqa: E402
from pipeline.pool.embed import ClipEmbedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--placer", choices=["clairic", "rules"], default="clairic")
    ap.add_argument("--base-rank", type=int, default=0, help="土台の候補の何番目を使うか(0 が最上位)")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--color-strength", type=float, default=0.2)
    a = ap.parse_args()

    db = PoolDB(Path("data/pool") / a.pool / "pool.sqlite")
    if a.placer == "clairic":
        from adapters.cloudflare import CloudflareEstimator
        placer = ClairicPlacer(CloudflareEstimator())
    else:
        placer = RuleBasedPlacer()
    res = compose(db, a.pool, ClipEmbedder(), placer, ComposeRequest.from_json(a.request), HeightTable.load(),
                  RenderConfig(color_strength=a.color_strength), base_rank=a.base_rank)
    out = Path(a.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    res.image.save(out / "collage.png")
    res.base.save(out / "base.png")
    (out / "log.json").write_text(json.dumps(res.log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out / "collage.png")
    for w in res.log["warnings"]:
        print("警告:", w)


if __name__ == "__main__":
    main()
