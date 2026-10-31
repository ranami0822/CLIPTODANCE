"""素材プールの操作: fetch(取得) → ingest(取り込み) → judge(属性判定) → sheet / search / set-horizon / stats。

  python scripts/pool.py fetch  --pool meiga --query landscape --dept 11 --n 20
  python scripts/pool.py ingest --pool meiga [--limit N] [--force]
  python scripts/pool.py judge  --pool meiga --model clef-flash --max-neurons 3000
  python scripts/pool.py sheet  --pool meiga --kind horizon|cutouts
  python scripts/pool.py search --pool meiga --text "a horse" [--k 12]
  python scripts/pool.py set-horizon --pool meiga --id met-436535 --y 0.62
  python scripts/pool.py stats  --pool meiga

データは data/pool/<pool>/ に置く(git 管理外)。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.pool.db import PoolDB  # noqa: E402
from pipeline.pool.sources.met import MetSource, SourceItem  # noqa: E402


def pool_root(pool: str) -> Path:
    return Path("data/pool") / pool


def open_db(pool: str) -> PoolDB:
    return PoolDB(pool_root(pool) / "pool.sqlite")


def cmd_fetch(a) -> None:
    root = pool_root(a.pool) / "incoming"
    root.mkdir(parents=True, exist_ok=True)
    met = MetSource()
    got = 0
    offset = 0
    while got < a.n:
        total, ids = met.search(a.query, department_id=a.dept, medium=a.medium, offset=offset, limit=100)
        if not ids:
            break
        print(f"search '{a.query}': total={total} offset={offset}")
        for oid in ids:
            if got >= a.n:
                break
            if (root / f"met-{oid}.json").exists():
                continue
            obj = met.get_object(oid)
            item = MetSource.to_item(obj, classification=a.classification) if obj else None
            if item is None:
                continue
            url = item.meta.get("primaryImageSmall") if a.small and item.meta.get("primaryImageSmall") else item.image_url
            data = met.download(url)
            (root / f"met-{oid}.jpg").write_bytes(data)
            (root / f"met-{oid}.json").write_text(json.dumps(dataclasses.asdict(item), ensure_ascii=False, indent=1), encoding="utf-8")
            got += 1
            print(f"  [{got}/{a.n}] met-{oid} {len(data)/1e6:.1f}MB {item.title[:50]!r}")
        offset += 100
        if offset >= 10000:
            break
    print(f"fetched {got}")


def cmd_ingest(a) -> None:
    from pipeline.pool.embed import ClipEmbedder
    from pipeline.pool.ingest import IngestConfig, ingest_image
    from pipeline.pool.segment import SamSegmenter

    root = pool_root(a.pool)
    db = open_db(a.pool)
    todo = sorted((root / "incoming").glob("*.json"))
    if not a.force:
        todo = [p for p in todo if db.get("images", p.stem) is None]
    if a.limit:
        todo = todo[: a.limit]
    print(f"to ingest: {len(todo)}")
    if not todo:
        return
    t0 = time.monotonic()
    segmenter = SamSegmenter(points_per_side=a.points_per_side)
    embedder = ClipEmbedder()
    print(f"models loaded in {time.monotonic()-t0:.1f}s ({segmenter.name}, {embedder.name})")
    cfg = IngestConfig()
    for i, meta_path in enumerate(todo, 1):
        item = SourceItem(**json.loads(meta_path.read_text(encoding="utf-8")))
        res = ingest_image(db, root, a.pool, item, meta_path.with_suffix(".jpg").read_bytes(), segmenter, embedder, cfg, force=a.force)
        s = " ".join(f"{k}={v:.1f}s" for k, v in res.seconds.items())
        print(f"[{i}/{len(todo)}] {res.image_id}: candidates={res.n_candidates} cutouts={res.n_cutouts} {s}", flush=True)


def cmd_judge(a) -> None:
    from adapters import CachedJudge
    from adapters.cloudflare import CloudflareJudge
    from pipeline.pool.ingest import judge_pool

    root = pool_root(a.pool)
    judge = CachedJudge(CloudflareJudge(model=a.model), root / "judge_cache.sqlite")
    db = open_db(a.pool)
    print(json.dumps(judge_pool(db, judge, a.pool, max_neurons=a.max_neurons, limit=a.limit), ensure_ascii=False, indent=1))


def cmd_sheet(a) -> None:
    from pipeline.pool.sheet import cutout_sheet, horizon_sheet

    db = open_db(a.pool)
    out = pool_root(a.pool) / "sheets"
    if a.kind == "horizon":
        print(horizon_sheet(db.rows("images", "pool = ?", (a.pool,)), out / "horizon.png"))
    else:
        print(cutout_sheet(db.rows("cutouts", "pool = ?", (a.pool,))[: a.max], out / "cutouts.png"))


def cmd_search(a) -> None:
    from pipeline.pool.embed import ClipEmbedder
    from pipeline.pool.index import VectorIndex
    from pipeline.pool.sheet import cutout_sheet

    db = open_db(a.pool)
    idx = VectorIndex.from_db(db, "cutouts", "pool = ?", (a.pool,))
    hits = idx.search(ClipEmbedder().encode_texts([a.text])[0], a.k)
    rows = [db.get("cutouts", cid) for cid, _ in hits]
    scores = dict(hits)
    for cid, s in hits:
        print(f"{s:.3f} {cid}")
    print(cutout_sheet(rows, pool_root(a.pool) / "sheets" / "search.png", scores=scores))


def cmd_set_horizon(a) -> None:
    db = open_db(a.pool)
    if db.get("images", a.id) is None:
        raise SystemExit(f"no such image: {a.id}")
    db.update("images", a.id, horizon_y=a.y, horizon_checked=1)
    db.commit()
    print(f"{a.id}: horizon_y={a.y} (checked)")


def cmd_stats(a) -> None:
    db = open_db(a.pool)
    imgs = db.rows("images")
    cuts = db.rows("cutouts")
    print(f"images={len(imgs)} cutouts={len(cuts)} judged={sum(1 for c in cuts if c['judge_version'])} "
          f"horizon_checked={sum(1 for i in imgs if i['horizon_checked'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sub.add_parser(name)
        p.add_argument("--pool", required=True)
        p.set_defaults(fn=fn)
        return p

    p = add("fetch", cmd_fetch)
    p.add_argument("--query", required=True); p.add_argument("--dept", type=int, default=11)  # 11 = European Paintings
    p.add_argument("--medium", default=None); p.add_argument("--classification", default="Paintings")
    p.add_argument("--n", type=int, default=20); p.add_argument("--small", action="store_true", help="小さい画像(primaryImageSmall)を使う")
    p = add("ingest", cmd_ingest)
    p.add_argument("--limit", type=int); p.add_argument("--force", action="store_true"); p.add_argument("--points-per-side", type=int, default=16)
    p = add("judge", cmd_judge)
    p.add_argument("--model", default="clef-flash", choices=["clef", "clef-flash"]); p.add_argument("--max-neurons", type=float, default=3000)
    p.add_argument("--limit", type=int)
    p = add("sheet", cmd_sheet)
    p.add_argument("--kind", choices=["horizon", "cutouts"], required=True); p.add_argument("--max", type=int, default=60)
    p = add("search", cmd_search)
    p.add_argument("--text", required=True); p.add_argument("--k", type=int, default=12)
    p = add("set-horizon", cmd_set_horizon)
    p.add_argument("--id", required=True); p.add_argument("--y", type=float, required=True)
    add("stats", cmd_stats)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
