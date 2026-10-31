"""素材プールの操作: fetch(取得) → ingest(取り込み) → judge(属性判定) → sheet / search / set-horizon / stats。

  python scripts/pool.py fetch  --pool meiga --query landscape --dept 11 --n 20
  python scripts/pool.py ingest --pool meiga [--limit N] [--force]
  python scripts/pool.py judge  --pool meiga --model clef-flash --max-neurons 3000
  python scripts/pool.py sheet  --pool meiga --kind horizon|cutouts
  python scripts/pool.py search --pool meiga --text "a horse" [--k 12]
  python scripts/pool.py set-horizon --pool meiga --id met-436535 --y 0.62
  python scripts/pool.py label-sheets --pool meiga        # 番号つきの一覧を作る
  python scripts/pool.py import-labels --pool meiga --usable 1,3,5-9 --horizon 3=0.45 --horizon-na 2
  python scripts/pool.py eval   --pool meiga              # 自動の値と人の判断の一致を測る
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


def cmd_label_sheets(a) -> None:
    from pipeline.pool import labels
    from pipeline.pool.sheet import numbered_cutout_sheets, numbered_horizon_sheet

    db = open_db(a.pool)
    out = pool_root(a.pool) / "label"
    cut_nums = labels.numbering(db, a.pool)
    img_nums = labels.image_numbering(db, a.pool)
    labels.save_numbering(out / "numbering.json", cut_nums, img_nums)
    cuts = [(n, db.get("cutouts", cid)) for n, cid in cut_nums.items()]
    imgs = [(n, db.get("images", iid)) for n, iid in img_nums.items()]
    for p in numbered_cutout_sheets(cuts, out, per_sheet=a.per_sheet):
        print(p)
    print(numbered_horizon_sheet(imgs, out / "horizon.png"))


def cmd_import_labels(a) -> None:
    from pipeline.pool import labels

    db = open_db(a.pool)
    if a.usable is not None:
        print(f"usable({a.source}):", labels.import_usable(db, a.pool, labels.parse_numbers(a.usable), upto=a.upto, source=a.source))
    if a.repeat is not None:
        print(f"repeat({a.source}):", labels.import_repeat(db, a.pool, labels.parse_numbers(a.repeat), upto=a.upto, source=a.source))
    if a.horizon or a.horizon_na or a.horizon_rest_ok:
        vals = labels.parse_horizon_values(a.horizon) if a.horizon else {}
        na = labels.parse_numbers(a.horizon_na) if a.horizon_na else []
        print(f"horizon({a.source}):", labels.import_horizons(db, a.pool, vals, na, accept_rest=a.horizon_rest_ok, source=a.source))


def cmd_eval(a) -> None:
    from pipeline.pool import labels

    db = open_db(a.pool)
    res = {"cutouts": labels.evaluate_cutouts(db, a.pool, a.source), "horizon": labels.evaluate_horizons(db, a.pool, a.source)}
    out = pool_root(a.pool) / f"eval_{a.source}.json"
    print(f"== 正解の出どころ: {a.source} ==")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    c, h = res["cutouts"], res["horizon"]
    print(f"[切り抜き] 判定済み {c['n_labeled']} 個(使える {c.get('n_usable', 0)} 個)")
    for f in c.get("features", [])[:12]:
        print(f"  {f['feature']:24s} AUC={f['auc']:.3f} ({f['direction']}) 強さ={f['strength']:.2f} n={f['n']}")
    if "roles" in c:
        print(f"  役割の内訳: {c['roles']}")
    for label, key in (("反復して使える", "features_repeat"), ("単体か反復で使える", "features_single_or_repeat")):
        for f in c.get(key, [])[:6]:
            print(f"  [{label}] {f['feature']:24s} AUC={f['auc']:.3f} ({f['direction']}) n={f['n']}")
    for k in ("clef_rule_single_object>0.5", "clef_rule_single>0.5_and_cutoff<0.5"):
        if k in c:
            print(f"  {k}: {c[k]}")
    if "majority_class_accuracy" in c:
        print(f"  (参考)全部を多数派に分類したときの正解率: {c['majority_class_accuracy']}")
    print(f"[地平線] 確認済み {h['n_labeled']} 枚(対象外 {h['n_not_applicable']} 枚)")
    for k in ("mean_abs_error", "median_abs_error", "max_error", "within_0.05", "within_0.10", "confidence_vs_accuracy_spearman"):
        if k in h:
            print(f"  {k}: {h[k]}")
    print(f"詳細: {out}")


def cmd_compare(a) -> None:
    from pipeline.pool import labels

    res = labels.compare_labelers(open_db(a.pool), a.pool)
    print(json.dumps(res, ensure_ascii=False, indent=1))


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
    p = add("label-sheets", cmd_label_sheets)
    p.add_argument("--per-sheet", type=int, default=30)
    p = add("import-labels", cmd_import_labels)
    p.add_argument("--usable", help="使える切り抜きの番号。例: 1,3,5-9(ほかは使えない扱い)")
    p.add_argument("--upto", type=int, help="この番号までを判定済みとする(省略時は全部)")
    p.add_argument("--repeat", help="反復素材として使える切り抜きの番号(単体では使えないもののうち)。例: 2,5,9-11")
    p.add_argument("--horizon", help="地平線の正解。例: 3=0.45,5=0.60(画像の番号=高さ、上が0)")
    p.add_argument("--horizon-na", help="地平線のない絵(対象外)の画像の番号。例: 2,7")
    p.add_argument("--horizon-rest-ok", action="store_true", help="上のどちらにも挙げなかった画像は、自動の線で良いものとして確認済みにする")
    p.add_argument("--source", choices=["human", "ai"], default="human", help="印の出どころ。ai は別のモデルの判断で、人の印とは別に保存する")
    p = add("eval", cmd_eval)
    p.add_argument("--source", choices=["human", "ai"], default="human")
    add("compare", cmd_compare)
    add("stats", cmd_stats)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
