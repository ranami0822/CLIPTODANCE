"""人の判断(正解)の取り込みと、自動の指標・Clef の判定・地平線の推定との一致の測定(M1 のゲート)。

正解は2つ:
  - 切り抜きが「使える」か(cutouts.usable)。番号つきの一覧を見て、使えるものの番号だけを答えてもらう。
  - 地平線の高さ(images.horizon_y、horizon_checked = 1)。自動推定値は horizon_auto_y に残るので、誤差が測れる。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from .db import PoolDB


def parse_numbers(text: str) -> list[int]:
    """"1,3,5-9" → [1, 3, 5, 6, 7, 8, 9]。空白と全角の記号も許す。"""
    t = text.replace("、", ",").replace("，", ",").replace("〜", "-").replace("~", "-").replace("ー", "-").replace(" ", "")
    out: list[int] = []
    for part in filter(None, t.split(",")):
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"範囲が逆: {part}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def parse_horizon_values(text: str) -> dict[int, float]:
    """"3=0.45, 5=0.60" → {3: 0.45, 5: 0.6}。0〜1(上が0)。"""
    out: dict[int, float] = {}
    for part in filter(None, text.replace("、", ",").replace(" ", "").split(",")):
        k, v = part.split("=", 1)
        y = float(v)
        if not 0.0 <= y <= 1.0:
            raise ValueError(f"地平線の高さは 0〜1: {part}")
        out[int(k)] = y
    return out


def numbering(db: PoolDB, pool: str) -> dict[int, str]:
    """切り抜きの番号(1 始まり、id の順)→ id。一覧と取り込みで同じ番号を使う。"""
    return {i + 1: r["id"] for i, r in enumerate(db.rows("cutouts", "pool = ?", (pool,)))}


def image_numbering(db: PoolDB, pool: str) -> dict[int, str]:
    return {i + 1: r["id"] for i, r in enumerate(db.rows("images", "pool = ?", (pool,)))}


def save_numbering(path: Path, cutouts: dict[int, str], images: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cutouts": cutouts, "images": images}, ensure_ascii=False, indent=1), encoding="utf-8")


USABLE_COL = {"human": "usable", "ai": "usable_ai", "claude": "usable_claude"}   # ai = Haiku
REPEAT_COL = {"human": "repeat", "ai": "repeat_ai", "claude": "repeat_claude"}
HORIZON_COLS = {"ai": ("horizon_ai_y", "horizon_ai_na"), "claude": ("horizon_claude_y", "horizon_claude_na")}
ROLES = ("both", "single", "repeat", "none")  # 単体でも反復でも使える / 単体だけ / 反復だけ / 使えない


def role_of(row: dict, source: str = "human") -> str | None:
    """切り抜きの役割。「単体で使える」と「反復して使える」は独立した2つの印で、両方が付いているときだけ決まる。"""
    u, r = row.get(USABLE_COL[source]), row.get(REPEAT_COL[source])
    if u is None or r is None:
        return None
    return {(1, 1): "both", (1, 0): "single", (0, 1): "repeat", (0, 0): "none"}[(int(u), int(r))]


def import_repeat(db: PoolDB, pool: str, repeat_numbers: list[int], upto: int | None = None, source: str = "human") -> dict:
    """番号 1〜upto のうち、repeat_numbers にあるものを「反復して使える」(1)、ほかを 0 にする。
    「単体で使える」の印とは独立している(単体で使えるものも、反復に使える。例: 馬を並べて群れにする)。"""
    rcol = REPEAT_COL[source]
    nums = numbering(db, pool)
    limit = upto if upto is not None else max(nums)
    unknown = [n for n in repeat_numbers if n not in nums or n > limit]
    if unknown:
        raise ValueError(f"範囲外の番号: {unknown}")
    rep = set(repeat_numbers)
    for n in range(1, limit + 1):
        db.update("cutouts", nums[n], **{rcol: 1 if n in rep else 0})
    db.commit()
    return {"covered": limit, "repeat": len(rep)}


def import_usable(db: PoolDB, pool: str, usable_numbers: list[int], upto: int | None = None, source: str = "human") -> dict:
    """番号 1〜upto(省略時は全部)のうち、usable_numbers にあるものを使える(1)、ほかを使えない(0)にする。
    source = "human"(あなたの判断)か "ai"(別のモデルの判断。人の印とは別の列に入れる)。"""
    col = USABLE_COL[source]
    nums = numbering(db, pool)
    limit = upto if upto is not None else max(nums)
    unknown = [n for n in usable_numbers if n not in nums or n > limit]
    if unknown:
        raise ValueError(f"範囲外の番号: {unknown}")
    good = set(usable_numbers)
    for n in range(1, limit + 1):
        db.update("cutouts", nums[n], **{col: 1 if n in good else 0})
    db.commit()
    return {"labeled": limit, "usable": len(good), "not_usable": limit - len(good)}


def import_horizons(db: PoolDB, pool: str, values: dict[int, float], not_applicable: list[int], accept_rest: bool = False,
                    source: str = "human") -> dict:
    """values = 直す画像(番号 → 正しい高さ)、not_applicable = 地平線のない絵。
    accept_rest=True なら、どちらにも挙がらない画像は「自動の線で良い」として確認済み(値は自動の値のまま)にする。"""
    nums = image_numbering(db, pool)
    bad = [n for n in [*values, *not_applicable] if n not in nums]
    if bad:
        raise ValueError(f"範囲外の番号: {bad}")
    accepted = 0
    if source == "human":
        for n, y in values.items():
            db.update("images", nums[n], horizon_y=y, horizon_checked=1)
        for n in not_applicable:
            db.update("images", nums[n], horizon_checked=2)
        if accept_rest:
            for n, iid in nums.items():
                if n not in values and n not in not_applicable:
                    db.update("images", iid, horizon_checked=1)  # horizon_y は自動の値のまま
                    accepted += 1
    else:  # 人以外の印は別の列。horizon_y / horizon_checked(人の印)には触れない
        ycol, nacol = HORIZON_COLS[source]
        for n, y in values.items():
            db.update("images", nums[n], **{ycol: y, nacol: 0})
        for n in not_applicable:
            db.update("images", nums[n], **{ycol: None, nacol: 1})
        if accept_rest:
            for n, iid in nums.items():
                if n not in values and n not in not_applicable:
                    db.update("images", iid, **{ycol: db.get("images", iid)["horizon_auto_y"], nacol: 0})
                    accepted += 1
    db.commit()
    return {"set": len(values), "not_applicable": len(not_applicable), "accepted_auto": accepted}


# ---------------------------------------------------------------- 評価

def auc(scores: list[float], positive: list[bool]) -> float | None:
    """「使える」を高く並べられるか(0.5 = 無関係、1 = 完全、0 = 完全に逆)。順位に基づく(Mann-Whitney)。"""
    s = np.asarray(scores, float)
    y = np.asarray(positive, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _features(r: dict) -> dict[str, float | None]:
    a = r.get("attrs") or {}
    f = {k: r.get(k) for k in ("area_ratio", "centrality", "clutter", "edge_score", "border_touch", "aspect", "thickness", "mask_quality")}
    if a:
        f["clef:single_object"] = a.get("single_object")
        f["clef:cut_off"] = a.get("cut_off")
        f["clef:flip_unnatural"] = a.get("flip_unnatural")
        f["clef:edge_quality"] = (a.get("edge_quality") or {}).get("normalized")
        f["clef:facing_certainty"] = max(a["facing"]["probs"].values()) if a.get("facing") else None
        f["clef:light_certainty"] = max(a["light_dir"]["probs"].values()) if a.get("light_dir") else None
    return f


def evaluate_cutouts(db: PoolDB, pool: str, source: str = "human") -> dict:
    col = USABLE_COL[source]
    rows = [r for r in db.rows("cutouts", f"pool = ? AND {col} IS NOT NULL", (pool,))]
    out: dict = {"source": source, "n_labeled": len(rows), "n_usable": sum(r[col] for r in rows)}
    if not rows:
        return out
    roles = [role_of(r, source) for r in rows]
    out["roles"] = {k: sum(1 for x in roles if x == k) for k in ROLES} | {"repeat_unlabeled": sum(1 for x in roles if x is None)}
    y = [bool(r[col]) for r in rows]
    feats = [_features(r) for r in rows]

    def feature_table(target: list[bool], row_idx: list[int]) -> list[dict]:
        table = []
        for name in sorted({k for f in feats for k in f}):
            idx = [(j, i) for j, i in enumerate(row_idx) if feats[i].get(name) is not None]
            a = auc([feats[i][name] for _, i in idx], [target[j] for j, _ in idx]) if idx else None
            if a is not None:
                table.append({"feature": name, "n": len(idx), "auc": round(a, 3), "direction": "高いほど使える" if a >= 0.5 else "低いほど使える",
                              "strength": round(abs(a - 0.5) * 2, 3)})
        return sorted(table, key=lambda t: -t["strength"])

    out["features"] = feature_table(y, list(range(len(rows))))  # 「単体で使える」の予測
    rep_rows = [i for i, r in enumerate(rows) if r[REPEAT_COL[source]] is not None]
    if rep_rows and any(rows[i][REPEAT_COL[source]] == 1 for i in rep_rows):
        out["features_repeat"] = feature_table([bool(rows[i][REPEAT_COL[source]]) for i in rep_rows], rep_rows)
        out["features_single_or_repeat"] = feature_table([bool(rows[i][col] or rows[i][REPEAT_COL[source]]) for i in rep_rows], rep_rows)

    judged = [(r, f) for r, f in zip(rows, feats) if f.get("clef:single_object") is not None]
    if judged:
        def rate(pred):
            tp = sum(1 for r, f in judged if pred(f) and r[col]); fp = sum(1 for r, f in judged if pred(f) and not r[col])
            fn = sum(1 for r, f in judged if not pred(f) and r[col]); tn = sum(1 for r, f in judged if not pred(f) and not r[col])
            n = tp + fp + fn + tn
            return {"n": n, "accuracy": round((tp + tn) / n, 3), "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 3) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        out["clef_rule_single_object>0.5"] = rate(lambda f: f["clef:single_object"] > 0.5)
        out["clef_rule_single>0.5_and_cutoff<0.5"] = rate(lambda f: f["clef:single_object"] > 0.5 and f["clef:cut_off"] < 0.5)
        out["majority_class_accuracy"] = round(max(sum(y), len(y) - sum(y)) / len(y), 3)
    return out


def evaluate_horizons(db: PoolDB, pool: str, source: str = "human") -> dict:
    if source == "human":
        rows = db.rows("images", "pool = ? AND horizon_checked = 1 AND horizon_auto_y IS NOT NULL", (pool,))
        n_na = len(db.rows("images", "pool = ? AND horizon_checked = 2", (pool,)))
        truth = lambda r: r["horizon_y"]
    else:
        ycol, nacol = HORIZON_COLS[source]
        rows = db.rows("images", f"pool = ? AND {ycol} IS NOT NULL AND horizon_auto_y IS NOT NULL", (pool,))
        n_na = len(db.rows("images", f"pool = ? AND {nacol} = 1", (pool,)))
        truth = lambda r: r[ycol]
    out: dict = {"source": source, "n_labeled": len(rows), "n_not_applicable": n_na}
    if not rows:
        return out
    err = np.array([abs(r["horizon_auto_y"] - truth(r)) for r in rows])
    conf = np.array([r["horizon_conf"] for r in rows])
    out.update({
        "mean_abs_error": round(float(err.mean()), 3), "median_abs_error": round(float(np.median(err)), 3),
        "within_0.05": int((err <= 0.05).sum()), "within_0.10": int((err <= 0.10).sum()), "max_error": round(float(err.max()), 3),
    })
    if len(rows) >= 5 and conf.std() > 0 and err.std() > 0:
        rho = spearmanr(conf, -err).statistic  # 確信度が高いほど誤差が小さければ正
        out["confidence_vs_accuracy_spearman"] = round(float(rho), 3)
    out["per_image"] = [{"id": r["id"], "auto": round(r["horizon_auto_y"], 3), "label": round(truth(r), 3),
                         "error": round(abs(r["horizon_auto_y"] - truth(r)), 3), "conf": round(r["horizon_conf"], 2)} for r in rows]
    return out


def compare_labelers(db: PoolDB, pool: str, labeler: str = "ai") -> dict:
    """labeler(ai = Haiku、claude)の印が、人の印とどれだけ一致するか(人の印を正解として)。両方が付いているものだけを数える。

    切り抜き: 一致率・Cohen の κ(偶然の一致を除いた一致。0 = 偶然と同じ、1 = 完全)・AI が「使える」としたものの適合率と再現率。
    地平線: 人が値を付けた画像で、AI の値との誤差。人と AI が「地平線なし」で一致するか。
    """
    uc, rc = USABLE_COL[labeler], REPEAT_COL[labeler]
    ycol, nacol = HORIZON_COLS[labeler]
    rows = db.rows("cutouts", f"pool = ? AND usable IS NOT NULL AND {uc} IS NOT NULL", (pool,))
    out: dict = {"labeler": labeler, "cutouts": {"n": len(rows)}}
    if rows:
        tp = sum(1 for r in rows if r[uc] and r["usable"]); fp = sum(1 for r in rows if r[uc] and not r["usable"])
        fn = sum(1 for r in rows if not r[uc] and r["usable"]); tn = sum(1 for r in rows if not r[uc] and not r["usable"])
        n = len(rows)
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
        out["cutouts"].update({
            "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
            "ai_precision": round(tp / (tp + fp), 3) if tp + fp else None, "ai_recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "human_usable_rate": round((tp + fn) / n, 3),
        })
    rrows = db.rows("cutouts", f"pool = ? AND repeat IS NOT NULL AND {rc} IS NOT NULL", (pool,))
    if rrows:
        tp = sum(1 for r in rrows if r[rc] and r["repeat"]); fp = sum(1 for r in rrows if r[rc] and not r["repeat"])
        fn = sum(1 for r in rrows if not r[rc] and r["repeat"]); tn = sum(1 for r in rrows if not r[rc] and not r["repeat"])
        n = len(rrows); po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
        out["repeat"] = {"n": n, "agreement": round(po, 3), "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
                         "ai_precision": round(tp / (tp + fp), 3) if tp + fp else None, "ai_recall": round(tp / (tp + fn), 3) if tp + fn else None,
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn, "human_repeat_rate": round((tp + fn) / n, 3)}
    imgs = db.rows("images", "pool = ?", (pool,))
    both = [r for r in imgs if r["horizon_checked"] == 1 and r[ycol] is not None]
    na_pairs = [r for r in imgs if r["horizon_checked"] in (1, 2) and (r[nacol] is not None)]
    h: dict = {"n_values": len(both), "n_na_compared": len(na_pairs)}
    if both:
        e = np.array([abs(r["horizon_y"] - r[ycol]) for r in both])
        h.update({"mean_abs_error": round(float(e.mean()), 3), "within_0.05": int((e <= 0.05).sum()), "within_0.10": int((e <= 0.10).sum())})
    if na_pairs:
        h["na_agreement"] = round(sum(1 for r in na_pairs if (r["horizon_checked"] == 2) == bool(r[nacol])) / len(na_pairs), 3)
    out["horizon"] = h
    return out
