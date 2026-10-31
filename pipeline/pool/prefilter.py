"""機械的な指標(と検出器の情報)だけで、「単体で使える」切り抜きを事前に絞る足切り。

目的: 数が増えたプールで、印を付ける(または判定を呼ぶ)前に、見込みの薄い候補を落とす。
Clef の判定は使わない(新しい候補には未実施で、費用がかかるため)。ロジスティック回帰(L2)を scipy で解く小さなモデル。
評価は「画像ごとに分けた」交差検証(同じ絵の切り抜きは似ているので、学習と評価に同じ絵を混ぜない)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

FEATURES = (
    "log_area", "log_thickness", "log_clutter", "log_aspect", "edge_score", "border_touch", "centrality",
    "sam_quality", "from_detector", "det_score",
)


def effective_label(row: dict) -> int | None:
    """単体で使えるか(1/0)。人の印があれば人、なければ Claude の印。どちらもなければ None。"""
    if row.get("usable") is not None:
        return int(row["usable"])
    if row.get("usable_claude") is not None:
        return int(row["usable_claude"])
    return None


def features(row: dict) -> list[float]:
    det_score = row.get("det_score") or 0.0
    from_det = 1.0 if row.get("source") == "maskrcnn" else 0.0
    sam_q = float(row.get("mask_quality") or 0.0) if row.get("source") != "maskrcnn" else 0.0
    return [
        math.log10(max(row["area_ratio"], 1e-6)), math.log10(max(row.get("thickness") or 1e-3, 1e-3)),
        math.log1p(row.get("clutter") or 0), math.log(max(row.get("aspect") or 1.0, 1.0)),
        float(row.get("edge_score") or 0.0), float(row.get("border_touch") or 0), float(row.get("centrality") or 0.0),
        sam_q, from_det, float(det_score),
    ]


@dataclass
class Model:
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float
    l2: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mean) / self.std) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def to_json(self) -> dict:
        return {"features": list(FEATURES), "mean": self.mean.tolist(), "std": self.std.tolist(), "w": self.w.tolist(), "b": self.b, "l2": self.l2}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        assert d["features"] == list(FEATURES), "特徴量の定義が変わっている"
        return cls(np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]), d["b"], d["l2"])


def fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> Model:
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = (X - mean) / std
    n, d = Z.shape

    def loss(p):
        w, b = p[:d], p[d]
        z = Z @ w + b
        # 数値的に安定な -log-likelihood
        nll = np.sum(np.logaddexp(0, z) - y * z) / n + 0.5 * l2 * np.sum(w * w) / n
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gw = Z.T @ (s - y) / n + l2 * w / n
        gb = np.sum(s - y) / n
        return nll, np.append(gw, gb)

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return Model(mean, std, res.x[:d], float(res.x[d]), l2)


def auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import rankdata
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((rankdata(scores)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cross_validate_by_group(X: np.ndarray, y: np.ndarray, groups: list[str], l2: float = 1.0) -> np.ndarray:
    """グループ(画像)ごとに 1 つずつ抜いて学習・予測し、全体の予測を返す(leave-one-group-out)。"""
    pred = np.zeros(len(y))
    for g in sorted(set(groups)):
        test = np.array([x == g for x in groups])
        if y[~test].sum() == 0 or (1 - y[~test]).sum() == 0:
            pred[test] = y[~test].mean() if (~test).any() else 0.5
            continue
        pred[test] = fit(X[~test], y[~test], l2).predict(X[test])
    return pred


def pruning_curve(scores: np.ndarray, y: np.ndarray, keep_recalls=(1.0, 0.95, 0.9, 0.8)) -> list[dict]:
    """「単体で使える」を recall 以上残すときの、しきい値と、落とせる候補の割合(precision も)。"""
    out = []
    order = np.argsort(-scores)
    total_pos = int(y.sum())
    for r in keep_recalls:
        need = math.ceil(r * total_pos)
        cum = np.cumsum(y[order])
        k = int(np.searchsorted(cum, need) + 1)   # 上位 k 個を残せば、必要な数の正例が入る
        kept = min(k, len(y))
        out.append({"keep_recall": r, "threshold": float(scores[order][kept - 1]), "kept": kept, "pruned_fraction": round(1 - kept / len(y), 3),
                    "precision_of_kept": round(float(y[order][:kept].sum() / kept), 3)})
    return out


def dataset(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    labeled = [r for r in rows if effective_label(r) is not None]
    X = np.array([features(r) for r in labeled], float)
    y = np.array([effective_label(r) for r in labeled], float)
    return X, y, [r["image_id"] for r in labeled], labeled


def save_model(model: Model, path: str | Path) -> None:
    Path(path).write_text(json.dumps(model.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")
