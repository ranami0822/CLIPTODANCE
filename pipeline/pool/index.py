"""ベクトル索引(FAISS)。埋め込みは DB に BLOB で持ち、検索の前にそこから作る(数千件までなら一瞬)。"""
from __future__ import annotations

import faiss
import numpy as np

from .db import PoolDB, blob_to_vec


class VectorIndex:
    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = ids
        self._index = faiss.IndexFlatIP(vectors.shape[1])  # 正規化済みなので内積 = コサイン類似度
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: np.ndarray, k: int = 50) -> list[tuple[str, float]]:
        q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, idx = self._index.search(q, min(k, len(self.ids)))
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    @classmethod
    def from_db(cls, db: PoolDB, table: str, where: str = "", params=()) -> "VectorIndex":
        clause = "embedding IS NOT NULL" + (f" AND ({where})" if where else "")
        rows = db.rows(table, clause, params)
        if not rows:
            raise ValueError(f"{table} に埋め込みがない")
        return cls([r["id"] for r in rows], np.stack([blob_to_vec(r["embedding"]) for r in rows]))
