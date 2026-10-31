"""判断結果のキャッシュ(SQLite)。

キー = sha256(model_id, 正規化した質問, 画像ごとのハッシュ(順序つき), state)。
質問の id はキーに入れない。同じ質問・同じ画像なら、id が違っても同じ結果を返す。
失敗(例外)はキャッシュしない。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Sequence

from .base import (
    RESULT_CLASS,
    ImageInput,
    Judge,
    JudgeCapabilities,
    JudgeResponse,
    JudgeResult,
    Question,
    State,
    Usage,
    ask_in_chunks,
    image_digest,
    validate_batch,
)

_EXCLUDED_FIELDS = ("question_id", "cached")


class CachedJudge:
    """`Judge` を包み、キャッシュに無い質問だけを中身に尋ねる。上限を超える質問数は自動で分ける。"""

    def __init__(self, inner: Judge, path: str | Path):
        self._inner = inner
        self.model_id = inner.model_id
        # 質問数は内部で分けるので上限なし。画像数の上限は中身のものを引き継ぐ。
        self.capabilities = JudgeCapabilities(max_questions=None, max_images=inner.capabilities.max_images)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)"
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def _key(self, question: Question, image_hashes: Sequence[str], state: State) -> str:
        raw = json.dumps(
            {"model": self.model_id, "q": question.canonical(), "images": list(image_hashes), "state": state},
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self, key: str, question: Question) -> JudgeResult | None:
        row = self._db.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return RESULT_CLASS[question.kind](question_id=question.id, cached=True, **payload)

    def _store(self, key: str, result: JudgeResult) -> None:
        payload = {k: v for k, v in dataclasses.asdict(result).items() if k not in _EXCLUDED_FIELDS}
        self._db.execute(
            "INSERT OR REPLACE INTO cache (key, payload, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False), time.time()),
        )

    def ask(
        self, questions: Sequence[Question], images: Sequence[ImageInput] = (), state: State = ""
    ) -> JudgeResponse:
        validate_batch(questions)
        hashes = [image_digest(img) for img in images]
        keys = [self._key(q, hashes, state) for q in questions]

        found: dict[int, JudgeResult] = {}
        for i, (q, key) in enumerate(zip(questions, keys)):
            hit = self._load(key, q)
            if hit is not None:
                found[i] = hit

        miss_idx = [i for i in range(len(questions)) if i not in found]
        usage = Usage(cache_hits=len(found))
        if miss_idx:
            resp = ask_in_chunks(self._inner, [questions[i] for i in miss_idx], images, state)
            usage = usage + resp.usage
            for i, result in zip(miss_idx, resp.results):
                found[i] = result
                self._store(keys[i], result)
            self._db.commit()  # 例外が出たときはここに来ない = 失敗は保存されない
        return JudgeResponse(
            results=tuple(found[i] for i in range(len(questions))), usage=usage, model_id=self.model_id
        )
