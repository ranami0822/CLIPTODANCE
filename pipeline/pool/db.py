"""素材プールの DB(SQLite)。プールごとに1ファイル: data/pool/<pool>/pool.sqlite

素材レコードは設計書の項目を、元画像(images)と切り抜き(cutouts)に分けて持つ。
埋め込みは float32 の BLOB で持ち、検索用の索引はそこから作る(索引ファイルは持たない)。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    pool TEXT NOT NULL,
    source TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_url TEXT,
    license TEXT,
    title TEXT, artist TEXT, date_text TEXT,
    width INTEGER, height INTEGER,          -- 作業用画像の大きさ
    orig_path TEXT, work_path TEXT, sha256 TEXT,
    horizon_y REAL, horizon_conf REAL,      -- 0-1(上が0)。初期値は軽い処理(R3a)
    horizon_checked INTEGER NOT NULL DEFAULT 0,  -- 0 = 未確認、1 = 目視で確認(値は horizon_y)、2 = 地平線のない絵(対象外)
    horizon_auto_y REAL,                    -- 自動推定の値(手で直しても残す。誤差の測定用)
    horizon_ai_y REAL, horizon_ai_na INTEGER,  -- AI(Haiku)が付けた印。人の印(horizon_y / horizon_checked)とは混ぜない
    horizon_claude_y REAL, horizon_claude_na INTEGER,  -- Claude が付けた印
    crop_box TEXT,                          -- 元画像上の内容部分 [x0,y0,x1,y1](額縁・余白を除いた範囲)
    embedding BLOB, embedding_model TEXT,   -- 画像全体の埋め込み(土台の検索用)
    attrs TEXT, judge_version TEXT,
    meta TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cutouts (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id),
    pool TEXT NOT NULL,
    cutout_path TEXT NOT NULL,
    x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER,   -- 作業用画像上の外接矩形(x1,y1 は含まない)
    area_ratio REAL, centrality REAL, clutter INTEGER, edge_score REAL, border_touch INTEGER,
    aspect REAL, thickness REAL,                      -- 最小外接回転矩形の 長辺/短辺、短辺/画像の長辺
    mask_quality REAL,                                -- SAM の予測 IoU × 安定度
    embedding BLOB, embedding_model TEXT,             -- 切り抜きの埋め込み(検索用)
    attrs TEXT, judge_version TEXT,
    usable INTEGER,                                   -- 人の判定: 1 = 単体でそのまま使える、0 = 単体では使えない、NULL = 未判定
    repeat INTEGER,                                   -- 人の判定: 1 = 反復して使える(繁み・雲・布、馬の群れなど)。usable とは独立
    usable_ai INTEGER, repeat_ai INTEGER,             -- AI(Haiku)の判定。人の判定とは混ぜない
    usable_claude INTEGER, repeat_claude INTEGER,     -- Claude(この作業をしているモデル)の判定。人の判定・Haiku の判定とは混ぜない
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cutouts_image ON cutouts(image_id);
"""

_JSON_COLS = {"attrs", "meta", "crop_box"}

# 後から足した列(古い DB に ALTER TABLE で足す)
_ADDED_COLUMNS = [("images", "crop_box", "TEXT"), ("cutouts", "aspect", "REAL"), ("cutouts", "thickness", "REAL"),
                  ("images", "horizon_auto_y", "REAL"), ("cutouts", "usable", "INTEGER"),
                  ("cutouts", "usable_ai", "INTEGER"), ("images", "horizon_ai_y", "REAL"), ("images", "horizon_ai_na", "INTEGER"),
                  ("cutouts", "repeat", "INTEGER"), ("cutouts", "repeat_ai", "INTEGER"),
                  ("cutouts", "usable_claude", "INTEGER"), ("cutouts", "repeat_claude", "INTEGER"),
                  ("images", "horizon_claude_y", "REAL"), ("images", "horizon_claude_na", "INTEGER")]


def vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class PoolDB:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        for table, col, typ in _ADDED_COLUMNS:
            have = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        # 自動推定値の列を足す前に取り込んだ画像: 未確認(手で直していない)なら、いまの値が自動推定値
        have = {r["name"] for r in self._db.execute("PRAGMA table_info(images)")}
        if {"horizon_y", "horizon_checked", "horizon_auto_y"} <= have:
            self._db.execute("UPDATE images SET horizon_auto_y = horizon_y WHERE horizon_auto_y IS NULL AND horizon_checked = 0")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # ---- 書き込み

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("created_at", time.time())
        for c in _JSON_COLS & row.keys():
            if row[c] is not None and not isinstance(row[c], str):
                row[c] = json.dumps(row[c], ensure_ascii=False)
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        self._db.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({qs})", list(row.values()))

    def update(self, table: str, id: str, **fields: Any) -> None:
        for c in _JSON_COLS & fields.keys():
            if fields[c] is not None and not isinstance(fields[c], str):
                fields[c] = json.dumps(fields[c], ensure_ascii=False)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._db.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*fields.values(), id])

    def commit(self) -> None:
        self._db.commit()

    def delete_cutouts_of(self, image_id: str) -> list[str]:
        paths = [r["cutout_path"] for r in self._db.execute("SELECT cutout_path FROM cutouts WHERE image_id = ?", (image_id,))]
        self._db.execute("DELETE FROM cutouts WHERE image_id = ?", (image_id,))
        return paths

    # ---- 読み出し

    def get(self, table: str, id: str) -> dict | None:
        row = self._db.execute(f"SELECT * FROM {table} WHERE id = ?", (id,)).fetchone()
        return self._decode(row)

    def rows(self, table: str, where: str = "", params: Iterable[Any] = ()) -> list[dict]:
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "") + " ORDER BY id"
        return [self._decode(r) for r in self._db.execute(sql, list(params))]

    def count(self, table: str) -> int:
        return self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for c in _JSON_COLS & d.keys():
            if d[c]:
                d[c] = json.loads(d[c])
        return d
