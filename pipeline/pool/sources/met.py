"""The Met Open Access API からの取得(CC0。画像は isPublicDomain のものだけ)。

仕様の出典: https://metmuseum.github.io/ (2026-10-31 に確認)
  - 検索は /public/collection/v1.1/search(offset / limit。v1/search は 2026-10-01 に廃止)
  - 1 秒あたり 80 リクエスト以下にする(ここでは控えめに 5 件/秒)
  - 権利: 「To the extent possible under law, The Met has waived all copyright ... using the Creative Commons Zero license」
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import requests

BASE = "https://collectionapi.metmuseum.org/public/collection"
LICENSE = "CC0 (The Met Open Access)"


@dataclass
class SourceItem:
    source: str
    source_id: str
    source_url: str
    license: str
    title: str
    artist: str
    date_text: str
    image_url: str
    meta: dict


class MetSource:
    def __init__(self, min_interval_s: float = 0.2, session: requests.Session | None = None):
        self._s = session or requests.Session()
        self._min_interval = min_interval_s
        self._last = 0.0

    def _get(self, url: str, **kw):
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        r = self._s.get(url, timeout=60, **kw)
        self._last = time.monotonic()
        return r

    def search(self, q: str, department_id: int | None = None, medium: str | None = None,
               offset: int = 0, limit: int = 100) -> tuple[int, list[int]]:
        params: dict = {"q": q, "hasImages": "true", "offset": offset, "limit": min(limit, 500)}
        if department_id is not None:
            params["departmentId"] = department_id
        if medium:
            params["medium"] = medium
        r = self._get(f"{BASE}/v1.1/search", params=params)
        r.raise_for_status()
        d = r.json()
        return int(d.get("total") or 0), list(d.get("objectIDs") or [])

    def get_object(self, object_id: int) -> dict | None:
        r = self._get(f"{BASE}/v1/objects/{object_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    @staticmethod
    def to_item(obj: dict, classification: str | None = "Paintings") -> SourceItem | None:
        """公開(isPublicDomain)で、画像があり、分類が合うものだけ SourceItem にする。"""
        if not obj.get("isPublicDomain") or not obj.get("primaryImage"):
            return None
        if classification and obj.get("classification") != classification:
            return None
        return SourceItem(
            source="met",
            source_id=str(obj["objectID"]),
            source_url=obj.get("objectURL", ""),
            license=LICENSE,
            title=obj.get("title", ""),
            artist=obj.get("artistDisplayName", ""),
            date_text=obj.get("objectDate", ""),
            image_url=obj["primaryImage"],
            meta={k: obj.get(k) for k in ("objectID", "department", "classification", "medium", "creditLine", "rightsAndReproduction", "tags", "primaryImageSmall")},
        )

    def download(self, url: str) -> bytes:
        r = self._get(url)
        r.raise_for_status()
        return r.content
