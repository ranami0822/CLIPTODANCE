"""物の実寸の表(configs/height_table.yaml)。表の値はすべて仮置きで、出典はない(ファイル内の source を参照)。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class HeightEntry:
    mean: float | None        # 地上の物の実寸の高さ(m)
    sd: float | None
    width_m: float | None     # 幅(長さ)で大きさを決める種類(船など)の実寸(m)。あれば高さでなく幅で決める
    height_frac: float | None  # 空中の物の画面に対する高さの比
    airborne: bool
    source: str


class HeightTable:
    def __init__(self, entries: dict[str, HeightEntry]):
        self._e = entries

    @classmethod
    def load(cls, path: str | Path = "configs/height_table.yaml") -> "HeightTable":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["entries"]
        return cls({
            k: HeightEntry(v.get("mean"), v.get("sd"), v.get("width_m"), v.get("height_frac"), bool(v.get("airborne", False)), v.get("source", ""))
            for k, v in raw.items()
        })

    def __contains__(self, kind: str) -> bool:
        return kind in self._e

    def get(self, kind: str) -> HeightEntry:
        if kind not in self._e:
            raise KeyError(f"実寸の表にない種類: {kind!r}(表にない種類は Clairic に尋ねる予定だが、まだ未実装)")
        return self._e[kind]
