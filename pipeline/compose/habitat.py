"""水と陸の補正(R10 の「規則が直す」側)。船(水)の足元が陸にある、馬(陸)の足元が水の中にある、を水の地図で直す。"""
from __future__ import annotations

from dataclasses import replace

from .geometry import Limits
from .types import Instance, Placement
from .water import WaterMap


def apply_habitat(instances: list[Instance], placements: list[Placement], water: WaterMap, limits: Limits | None = None) -> tuple[list[Placement], list[dict]]:
    """habitat が "water" / "land" の地上物の足元を、その場所に入れる。直した分を log に残す。空中の物と habitat なしは触らない。"""
    lim = limits or Limits()
    min_y = water.horizon_y + lim.min_ground_gap
    out: list[Placement] = []
    log: list[dict] = []
    for inst, pl in zip(instances, placements):
        want = inst.habitat
        if inst.airborne or want not in ("water", "land"):
            out.append(pl)
            continue
        want_water = want == "water"
        y = max(pl.y, min_y)
        if water.is_water(pl.x, y) == want_water:
            out.append(pl if y == pl.y else replace(pl, y=y))     # 地平線より上なら、地平線の下に入れる(描画側も同じ位置にそろえる)
            continue
        target = water.nearest(pl.x, y, water=want_water, min_y=min_y)
        if target is None:
            log.append({"instance": inst.id, "habitat": want, "status": "no_place_found"})
            out.append(pl)
            continue
        log.append({"instance": inst.id, "habitat": want, "moved": [[round(pl.x, 3), round(pl.y, 3)], [round(target[0], 3), round(target[1], 3)]]})
        out.append(replace(pl, x=target[0], y=target[1], x_range80=None, y_range80=None))
    return out, log


def foot_ok_for(water: WaterMap | None):
    """render_collage の foot_ok に渡す関数。habitat を守る位置だけを許す(地図がなければ、すべて許す)。"""
    if water is None:
        return None

    def ok(inst: Instance, x: float, y: float) -> bool:
        if inst.airborne or inst.habitat not in ("water", "land"):
            return True
        return water.is_water(x, y) == (inst.habitat == "water")
    return ok
