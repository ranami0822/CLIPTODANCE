"""配置の推定(Clairic)と、その後処理・モック。

流れ:
  スケッチ + 素材のコンタクトシート + テーマ文 → Clairic が素材ごとに x / y / z / scale / rot を 80% 範囲つきで出す
  → ここで範囲に収める(補正した事実はログに残す)→ UI が動かして見せる

座標の約束(UI と共通):
  x, y   素材の中心。キャンバスの幅・高さに対する 0–1。原点は左上。
  z      奥行き。0 = いちばん奥、1 = いちばん手前。描く順(order)と、UI のスライダーに使う。
  scale  素材の長いほうの辺 ÷ キャンバスの長いほうの辺。
  rot    回転。度。時計回りが正。0 = 素材の取り込み時の向き(コンタクトシートで見える向き)。

スケッチがあるときは、線のつながった塊(領域)をこちらで検出して番号を振る。
  1. Clef(選択)が「各素材はどの番号の塊に当たるか」を確率で答える。こちらは塊が重複しない最適な割り当てを解く。
  2. Clairic(数値)が各素材の z・rot(と、塊に当たらない素材の x・y・scale)を、80% 範囲つきで答える。
  3. 対応した塊の矩形から x・y・大きさを決める(Clairic に座標を直接数字で当てさせると、実測で
     位置・大きさがスケッチと合わなかったため。Clairic に番号を数値として見積もらせる方式も当てずっぽうだった。
     経緯は docs/design.md)。
塊に当たらない素材は、Clairic が出した x・y・scale をそのまま使う。

Clef / Clairic のリクエスト形式は、このリポジトリの本筋(adapters/cloudflare.py)で確認済みのものを、
ここで独立に書き直している(本筋のコードは import しない)。
"""
from __future__ import annotations

import json
import math
import os
import random
import string
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from . import imaging

MAX_OBJECTS = 8   # 項目数は 5×8=40 まで。コンタクトシートの札が読めなくなる前の上限
BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"

# 項目ごとの範囲(Clairic の estimate は min / max が必須)
RANGES = {
    "x": (0.0, 1.0),
    "y": (0.0, 1.0),
    "z": (0.0, 1.0),
    "scale": (0.04, 1.0),
    "rot": (-180.0, 180.0),
}
DEFAULTS = {"x": 0.5, "y": 0.5, "z": 0.5, "scale": 0.3, "rot": 0.0}
# 回転の 80% 範囲(p90 - p10)がこの度数を超えたら、推定に根拠がないとみて 0° に戻す(仮置き。実測: 全素材が 11〜15° で範囲 60〜100° だった)
ROT_MAX_RANGE_DEG = 60.0
KEYS = ("x", "y", "z", "scale", "rot")


class ArrangeError(Exception):
    """配置の推定の失敗(呼び出し側が利用者に見せる文面を持つ)。"""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------- データ

@dataclass
class Obj:
    id: str           # "A", "B", ...
    name: str
    image: Image.Image


@dataclass
class Placement:
    id: str
    name: str
    x: float
    y: float
    z: float
    scale: float
    rot: float
    shape: int = 0                                # 対応したスケッチの塊の番号(0 = なし)
    shape_p: float | None = None                  # その対応の確率(Clef)。対応づけをしなかったときは None
    geom: str = "estimate"                        # x・y・scale の出どころ: "sketch"(塊の矩形)/ "estimate"(Clairic の推定)
    range80: dict = field(default_factory=dict)   # {"x": [lo, hi], ...}。得られた項目だけ
    missing: list = field(default_factory=list)   # Clairic が値を出さなかった項目
    order: int = 0                                # 描く順(0 = 最初 = 奥)

    def to_json(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "x": self.x, "y": self.y, "z": self.z, "scale": self.scale, "rot": self.rot,
            "shape": self.shape, "shape_p": self.shape_p, "geom": self.geom,
            "range80": self.range80, "missing": self.missing, "order": self.order,
        }


# ---------------------------------------------------------------- Clairic への質問

def field_name(obj_id: str, key: str) -> str:
    return f"{obj_id}_{key}"


def build_fields(objs: list[Obj], canvas_aspect: float) -> dict:
    """Clairic の fields。名前は `A_x` のように 素材ID_項目。"""
    a = canvas_aspect
    desc = {
        "x": "horizontal position of the object's center, as a fraction of canvas width (0 = left edge, 1 = right edge)",
        "y": "vertical position of the object's center, as a fraction of canvas height (0 = top edge, 1 = bottom edge)",
        "z": "depth order: 0 = farthest back (behind everything), 1 = nearest to the viewer (in front of everything)",
        "scale": f"size of the object's longer side as a fraction of the canvas's longer side (canvas is {a:.2f}:1 wide:high)",
        "rot": "rotation in degrees, clockwise positive; 0 = upright exactly as shown on the contact sheet",
    }
    fields: dict = {}
    for o in objs:
        for k in KEYS:
            lo, hi = RANGES[k]
            spec = {
                "type": "estimate",
                "min": lo, "max": hi,
                "instructions": f"Object {o.id} ({o.name}): {desc[k]}.",
            }
            if k == "rot":
                spec["step"] = 1
            fields[field_name(o.id, k)] = spec
    return fields


def build_state(theme: str, objs: list[Obj], has_sketch: bool, canvas_aspect: float,
                regions: list[dict] = (), assigned: dict | None = None) -> str:
    """Clairic の state。assigned = {素材ID: 塊の番号}(Clef の対応づけ)。0 や欠けは「対応なし」。"""
    assigned = assigned or {}
    lines = [
        "Task: lay out cut-out objects on one canvas so that the arrangement realises the user's rough sketch and theme.",
        f"The canvas is {canvas_aspect:.2f}:1 (width:height). Coordinates have their origin at the top-left.",
    ]
    if has_sketch:
        lines.append(f"Image 1 is the rough sketch (dark strokes on white); its {len(regions)} shapes are outlined in blue and numbered.")
        lines.append("Image 2 is a contact sheet of the cut-out objects, each labelled with a letter in its top-left corner.")
    else:
        lines.append("Image 1 is a contact sheet of the cut-out objects, each labelled with a letter in its top-left corner. "
                     "There is no sketch, so compose freely from the theme.")
    lines.append("Objects: " + "; ".join(f"{o.id} = {o.name or 'unnamed'}" for o in objs) + ".")
    if has_sketch:
        got = [f"{o.id} replaces sketch shape {assigned[o.id]}" for o in objs if assigned.get(o.id)]
        free = [o.id for o in objs if not assigned.get(o.id)]
        if got:
            lines.append("Already decided: " + "; ".join(got) + ". An object placed over its shape takes the shape's position and size, "
                         "so for those objects what matters is depth (z) and tilt (rot).")
        if free:
            lines.append("Not matched to any sketch shape (place them where the theme suggests, in empty areas): " + ", ".join(free) + ".")
    lines.append(f"Theme / idea written by the user: {theme.strip() or '(none)'}")
    lines.append("Nearer objects (larger z) are usually lower in the picture and larger.")
    return "\n".join(lines)


# ---------------------------------------------------------------- Clairic の呼び出し

Transport = Callable[[str, dict, dict, float], "tuple[int, dict, object]"]


def urllib_transport(url: str, headers: dict, body: dict, timeout: float):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, hdrs, raw = r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        status, hdrs, raw = e.code, dict(e.headers), e.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        data = {"non_json": raw[:300].decode("utf-8", "replace")}
    return status, {k.lower(): v for k, v in hdrs.items()}, data


class CloudflareClient:
    def __init__(self, account_id: str | None = None, token: str | None = None,
                 transport: Transport | None = None, timeout_s: float = 90.0):
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        self._token = token or os.environ.get("CLOUDFLARE_API_TOKEN", "")
        self._transport = transport or urllib_transport
        self.timeout_s = timeout_s

    @property
    def configured(self) -> bool:
        return bool(self.account_id and self._token)

    def run(self, model: str, body: dict) -> tuple[dict, dict]:
        """model は "clairic" か "clef"。(result, meta) を返す。meta は neurons と latency_s。"""
        if not self.configured:
            raise ArrangeError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN が未設定です(モックなら動きます)", 503)
        url = BASE_URL.format(account=self.account_id, model=model)
        t0 = time.monotonic()
        status, headers, data = self._transport(url, {"Authorization": f"Bearer {self._token}"}, body, self.timeout_s)
        latency = time.monotonic() - t0
        errors = data.get("errors") if isinstance(data, dict) else None
        if status != 200 or (isinstance(data, dict) and data.get("success") is False):
            codes = [e.get("code") for e in (errors or []) if isinstance(e, dict)]
            msg = "; ".join(str(e.get("message")) for e in (errors or []) if isinstance(e, dict))[:200]
            if status == 429 or 3036 in codes:
                raise ArrangeError("Clairic の無料枠(1日)を使い切ったか、一時的に混んでいます。あとでやり直してください。" + (f" [{msg}]" if msg else ""), 429)
            if status in (400, 413) or any(c in (3003, 3006, 5004, 5006, 5007) for c in codes):
                raise ArrangeError(f"Clairic が入力を受け付けませんでした: {msg or status}", 400)
            raise ArrangeError(f"Clairic の呼び出しに失敗しました(HTTP {status}): {msg}", 502)
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise ArrangeError(f"{model} の応答の形が想定と違います", 502)
        if result.get("truncated"):
            raise ArrangeError("入力が長すぎて切り詰められました。テーマ文を短くするか素材を減らしてください", 400)
        neurons = headers.get("cf-ai-neurons")
        return result, {"neurons": float(neurons) if neurons else None, "latency_s": round(latency, 2)}


def parse_answers(objs: list[Obj], answers: dict) -> list[Placement]:
    out: list[Placement] = []
    for o in objs:
        vals, rng, missing = {}, {}, []
        for k in KEYS:
            ans = answers.get(field_name(o.id, k))
            v = ans.get("value") if isinstance(ans, dict) else None
            if v is None:
                vals[k] = DEFAULTS[k]
                missing.append(k)
                continue
            vals[k] = float(v)
            p10, p90 = ans.get("p10"), ans.get("p90")
            if p10 is not None and p90 is not None:
                rng[k] = [float(p10), float(p90)]
        out.append(Placement(id=o.id, name=o.name, range80=rng, missing=missing, **vals))
    return out


# ---------------------------------------------------------------- 塊への対応づけ(Clef)

def build_assign_body(objs: list[Obj], theme: str, regions: list[dict], images: list[bytes]) -> dict:
    import base64
    state = (f"Image 1 is a rough sketch whose {len(regions)} shapes are outlined in blue and numbered. "
             "Image 2 is a contact sheet of cut-out objects, each labelled with a letter. "
             "Each object should replace at most one sketch shape, and each shape is used by at most one object. "
             f"Theme written by the user: {theme.strip() or '(none)'} "
             "Objects: " + "; ".join(f"{o.id} = {o.name or 'unnamed'}" for o in objs) + ".")
    criteria = {"0": "no shape in the sketch fits this object"}
    criteria.update({str(r["id"]): f"numbered shape {r['id']} in the sketch" for r in regions})
    questions = {
        field_name(o.id, "shape"): {
            "type": "choice",
            "instructions": f"Which sketch shape does object {o.id} ({o.name or 'unnamed'}) replace?",
            "criteria": criteria,
        }
        for o in objs
    }
    return {"model": "clef", "state": state, "questions": questions,
            "images": [{"content_type": "image/png", "base64": base64.b64encode(b).decode("ascii")} for b in images]}


def solve_assignment(probs: dict[str, dict[int, float]], ids: list[str]) -> dict[str, int]:
    """素材ごとの確率 {id: {塊番号: p}}(0 = 対応なし)から、塊が重複しない確率の積(対数和)が最大の割り当てを選ぶ。

    0 は何個でも使える。塊は多くて 12 なので、使った塊のビット集合で全探索する。
    """
    import math as _m
    from functools import lru_cache

    shapes = sorted({k for d in probs.values() for k in d if k != 0})
    bit = {k: 1 << i for i, k in enumerate(shapes)}

    @lru_cache(maxsize=None)
    def best(i: int, used: int):
        if i == len(ids):
            return 0.0, ()
        d = probs[ids[i]]
        cands = []
        for k, p in d.items():
            if k != 0 and used & bit[k]:
                continue
            sub, rest = best(i + 1, used | (bit[k] if k != 0 else 0))
            cands.append((_m.log(max(p, 1e-9)) + sub, (k,) + rest))
        return max(cands, key=lambda c: c[0])

    _, picks = best(0, 0)
    return dict(zip(ids, picks))


def assign_shapes(client: "CloudflareClient", objs: list[Obj], theme: str, regions: list[dict],
                  images: list[bytes]) -> tuple[dict[str, int], dict[str, float], dict]:
    """Clef の choice で素材ごとに塊の確率を得て、重複なしの割り当てを解く。(割り当て, その確率, meta) を返す。"""
    result, meta = client.run("clef", build_assign_body(objs, theme, regions, images))
    probs: dict[str, dict[int, float]] = {}
    for o in objs:
        ans = result["answers"].get(field_name(o.id, "shape"))
        if not isinstance(ans, dict) or not isinstance(ans.get("probabilities"), dict):
            raise ArrangeError(f"Clef の答えが想定と違います({o.id}): {str(ans)[:120]}", 502)
        probs[o.id] = {int(k): float(v) for k, v in ans["probabilities"].items()}
    assigned = solve_assignment(probs, [o.id for o in objs])
    return assigned, {i: probs[i][k] for i, k in assigned.items()}, meta


# ---------------------------------------------------------------- 塊への対応(規則側)

def apply_regions(placements: list[Placement], objs: list[Obj], regions: list[dict]) -> list[str]:
    """shape で対応した塊の矩形から x・y・scale を決める。範囲外・重複の対応は外す(ログに残す)。"""
    log: list[str] = []
    if not regions:
        for p in placements:
            p.shape = 0
        return log
    by_id = {r["id"]: r for r in regions}
    taken: dict[int, str] = {}
    for p, o in zip(placements, objs):
        k = p.shape
        if k == 0:
            continue
        if k not in by_id:
            log.append(f"{p.id}({p.name}): shape={k} は存在しない塊なので外した")
            p.shape = 0
            continue
        if k in taken:
            log.append(f"{p.id}({p.name}): 塊 {k} は {taken[k]} が使用済みなので外した")
            p.shape = 0
            continue
        taken[k] = p.id
        r = by_id[k]
        ar = o.image.width / max(1, o.image.height)             # 素材の縦横比(幅 / 高さ)
        # 素材の外接矩形が塊の矩形に収まる最大の大きさ(長辺 L。ar >= 1 なら 幅 = L・高さ = L/ar)
        L = min(r["w_long"] / (1.0 if ar >= 1 else ar), r["h_long"] / (1.0 / ar if ar >= 1 else 1.0))
        p.x, p.y, p.scale, p.geom = r["cx"], r["cy"], L, "sketch"
        p.range80["x"] = [r["x0"], r["x1"]]
        p.range80["y"] = [r["y0"], r["y1"]]
    return log


# ---------------------------------------------------------------- 範囲に収める(規則側)

def postprocess(placements: list[Placement]) -> list[str]:
    """範囲に収め、描く順(order)を付ける。補正した事実を文字列で返す(ログ用)。"""
    log: list[str] = []

    def clamp(p: Placement, k: str, lo: float, hi: float):
        v = getattr(p, k)
        nv = min(max(v, lo), hi)
        if abs(nv - v) > 1e-9:
            log.append(f"{p.id}({p.name}) {k}: {v:.3f} → {nv:.3f}")
            setattr(p, k, nv)

    for p in placements:
        rr = p.range80.get("rot")
        if rr and rr[1] - rr[0] > ROT_MAX_RANGE_DEG and abs(p.rot) > 1e-9:
            log.append(f"{p.id}({p.name}) rot: 80%範囲 [{rr[0]:.0f}, {rr[1]:.0f}] が広いので {p.rot:.0f}° → 0°")
            p.rot = 0.0
        clamp(p, "x", 0.02, 0.98)       # 中心はキャンバス内に(はみ出し自体は許す)
        clamp(p, "y", 0.02, 0.98)
        clamp(p, "z", 0.0, 1.0)
        clamp(p, "scale", RANGES["scale"][0], RANGES["scale"][1])
        clamp(p, "rot", -180.0, 180.0)
        if p.missing:
            log.append(f"{p.id}({p.name}): Clairic が値を出さなかった項目 {p.missing} は既定値で補った")
    ranked = sorted(range(len(placements)), key=lambda i: (placements[i].z, i))
    for rank, i in enumerate(ranked):
        placements[i].order = rank
    return log


# ---------------------------------------------------------------- モック(Clairic を呼ばないとき)

def mock_placements(objs: list[Obj], regions: list[dict], seed: int = 7) -> list[Placement]:
    """Clairic を呼ばないときの配置。スケッチの塊に、素材を大きい順に当てはめる(対応づけに意味の判断はしない)。

    UI の確認と無料枠の節約用で、Clairic の代わりではない。塊より素材が多ければ余りは楕円状に置く。
    """
    rng = random.Random(seed)
    n = len(objs)
    order = sorted(range(n), key=lambda i: -(objs[i].image.width * objs[i].image.height))
    slots = sorted(regions, key=lambda r: -(r["w_long"] * r["h_long"]))
    out: list[Placement | None] = [None] * n
    for rank, i in enumerate(order):
        rot = max(-20.0, min(20.0, rng.gauss(0, 8)))
        if rank < len(slots):
            shape = slots[rank]["id"]
            x = y = scale = 0.5   # 実際の値は apply_regions が塊から決める
            z = slots[rank]["cy"]
        else:
            t = 2 * math.pi * ((rank - len(slots)) / max(1, n)) + 0.6
            shape, x, y, scale = 0, 0.5 + 0.28 * math.cos(t), 0.5 + 0.24 * math.sin(t), 0.2
            z = y
        z = min(1.0, max(0.0, z + rng.uniform(-0.05, 0.05)))
        out[i] = Placement(id=objs[i].id, name=objs[i].name, x=x, y=y, z=z, scale=scale, rot=rot, shape=shape,
                           range80={"z": [z - 0.15, z + 0.15], "rot": [rot - 12, rot + 12]})
    return [p for p in out if p is not None]


# ---------------------------------------------------------------- 入口

def arrange(theme: str, objs: list[Obj], sketch: Image.Image | None, canvas_w: int, canvas_h: int,
            client: CloudflareClient | None, mock: bool = False) -> dict:
    if not objs:
        raise ArrangeError("素材がありません", 400)
    if len(objs) > MAX_OBJECTS:
        raise ArrangeError(f"素材は {MAX_OBJECTS} 個までです", 400)
    aspect = canvas_w / max(1, canvas_h)

    regions: list[dict] = []
    sketch_png = None
    if sketch is not None:
        regions = imaging.sketch_regions(sketch)       # 線がほぼない・小さな点だけのときは空 = スケッチなし
        if regions:
            sketch_png, _ = imaging.flatten_sketch(sketch, regions)

    meta: dict = {"neurons": None, "latency_s": 0.0}
    if mock:
        source = "mock"
        placements = mock_placements(objs, regions)
    else:
        if client is None:
            raise ArrangeError("Clairic のクライアントがありません", 503)
        sheet = imaging.contact_sheet([(o.id, o.image) for o in objs])
        images = ([sketch_png] if sketch_png else []) + [sheet]
        assigned: dict[str, int] = {}
        assign_p: dict[str, float] = {}
        if regions:
            assigned, assign_p, m1 = assign_shapes(client, objs, theme, regions, images)
            meta = _add_meta(meta, m1)
        import base64
        body = {
            "model": "clairic",
            "state": build_state(theme, objs, sketch_png is not None, aspect, regions, assigned),
            "fields": build_fields(objs, aspect),
            "images": [{"content_type": "image/png", "base64": base64.b64encode(b).decode("ascii")} for b in images],
        }
        result, m2 = client.run("clairic", body)
        meta = _add_meta(meta, m2)
        placements = parse_answers(objs, result["answers"])
        for p in placements:
            p.shape, p.shape_p = assigned.get(p.id, 0), assign_p.get(p.id)
        source = "clairic"

    corrections = apply_regions(placements, objs, regions) + postprocess(placements)
    return {
        "source": source,
        "placements": [p.to_json() for p in placements],
        "corrections": corrections,
        "regions": [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in regions],
        "sketch_used": sketch_png is not None,
        **meta,
    }


def _add_meta(a: dict, b: dict) -> dict:
    n = None if a["neurons"] is None and b["neurons"] is None else (a["neurons"] or 0.0) + (b["neurons"] or 0.0)
    return {"neurons": None if n is None else round(n, 2), "latency_s": round(a["latency_s"] + b["latency_s"], 2)}


def make_ids(n: int) -> list[str]:
    return list(string.ascii_uppercase[:n])
