"""目視確認用のコンタクトシート(設計書: 地平線と切り抜きは一覧で確認して直す)。ラベルは ASCII のみ。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

PAD = 6
LABEL_H = 30


def _checker(size: tuple[int, int], cell: int = 8) -> Image.Image:
    im = Image.new("RGB", size, (200, 200, 200))
    d = ImageDraw.Draw(im)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                d.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(160, 160, 160))
    return im


def _grid(tiles: list[tuple[Image.Image, str]], thumb: int, cols: int) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    cell_w, cell_h = thumb + PAD, thumb + LABEL_H + PAD
    sheet = Image.new("RGB", (cols * cell_w + PAD, max(1, rows) * cell_h + PAD), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(tiles):
        x, y = PAD + (i % cols) * cell_w, PAD + (i // cols) * cell_h
        sheet.paste(im, (x, y))
        d.text((x, y + thumb + 2), label, fill=(235, 235, 235))
    return sheet


def horizon_sheet(images: list[dict], path: str | Path, thumb: int = 240, cols: int = 5) -> Path:
    tiles = []
    for r in images:
        im = Image.open(r["work_path"]).convert("RGB")
        im.thumbnail((thumb, thumb))
        canvas = Image.new("RGB", (thumb, thumb), (60, 60, 60))
        ox, oy = (thumb - im.width) // 2, (thumb - im.height) // 2
        canvas.paste(im, (ox, oy))
        y = oy + r["horizon_y"] * im.height
        ImageDraw.Draw(canvas).line([(ox, y), (ox + im.width, y)], fill=(255, 40, 40), width=2)
        mark = "OK" if r["horizon_checked"] else "?"
        tiles.append((canvas, f"{r['id']}\ny={r['horizon_y']:.2f} conf={r['horizon_conf']:.2f} [{mark}]"))
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    _grid(tiles, thumb, cols).save(out)
    return out


def cutout_sheet(cutouts: list[dict], path: str | Path, thumb: int = 200, cols: int = 6, scores: dict[str, float] | None = None) -> Path:
    tiles = []
    for r in cutouts:
        rgba = Image.open(r["cutout_path"]).convert("RGBA")
        rgba.thumbnail((thumb, thumb))
        canvas = _checker((thumb, thumb))
        canvas.paste(rgba, ((thumb - rgba.width) // 2, (thumb - rgba.height) // 2), rgba)
        extra = f" s={scores[r['id']]:.3f}" if scores and r["id"] in scores else ""
        tiles.append((canvas, f"{r['id']}{extra}\narea={r['area_ratio']:.3f} edge={r['edge_score']:.2f} clut={r['clutter']} bt={r['border_touch']}"))
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    _grid(tiles, thumb, cols).save(out)
    return out
