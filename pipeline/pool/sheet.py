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


# ---------------------------------------------------------------- 印付け用(番号を大きく入れる)

def _badge(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 26) -> None:
    from PIL import ImageFont

    font = ImageFont.load_default(size=size)
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle([box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3], fill=(20, 20, 20))
    draw.text(xy, text, fill=(255, 235, 80), font=font)


def numbered_cutout_sheets(numbered: list[tuple[int, dict]], out_dir: str | Path, per_sheet: int = 30,
                           cols: int = 6, thumb: int = 210) -> list[Path]:
    """番号つきの切り抜きの一覧を、per_sheet 個ずつの画像にして書き出す。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(0, len(numbered), per_sheet):
        chunk = numbered[s:s + per_sheet]
        tiles = []
        for n, r in chunk:
            rgba = Image.open(r["cutout_path"]).convert("RGBA")
            rgba.thumbnail((thumb, thumb))
            canvas = _checker((thumb, thumb))
            canvas.paste(rgba, ((thumb - rgba.width) // 2, (thumb - rgba.height) // 2), rgba)
            _badge(ImageDraw.Draw(canvas), (6, 4), str(n))
            tiles.append((canvas, ""))
        sheet = _grid(tiles, thumb, cols)
        path = out_dir / f"cutouts_{chunk[0][0]:03d}-{chunk[-1][0]:03d}.png"
        sheet.crop((0, 0, sheet.width, sheet.height - LABEL_H)).save(path)  # 文字の欄は使わないので切る
        paths.append(path)
    return paths


def numbered_horizon_sheet(numbered: list[tuple[int, dict]], path: str | Path, thumb: int = 330, cols: int = 4) -> Path:
    """番号つきの絵に、0.1 刻みの目盛り(白の点線、0.0 が上端・1.0 が下端)と、自動推定の線(赤)を重ねる。"""
    tiles = []
    for n, r in numbered:
        im = Image.open(r["work_path"]).convert("RGB")
        im.thumbnail((thumb - 40, thumb - 28))  # 上下に目盛りの文字の余白
        canvas = Image.new("RGB", (thumb, thumb), (50, 50, 50))
        ox, oy = 36, (thumb - im.height) // 2
        canvas.paste(im, (ox, oy))
        d = ImageDraw.Draw(canvas)
        for k in range(0, 11):
            y = oy + k / 10 * im.height
            if 0 < k < 10:
                for x in range(ox, ox + im.width, 8):
                    d.line([(x, y), (x + 3, y)], fill=(255, 255, 255), width=1)
            d.text((2, y - 6), f"{k/10:.1f}", fill=(200, 200, 200))
        ya = oy + r["horizon_auto_y"] * im.height
        d.line([(ox, ya), (ox + im.width, ya)], fill=(255, 40, 40), width=2)
        d.text((ox + 4, ya + 3), f"auto={r['horizon_auto_y']:.2f}", fill=(255, 120, 120))
        _badge(d, (ox + 4, oy + 4), f"#{n}")
        tiles.append((canvas, ""))
    sheet = _grid(tiles, thumb, cols)
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    sheet.crop((0, 0, sheet.width, sheet.height - LABEL_H)).save(out)
    return out
