"""動作確認用のサンプル素材を作る(PIL だけ)。

  python scripts/make_samples.py samples/

  tree.png   透過 PNG(木)
  sun.png    透過 PNG(太陽)
  house.jpg  白背景の家(自動切り抜きの確認用。透過なし)
  sketch.png 白地に丸・三角・四角の粗いスケッチ
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw


def tree(size=240):
    im = Image.new("RGBA", (size, size * 3 // 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((size * 0.42, size * 1.0, size * 0.58, size * 1.45), fill=(110, 70, 35, 255))
    d.ellipse((size * 0.1, size * 0.1, size * 0.9, size * 1.1), fill=(40, 140, 60, 255))
    return im


def sun(size=200):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((10, 10, size - 10, size - 10), fill=(250, 190, 40, 255))
    return im


def house(w=320, h=280):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((50, 120, 270, 250), fill=(200, 90, 70))
    d.polygon([(30, 125), (160, 30), (290, 125)], fill=(90, 50, 40))
    d.rectangle((140, 170, 185, 250), fill=(60, 40, 30))
    return im


def sketch(w=960, h=720):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse((760, 60, 880, 180), outline=(30, 30, 30), width=6)          # 右上: 太陽の丸
    d.polygon([(150, 560), (230, 330), (310, 560)], outline=(30, 30, 30), width=6)  # 左下: 木の三角
    d.rectangle((520, 380, 760, 560), outline=(30, 30, 30), width=6)       # 中央右: 家の四角
    return im


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    out.mkdir(parents=True, exist_ok=True)
    tree().save(out / "tree.png")
    sun().save(out / "sun.png")
    house().save(out / "house.jpg", quality=92)
    sketch().save(out / "sketch.png")
    print("wrote", sorted(p.name for p in out.iterdir()))
