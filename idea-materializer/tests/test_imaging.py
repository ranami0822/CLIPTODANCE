import io

import numpy as np
from PIL import Image

from server import imaging
from scripts import make_samples as ms


def test_alpha_png_is_kept_and_trimmed():
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    im.paste((255, 0, 0, 255), (60, 80, 120, 140))
    out, info = imaging.prepare_object(im)
    assert info["method"] == "alpha" and info["ok"]
    assert out.width <= 64 and out.height <= 64          # 60x60 + pad 2*2


def test_border_flood_cuts_white_background():
    out, info = imaging.prepare_object(ms.house())
    assert info["ok"], info
    a = np.asarray(out.getchannel("A"))
    # 四隅は透明、家の中心は不透明
    assert a[0, 0] == 0 and a[-1, -1] == 0
    assert a[a.shape[0] // 2, a.shape[1] // 2] > 200
    assert 0.2 < info["fg_fraction"] < 0.9


def test_border_flood_does_not_eat_same_color_inside():
    # 背景と同じ白が物の内側にあっても(縁につながっていなければ)抜かない
    im = Image.new("RGB", (200, 200), (255, 255, 255))
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, 160, 160), fill=(20, 20, 160))
    d.rectangle((90, 90, 110, 110), fill=(255, 255, 255))   # 内側の白い穴(閉じている)
    out, info = imaging.cutout_border_flood(im.convert("RGBA"))
    assert info["ok"]
    a = np.asarray(out.getchannel("A"))
    assert a[a.shape[0] // 2, a.shape[1] // 2] > 200


def test_noisy_background_is_reported_not_forced():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    out, info = imaging.cutout_border_flood(Image.fromarray(noise).convert("RGBA"))
    assert not info["ok"] and "一様" in info["note"]


def test_contact_sheet_is_png_with_all_tiles():
    items = [("A", ms.tree()), ("B", ms.sun()), ("C", ms.house().convert("RGBA"))]
    data = imaging.contact_sheet(items)
    im = Image.open(io.BytesIO(data))
    assert im.format == "PNG" and im.width == 2 * imaging.TILE and im.height == 2 * imaging.TILE


def test_flatten_sketch_ink_fraction():
    png, ink = imaging.flatten_sketch(ms.sketch().convert("RGBA"))
    assert 0.003 < ink < 0.2
    _, blank = imaging.flatten_sketch(Image.new("RGBA", (100, 100), (0, 0, 0, 0)))
    assert blank == 0.0


def test_data_url_roundtrip_and_rejects_garbage():
    url = imaging.to_data_url(ms.sun())
    assert imaging.open_rgba(imaging.decode_data_url(url)).size == ms.sun().size
    import pytest
    with pytest.raises(imaging.ImageError):
        imaging.decode_data_url("http://example.com/x.png")


def test_sketch_regions_find_three_shapes_in_reading_order():
    r = imaging.sketch_regions(ms.sketch().convert("RGBA"))
    assert [x["id"] for x in r] == [1, 2, 3]
    assert (r[0]["cx"], r[0]["cy"]) == __import__("pytest").approx((0.86, 0.17), abs=0.03)   # 右上の丸
    assert (r[1]["cx"], r[1]["cy"]) == __import__("pytest").approx((0.24, 0.62), abs=0.03)   # 左下の三角
    assert (r[2]["cx"], r[2]["cy"]) == __import__("pytest").approx((0.67, 0.65), abs=0.03)   # 右寄りの四角


def test_sketch_regions_merge_nearby_strokes_and_drop_specks():
    from PIL import ImageDraw
    im = Image.new("RGB", (960, 720), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle((200, 200, 400, 400), outline=(0, 0, 0), width=5)
    d.line((250, 250, 350, 350), fill=(0, 0, 0), width=5)          # 同じ塊の内側の線
    d.line((420, 300, 520, 300), fill=(0, 0, 0), width=5)          # すぐ隣の線(同じ塊にまとまる)
    d.point((800, 100), fill=(0, 0, 0))                            # ごみ
    r = imaging.sketch_regions(im.convert("RGBA"))
    assert len(r) == 1
    assert imaging.sketch_regions(Image.new("RGBA", (300, 200), (255, 255, 255, 255))) == []


def test_flatten_sketch_draws_region_boxes_without_changing_ink():
    s = ms.sketch().convert("RGBA")
    plain, ink0 = imaging.flatten_sketch(s)
    boxed, ink1 = imaging.flatten_sketch(s, imaging.sketch_regions(s))
    assert ink0 == ink1 and plain != boxed
