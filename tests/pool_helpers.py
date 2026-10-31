"""プール取り込みのテスト用の部品: 合成画像、偽のセグメンタ、偽の埋め込み。"""
import io

import numpy as np
from PIL import Image

from pipeline.pool.embed import _normalize
from pipeline.pool.segment import Candidate
from pipeline.pool.sources.met import SourceItem

H, W = 240, 360
HORIZON = 0.6


def landscape_array():
    """上が空(青)、下が地面(緑)、地平線は高さ 0.6。赤い円と黄色い四角を置く。"""
    img = np.zeros((H, W, 3), np.uint8)
    split = int(H * HORIZON)
    img[:split] = (110, 160, 230)
    img[split:] = (70, 130, 60)
    yy, xx = np.mgrid[:H, :W]
    circle = (yy - 170) ** 2 + (xx - 120) ** 2 <= 30 ** 2
    square = (abs(yy - 175) <= 20) & (abs(xx - 260) <= 20)
    img[circle] = (210, 30, 30)
    img[square] = (240, 220, 40)
    return img, circle, square


def jpeg_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")  # 可逆にして、色の判定を安定させる
    return buf.getvalue()


class FakeSegmenter:
    name = "fake-seg"

    def __init__(self, extra=()):
        self.extra = list(extra)
        self.calls = 0

    def segment(self, rgb):
        self.calls += 1
        _, circle, square = landscape_array()
        sky = np.zeros((H, W), bool)
        sky[: int(H * 0.7)] = True  # 面積比 0.7 > 0.6 なので落とされるはずのマスク
        tiny = np.zeros((H, W), bool)
        tiny[5:7, 5:7] = True           # 小さすぎるので落とされるはずのマスク
        dup = circle.copy()             # 円とほぼ同じ(重複)
        dup[170, 90] = False
        return [Candidate(circle, 0.9), Candidate(square, 0.8), Candidate(sky, 0.95),
                Candidate(tiny, 0.99), Candidate(dup, 0.85), *self.extra]


class FakeEmbedder:
    """色だけで決まる埋め込み。画像は平均色(灰色の背景は無視しない単純な平均)、文章は色の名前から。"""
    name = "fake-color"
    dim = 3
    WORDS = {"red": (1, 0, 0), "yellow": (1, 1, 0), "blue": (0, 0, 1), "green": (0, 1, 0)}

    def encode_images(self, images):
        out = []
        for im in images:
            a = np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3)
            # 灰色(127)の背景に埋もれないよう、彩度の高い画素だけの平均を使う
            sat = a.max(1) - a.min(1)
            sel = a[sat > 60] if (sat > 60).any() else a
            out.append(sel.mean(0))
        return _normalize(np.array(out, np.float32))

    def encode_texts(self, texts):
        def vec(t):
            for word, v in self.WORDS.items():   # 文の中の最初の色の語で決める("a painting of red" → red)
                if word in t:
                    return v
            raise KeyError(t)
        return _normalize(np.array([vec(t) for t in texts], np.float32))


def make_item(source_id="1"):
    return SourceItem(source="test", source_id=source_id, source_url="http://example/x", license="CC0 (test)",
                      title="Landscape", artist="A. Painter", date_text="1850", image_url="http://example/x.jpg",
                      meta={"department": "European Paintings"})
