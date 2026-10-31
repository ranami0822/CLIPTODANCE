"""実際の Cloudflare に合成データで 3 回呼ぶテスト。無料枠を数ニューロン使うので、明示したときだけ動く。

    CLIPTODANCE_LIVE=1 pytest tests/test_cloudflare_live.py
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from m0_probe import solid_png  # noqa: E402

from adapters import EstimateField, FieldKind, choice, noul, score  # noqa: E402
from adapters.cloudflare import CloudflareEstimator, CloudflareJudge  # noqa: E402

pytestmark = pytest.mark.skipif(os.environ.get("CLIPTODANCE_LIVE") != "1", reason="CLIPTODANCE_LIVE=1 のときだけ実行")


def test_clef_live_text_and_image():
    judge = CloudflareJudge()
    resp = judge.ask(
        [noul("has_tree", "Is there a tree in the scene?"),
         choice("tree_side", "Which side is the tree on?", ["left", "right", "center"]),
         score("sunny", "How sunny is the scene?", ["dark", "cloudy", "bright", "very bright"])],
        state="A sunny meadow with one large oak tree on the right.",
    )
    n, c, s = resp.results
    assert n.p_yes > 0.5 and c.top == "right" and 0 <= s.score <= 3
    assert resp.usage.neurons is not None and resp.usage.input_tokens > 0

    red = judge.ask([noul("is_red", "Is the image mostly red?")], images=[solid_png(64, 64, (200, 30, 30))])
    assert red.results[0].p_yes > 0.5


def test_clairic_live_estimate():
    est = CloudflareEstimator().estimate(
        "A sunny meadow with one large oak tree on the right side.",
        [EstimateField("x", FieldKind.ESTIMATE, "Horizontal position of the tree centre, 0 = left edge",
                       minimum=0, maximum=1, step=0.01)],
    )
    x = est.values["x"]
    assert x.range80 is not None and x.range80[0] <= x.value <= x.range80[1] and x.value > 0.5
