"""切り抜きの属性判定(Clef)。設計書「用途と質問設計」の事前判定の分。

質問は英語で書く(公式ドキュメントの例が英語。日本語との精度差は未確認)。
「この切り抜きは検索語の物か」は取り込み時には要望が分からないので、ここには含めない(実行時の確認に回す)。
1 切り抜きにつき 1 回の `ask` に全質問をまとめる(Clef は 1 リクエスト 64 問まで)。
"""
from __future__ import annotations

import hashlib

from adapters import Judge, JudgeResult, Question, Usage, choice, noul, score
from adapters.base import ChoiceResult, NoulResult, ScoreResult

FACING = ["front", "left", "right", "back", "unknown"]
LIGHT = ["left", "right", "top", "front", "unknown"]
EDGE_SCALE = [
    "very rough: ragged or jagged outline, parts of the background remain",
    "rough: visibly uneven outline",
    "acceptable outline",
    "clean outline",
    "very clean outline, nothing missing and nothing extra",
]


def cutout_questions() -> list[Question]:
    return [
        noul("watermark_text", "Does the image contain a watermark, a caption or written text?"),
        noul("cut_off", "Is the main object cut off by the edge of the image, so that part of it is missing?"),
        noul("single_object", "Is this one complete object, rather than a fragment or a group of several objects?"),
        noul("flip_unnatural", "Would mirroring this image left to right look unnatural, for example because of text, a one-sided feature or a known landmark?"),
        choice("facing", "Which way is the main object facing?", FACING),
        choice("light_dir", "Where does the main light on the object come from?", LIGHT),
        score("edge_quality", "How clean is the outline of this cut-out object?", EDGE_SCALE),
    ]


def questions_version() -> str:
    """質問の中身から作る短い版。質問を変えたら変わる(再判定の判断用)。"""
    raw = "\n".join(q.canonical() for q in cutout_questions())
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def judge_version(judge: Judge) -> str:
    return f"{judge.model_id}|q-{questions_version()}"


def results_to_attrs(results: tuple[JudgeResult, ...]) -> dict:
    attrs: dict = {}
    for r in results:
        if isinstance(r, NoulResult):
            attrs[r.question_id] = round(r.p_yes, 4)
        elif isinstance(r, ChoiceResult):
            attrs[r.question_id] = {"top": r.top, "probs": {k: round(v, 4) for k, v in r.probs.items()}, "confidence": r.confidence}
        elif isinstance(r, ScoreResult):
            # score は 0 始まり(0 = 最低段階)。0〜1 に直した値も持つ
            attrs[r.question_id] = {"score": round(r.score, 4), "normalized": round(r.score / (len(EDGE_SCALE) - 1), 4), "confidence": r.confidence}
    return attrs


def judge_image(judge: Judge, image_png: bytes) -> tuple[dict, Usage]:
    """1 枚の切り抜き画像(灰色の背景に載せた PNG)を判定し、(属性, 使用量)を返す。"""
    resp = judge.ask(cutout_questions(), images=[image_png], state="")
    return results_to_attrs(resp.results), resp.usage
