"""テスト・ドライラン用の偽の Judge。外部通信をしない。

Cloudflare 実装と同じ上限の検査(質問数・画像数)をするので、
呼び出し側が上限を守っているかをテストで確かめられる。
"""
from __future__ import annotations

from typing import Sequence

from .base import (
    ChoiceResult,
    ImageInput,
    JudgeCapabilities,
    JudgeError,
    JudgeResponse,
    JudgeResult,
    NoulResult,
    Question,
    QuestionKind,
    ScoreResult,
    Usage,
    check_capabilities,
    validate_batch,
)


class FakeJudge:
    def __init__(
        self,
        model_id: str = "fake",
        max_questions: int | None = 64,
        max_images: int | None = 4,
        neurons_per_call: float | None = None,
        fail_with: JudgeError | None = None,
    ):
        self.model_id = model_id
        self.capabilities = JudgeCapabilities(max_questions=max_questions, max_images=max_images)
        self.neurons_per_call = neurons_per_call
        self.fail_with = fail_with
        self.calls: list[list[Question]] = []  # 実際に受けた呼び出しごとの質問

    def ask(
        self, questions: Sequence[Question], images: Sequence[ImageInput] = (), context: str = ""
    ) -> JudgeResponse:
        validate_batch(questions)
        check_capabilities(self.capabilities, len(questions), len(images))
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append(list(questions))
        results: list[JudgeResult] = []
        for q in questions:
            if q.kind is QuestionKind.NOUL:
                results.append(NoulResult(question_id=q.id, p_yes=0.5))
            elif q.kind is QuestionKind.CHOICE:
                p = 1.0 / len(q.options)
                results.append(ChoiceResult(question_id=q.id, probs={o: p for o in q.options}, confidence=0.0))
            else:
                p = 1.0 / len(q.scale)
                mid = (len(q.scale) + 1) / 2
                results.append(ScoreResult(question_id=q.id, score=mid, level_probs={s: p for s in q.scale}))
        return JudgeResponse(
            results=tuple(results),
            usage=Usage(api_calls=1, neurons=self.neurons_per_call),
            model_id=self.model_id,
        )
