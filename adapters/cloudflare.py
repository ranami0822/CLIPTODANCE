"""Cloudflare Workers AI の REST 実装(M0 で実装)。

TODO(M0): 質問の正確なリクエスト形式は未確認。実装前に公式ドキュメントで確認すること。
  - Clef:    /ai/run/@cf/cloudflare/clef (設計書の記載。未検証)
  - Clairic: Cloudflare 更新情報 2026-10-19(プレビュー)。モデルIDは未確認
  認証: 環境変数 CLOUDFLARE_API_TOKEN(Workers AI の Read + Edit)/ CLOUDFLARE_ACCOUNT_ID
  M0 のゲート: 画像1枚の結果をログに残す + 1回あたりのニューロン消費を実測。

実装するときの方針:
  - 応答の消費量が取れるなら `Usage.neurons` に入れる。取れないなら None のまま(推測で埋めない)。
  - 無料枠を超えたら `BudgetExceeded`、頻度制限は `RateLimited`、その他は `ApiError`。握りつぶさない。
  - キャッシュは `CachedJudge` で包んで使う。
"""
from __future__ import annotations

from typing import Sequence

from .base import (
    EstimateField,
    EstimateResponse,
    ImageInput,
    JudgeCapabilities,
    JudgeResponse,
    Question,
)

_MSG = "リクエスト形式を公式ドキュメントで確認してから実装する(docs/design.md M0)"

CLEF_MODEL_ID = "@cf/cloudflare/clef"  # 設計書の記載。未検証
CLAIRIC_MODEL_ID = "UNVERIFIED:clairic"  # 実際のIDは公式ドキュメントで確認する

# 設計書「確認済みの仕様」: 1リクエスト最大64問、状態と一緒に最大4枚の画像
CLEF_CAPABILITIES = JudgeCapabilities(max_questions=64, max_images=4)
# Clairic: テキストと最大4枚の画像。質問数の概念はない
CLAIRIC_CAPABILITIES = JudgeCapabilities(max_questions=None, max_images=4)


class CloudflareJudge:
    model_id = CLEF_MODEL_ID
    capabilities = CLEF_CAPABILITIES

    def ask(
        self, questions: Sequence[Question], images: Sequence[ImageInput] = (), context: str = ""
    ) -> JudgeResponse:
        raise NotImplementedError(_MSG)


class CloudflareEstimator:
    model_id = CLAIRIC_MODEL_ID
    capabilities = CLAIRIC_CAPABILITIES

    def estimate(
        self,
        text: str,
        fields: Sequence[EstimateField],
        images: Sequence[ImageInput] = (),
        context: str = "",
    ) -> EstimateResponse:
        raise NotImplementedError(_MSG)
