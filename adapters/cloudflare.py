"""Cloudflare Workers AI の REST 実装(M0 で実装)。

TODO(M0): 質問の正確なリクエスト形式は未確認。実装前に公式ドキュメントで確認すること。
  - Clef:    /ai/run/@cf/cloudflare/clef
  - Clairic: Cloudflare 更新情報 2026-10-19(プレビュー)
  認証: 環境変数 CLOUDFLARE_API_TOKEN(Workers AI の Read + Edit)/ CLOUDFLARE_ACCOUNT_ID
  M0 のゲート: 画像1枚の結果をログに残す + 1回あたりのニューロン消費を実測。
  結果は (画像ハッシュ, 質問セットとモデルの版) をキーにキャッシュする。
"""
from __future__ import annotations

from .base import ChoiceResult, EstimateField, EstimateResult, ImageRef, NoulResult, ScoreResult
from typing import Sequence

_MSG = "リクエスト形式を公式ドキュメントで確認してから実装する(docs/design.md M0)"


class CloudflareJudge:
    def noul(self, question: str, images: Sequence[ImageRef], context: str = "") -> NoulResult:
        raise NotImplementedError(_MSG)

    def choice(self, question, options, images, context: str = "") -> ChoiceResult:
        raise NotImplementedError(_MSG)

    def score(self, question, scale, images, context: str = "") -> ScoreResult:
        raise NotImplementedError(_MSG)


class CloudflareEstimator:
    def estimate(
        self, text: str, fields: Sequence[EstimateField], images: Sequence[ImageRef] = ()
    ) -> dict[str, EstimateResult]:
        raise NotImplementedError(_MSG)
