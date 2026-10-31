# CLIPTODANCE

テキストの要望から、検索した既存画像の切り抜きを配置・合成して1枚のコラージュを作るパイプライン(Python / PyTorch)。生成モデルは学習しない。

**設計は草案で、これから詰める → [docs/design.md](docs/design.md)**

## 構成(M0)

| パス | 内容 |
| --- | --- |
| `docs/design.md` | 設計方針(草案) |
| `pipeline/` | パイプライン本体(M1 以降で追加) |
| `adapters/` | 判断モデルの差し替え層。`Judge`(Clef 相当)/ `Estimator`(Clairic 相当)のインターフェース、`CachedJudge`(SQLite キャッシュ)、`FakeJudge`(テスト用)、Cloudflare 実装(Clef / Clairic) |
| `configs/default.yaml` | 設定。数値は設計書の仮置き |
| `data/` | 素材・索引・キャッシュ(git 管理外) |
| `scripts/` | 取り込み・実験スクリプト |

## 現状

- 判断側(Clef / Clairic)のアダプタまで実装済み。合成データで実際の API を呼んで確認した。
- 実際の API を呼ぶ確認: `python scripts/m0_probe.py`(数ニューロン消費。要 `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID`)、`CLIPTODANCE_LIVE=1 pytest tests/test_cloudflare_live.py`。
- 重い依存(torch / CLIP / SAM / FAISS)は本番(GPU・Colab)で M1 から追加する。

## 開発

```bash
pip install -e ".[dev]"
pytest
```
