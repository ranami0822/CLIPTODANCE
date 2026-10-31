# CLIPTODANCE

テキストの要望から、検索した既存画像の切り抜きを配置・合成して1枚のコラージュを作るパイプライン(Python / PyTorch)。生成モデルは学習しない。

**設計は草案で、これから詰める → [docs/design.md](docs/design.md)**

## 構成(M0)

| パス | 内容 |
| --- | --- |
| `docs/design.md` | 設計方針(草案) |
| `pipeline/` | パイプライン本体(M1 以降で追加) |
| `adapters/` | 判断モデルの差し替え層。`Judge`(Clef 相当)/ `Estimator`(Clairic 相当)のインターフェース、`CachedJudge`(SQLite キャッシュ)、`FakeJudge`(テスト用)。Cloudflare 実装は未実装 |
| `configs/default.yaml` | 設定。数値は設計書の仮置き |
| `data/` | 素材・索引・キャッシュ(git 管理外) |
| `scripts/` | 取り込み・実験スクリプト |

## 現状

- インターフェース・キャッシュ・テスト用の偽実装まで。Cloudflare 実装は、リクエスト形式を公式ドキュメントで確認するまで `NotImplementedError`。
- 重い依存(torch / CLIP / SAM / FAISS)は本番(GPU・Colab)で M1 から追加する。

## 開発

```bash
pip install -e ".[dev]"
pytest
```
