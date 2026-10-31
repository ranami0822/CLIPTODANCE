# アイデア具現化マシーン

素材の画像と、粗いスケッチと、ひとことのテーマを入れると、素材をスケッチに合わせて動かして並べるツール。

> **本筋(リポジトリ直下の CLIPTODANCE)とは独立した別プロジェクト。** このフォルダの外のファイルは編集していない。
> 本筋のコードも import していない(Clef / Clairic の呼び出し形式を、本筋の `adapters/cloudflare.py` を読んで確かめたうえで、ここで書き直している)。

```
素材の画像 ──(切り抜き)──┐
スケッチ ──(塊に分ける)──┼─→ Clef: どの素材がどの塊か ─→ Clairic: 奥行き z・傾き rot ─→ 配置(x, y, z, 大きさ, 回転)
テーマの文 ──────────────┘                                                       └→ ブラウザで動かして直す → PNG / JSON
```

- 素材: 透過 PNG はそのまま。背景が単色の写真・イラストは、縁からの塗りつぶしで自動で切り抜く。
- スケッチ: ブラウザの上でペンで描く(丸・三角・四角の粗さでよい)。つながった線を「塊」として検出し、番号を振る。
- 配置: 塊に素材を対応させ(Clef)、奥行きと傾きを推定する(Clairic)。各値に 80% 範囲が付き、画面に点線で出る。
- 仕上げ: 素材はドラッグ・回転・拡大縮小・スライダーで直せる。PNG と、位置・奥行きの JSON を書き出せる。

設計の経緯と実測は [docs/design.md](docs/design.md)。

## 動かす

必要なもの: Python 3.10 以上、`pillow`、`numpy`。テストに `pytest`。

```bash
cd idea-materializer
pip install pillow numpy pytest
python -m server.app            # http://127.0.0.1:8765/ を開く
```

Clef / Clairic を使うには、Cloudflare Workers AI の資格情報を環境変数で渡す(トークンは Workers AI の Read と Edit が必要)。

```bash
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_API_TOKEN=...
python -m server.app
```

資格情報がなければ「モック配置」だけが動く(スケッチの塊に素材を大きい順に当てはめるだけで、意味の判断はしない)。画面左の「モック配置」で切り替える。

動作確認用のサンプル素材とスケッチ:

```bash
python scripts/make_samples.py samples/   # tree.png sun.png house.jpg sketch.png
```

## テスト

```bash
python -m pytest -q       # idea-materializer/ の中で。通信は差し替えるので資格情報は要らない
```

## 費用の目安(2026-10-31 実測)

素材 3 個 + スケッチ 1 枚で、1 回の「配置する」が Clef + Clairic の 2 呼び出しで約 50 ニューロン・1〜1.5 秒。Cloudflare の無料枠は 1 日 10,000 ニューロン(全モデル共通)。

## ファイル

| パス | 内容 |
| --- | --- |
| `server/imaging.py` | 切り抜き・スケッチの塊の検出・Clairic に渡す画像(コンタクトシート)づくり |
| `server/arrange.py` | Clef での対応づけ、Clairic での推定、範囲に収める規則、モック |
| `server/app.py` | 標準ライブラリだけのサーバー(静的ファイル + `/api/cutout` `/api/arrange` `/api/status`) |
| `web/` | ブラウザの画面(ビルド不要。`index.html` `app.js` `style.css`) |
| `scripts/make_samples.py` | サンプル素材の生成 |
| `tests/` | pytest(27 件) |
