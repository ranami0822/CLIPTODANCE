"""ローカルサーバー(標準ライブラリだけ)。静的ファイル(web/)と API を出す。

  python -m server.app [--host 127.0.0.1] [--port 8765]

  GET  /api/status   {clairic: 資格情報があるか}
  POST /api/cutout   {image: dataURL, tol?: number} → {png: dataURL, ok, note, method, ...}
  POST /api/arrange  {theme, canvas:{w,h}, sketch: dataURL|null, objects:[{name, png: dataURL}], mock?: bool}
                     → {source, placements[], corrections[], neurons, latency_s, sketch_used}
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import arrange as arr
from . import imaging

WEB_DIR = (Path(__file__).resolve().parent.parent / "web").resolve()
MAX_BODY = 40 * 1024 * 1024


def handle_cutout(payload: dict) -> dict:
    im = imaging.open_rgba(imaging.decode_data_url(payload.get("image", "")))
    tol = payload.get("tol")
    out, info = imaging.prepare_object(im, tol=float(tol) if tol not in (None, "") else None)
    return {**info, "png": imaging.to_data_url(out), "width": out.width, "height": out.height}


def handle_arrange(payload: dict, client: arr.CloudflareClient) -> dict:
    raw = payload.get("objects") or []
    if not isinstance(raw, list) or not raw:
        raise arr.ArrangeError("素材がありません", 400)
    if len(raw) > arr.MAX_OBJECTS:
        raise arr.ArrangeError(f"素材は {arr.MAX_OBJECTS} 個までです", 400)
    ids = arr.make_ids(len(raw))
    objs = [
        arr.Obj(id=ids[i], name=str(o.get("name") or "").strip()[:40],
                image=imaging.open_rgba(imaging.decode_data_url(o.get("png", ""))))
        for i, o in enumerate(raw)
    ]
    sketch = None
    if payload.get("sketch"):
        sketch = imaging.open_rgba(imaging.decode_data_url(payload["sketch"]))
    canvas = payload.get("canvas") or {}
    w, h = int(canvas.get("w") or 960), int(canvas.get("h") or 720)
    return arr.arrange(str(payload.get("theme") or "")[:2000], objs, sketch, w, h, client, mock=bool(payload.get("mock")))


def make_handler(client: arr.CloudflareClient):
    class Handler(BaseHTTPRequestHandler):
        server_version = "IdeaMaterializer/0.1"

        def log_message(self, fmt, *args):  # 静かに
            if os.environ.get("IM_VERBOSE"):
                super().log_message(fmt, *args)

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/api/status":
                return self._json(200, {"clairic": client.configured})
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            target = (WEB_DIR / rel).resolve()
            if WEB_DIR not in target.parents and target != WEB_DIR or not target.is_file():
                return self._json(404, {"error": "not found"})
            data = target.read_bytes()
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > MAX_BODY:
                    return self._json(413, {"error": "本文の大きさが範囲外です"})
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                if path == "/api/cutout":
                    return self._json(200, handle_cutout(payload))
                if path == "/api/arrange":
                    return self._json(200, handle_arrange(payload, client))
                return self._json(404, {"error": "not found"})
            except arr.ArrangeError as e:
                return self._json(e.status, {"error": str(e)})
            except (imaging.ImageError, ValueError, KeyError) as e:
                return self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": f"内部エラー: {type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    client = arr.CloudflareClient()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(client))
    print(f"http://{a.host}:{a.port}/  (Clairic: {'設定あり' if client.configured else '未設定 → モックのみ'})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
