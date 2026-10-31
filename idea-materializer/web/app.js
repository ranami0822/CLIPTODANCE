"use strict";
// アイデア具現化マシーン — フロント。依存なし。
// 座標の約束は server/arrange.py の冒頭と同じ:
//   x,y = 中心(キャンバスの幅・高さに対する 0–1) / z = 奥(0)↔手前(1) / scale = 長辺÷キャンバスの長辺 / rot = 度(時計回り)

const $ = (s) => document.querySelector(s);
const S = {
  CW: 960, CH: 720,
  objects: [],          // {uid, name, img, dataUrl, origUrl, x,y,z,scale,rot, range, msg, warn}
  selected: null,       // uid
  tool: "select",
  drag: null,
  sketchInk: false,
  server: { clairic: false, reachable: false },
  busy: false,
  nextUid: 1,
};

const sketchCv = $("#sketch"), sceneCv = $("#scene");
const sk = sketchCv.getContext("2d"), sc = sceneCv.getContext("2d");

// ------------------------------------------------------------ キャンバスの大きさ
function setCanvasSize(w, h) {
  const old = document.createElement("canvas");
  old.width = sketchCv.width; old.height = sketchCv.height;
  old.getContext("2d").drawImage(sketchCv, 0, 0);
  S.CW = w; S.CH = h;
  for (const c of [sketchCv, sceneCv]) { c.width = w; c.height = h; }
  sk.lineCap = "round"; sk.lineJoin = "round";
  if (old.width > 1) sk.drawImage(old, 0, 0, w, h);
  $("#stage").style.aspectRatio = `${w} / ${h}`;
  render();
}
function aspectToSize(a) {
  const [w, h] = a.split(":").map(Number);
  const long = 960;
  return w >= h ? [long, Math.round((long * h) / w)] : [Math.round((long * w) / h), long];
}

// ------------------------------------------------------------ 素材の取り込み
const readAsDataURL = (f) => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(f); });
const loadImg = (url) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });

async function hasTransparency(img) {
  const c = document.createElement("canvas");
  const k = Math.min(1, 256 / Math.max(img.naturalWidth, img.naturalHeight));
  c.width = Math.max(1, Math.round(img.naturalWidth * k)); c.height = Math.max(1, Math.round(img.naturalHeight * k));
  const x = c.getContext("2d"); x.drawImage(img, 0, 0, c.width, c.height);
  const d = x.getImageData(0, 0, c.width, c.height).data;
  let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] < 250) n++;
  return n / (d.length / 4) > 0.01;
}

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
  return j;
}

async function addFiles(files) {
  for (const f of files) {
    if (!/^image\/(png|jpeg|webp)$/.test(f.type)) { note(`「${f.name}」は png / jpeg / webp ではありません`); continue; }
    if (S.objects.length >= 8) { note("素材は 8 個までです"); break; }
    const orig = await readAsDataURL(f);
    const o = {
      uid: S.nextUid++, name: f.name.replace(/\.[^.]+$/, "").slice(0, 20),
      origUrl: orig, dataUrl: orig, img: null, tol: null,
      x: 0.5, y: 0.7, z: 0.5, scale: 0.2, rot: 0, range: null, msg: "読み込み中…", warn: false,
    };
    S.objects.push(o);
    await processObject(o);
    S.selected = o.uid;
  }
  spreadDefaults();
  renderList(); render();
}

async function processObject(o, tol = null) {
  try {
    const raw = await loadImg(o.origUrl);
    if (tol === null && await hasTransparency(raw)) {
      // 透過 PNG はそのまま(サーバーなしでも動く)。余白だけサーバー側で詰めたいが、なくても困らない
      o.dataUrl = o.origUrl; o.img = raw; o.msg = "透過 PNG をそのまま使用"; o.warn = false; o.canRetry = false;
    } else {
      if (!S.server.reachable) throw new Error("サーバーに接続できません(自動切り抜きにはサーバーが必要)");
      const r = await api("/api/cutout", { image: o.origUrl, tol });
      o.dataUrl = r.png; o.img = await loadImg(r.png);
      o.canRetry = r.method === "border_flood";
      o.msg = r.ok ? `自動で切り抜きました(許容 ${r.tol})` : r.note; o.warn = !r.ok;
      if (!r.ok) { o.dataUrl = o.origUrl; o.img = raw; }
    }
  } catch (e) {
    o.msg = String(e.message || e); o.warn = true;
    try { o.img = o.img || await loadImg(o.origUrl); } catch (_) { /* 読めない画像 */ }
  }
}

function spreadDefaults() {
  // まだ一度も配置していない素材を、下の段に等間隔で並べる
  const fresh = S.objects.filter((o) => !o.placed);
  fresh.forEach((o, i) => { o.x = (i + 1) / (fresh.length + 1); o.y = 0.72; o.placed = false; });
}

// ------------------------------------------------------------ 描画
const byDepth = () => [...S.objects].sort((a, b) => a.z - b.z || a.uid - b.uid);
function objSize(o) {
  const k = o.img.naturalWidth / o.img.naturalHeight, L = o.scale * Math.max(S.CW, S.CH);
  return k >= 1 ? [L, L / k] : [L * k, L];
}
function drawObjects(ctx, withUi) {
  for (const o of byDepth()) {
    if (!o.img) continue;
    const [w, h] = objSize(o);
    ctx.save();
    ctx.translate(o.x * S.CW, o.y * S.CH); ctx.rotate((o.rot * Math.PI) / 180);
    ctx.drawImage(o.img, -w / 2, -h / 2, w, h);
    if (withUi && o.uid === S.selected) {
      ctx.strokeStyle = "#c4572b"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.strokeRect(-w / 2, -h / 2, w, h); ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(0, -h / 2); ctx.lineTo(0, -h / 2 - 28); ctx.stroke();
      ctx.fillStyle = "#2b7a5b"; ctx.beginPath(); ctx.arc(0, -h / 2 - 28, 7, 0, 7); ctx.fill();
    }
    ctx.restore();
  }
}
function drawUncertainty(ctx) {
  if (!$("#uncert").checked) return;
  ctx.save(); ctx.lineWidth = 1.5; ctx.setLineDash([3, 5]);
  for (const o of S.objects) {
    const r = o.range; if (!r || !r.x || !r.y) continue;
    ctx.strokeStyle = o.uid === S.selected ? "#c4572b" : "#8a7f68";
    const cx = ((r.x[0] + r.x[1]) / 2) * S.CW, cy = ((r.y[0] + r.y[1]) / 2) * S.CH;
    ctx.beginPath(); ctx.ellipse(cx, cy, Math.max(4, ((r.x[1] - r.x[0]) / 2) * S.CW), Math.max(4, ((r.y[1] - r.y[0]) / 2) * S.CH), 0, 0, 7); ctx.stroke();
  }
  ctx.restore();
}
function render() {
  sc.clearRect(0, 0, S.CW, S.CH);
  drawObjects(sc, true);
  drawUncertainty(sc);
  sketchCv.style.opacity = $("#skop").value;
  syncInspector();
}

// ------------------------------------------------------------ ポインタ操作
function pt(e) {
  const r = sceneCv.getBoundingClientRect();
  return [((e.clientX - r.left) * S.CW) / r.width, ((e.clientY - r.top) * S.CH) / r.height];
}
function toLocal(o, px, py) {
  const dx = px - o.x * S.CW, dy = py - o.y * S.CH, a = (-o.rot * Math.PI) / 180;
  return [dx * Math.cos(a) - dy * Math.sin(a), dx * Math.sin(a) + dy * Math.cos(a)];
}
function hit(px, py) {
  const sel = S.objects.find((o) => o.uid === S.selected);
  if (sel && sel.img) {
    const [lx, ly] = toLocal(sel, px, py), [, h] = objSize(sel);
    if (Math.hypot(lx, ly + h / 2 + 28) < 14) return { o: sel, mode: "rotate" };
  }
  for (const o of byDepth().reverse()) {
    if (!o.img) continue;
    const [lx, ly] = toLocal(o, px, py), [w, h] = objSize(o);
    if (Math.abs(lx) <= w / 2 && Math.abs(ly) <= h / 2) return { o, mode: "move" };
  }
  return null;
}

sceneCv.addEventListener("pointerdown", (e) => {
  sceneCv.setPointerCapture(e.pointerId);
  const [px, py] = pt(e);
  if (S.tool === "pen" || S.tool === "eraser") {
    S.drag = { kind: "draw", last: [px, py] };
    sk.globalCompositeOperation = S.tool === "eraser" ? "destination-out" : "source-over";
    sk.strokeStyle = "#222"; sk.lineWidth = +$("#pensize").value * (S.tool === "eraser" ? 2.5 : 1) * (S.CW / 960);
    sk.beginPath(); sk.moveTo(px, py); sk.lineTo(px + 0.01, py); sk.stroke();
    if (S.tool === "pen") S.sketchInk = true;
    return;
  }
  const h = hit(px, py);
  if (!h) { S.selected = null; renderList(); render(); return; }
  S.selected = h.o.uid; renderList();
  S.drag = { kind: h.mode, o: h.o, off: [px - h.o.x * S.CW, py - h.o.y * S.CH] };
  render();
});
sceneCv.addEventListener("pointermove", (e) => {
  const d = S.drag; if (!d) return;
  const [px, py] = pt(e);
  if (d.kind === "draw") { sk.beginPath(); sk.moveTo(...d.last); sk.lineTo(px, py); sk.stroke(); d.last = [px, py]; return; }
  if (d.kind === "move") { d.o.x = clamp((px - d.off[0]) / S.CW, 0, 1); d.o.y = clamp((py - d.off[1]) / S.CH, 0, 1); d.o.placed = true; d.o.range = null; }
  if (d.kind === "rotate") {
    const ang = (Math.atan2(py - d.o.y * S.CH, px - d.o.x * S.CW) * 180) / Math.PI + 90;
    d.o.rot = Math.round(((ang + 540) % 360) - 180); d.o.range = null;
  }
  render();
});
const endDrag = () => { S.drag = null; };
sceneCv.addEventListener("pointerup", endDrag);
sceneCv.addEventListener("pointercancel", endDrag);
sceneCv.addEventListener("wheel", (e) => {
  const o = S.objects.find((x) => x.uid === S.selected);
  if (!o || S.tool !== "select") return;
  e.preventDefault();
  o.scale = clamp(o.scale * (e.deltaY < 0 ? 1.06 : 1 / 1.06), 0.04, 1); o.range = null; render();
}, { passive: false });
window.addEventListener("keydown", (e) => {
  if ((e.key === "Delete" || e.key === "Backspace") && S.selected && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) removeObject(S.selected);
});
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

// ------------------------------------------------------------ 左パネル
function renderList() {
  const ul = $("#objlist"); ul.innerHTML = "";
  S.objects.forEach((o, i) => {
    const li = document.createElement("li"); li.className = o.uid === S.selected ? "sel" : "";
    li.innerHTML = `<img class="th" alt=""><input type="text" maxlength="20" placeholder="名前(例: 木)"><div class="acts"></div><div class="msg"></div>`;
    li.querySelector("img").src = o.dataUrl;
    const inp = li.querySelector("input"); inp.value = o.name;
    inp.addEventListener("input", () => { o.name = inp.value; });
    const acts = li.querySelector(".acts");
    if (o.canRetry) {
      const b = document.createElement("button"); b.className = "mini"; b.textContent = "許容±"; b.title = "切り抜きの許容値を変えて再実行";
      b.onclick = async () => {
        const v = prompt("背景色との差の許容値(大きいほど広く抜く。目安 16〜80)", String(o.tol ?? 40));
        if (v === null) return; o.tol = +v; o.msg = "再切り抜き中…"; renderList(); await processObject(o, o.tol); renderList(); render();
      };
      acts.appendChild(b);
    }
    const del = document.createElement("button"); del.className = "mini"; del.textContent = "×"; del.onclick = () => removeObject(o.uid);
    acts.appendChild(del);
    const m = li.querySelector(".msg"); m.textContent = `${String.fromCharCode(65 + i)}: ${o.msg || ""}`; if (o.warn) m.classList.add("warn");
    li.querySelector("img").onclick = () => { S.selected = o.uid; renderList(); render(); };
    ul.appendChild(li);
  });
}
function removeObject(uid) {
  S.objects = S.objects.filter((o) => o.uid !== uid);
  if (S.selected === uid) S.selected = null;
  renderList(); render();
}
function note(t) { $("#notes").textContent = t; }
function status(t) { $("#status").textContent = t; }

const SL = ["x", "y", "z", "scale", "rot"];
function syncInspector() {
  const o = S.objects.find((x) => x.uid === S.selected);
  $("#inspector").hidden = !o;
  if (!o) return;
  $("#selname").textContent = o.name || "(名前なし)";
  for (const k of SL) {
    const el = $("#s_" + k); if (document.activeElement !== el) el.value = o[k];
    el.nextElementSibling.textContent = k === "rot" ? Math.round(o[k]) + "°" : (+o[k]).toFixed(2);
  }
}
for (const k of SL) $("#s_" + k).addEventListener("input", (e) => {
  const o = S.objects.find((x) => x.uid === S.selected); if (!o) return;
  o[k] = +e.target.value; o.placed = true; if (k !== "z") o.range = null; render();
});

// ------------------------------------------------------------ 具現化
function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
function animateTo(targets, ms = 700) {
  const from = targets.map((t) => ({ o: t.o, ...Object.fromEntries(SL.map((k) => [k, t.o[k]])) }));
  const t0 = performance.now();
  const tick = (now) => {
    const u = clamp((now - t0) / ms, 0, 1), e = easeOut(u);
    targets.forEach((t, i) => SL.forEach((k) => { t.o[k] = from[i][k] + (t[k] - from[i][k]) * e; }));
    render();
    if (u < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

async function arrange() {
  if (S.busy) return;
  if (!S.objects.length) { note("先に素材を入れてください"); return; }
  const mock = $("#mock").checked;
  S.busy = true; $("#arrange").disabled = true; status("配置を考えています…");
  try {
    const r = await api("/api/arrange", {
      theme: $("#theme").value, mock,
      canvas: { w: S.CW, h: S.CH },
      sketch: S.sketchInk ? sketchCv.toDataURL("image/png") : null,
      objects: S.objects.map((o) => ({ name: o.name, png: o.dataUrl })),
    });
    const targets = r.placements.map((p, i) => ({ o: S.objects[i], x: p.x, y: p.y, z: p.z, scale: p.scale, rot: p.rot }));
    r.placements.forEach((p, i) => { S.objects[i].range = p.range80; S.objects[i].placed = true; });
    animateTo(targets);
    const b = $("#badge"); b.hidden = false; b.textContent = r.source === "clairic" ? "Clairic" : "モック"; b.className = "badge" + (r.source === "mock" ? " mock" : "");
    const lines = [];
    lines.push(r.sketch_used ? `スケッチを使いました(塊 ${r.regions.length} 個)` : "スケッチなし(テーマだけで配置)");
    if (r.sketch_used) r.placements.forEach((p, i) => {
      const nm = S.objects[i].name || String.fromCharCode(65 + i);
      lines.push(` ・${nm} → ` + (p.geom === "sketch" ? `塊${p.shape}` + (p.shape_p != null ? `(確信 ${Math.round(p.shape_p * 100)}%)` : "") : "対応なし(推定の位置)"));
    });
    if (r.neurons != null) lines.push(`消費 ${r.neurons} ニューロン / ${r.latency_s}s`);
    if (r.corrections.length) lines.push("補正:\n" + r.corrections.map((c) => " ・" + c).join("\n"));
    note(lines.join("\n")); status("");
  } catch (e) {
    note("失敗: " + e.message); status("");
  } finally {
    S.busy = false; $("#arrange").disabled = false;
  }
}

// ------------------------------------------------------------ 書き出し
function flatCanvas(withSketch) {
  const c = document.createElement("canvas"); c.width = S.CW; c.height = S.CH;
  const x = c.getContext("2d"); x.fillStyle = "#fff"; x.fillRect(0, 0, S.CW, S.CH);
  if (withSketch) { x.globalAlpha = +$("#skop").value; x.drawImage(sketchCv, 0, 0); x.globalAlpha = 1; }
  drawObjects(x, false);
  return c;
}
function download(name, url) { const a = document.createElement("a"); a.href = url; a.download = name; a.click(); }
$("#exportPng").onclick = () => download("materialized.png", flatCanvas(false).toDataURL("image/png"));
$("#exportJson").onclick = () => {
  const data = {
    canvas: { w: S.CW, h: S.CH }, theme: $("#theme").value,
    objects: byDepth().map((o, order) => ({ name: o.name, x: +o.x.toFixed(4), y: +o.y.toFixed(4), z: +o.z.toFixed(4), scale: +o.scale.toFixed(4), rot: Math.round(o.rot), order })),
  };
  download("layout.json", "data:application/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2)));
};

// ------------------------------------------------------------ 配線
const file = $("#file"), drop = $("#drop");
file.addEventListener("change", () => { addFiles([...file.files]); file.value = ""; });
drop.addEventListener("click", (e) => { if (e.target !== file) file.click(); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); });
drop.addEventListener("drop", (e) => addFiles([...e.dataTransfer.files]));
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => { e.preventDefault(); if (e.target !== drop && !drop.contains(e.target)) addFiles([...e.dataTransfer.files]); });

$("#tools").addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  S.tool = b.dataset.tool;
  for (const x of $("#tools").children) x.classList.toggle("on", x === b);
  sceneCv.style.cursor = S.tool === "select" ? "default" : "crosshair";
});
$("#clearSketch").onclick = () => { sk.globalCompositeOperation = "source-over"; sk.clearRect(0, 0, S.CW, S.CH); S.sketchInk = false; };
$("#aspect").addEventListener("change", (e) => setCanvasSize(...aspectToSize(e.target.value)));
$("#skop").addEventListener("input", render);
$("#uncert").addEventListener("change", render);
$("#arrange").addEventListener("click", arrange);

(async function init() {
  setCanvasSize(960, 720);
  try {
    const r = await fetch("/api/status"); const j = await r.json();
    S.server = { reachable: true, clairic: !!j.clairic };
  } catch (_) { S.server = { reachable: false, clairic: false }; }
  if (!S.server.reachable) status("サーバーに接続できません(python -m server.app で起動)");
  else if (!S.server.clairic) { $("#mock").checked = true; status("Clairic の資格情報が未設定なのでモック配置になります"); }
  window.__S = S; // 動作確認用
})();
