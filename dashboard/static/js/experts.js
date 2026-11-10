// ── Experts: which of Qwen Next's experts each layer routes to, as it happens ─────────
// The flash-local lane's engine counts, per MoE layer, the tokens routed to each expert
// (flash-local-experts/, a vLLM plugin) and writes the matrix twice a second to a file the
// cockpit serves whole at /api/experts. This view polls that file once a second while it is
// open, keeps the previous frame, and draws the difference: a tile per layer, a cell per
// expert in its fixed place, over the model's own stack as the engine's config describes it.
// The stream carries only the file's header (the experts collector): the rail entry and the
// caption read that, and, while no engine writes a counter, the checkpoint's shape, which the
// view draws unlit so the stack is there before the counts come.
const EX = {head: {}, model: '', shape: null, cur: null, prev: null, mode: 'live', timer: null, inflight: false,
            hover: null, layout: null, geom: null, off: null, failed: 0};
const EX_QWEN_NEXT = /flash-next|qwen4|qwen3[-_.]?next/i;

const exVar = name => getComputedStyle(document.body).getPropertyValue(name).trim() || '#fff';
const exQwenNext = () => EX_QWEN_NEXT.test(EX.model) || /qwen4_exp|qwen3_next/.test(EX.head.model_type || '');

function exParse(buf){
  if (buf.byteLength < 48) return null;
  const dv = new DataView(buf);
  if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'QEXP' || dv.getUint32(4, true) !== 1) return null;
  const rows = dv.getUint32(8, true), experts = dv.getUint32(12, true), topk = dv.getUint32(16, true), pid = dv.getUint32(20, true);
  const seq = Number(dv.getBigUint64(24, true)), wall = dv.getFloat64(32, true), metaLen = dv.getUint32(40, true);
  const start = 48 + ((metaLen + 3) & ~3);
  if (buf.byteLength !== start + rows * experts * 4) return null;
  let meta;
  try { meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 48, metaLen))); } catch { return null; }
  return {rows, experts, topk, pid, seq, wall, meta: meta || {}, counts: new Int32Array(buf, start, rows * experts)};
}

// The rows as the engine built them, sorted into the model's stack: the decoder layers by
// their index, then the rows that are not decoder layers (the MTP head's own MoE).
const exKind = t => t === 'full_attention' ? 'QSA' : t === 'linear_attention' ? 'GDN' : t || 'layer';
function exLayout(frame){
  const names = Array.isArray(frame.meta.rows) ? frame.meta.rows : [];
  const types = Array.isArray(frame.meta.layer_types) ? frame.meta.layer_types : [];
  const ple = new Set(Array.isArray(frame.meta.ple_layer_ids) ? frame.meta.ple_layer_ids : []);
  const main = [], extra = [];
  for (let r = 0; r < frame.rows; r++){
    const name = String(names[r] || `row ${r}`);
    const m = /\.layers\.(\d+)\./.exec(name);
    if (m && !/mtp|draft/i.test(name)){
      const idx = +m[1];
      main.push({row: r, idx, label: `${idx}`, kind: exKind(types[idx] || ''), ple: ple.has(idx)});
    } else extra.push({row: r, idx: null, label: 'MTP', kind: /mtp/i.test(name) ? 'MTP head' : name, ple: false});
  }
  main.sort((a, b) => a.idx - b.idx);
  return {main, extra, all: main.concat(extra)};
}
// The same stack before any counter writes, from the checkpoint's config: every decoder layer,
// and the MTP head's own experts where the unit drafts with it (another drafter has none).
function exSkeleton(shape){
  const n = +shape.num_hidden_layers || 0, types = Array.isArray(shape.layer_types) ? shape.layer_types : [];
  const ple = new Set(Array.isArray(shape.ple_layer_ids) ? shape.ple_layer_ids : []);
  const main = [], extra = [];
  for (let idx = 0; idx < n; idx++) main.push({row: idx, idx, label: `${idx}`, kind: exKind(types[idx] || ''), ple: ple.has(idx)});
  const mtp = shape.spec_method === 'mtp' ? Math.max(1, +shape.mtp_num_hidden_layers || 1) : 0;
  for (let k = 0; k < mtp; k++) extra.push({row: n + k, idx: null, label: 'MTP', kind: 'MTP head', ple: false});
  return {main, extra, all: main.concat(extra)};
}

// Per cell, what the drawing and the table read: in 'live', tokens since the previous frame;
// in 'total', tokens since the engine started. Both as Int32Array of rows x experts.
function exValues(){
  const cur = EX.cur; if (!cur) return null;
  if (EX.mode === 'total' || !EX.prev) return {v: cur.counts, dt: null, partial: EX.mode !== 'total'};
  const n = cur.counts.length, v = new Int32Array(n), p = EX.prev.counts;
  for (let i = 0; i < n; i++){ const d = cur.counts[i] - p[i]; v[i] = d > 0 ? d : 0; }
  return {v, dt: Math.max(0.05, cur.wall - EX.prev.wall), partial: false};
}

function exStats(values, layout){
  const cur = EX.cur, E = cur.experts, v = values.v;
  let routes = 0, used = 0, hot = null;
  const shares = [];
  for (const L of layout.main){
    const base = L.row * E; let total = 0;
    for (let e = 0; e < E; e++) total += v[base + e];
    if (!total) continue;
    const fair = total / E, row = [];
    for (let e = 0; e < E; e++){
      const x = v[base + e]; if (x > 0) used++;
      row.push(x);
      const ratio = x / fair;
      if (!hot || ratio > hot.ratio) hot = {layer: L, expert: e, tokens: x, ratio};
    }
    routes += total;
    row.sort((a, b) => b - a);
    let top = 0; const k = Math.max(1, Math.round(E / 10));
    for (let i = 0; i < k; i++) top += row[i];
    shares.push(top / total);
  }
  const layers = layout.main.length || 1;
  const tokens = routes / Math.max(1, cur.topk) / layers;
  return {routes, used, hot, top10: shares.length ? shares.reduce((a, b) => a + b, 0) / shares.length : null,
          tokens, tps: values.dt ? tokens / values.dt : null, cells: layers * E};
}

// ── the drawing ────────────────────────────────────────────────────────────────────────
// A tile per layer, the layer's experts as 16 rows of 32 cells, expert 0 at the top left and
// the last at the bottom right: an expert keeps its place from tile to tile, so one cell lit
// in tile after tile is one expert the whole stack leans on. Eight tiles to a row where the
// panel is wide enough (six rows of eight for 48 layers, the QSA layers in two columns of
// their own), four or two where it is not. The MTP head's own experts are a tile below the
// stack. Without counts (no counter yet) the tiles are drawn unlit, the stack in place.
const EX_TOP = 26, EX_HEAD = 14, EX_GAP = 12, EX_GAPY = 10, EX_BREAK = 24, EX_BOTTOM = 34;

function exGeom(W, E, nMain, nExtra){
  const gw = E >= 32 && E % 32 === 0 ? 32 : E, gh = Math.ceil(E / gw);
  const fits = n => n * gw * 3 + (n - 1) * EX_GAP <= W;
  const cols = fits(8) ? 8 : fits(4) ? 4 : fits(2) ? 2 : 1;
  const cs = Math.max(1, Math.min(6, Math.floor((W - (cols - 1) * EX_GAP) / cols / gw)));
  const tileW = gw * cs, tileH = EX_HEAD + gh * cs, pitchX = tileW + EX_GAP, pitchY = tileH + EX_GAPY;
  const mainRows = Math.ceil(nMain / cols), extraRows = Math.ceil(nExtra / cols);
  const yExtra = EX_TOP + mainRows * pitchY + EX_BREAK;
  const pos = i => i < nMain ? {x: (i % cols) * pitchX, y: EX_TOP + Math.floor(i / cols) * pitchY}
                             : {x: ((i - nMain) % cols) * pitchX, y: yExtra + Math.floor((i - nMain) / cols) * pitchY};
  const H = (nExtra ? yExtra + extraRows * pitchY : EX_TOP + mainRows * pitchY) + EX_BOTTOM;
  const gridW = Math.min(cols, Math.max(nMain, nExtra, 1)) * pitchX - EX_GAP;
  return {W, H, E, gw, gh, cols, cs, tileW, tileH, pitchX, pitchY, nMain, nExtra, yExtra, gridW, pos};
}

function exColor(ratio){
  // one hue, the flash-local lane's: dim under an even share, full at four times it
  if (!(ratio > 0)) return [255, 255, 255, 8];
  const t = Math.max(0, Math.min(1, (Math.log2(ratio) + 2) / 4));
  return [34 + (190 - 34) * t * t, 211 + (250 - 211) * t, 238 + (255 - 238) * t, Math.round(60 + 195 * t)];
}

function exDraw(){
  const canvas = $('ex-heat'); if (!canvas || !canvas.getContext || !EX.layout) return;
  const values = EX.cur ? exValues() : null, layout = EX.layout;
  const m = EX.cur ? EX.cur.meta || {} : EX.shape || {};
  const E = EX.cur ? EX.cur.experts : +m.num_experts || 0, topk = EX.cur ? EX.cur.topk : m.num_experts_per_tok;
  if (!E) return;
  const wrap = $('ex-wrap'), W = Math.max(320, wrap.clientWidth || 800);
  const g = exGeom(W, E, layout.main.length, layout.extra.length), H = g.H;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)){ canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); canvas.style.height = H + 'px'; }
  const c = canvas.getContext('2d'); if (!c) return;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const fg2 = exVar('--fg-2'), fg3 = exVar('--fg-3'), brand = exVar('--brand'), orange = exVar('--orange'), ground = exVar('--ink-0');
  const mono = `500 10px ${exVar('--f-code')}`, ui = `500 11px ${exVar('--f-ui')}`;
  const rowsAll = layout.all.length, used = new Int32Array(rowsAll), cells = g.gw * g.gh;

  // the cells: one image at cell resolution, each layer's experts as gh rows of gw, scaled without smoothing
  let off = null;
  if (values){
    if (!EX.off) EX.off = document.createElement('canvas');
    off = EX.off;
    if (off.width !== g.gw || off.height !== g.gh * rowsAll){ off.width = g.gw; off.height = g.gh * rowsAll; }
    const oc = off.getContext('2d'), img = oc.createImageData(g.gw, g.gh * rowsAll), px = img.data;
    layout.all.forEach((L, i) => {
      const base = L.row * E; let total = 0;
      for (let e = 0; e < E; e++) total += values.v[base + e];
      const fair = total / E;
      for (let e = 0; e < E; e++){
        const x = values.v[base + e]; if (x > 0) used[i]++;
        const col = exColor(total ? x / fair : 0), o = (i * cells + e) * 4;
        px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; px[o + 3] = col[3];
      }
    });
    oc.putImageData(img, 0, 0);
  }
  c.imageSmoothingEnabled = false;
  c.font = mono; c.textBaseline = 'middle';
  layout.all.forEach((L, i) => {
    const p = g.pos(i), cy = p.y + EX_HEAD, ty = p.y + EX_HEAD / 2;
    c.fillStyle = ground; c.fillRect(p.x, cy, g.tileW, g.gh * g.cs);
    if (off) c.drawImage(off, 0, i * g.gh, g.gw, g.gh, p.x, cy, g.tileW, g.gh * g.cs);
    // the tile's head: the layer's kind as a mark and a name, on the right the experts it used
    const plain = L.idx != null && L.kind !== 'QSA' && !L.ple;
    const name = L.idx == null ? L.kind : L.ple ? `${L.idx} PLE` : L.kind === 'QSA' ? `${L.idx} QSA` : `${L.idx}`;
    c.fillStyle = L.idx == null || L.kind === 'QSA' ? brand : L.ple ? orange : fg3;
    if (plain){ c.globalAlpha = .6; c.fillRect(p.x + 1, p.y + 5, 4, 4); c.globalAlpha = 1; }
    else c.fillRect(p.x, p.y + 4, 6, 6);
    c.fillStyle = plain ? fg3 : fg2; c.textAlign = 'left'; c.fillText(name, p.x + 10, ty);
    if (used[i]){ c.fillStyle = fg3; c.textAlign = 'right'; c.fillText(`${used[i]}`, p.x + g.tileW, ty); }
  });

  // around the stack: the embedding and the residual streams above, the heads and the drafter below
  c.font = ui; c.fillStyle = fg2; c.textBaseline = 'alphabetic';
  c.textAlign = 'left'; c.fillText((m.vocab_size ? `embed ${fmtN(m.vocab_size)} x ${m.hidden_size || '?'}` : 'embedding') + (m.hc_count ? `, ${m.hc_count} residual streams` : ''), 0, 12);
  c.textAlign = 'right'; c.fillText(`${E} experts, top ${topk || '?'} per token${m.shared_expert_intermediate_size ? ', one shared' : ''}`, g.gridW, 12);
  c.textAlign = 'left'; c.fillText('norm, lm_head (shared with the drafts)', 0, g.nExtra ? g.yExtra - 9 : H - EX_BOTTOM + 14);
  const spec = m.spec_method ? (m.spec_method === 'mtp' ? `drafts ${m.num_speculative_tokens || '?'} with the MTP head` :
    `drafts ${m.num_speculative_tokens || '?'} with ${m.draft_model || m.spec_method} (no experts of its own)`) : 'no drafts';
  c.fillStyle = fg3; c.textAlign = 'left'; c.fillText(spec, 0, H - 8);
  c.textAlign = 'right'; c.fillText(!values ? 'waiting for the counter' : values.dt ? `last ${values.dt.toFixed(1)} s` : EX.mode === 'total' ? 'since the engine started' : 'first frame', g.gridW, H - 8);
  EX.geom = g;
  if (EX.hover && values) exHighlight(c);
}

function exHighlight(c){
  const h = EX.hover, g = EX.geom; if (!h || !g) return;
  const p = g.pos(h.i), x = p.x + (h.e % g.gw) * g.cs, y = p.y + EX_HEAD + Math.floor(h.e / g.gw) * g.cs;
  c.strokeStyle = exVar('--fg-1'); c.lineWidth = 1; c.strokeRect(x - .5, y - .5, g.cs + 1, g.cs + 1);
}

// the tile and the cell under a point of the canvas, or null
function exCellAt(x, y){
  const g = EX.geom; if (!g || !EX.layout) return null;
  for (let i = 0; i < EX.layout.all.length; i++){
    const p = g.pos(i), cy = p.y + EX_HEAD;
    if (x < p.x || x >= p.x + g.tileW || y < cy || y >= cy + g.gh * g.cs) continue;
    const e = Math.floor((y - cy) / g.cs) * g.gw + Math.floor((x - p.x) / g.cs);
    return e < g.E ? {i, e} : null;
  }
  return null;
}

function exHover(ev){
  const g = EX.geom, canvas = $('ex-heat'); if (!g || !EX.cur) return;
  const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const at = exCellAt(x, y), tip = $('ex-tip');
  if (!at){ if (EX.hover){ EX.hover = null; tip.hidden = true; exDraw(); } return; }
  if (EX.hover && EX.hover.i === at.i && EX.hover.e === at.e) return;
  EX.hover = at;
  const E = g.E, L = EX.layout.all[at.i], values = exValues(), base = L.row * E;
  let total = 0; for (let k = 0; k < E; k++) total += values.v[base + k];
  const v = values.v[base + at.e], ratio = total ? v / (total / E) : 0;
  tip.textContent = `${L.idx == null ? L.kind : 'layer ' + L.idx + ' ' + L.kind}, expert ${at.e}: ${fmtN(v)} ${values.dt ? 'in ' + values.dt.toFixed(1) + ' s' : 'tokens'}` + (total ? `, ${ratio.toFixed(2)}x the even share` : '');
  tip.hidden = false;
  tip.style.left = Math.min(x + 14, g.W - tip.offsetWidth - 8) + 'px'; tip.style.top = Math.max(0, y - 34) + 'px';
  exDraw();
}

function exTable(values, layout){
  const tb = $('ex-table').querySelector('tbody'); clear(tb);
  if (!values) return;
  const E = EX.cur.experts, top = [];
  for (const L of layout.all){
    const base = L.row * E; let total = 0;
    for (let e = 0; e < E; e++) total += values.v[base + e];
    if (!total) continue;
    for (let e = 0; e < E; e++){ const v = values.v[base + e]; if (v > 0) top.push({L, e, v, ratio: v / (total / E)}); }
  }
  top.sort((a, b) => b.ratio - a.ratio);
  for (const t of top.slice(0, 8)){
    const tr = el('tr');
    tr.append(el('td', null, t.L.idx == null ? t.L.kind : `layer ${t.L.idx}`), el('td', null, t.L.idx == null ? '' : t.L.kind),
              el('td', 'r', `${t.e}`), el('td', 'r', fmtN(t.v)), el('td', 'r', `${t.ratio.toFixed(1)}x`));
    tb.append(tr);
  }
}

function exRender(){
  if (!EX.layout) return;
  const values = EX.cur ? exValues() : null;
  if (!values){ ['ex-tps', 'ex-active', 'ex-top10', 'ex-hot'].forEach(id => setText(id, '…')); exDraw(); exTable(null, EX.layout); return; }
  const s = exStats(values, EX.layout);
  setText('ex-tps', values.partial ? '…' : s.tps != null ? fmtN(Math.round(s.tps)) : fmtN(Math.round(s.tokens)));
  setText('ex-active', values.partial ? '…' : `${Math.round(100 * s.used / s.cells)} %`);
  setText('ex-top10', values.partial || s.top10 == null ? '…' : `${Math.round(100 * s.top10)} %`);
  setText('ex-hot', values.partial || !s.hot ? '…' : `${s.hot.ratio.toFixed(1)}x`);
  const k = $('ex-tps').nextElementSibling; if (k) setText(k, EX.mode === 'total' ? 'Tokens through the experts since the start' : 'Tokens through the experts per second');
  const k2 = $('ex-active').nextElementSibling; if (k2) setText(k2, EX.mode === 'total' ? 'Experts used since the start' : 'Experts used in the last second');
  exDraw(); exTable(values, EX.layout);
}

// ── the poll, while the view is open ───────────────────────────────────────────────────
async function exPoll(){
  if (EX.inflight) return; EX.inflight = true;
  try{
    const r = await fetch('/api/experts');
    if (r.status === 401) return;
    if (!r.ok){ EX.failed++; exCaption(); return; }
    const frame = exParse(await r.arrayBuffer());
    if (!frame){ EX.failed++; exCaption(); return; }
    EX.failed = 0;
    if (EX.cur && EX.cur.seq === frame.seq && EX.cur.pid === frame.pid) return;   // the writer has not written since
    // a new writer (the engine restarted) starts the difference afresh
    EX.prev = EX.cur && EX.cur.pid === frame.pid && EX.cur.rows === frame.rows ? EX.cur : null;
    EX.cur = frame; EX.layout = exLayout(frame);
    show('ex-panel', true); exRender(); exCaption();
  }catch(e){ EX.failed++; warnOnce('experts', e.message); exCaption(); }
  finally{ EX.inflight = false; }
}
function exWatch(){
  const want = !document.hidden && activeView === 'experts';
  if (want && !EX.timer){ exPoll(); EX.timer = setInterval(() => { if (document.hidden || activeView !== 'experts'){ clearInterval(EX.timer); EX.timer = null; return; } exPoll(); }, 1000); }
  if (!want && EX.timer){ clearInterval(EX.timer); EX.timer = null; }
}

function exCaption(){
  const d = EX.head, qwen = exQwenNext();
  const btn = $('nav-experts'); if (btn) btn.hidden = !(d.present || qwen);
  let text, kind, note;
  if (d.live){ text = 'live'; kind = 'ok'; note = `${d.model || 'the engine'}: ${d.rows} layers x ${d.experts} experts, top ${d.top_k}, written ${d.age < 1.5 ? 'just now' : fmtDur(d.age) + ' ago'}.`; }
  else if (d.present){ text = `stale, ${fmtDur(d.age || 0)}`; kind = 'warn'; note = `The counter's last write was ${fmtDur(d.age || 0)} ago: the engine that wrote it is gone, or still loading. The picture is its last.`; }
  else if (qwen){ text = 'no counter'; kind = 'warn'; note = 'This engine writes no routing counter: the tiles are the checkpoint\'s layers, unlit until an engine with the counter serves. The flash-local lane\'s unit mounts one since this version: a Load from this checkout, then a restart of the lane, bring it.'; }
  else { text = 'not this model'; kind = 'idle'; note = 'The lane that serves is not a Qwen Next model: there are no experts to count.'; }
  if (EX.failed > 2 && d.present){ text = 'unreadable'; kind = 'err'; note = 'The counter file is there but /api/experts could not read it whole: the cockpit\'s journal says why.'; }
  cap('ex-cap', text, kind); setText('ex-empty', note);
  // the panel: the counts once a counter writes; until then the checkpoint's stack, unlit
  if (EX.cur || d.present) return;
  if (qwen && EX.shape && +EX.shape.num_experts){ EX.layout = exSkeleton(EX.shape); show('ex-panel', true); exRender(); }
  else show('ex-panel', false);
}

on('experts', d => { EX.head = d || {}; EX.shape = EX.head.shape || null; exCaption(); exWatch(); });
on('engine_info', d => { EX.model = ((d || {}).info || {}).served_model_name || ''; exCaption(); });
onShow('experts', () => { exWatch(); if (EX.layout) exDraw(); });   // drawn while hidden, the panel had no width
document.addEventListener('visibilitychange', exWatch);
window.addEventListener('resize', () => { if (activeView === 'experts' && EX.layout) exDraw(); });
document.addEventListener('DOMContentLoaded', () => {
  const canvas = $('ex-heat');
  if (canvas){
    canvas.addEventListener('pointermove', exHover);
    canvas.addEventListener('pointerleave', () => { if (EX.hover){ EX.hover = null; $('ex-tip').hidden = true; exDraw(); } });
  }
  const seg = $('ex-mode');
  if (seg) seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    EX.mode = b.dataset.mode; seg.querySelectorAll('button').forEach(x => x.setAttribute('aria-checked', String(x === b)));
    exRender();
  }));
});
