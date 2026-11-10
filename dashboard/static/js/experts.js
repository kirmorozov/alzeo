// ── Experts: which of Qwen Next's experts each layer routes to, as it happens ─────────
// The flash-local lane's engine counts, per MoE layer, the tokens routed to each expert
// (flash-local-experts/, a vLLM plugin) and writes the matrix twice a second to a file the
// cockpit serves whole at /api/experts. This view polls that file once a second while it is
// open, keeps the previous frame, and draws the difference: a row per layer, a cell per
// expert, over the model's own stack as the engine's config describes it. The stream carries
// only the file's header (the experts collector): the rail entry and the caption read that.
const EX = {head: {}, model: '', cur: null, prev: null, mode: 'live', timer: null, inflight: false,
            hover: null, layout: null, off: null, failed: 0};
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
function exLayout(frame){
  const names = Array.isArray(frame.meta.rows) ? frame.meta.rows : [];
  const types = Array.isArray(frame.meta.layer_types) ? frame.meta.layer_types : [];
  const ple = new Set(Array.isArray(frame.meta.ple_layer_ids) ? frame.meta.ple_layer_ids : []);
  const main = [], extra = [];
  for (let r = 0; r < frame.rows; r++){
    const name = String(names[r] || `row ${r}`);
    const m = /\.layers\.(\d+)\./.exec(name);
    if (m && !/mtp|draft/i.test(name)){
      const idx = +m[1], t = types[idx] || '';
      main.push({row: r, idx, label: `${idx}`, kind: t === 'full_attention' ? 'QSA' : t === 'linear_attention' ? 'GDN' : t || 'layer', ple: ple.has(idx)});
    } else extra.push({row: r, idx: null, label: 'MTP', kind: /mtp/i.test(name) ? 'MTP head' : name, ple: false});
  }
  main.sort((a, b) => a.idx - b.idx);
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
const EX_L = 118, EX_R = 96, EX_TOP = 30, EX_ROW = 8, EX_GAP = 16, EX_BOTTOM = 46;

function exColor(ratio){
  // one hue, the flash-local lane's: dim under an even share, full at four times it
  if (!(ratio > 0)) return [255, 255, 255, 8];
  const t = Math.max(0, Math.min(1, (Math.log2(ratio) + 2) / 4));
  return [34 + (190 - 34) * t * t, 211 + (250 - 211) * t, 238 + (255 - 238) * t, Math.round(60 + 195 * t)];
}

function exDraw(){
  const canvas = $('ex-heat'); if (!canvas || !canvas.getContext || !EX.cur) return;
  const values = exValues(); if (!values) return;
  const layout = EX.layout, cur = EX.cur, E = cur.experts;
  const nMain = layout.main.length, nExtra = layout.extra.length;
  const wrap = $('ex-wrap'), W = Math.max(320, wrap.clientWidth || 800);
  const H = EX_TOP + nMain * EX_ROW + (nExtra ? EX_GAP + nExtra * EX_ROW : 0) + EX_BOTTOM;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)){ canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); canvas.style.height = H + 'px'; }
  const c = canvas.getContext('2d'); if (!c) return;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const heatW = W - EX_L - EX_R, fg2 = exVar('--fg-2'), fg3 = exVar('--fg-3'), brand = exVar('--brand'), orange = exVar('--orange'), rule = exVar('--rule-2');
  const mono = `500 10px ${exVar('--f-code')}`, ui = `500 11px ${exVar('--f-ui')}`;
  const rowY = i => EX_TOP + i * EX_ROW;
  const extraY = i => EX_TOP + nMain * EX_ROW + EX_GAP + i * EX_ROW;

  // the cells: one image at cell resolution, scaled without smoothing
  if (!EX.off) EX.off = document.createElement('canvas');
  const off = EX.off, rowsAll = layout.all.length;
  if (off.width !== E || off.height !== rowsAll){ off.width = E; off.height = rowsAll; }
  const oc = off.getContext('2d'), img = oc.createImageData(E, rowsAll), px = img.data;
  const used = new Int32Array(rowsAll);
  layout.all.forEach((L, i) => {
    const base = L.row * E; let total = 0;
    for (let e = 0; e < E; e++) total += values.v[base + e];
    const fair = total / E;
    for (let e = 0; e < E; e++){
      const x = values.v[base + e]; if (x > 0) used[i]++;
      const col = exColor(total ? x / fair : 0), o = (i * E + e) * 4;
      px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; px[o + 3] = col[3];
    }
  });
  oc.putImageData(img, 0, 0);
  c.imageSmoothingEnabled = false;
  c.drawImage(off, 0, 0, E, nMain, EX_L, EX_TOP, heatW, nMain * EX_ROW);
  if (nExtra) c.drawImage(off, 0, nMain, E, nExtra, EX_L, extraY(0), heatW, nExtra * EX_ROW);
  // a hair between rows, in the surface, so a row reads as a layer
  c.strokeStyle = exVar('--ink-0'); c.lineWidth = 1;
  for (let i = 1; i < nMain; i++){ c.beginPath(); c.moveTo(EX_L, rowY(i) + .5); c.lineTo(EX_L + heatW, rowY(i) + .5); c.stroke(); }

  // the stack beside it: the four residual streams, each layer's kind, the heads
  const m = cur.meta || {};
  c.font = ui; c.fillStyle = fg2; c.textBaseline = 'alphabetic';
  c.textAlign = 'left'; c.fillText(m.vocab_size ? `embed ${fmtN(m.vocab_size)} x ${m.hidden_size || '?'}` : 'embedding', EX_L, 12);
  c.textAlign = 'right'; c.fillText(`${E} experts, top ${cur.topk} per token${m.shared_expert_intermediate_size ? ', one shared' : ''}`, EX_L + heatW, 12);
  const hc = +m.hc_count || 0;
  if (hc){
    c.strokeStyle = brand; c.globalAlpha = .55; c.lineWidth = 1;
    for (let k = 0; k < hc; k++){ const x = 10 + k * 4 + .5; c.beginPath(); c.moveTo(x, EX_TOP - 6); c.lineTo(x, rowY(nMain) + 4); c.stroke(); }
    c.globalAlpha = 1; c.font = mono; c.fillStyle = fg3; c.textAlign = 'left'; c.fillText(`${hc} streams`, 8, 12);
  }
  c.font = mono; c.textBaseline = 'middle';
  layout.main.forEach((L, i) => {
    const y = rowY(i) + EX_ROW / 2;
    if (L.kind === 'QSA'){
      c.fillStyle = brand; c.fillRect(EX_L - 10, rowY(i) + 1, 6, EX_ROW - 2);
      c.textAlign = 'right'; c.fillText(`${L.idx} QSA`, EX_L - 16, y);
    } else {
      c.fillStyle = fg3; c.globalAlpha = .6; c.fillRect(EX_L - 8, rowY(i) + 3, 4, EX_ROW - 6); c.globalAlpha = 1;
      if (L.idx % 4 === 0 && !L.ple){ c.fillStyle = fg3; c.textAlign = 'right'; c.fillText(`${L.idx}`, EX_L - 16, y); }
    }
    if (L.ple){ c.fillStyle = orange; c.textAlign = 'right'; c.fillText(`${L.idx} PLE`, EX_L - 16, y); }
  });
  // the right gutter: experts used in the window, a thin bar per layer
  const barW = EX_R - 44;
  layout.all.forEach((L, i) => {
    const y = i < nMain ? rowY(i) : extraY(i - nMain);
    c.fillStyle = fg3; c.globalAlpha = .18; c.fillRect(EX_L + heatW + 8, y + 2, barW, EX_ROW - 4);
    c.globalAlpha = .7; c.fillRect(EX_L + heatW + 8, y + 2, barW * used[i] / E, EX_ROW - 4); c.globalAlpha = 1;
    if ((L.kind === 'QSA' || i >= nMain) && used[i]){ c.fillStyle = fg2; c.textAlign = 'left'; c.fillText(`${used[i]}`, EX_L + heatW + 12 + barW, y + EX_ROW / 2); }
  });
  c.fillStyle = fg3; c.textAlign = 'left'; c.textBaseline = 'alphabetic'; c.fillText('used', EX_L + heatW + 8, 12);
  // under the stack: the heads
  c.font = ui; c.fillStyle = fg2; c.textBaseline = 'alphabetic';
  const yh = rowY(nMain) + (nExtra ? 11 : 16);
  c.textAlign = 'left'; c.fillText('norm, lm_head (shared with the drafts)', EX_L, yh);
  if (nExtra){
    c.textAlign = 'right'; c.fillStyle = brand; c.font = mono; c.textBaseline = 'middle';
    layout.extra.forEach((L, i) => c.fillText(L.kind, EX_L - 16, extraY(i) + EX_ROW / 2));
    c.textBaseline = 'alphabetic';
  }
  const spec = m.spec_method ? (m.spec_method === 'mtp' ? `drafts ${m.num_speculative_tokens || '?'} with the MTP head` :
    `drafts ${m.num_speculative_tokens || '?'} with ${m.draft_model || m.spec_method} (no experts of its own)`) : 'no drafts';
  c.font = ui; c.fillStyle = fg3; c.textAlign = 'left'; c.fillText(spec, EX_L, H - 10);
  c.textAlign = 'right'; c.fillText(values.dt ? `last ${values.dt.toFixed(1)} s` : EX.mode === 'total' ? 'since the engine started' : 'first frame', EX_L + heatW, H - 10);
  if (EX.hover) exHighlight(c, layout, heatW, E);
  EX.geom = {W, H, heatW, E, nMain, nExtra, rowY, extraY};
}

function exHighlight(c, layout, heatW, E){
  const h = EX.hover, g = EX.geom; if (!h || !g) return;
  const cw = heatW / E, y = h.i < g.nMain ? g.rowY(h.i) : g.extraY(h.i - g.nMain);
  c.strokeStyle = exVar('--fg-1'); c.lineWidth = 1; c.strokeRect(EX_L + h.e * cw - .5, y - .5, Math.max(2, cw) + 1, EX_ROW + 1);
}

function exHover(ev){
  const g = EX.geom, canvas = $('ex-heat'); if (!g || !EX.cur) return;
  const r = canvas.getBoundingClientRect(), x = ev.clientX - r.left, y = ev.clientY - r.top;
  const E = g.E, cw = g.heatW / E;
  let i = -1;
  if (x >= EX_L && x < EX_L + g.heatW){
    if (y >= EX_TOP && y < g.rowY(g.nMain)) i = Math.floor((y - EX_TOP) / EX_ROW);
    else if (g.nExtra && y >= g.extraY(0) && y < g.extraY(g.nExtra)) i = g.nMain + Math.floor((y - g.extraY(0)) / EX_ROW);
  }
  const tip = $('ex-tip');
  if (i < 0){ if (EX.hover){ EX.hover = null; tip.hidden = true; exDraw(); } return; }
  const e = Math.min(E - 1, Math.floor((x - EX_L) / cw));
  if (EX.hover && EX.hover.i === i && EX.hover.e === e) return;
  EX.hover = {i, e};
  const L = EX.layout.all[i], values = exValues(), base = L.row * E;
  let total = 0; for (let k = 0; k < E; k++) total += values.v[base + k];
  const v = values.v[base + e], ratio = total ? v / (total / E) : 0;
  tip.textContent = `${L.idx == null ? L.kind : 'layer ' + L.idx + ' ' + L.kind}, expert ${e}: ${fmtN(v)} ${values.dt ? 'in ' + values.dt.toFixed(1) + ' s' : 'tokens'}` + (total ? `, ${ratio.toFixed(2)}x the even share` : '');
  tip.hidden = false;
  tip.style.left = Math.min(x + 14, g.W - tip.offsetWidth - 8) + 'px'; tip.style.top = Math.max(0, y - 34) + 'px';
  exDraw();
}

function exTable(values, layout){
  const tb = $('ex-table').querySelector('tbody'); clear(tb);
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
  const values = exValues(); if (!values || !EX.layout) return;
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
  else if (qwen){ text = 'no counter'; kind = 'warn'; note = 'This engine writes no routing counter. The flash-local lane\'s unit mounts one since this version: a Load from this checkout, then a restart of the lane, bring it.'; }
  else { text = 'not this model'; kind = 'idle'; note = 'The lane that serves is not a Qwen Next model: there are no experts to count.'; }
  if (EX.failed > 2 && d.present){ text = 'unreadable'; kind = 'err'; note = 'The counter file is there but /api/experts could not read it whole: the cockpit\'s journal says why.'; }
  cap('ex-cap', text, kind); setText('ex-empty', note);
  if (!d.present && !EX.cur) show('ex-panel', false);
}

on('experts', d => { EX.head = d || {}; exCaption(); exWatch(); });
on('engine_info', d => { EX.model = ((d || {}).info || {}).served_model_name || ''; exCaption(); });
onShow('experts', exWatch);
document.addEventListener('visibilitychange', exWatch);
window.addEventListener('resize', () => { if (activeView === 'experts' && EX.cur) exDraw(); });
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
