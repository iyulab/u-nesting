import { presets2d, presets3d } from './presets.js';
import { render2d, colorFor, boundaryExtent } from './render2d.js';
import { renderCutting } from './render_cutting.js';
import { render3d } from './render3d.js';
import { Viewport2D } from './viewport2d.js';
import { buildControls } from './controls.js';
import { buildPartsEditor } from './partsEditor.js';
import { downloadJson, downloadCanvasPng } from './exporter.js';

// --- Worker plumbing ---
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let reqId = 0;
worker.onmessage = (e) => {
  const resolve = pending.get(e.data.id);
  if (resolve) { pending.delete(e.data.id); resolve(e.data); }
};
function call(mode, request) {
  return new Promise((resolve) => { const id = ++reqId; pending.set(id, resolve); worker.postMessage({ id, mode, request }); });
}

// Strategies verified to work in the current WASM build. The 0.3.4 release fixed the
// runtime panics (sa/gdrr/alns 2D, sa 3D) and 3D orientation/out-of-bounds reporting
// (ep/ga/brkga), so all strategies are now enabled. The wasm runtime smoke test
// (test/wasm_runtime_smoke.mjs) guards every entry here.
const VALID = {
  '2d': ['blf', 'nfp', 'ga', 'brkga', 'sa', 'gdrr', 'alns'],
  '3d': ['blf', 'ep', 'ga', 'brkga', 'sa'],
};

// --- DOM refs ---
const $ = (id) => document.getElementById(id);
const canvas2d = $('canvas2d');
const canvas3d = $('canvas3d');
const ctx = canvas2d.getContext('2d');
const tooltip = $('tooltip');

// --- State ---
let mode = '2d';
let geometries = [];
let boundary = {};
let strategiesByDim = { '2d': [], '3d': [] };
const viewport = new Viewport2D();
let lastRender = { hitItems: [], issues: [], sheets: [0] };
let activeSheet = 0;
let lastPayload = null; // { mode, request, response, cutting }

// --- Mode config: controls spec + preset source ---
const STRATEGY_DIM = { '2d': '2d', '3d': '3d', cutting: '2d' };
function presetSource() { return mode === '3d' ? presets3d : presets2d; }

function controlSpec() {
  if (mode === '3d') {
    return [
      { key: 'time_limit_ms', label: 'Time limit (ms)', type: 'range', min: 100, max: 8000, step: 100, value: 2000 },
      { key: 'gravity', label: 'Gravity', type: 'toggle', value: false },
      { key: 'stability', label: 'Stability', type: 'toggle', value: false },
      { key: 'max_mass', label: 'Max mass (0 = none)', type: 'number', min: 0, value: 0 },
    ];
  }
  const base = [
    { key: 'spacing', label: 'Spacing', type: 'range', min: 0, max: 10, step: 0.5, value: 2 },
    { key: 'margin', label: 'Margin', type: 'range', min: 0, max: 20, step: 1, value: 0 },
    { key: 'population_size', label: 'GA population', type: 'range', min: 10, max: 120, step: 10, value: 50 },
    { key: 'max_generations', label: 'GA generations', type: 'range', min: 10, max: 300, step: 10, value: 100 },
    { key: 'target_utilization', label: 'Target util (0 = off)', type: 'range', min: 0, max: 1, step: 0.05, value: 0 },
    { key: 'time_limit_ms', label: 'Time limit (ms)', type: 'range', min: 100, max: 8000, step: 100, value: 2000 },
    { key: 'rotations', label: 'Rotations', type: 'checks', options: [0, 90, 180, 270], value: [0, 90] },
    { key: 'allow_flip', label: 'Allow flip', type: 'toggle', value: false },
  ];
  if (mode === 'cutting') {
    base.push(
      { key: 'kerf_width', label: 'Kerf width', type: 'range', min: 0, max: 3, step: 0.1, value: 0.5 },
      { key: 'cut_speed', label: 'Cut speed', type: 'range', min: 10, max: 300, step: 10, value: 100 },
      { key: 'rapid_speed', label: 'Rapid speed', type: 'range', min: 50, max: 1000, step: 50, value: 500 },
    );
  }
  return base;
}

let controlState = {};

// --- Strategy dropdown ---
function fillStrategies() {
  const dim = STRATEGY_DIM[mode];
  const all = strategiesByDim[dim] || [];
  const valid = VALID[dim];
  const sel = $('strategy');
  sel.replaceChildren();
  for (const s of all) {
    const o = document.createElement('option');
    o.value = s;
    o.textContent = s.toUpperCase() + (valid.includes(s) ? '' : ' — unavailable');
    if (!valid.includes(s)) o.disabled = true;
    sel.appendChild(o);
  }
  sel.value = valid[0];
}

// --- Preset dropdown ---
function fillPresets() {
  const src = presetSource();
  const sel = $('preset');
  sel.replaceChildren();
  for (const [key, val] of Object.entries(src)) {
    const o = document.createElement('option');
    o.value = key; o.textContent = val.label;
    sel.appendChild(o);
  }
  applyPreset(sel.value);
}
function applyPreset(key) {
  const p = presetSource()[key];
  geometries = JSON.parse(JSON.stringify(p.geometries)); // deep copy (editable)
  boundary = JSON.parse(JSON.stringify(p.boundary));
  partsEditor.refresh();
  fitView();
  draw();
}

// --- Parts editor ---
const partsEditor = buildPartsEditor($('parts'), () => geometries, (g) => { geometries = g; });

// --- Viewport fit ---
function fitView() {
  if (mode === '3d') return;
  const ext = boundaryExtent(boundary);
  const pad = 36;
  const sc = Math.min((canvas2d.width - 2 * pad) / ext.width, (canvas2d.height - 2 * pad) / ext.height);
  viewport.setBase(sc, (canvas2d.width - ext.width * sc) / 2, (canvas2d.height - ext.height * sc) / 2);
}

// --- Drawing ---
function draw() {
  if (mode === '3d' || !lastPayload) return;
  if (mode === 'cutting' && lastPayload.cutting) {
    renderCutting(ctx, geometries, lastPayload.response, lastPayload.cutting, viewport, boundary, activeSheet);
  } else if (lastPayload.response) {
    lastRender = render2d(ctx, geometries, lastPayload.response, viewport, boundary, activeSheet);
  }
}

function drawEmpty() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, canvas2d.width, canvas2d.height);
}

// --- Run ---
async function run() {
  const btn = $('run');
  btn.disabled = true;
  const metrics = $('metrics');
  metrics.textContent = 'Computing…';
  try {
    if (mode === '3d') { await run3d(metrics); return; }
    if (mode === 'cutting') { await runCutting(metrics); return; }
    await run2d(metrics);
  } finally { btn.disabled = false; }
}

function build2dRequest() {
  return {
    geometries: geometries.map((g) => ({
      ...g,
      rotations: controlState.rotations,
      allow_flip: controlState.allow_flip,
    })),
    boundary,
    config: {
      strategy: $('strategy').value,
      spacing: controlState.spacing,
      margin: controlState.margin,
      population_size: controlState.population_size,
      max_generations: controlState.max_generations,
      target_utilization: controlState.target_utilization > 0 ? controlState.target_utilization : undefined,
      time_limit_ms: controlState.time_limit_ms,
    },
  };
}

async function run2d(metrics) {
  const request = build2dRequest();
  const { ok, response, error } = await call('2d', request);
  if (!ok || !response.success) return showError(metrics, error || response.error);
  lastPayload = { mode: '2d', request, response };
  activeSheet = response.placements.length ? response.placements[0].sheet_index : 0;
  fitView(); draw();
  showMetrics2d(metrics, response, lastRender.issues);
  renderSheetTabs();
  renderLegend();
}

async function runCutting(metrics) {
  const request = build2dRequest();
  const solveRes = await call('2d', request);
  if (!solveRes.ok || !solveRes.response.success) return showError(metrics, solveRes.error || solveRes.response.error);
  const cutReq = {
    geometries: request.geometries,
    solve_result: solveRes.response,
    cutting_config: {
      kerf_width: controlState.kerf_width,
      cut_speed: controlState.cut_speed,
      rapid_speed: controlState.rapid_speed,
    },
  };
  const cutRes = await call('cutting', cutReq);
  if (!cutRes.ok || !cutRes.response.success) return showError(metrics, cutRes.error || cutRes.response.error);
  lastPayload = { mode: 'cutting', request, response: solveRes.response, cutting: cutRes.response };
  activeSheet = solveRes.response.placements.length ? solveRes.response.placements[0].sheet_index : 0;
  fitView(); draw();
  showMetricsCutting(metrics, solveRes.response, cutRes.response);
  renderSheetTabs();
  renderLegend();
}

async function run3d(metrics) {
  const request = {
    geometries,
    boundary: {
      ...boundary,
      gravity: controlState.gravity,
      stability: controlState.stability,
      max_mass: controlState.max_mass > 0 ? controlState.max_mass : undefined,
    },
    config: { strategy: $('strategy').value, time_limit_ms: controlState.time_limit_ms },
  };
  const { ok, response, error } = await call('3d', request);
  if (!ok || !response.success) return showError(metrics, error || response.error);
  lastPayload = { mode: '3d', request, response };
  const issues = render3d(geometries, response, canvas3d, boundary);
  showMetrics3d(metrics, response, issues);
  renderLegend();
}

// --- Metrics rendering ---
function row(k, v) { return `<div class="row"><span class="k">${k}</span><span>${v}</span></div>`; }
function showMetrics2d(el, r, issues) {
  el.innerHTML = row('Utilization', `${(r.utilization * 100).toFixed(1)}%`) +
    row('Sheets', r.sheets_used) + row('Placed', r.placements.length) +
    row('Unplaced', r.unplaced.length) + row('Time', `${r.elapsed_ms} ms`) +
    (issues.length ? `<div class="warn">⚠ self-check: ${issues.length} (AABB, suspected)</div>` : '');
  showUtil(r.utilization);
}
function showMetrics3d(el, r, issues) {
  el.innerHTML = row('Utilization', `${(r.utilization * 100).toFixed(1)}%`) +
    row('Bins', r.bins_used) + row('Placed', r.placements.length) +
    row('Unplaced', r.unplaced.length) + row('Time', `${r.elapsed_ms} ms`) +
    (issues.length ? `<div class="warn">⚠ self-check: ${issues.length}</div>` : '');
  showUtil(r.utilization);
}
function showMetricsCutting(el, s, c) {
  el.innerHTML = row('Utilization', `${(s.utilization * 100).toFixed(1)}%`) +
    row('Cut steps', c.sequence.length) + row('Pierces', c.total_pierces) +
    row('Cut dist', c.total_cut_distance.toFixed(1)) + row('Rapid dist', c.total_rapid_distance.toFixed(1)) +
    row('Efficiency', `${(c.efficiency * 100).toFixed(1)}%`) +
    (c.estimated_time_seconds != null ? row('Est. time', `${c.estimated_time_seconds.toFixed(1)} s`) : '');
  showUtil(s.utilization);
}
function showUtil(u) {
  const bar = $('utilbar');
  bar.hidden = false;
  bar.firstElementChild.style.width = `${Math.min(100, u * 100)}%`;
}
function showError(el, message) {
  el.replaceChildren();
  const span = document.createElement('span');
  span.className = 'err';
  span.textContent = 'Error: ' + message;
  el.appendChild(span);
}

// --- Legend ---
function renderLegend() {
  const legend = $('legend');
  legend.replaceChildren();
  const ids = [...new Set(geometries.map((g) => g.id))];
  for (const id of ids) {
    const item = document.createElement('div'); item.className = 'legend-item';
    const sw = document.createElement('span'); sw.className = 'legend-swatch'; sw.style.background = colorFor(id);
    const label = document.createElement('span'); label.textContent = id;
    item.append(sw, label); legend.appendChild(item);
  }
}

// --- Sheet tabs (2D / cutting, multi-sheet) ---
function renderSheetTabs() {
  const tabs = $('sheetTabs');
  tabs.replaceChildren();
  const sheets = lastRender.sheets || [0];
  if (sheets.length <= 1) return;
  for (const s of sheets) {
    const b = document.createElement('button');
    b.className = 'sheet-tab' + (s === activeSheet ? ' active' : '');
    b.textContent = `Sheet ${s + 1}`;
    b.onclick = () => { activeSheet = s; draw(); renderSheetTabs(); };
    tabs.appendChild(b);
  }
}

// --- Canvas interaction (2D / cutting) ---
canvas2d.addEventListener('wheel', (e) => {
  if (mode === '3d' || !lastPayload) return;
  e.preventDefault();
  const r = canvas2d.getBoundingClientRect();
  const sx = (e.clientX - r.left) * (canvas2d.width / r.width);
  const sy = (e.clientY - r.top) * (canvas2d.height / r.height);
  viewport.zoomAt(sx, sy, e.deltaY < 0 ? 1.1 : 1 / 1.1);
  draw();
}, { passive: false });

let dragging = false, lastX = 0, lastY = 0;
canvas2d.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => { dragging = false; });
canvas2d.addEventListener('mousemove', (e) => {
  if (mode === '3d' || !lastPayload) return;
  const r = canvas2d.getBoundingClientRect();
  const scaleX = canvas2d.width / r.width, scaleY = canvas2d.height / r.height;
  if (dragging) {
    viewport.panBy((e.clientX - lastX) * scaleX, (e.clientY - lastY) * scaleY);
    lastX = e.clientX; lastY = e.clientY;
    draw(); tooltip.hidden = true; return;
  }
  // hover hit-test
  const sx = (e.clientX - r.left) * scaleX, sy = (e.clientY - r.top) * scaleY;
  const [wx, wy] = viewport.toWorld(sx, sy);
  const hit = lastRender.hitItems.find((it) => pointInPolygon(wx, wy, it.poly));
  if (hit) {
    tooltip.hidden = false;
    tooltip.style.left = `${e.clientX - r.left + 12}px`;
    tooltip.style.top = `${e.clientY - r.top + 12}px`;
    tooltip.textContent = `${hit.id} #${hit.instance} · rot ${hit.rotation.toFixed(0)}°`;
  } else tooltip.hidden = true;
});
canvas2d.addEventListener('mouseleave', () => { tooltip.hidden = true; });

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// --- Mode switching ---
function setMode(m) {
  mode = m;
  document.querySelectorAll('.mode').forEach((b) => b.classList.toggle('active', b.dataset.mode === m));
  const is3d = m === '3d';
  canvas2d.hidden = is3d;
  canvas3d.hidden = !is3d;
  $('hint').style.display = is3d ? 'none' : '';
  lastPayload = null;
  $('metrics').textContent = 'Run to see results.';
  $('utilbar').hidden = true;
  $('sheetTabs').replaceChildren();
  controlState = buildControls($('controls'), controlSpec());
  fillStrategies();
  fillPresets();
  if (!is3d) drawEmpty();
}
document.querySelectorAll('.mode').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
$('preset').addEventListener('change', (e) => applyPreset(e.target.value));
$('run').addEventListener('click', run);

// --- Load JSON ---
$('loadJson').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const b = partsEditor.loadJson(ev.target.result);
      if (b) boundary = b;
      fitView(); drawEmpty();
    } catch (err) { showError($('metrics'), err.message); }
  };
  reader.readAsText(file);
});

// --- Export ---
$('exportJson').addEventListener('click', () => {
  if (!lastPayload) return;
  downloadJson(`u-nesting-${lastPayload.mode}.json`, lastPayload.cutting
    ? { request: lastPayload.request, solve: lastPayload.response, cutting: lastPayload.cutting }
    : { request: lastPayload.request, response: lastPayload.response });
});
$('exportPng').addEventListener('click', () => {
  downloadCanvasPng(`u-nesting-${mode}.png`, mode === '3d' ? canvas3d : canvas2d);
});

// --- Init ---
(async () => {
  const meta = await call('meta');
  if (meta.ok) {
    $('version').textContent = 'v' + meta.version;
    strategiesByDim = meta.strategies;
  }
  setMode('2d');
})();
