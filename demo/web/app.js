import { presets2d, presets3d } from './presets.js';
import { render2d } from './render2d.js';
import { render3d } from './render3d.js';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let reqId = 0;

worker.onmessage = (e) => {
  const { id } = e.data;
  const resolve = pending.get(id);
  if (resolve) { pending.delete(id); resolve(e.data); }
};

function solve(mode, request) {
  return new Promise((resolve) => {
    const id = ++reqId;
    pending.set(id, resolve);
    worker.postMessage({ id, mode, request });
  });
}

// 에러 메시지는 WASM에서 오므로 textContent로 안전하게 표시(XSS 방지)
function showError(el, message) {
  el.replaceChildren();
  const span = document.createElement('span');
  span.className = 'err';
  span.textContent = '오류: ' + message;
  el.appendChild(span);
}

// 탭 전환
document.querySelectorAll('.tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('panel-' + btn.dataset.tab).classList.add('active');
  });
});

// 프리셋 채우기
function fillPresets(selectId, presets) {
  const sel = document.getElementById(selectId);
  for (const [key, val] of Object.entries(presets)) {
    const opt = document.createElement('option');
    opt.value = key; opt.textContent = val.label;
    sel.appendChild(opt);
  }
}
fillPresets('preset2d', presets2d);
fillPresets('preset3d', presets3d);

// 2D 실행
document.getElementById('run2d').addEventListener('click', async () => {
  const btn = document.getElementById('run2d');
  const metrics = document.getElementById('metrics2d');
  const preset = presets2d[document.getElementById('preset2d').value];
  const request = {
    geometries: preset.geometries,
    boundary: preset.boundary,
    config: {
      strategy: document.getElementById('strategy2d').value,
      time_limit_ms: Number(document.getElementById('time2d').value),
      spacing: 2,
    },
  };
  btn.disabled = true; metrics.textContent = '계산 중…';
  const { ok, response, error } = await solve('2d', request);
  btn.disabled = false;
  if (!ok || !response.success) {
    showError(metrics, error || response.error);
    return;
  }
  const issues = render2d(preset.geometries, response, document.getElementById('canvas2d'), preset.boundary);
  let msg = `활용률 ${(response.utilization * 100).toFixed(1)}% · 시트 ${response.sheets_used} · ` +
            `배치 ${response.placements.length} · 미배치 ${response.unplaced.length} · ${response.elapsed_ms}ms`;
  if (issues.length) msg += ` <span class="warn">⚠ self-check ${issues.length}건(AABB 기준 의심)</span>`;
  metrics.innerHTML = msg;
});

// 3D 실행
document.getElementById('run3d').addEventListener('click', async () => {
  const btn = document.getElementById('run3d');
  const metrics = document.getElementById('metrics3d');
  const preset = presets3d[document.getElementById('preset3d').value];
  const request = {
    geometries: preset.geometries,
    boundary: preset.boundary,
    config: {
      strategy: document.getElementById('strategy3d').value,
      time_limit_ms: Number(document.getElementById('time3d').value),
    },
  };
  btn.disabled = true; metrics.textContent = '계산 중…';
  const { ok, response, error } = await solve('3d', request);
  btn.disabled = false;
  if (!ok || !response.success) {
    showError(metrics, error || response.error);
    return;
  }
  const issues = render3d(preset.geometries, response, document.getElementById('canvas3d'), preset.boundary);
  let msg = `활용률 ${(response.utilization * 100).toFixed(1)}% · bin ${response.bins_used} · ` +
            `배치 ${response.placements.length} · 미배치 ${response.unplaced.length} · ${response.elapsed_ms}ms`;
  if (issues.length) msg += ` <span class="warn">⚠ self-check ${issues.length}건</span>`;
  metrics.innerHTML = msg;
});
