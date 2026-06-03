// WASM solve/cutting 오프로딩 워커. pkg/는 `wasm-pack build --target web` 출력.
import init, { solve_2d, solve_3d, optimize_cutting_path, version, available_strategies }
  from './pkg/u_nesting_wasm.js';

let ready = null;
async function ensureReady() {
  if (!ready) ready = init();
  await ready;
}

const FN = { '2d': solve_2d, '3d': solve_3d, cutting: optimize_cutting_path };

self.onmessage = async (e) => {
  const { id, mode, request } = e.data;
  try {
    await ensureReady();
    if (mode === 'meta') {
      self.postMessage({ id, ok: true, version: version(), strategies: JSON.parse(available_strategies()) });
      return;
    }
    const out = FN[mode](JSON.stringify(request));
    self.postMessage({ id, ok: true, response: JSON.parse(out), version: version() });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
