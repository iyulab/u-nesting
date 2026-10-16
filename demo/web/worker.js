// Web Worker offloading WASM solve/cutting. pkg/ is the `wasm-pack build --target web` output.
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
      self.postMessage({ id, ok: true, version: version(), strategies: available_strategies() });
      return;
    }
    // A refused request throws (an Error carrying `code`); the catch reports it.
    const response = FN[mode](request);
    self.postMessage({ id, ok: true, response, version: version() });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
