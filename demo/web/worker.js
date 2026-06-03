// WASM solve 오프로딩 워커. pkg/는 `wasm-pack build --target web` 출력.
import init, { solve_2d, solve_3d, version } from './pkg/u_nesting_wasm.js';

let ready = null;

async function ensureReady() {
  if (!ready) ready = init(); // pkg/u_nesting_wasm_bg.wasm 자동 로드
  await ready;
}

self.onmessage = async (e) => {
  const { id, mode, request } = e.data;
  try {
    await ensureReady();
    const json = JSON.stringify(request);
    const out = mode === '3d' ? solve_3d(json) : solve_2d(json);
    self.postMessage({ id, ok: true, response: JSON.parse(out), version: version() });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.message ? err.message : err) });
  }
};
