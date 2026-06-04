// WASM runtime smoke test — exercises every strategy through an actual solve
// to catch runtime panics that `cargo check --target wasm32` cannot (Instant/thread).
//
// Usage:
//   1. wasm-pack build crates/wasm --target web --out-dir <pkg> --dev
//   2. node --test demo/web/test/wasm_runtime_smoke.mjs   (set PKG env to pkg dir)
//
// Defaults to ../pkg (the demo's own pkg) when PKG is unset.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const pkgDir = process.env.PKG ?? resolve(here, '../pkg');
const mod = await import(pathToFileURL(resolve(pkgDir, 'u_nesting_wasm.js')).href);
const { default: init, solve_2d, solve_3d, available_strategies } = mod;
await init({ module_or_path: readFileSync(resolve(pkgDir, 'u_nesting_wasm_bg.wasm')) });

const STRATEGIES_2D = ['blf', 'nfp', 'ga', 'brkga', 'sa', 'gdrr', 'alns'];
const STRATEGIES_3D = ['blf', 'ep', 'ga', 'brkga', 'sa'];

function req2d(strategy) {
  return JSON.stringify({
    geometries: [
      { id: 'A', polygon: [[0, 0], [40, 0], [40, 30], [0, 30]], quantity: 4 },
      { id: 'B', polygon: [[0, 0], [25, 0], [25, 25], [0, 25]], quantity: 3 },
    ],
    boundary: { width: 200, height: 200 },
    config: { strategy, time_limit_ms: 500 },
  });
}

function req3d(strategy) {
  return JSON.stringify({
    geometries: [{ id: 'mid', dimensions: [30, 20, 25], quantity: 6 }],
    boundary: { dimensions: [100, 100, 100] },
    config: { strategy, time_limit_ms: 500 },
  });
}

test('available_strategies returns 2D + 3D lists', () => {
  const s = JSON.parse(available_strategies());
  assert.ok(Array.isArray(s.strategies_2d) || Array.isArray(s['2d']) || s, 'strategies present');
});

for (const strategy of STRATEGIES_2D) {
  test(`solve_2d:${strategy} runs without panic`, () => {
    const out = solve_2d(req2d(strategy)); // throws RuntimeError on wasm panic
    const res = JSON.parse(out);
    assert.ok(res, `solve_2d(${strategy}) returned a parseable result`);
  });
}

for (const strategy of STRATEGIES_3D) {
  test(`solve_3d:${strategy} runs without panic`, () => {
    const out = solve_3d(req3d(strategy));
    const res = JSON.parse(out);
    assert.ok(res, `solve_3d(${strategy}) returned a parseable result`);
  });
}

// Reconstruct an oriented box footprint from the response orientation label
// (axis permutation, e.g. "zyx"), exactly as the demo self-check does.
function dimsForOrientation(base, label) {
  const axis = { x: 0, y: 1, z: 2 };
  const [a, b, c] = [...label].map((ch) => axis[ch]);
  return [base[a], base[b], base[c]];
}

// A 25x25x80 box only fits a 100x100x30 bin lying down, so the packer must rotate
// it and report the rotated orientation. Before the 0.3.4 fix, EP/GA/BRKGA reported
// the identity "xyz", which reconstructs to a 25x25x80 footprint that overflows the
// 30-tall bin — the exact "out-of-bounds" the demo self-check flagged on 0.3.3.
function req3dRotation(strategy) {
  return JSON.stringify({
    geometries: [{ id: 'tall', dimensions: [25, 25, 80], quantity: 3 }],
    boundary: { dimensions: [100, 100, 30] },
    config: { strategy, time_limit_ms: 500 },
  });
}

for (const strategy of STRATEGIES_3D) {
  test(`solve_3d:${strategy} reports orientation that keeps boxes in bounds`, () => {
    const res = JSON.parse(solve_3d(req3dRotation(strategy)));
    const base = [25, 25, 80];
    const [bx, by, bz] = [100, 100, 30];
    const e = 1e-6;
    for (const p of res.placements ?? []) {
      const [dx, dy, dz] = dimsForOrientation(base, p.orientation);
      assert.ok(
        p.x + dx <= bx + e && p.y + dy <= by + e && p.z + dz <= bz + e,
        `solve_3d(${strategy}) ${p.id}#${p.instance} out of bounds for reported ` +
          `orientation '${p.orientation}': pos(${p.x},${p.y},${p.z}) dims(${dx},${dy},${dz})`,
      );
    }
  });
}
