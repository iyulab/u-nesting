import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Viewport2D } from '../viewport2d.js';

const approx = (a, b, e = 1e-9) => Math.abs(a - b) < e;

test('initial transform: world->screen round trip matches', () => {
  const vp = new Viewport2D();
  vp.setBase(2, 10, 20); // scale=2, panPx=(10,20)
  const [sx, sy] = vp.toScreen(5, 7);
  const [wx, wy] = vp.toWorld(sx, sy);
  assert.ok(approx(wx, 5) && approx(wy, 7));
});

test('zoomAt keeps the world point under the cursor fixed', () => {
  const vp = new Viewport2D();
  vp.setBase(1, 0, 0);
  const before = vp.toWorld(100, 50);
  vp.zoomAt(100, 50, 1.5);
  const after = vp.toWorld(100, 50);
  assert.ok(approx(before[0], after[0]) && approx(before[1], after[1]));
  assert.ok(approx(vp.scale, 1.5));
});

test('panBy increases offset by the screen delta', () => {
  const vp = new Viewport2D();
  vp.setBase(2, 10, 20);
  vp.panBy(5, -5);
  assert.ok(approx(vp.panX, 15) && approx(vp.panY, 15));
});
