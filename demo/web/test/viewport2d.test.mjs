import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Viewport2D } from '../viewport2d.js';

const approx = (a, b, e = 1e-9) => Math.abs(a - b) < e;

test('초기 변환: world→screen 왕복 일치', () => {
  const vp = new Viewport2D();
  vp.setBase(2, 10, 20); // scale=2, panPx=(10,20)
  const [sx, sy] = vp.toScreen(5, 7);
  const [wx, wy] = vp.toWorld(sx, sy);
  assert.ok(approx(wx, 5) && approx(wy, 7));
});

test('zoomAt: 커서 아래 world점이 고정된다', () => {
  const vp = new Viewport2D();
  vp.setBase(1, 0, 0);
  const before = vp.toWorld(100, 50);
  vp.zoomAt(100, 50, 1.5); // 커서(100,50)에서 1.5배 확대
  const after = vp.toWorld(100, 50);
  assert.ok(approx(before[0], after[0]) && approx(before[1], after[1]));
  assert.ok(approx(vp.scale, 1.5));
});

test('panBy: 화면 이동량만큼 offset 증가', () => {
  const vp = new Viewport2D();
  vp.setBase(2, 10, 20);
  vp.panBy(5, -5);
  assert.ok(approx(vp.panX, 15) && approx(vp.panY, 15));
});
