import { test } from 'node:test';
import assert from 'node:assert/strict';
import { transformPolygon2d, dimsForOrientation } from '../transform.js';

const approx = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

test('회전 0°는 평행이동만 적용', () => {
  const out = transformPolygon2d([[0, 0], [2, 0], [2, 1]], { x: 10, y: 5, rotation: 0 });
  assert.deepEqual(out, [[10, 5], [12, 5], [12, 6]]);
});

test('회전 90°는 원점 기준 후 평행이동', () => {
  // (1,0) → 90° CCW → (0,1) → +(0,0) = (0,1)
  const out = transformPolygon2d([[1, 0]], { x: 0, y: 0, rotation: 90 });
  assert.ok(approx(out[0][0], 0));
  assert.ok(approx(out[0][1], 1));
});

test('flipped는 x축 반전 후 회전+이동', () => {
  // flip: (2,0)→(-2,0), rot 0, +(5,0) = (3,0)
  const out = transformPolygon2d([[2, 0]], { x: 5, y: 0, rotation: 0, flipped: true });
  assert.ok(approx(out[0][0], 3));
  assert.ok(approx(out[0][1], 0));
});

test('orientation xyz는 원본 dims', () => {
  assert.deepEqual(dimsForOrientation([10, 20, 30], 'xyz'), [10, 20, 30]);
});

test('orientation xzy는 y/z 교환', () => {
  assert.deepEqual(dimsForOrientation([10, 20, 30], 'xzy'), [10, 30, 20]);
});

test('orientation yxz는 x/y 교환', () => {
  assert.deepEqual(dimsForOrientation([10, 20, 30], 'yxz'), [20, 10, 30]);
});
