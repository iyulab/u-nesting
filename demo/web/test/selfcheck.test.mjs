import { test } from 'node:test';
import assert from 'node:assert/strict';
import { aabbOf, checkPlacements2d } from '../selfcheck.js';

test('aabbOf는 min/max 반환', () => {
  assert.deepEqual(aabbOf([[1, 2], [3, 0], [2, 5]]), { minX: 1, minY: 0, maxX: 3, maxY: 5 });
});

test('boundary 내부 + 비겹침이면 issues 없음', () => {
  const polys = [[[0, 0], [1, 0], [1, 1], [0, 1]], [[2, 0], [3, 0], [3, 1], [2, 1]]];
  const issues = checkPlacements2d(polys, { width: 10, height: 10 });
  assert.equal(issues.length, 0);
});

test('boundary 이탈 감지', () => {
  const polys = [[[0, 0], [20, 0], [20, 1], [0, 1]]];
  const issues = checkPlacements2d(polys, { width: 10, height: 10 });
  assert.ok(issues.some((i) => i.type === 'out_of_bounds'));
});

test('겹침 감지', () => {
  const polys = [[[0, 0], [5, 0], [5, 5], [0, 5]], [[2, 2], [7, 2], [7, 7], [2, 7]]];
  const issues = checkPlacements2d(polys, { width: 10, height: 10 });
  assert.ok(issues.some((i) => i.type === 'overlap'));
});
