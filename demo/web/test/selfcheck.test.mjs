import { test } from 'node:test';
import assert from 'node:assert/strict';
import { aabbOf, checkPlacements2d } from '../selfcheck.js';

test('aabbOf returns min/max', () => {
  assert.deepEqual(aabbOf([[1, 2], [3, 0], [2, 5]]), { minX: 1, minY: 0, maxX: 3, maxY: 5 });
});

test('inside boundary and non-overlapping yields no issues', () => {
  const polys = [[[0, 0], [1, 0], [1, 1], [0, 1]], [[2, 0], [3, 0], [3, 1], [2, 1]]];
  const issues = checkPlacements2d(polys, { width: 10, height: 10 });
  assert.equal(issues.length, 0);
});

test('detects out of bounds', () => {
  const polys = [[[0, 0], [20, 0], [20, 1], [0, 1]]];
  const issues = checkPlacements2d(polys, { width: 10, height: 10 });
  assert.ok(issues.some((i) => i.type === 'out_of_bounds'));
});

test('detects overlap', () => {
  const polys = [[[0, 0], [5, 0], [5, 5], [0, 5]], [[2, 2], [7, 2], [7, 7], [2, 7]]];
  const issues = checkPlacements2d(polys, { width: 10, height: 10 });
  assert.ok(issues.some((i) => i.type === 'overlap'));
});
