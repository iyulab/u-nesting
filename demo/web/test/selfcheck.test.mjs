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

// Regression: concave parts whose bounding boxes overlap but whose polygons are disjoint
// must NOT be flagged. An L-shape and a square sitting in the L's notch — the square's
// AABB (40..70) overlaps the L's AABB (0..80), but the polygons do not touch. An AABB
// check would false-positive here; the polygon check must report 0 overlaps.
test('does not false-positive on a part nested in a concave notch', () => {
  const lShape = [[0, 0], [80, 0], [80, 30], [30, 30], [30, 80], [0, 80]];
  const squareInNotch = [[40, 40], [70, 40], [70, 70], [40, 70]];
  const issues = checkPlacements2d([lShape, squareInNotch], { width: 100, height: 100 });
  assert.equal(issues.length, 0);
});
