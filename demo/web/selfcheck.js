// Demo diagnostic self-check. Catches transform-accuracy regressions: if a placement
// transform is wrong, parts end up overlapping or outside the sheet.
//
// Overlap uses a true polygon test (edge intersection + containment), not bounding
// boxes. An AABB test reports false positives for concave parts that legitimately
// interlock — e.g. two L-shapes nested via rotation have overlapping bounding boxes
// but disjoint polygons. Such packings are valid and must not be flagged.
const EPS = 1e-6;

export function aabbOf(points) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of points) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

// Proper (strict) segment intersection — touching endpoints/collinear do not count.
function segmentsIntersect(p1, p2, p3, p4) {
  const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const d1 = cross(p3, p4, p1), d2 = cross(p3, p4, p2);
  const d3 = cross(p1, p2, p3), d4 = cross(p1, p2, p4);
  return ((d1 > EPS && d2 < -EPS) || (d1 < -EPS && d2 > EPS)) &&
         ((d3 > EPS && d4 < -EPS) || (d3 < -EPS && d4 > EPS));
}

function pointInPolygon([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// True polygon overlap: any edges cross, or one polygon fully contains the other.
// Note: this tests outer polygons only — holes are not considered. A part legitimately
// nested inside another part's hole would be reported as overlap (false positive). The
// demo presets do not exercise that case; revisit if hole-nesting is showcased.
function polygonsOverlap(a, b) {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      if (segmentsIntersect(a[i], a[(i + 1) % a.length], b[j], b[(j + 1) % b.length])) return true;
    }
  }
  // No edges cross — one may be entirely inside the other.
  return pointInPolygon(a[0], b) || pointInPolygon(b[0], a);
}

// polys: transformed polygons (same sheet), boundary: { width, height } (origin-based rectangle)
export function checkPlacements2d(polys, boundary) {
  const issues = [];
  const boxes = polys.map(aabbOf);
  boxes.forEach((box, i) => {
    if (box.minX < -EPS || box.minY < -EPS ||
        box.maxX > boundary.width + EPS || box.maxY > boundary.height + EPS) {
      issues.push({ type: 'out_of_bounds', index: i });
    }
  });
  for (let i = 0; i < polys.length; i++) {
    for (let j = i + 1; j < polys.length; j++) {
      if (polygonsOverlap(polys[i], polys[j])) {
        issues.push({ type: 'overlap', index: i, other: j });
      }
    }
  }
  return issues;
}
