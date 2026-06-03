// 데모 진단용 AABB 기반 self-check. 변환 정확성 회귀를 잡는 용도.
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

function aabbOverlap(a, b) {
  return a.minX < b.maxX - EPS && a.maxX > b.minX + EPS &&
         a.minY < b.maxY - EPS && a.maxY > b.minY + EPS;
}

// polys: 변환된 폴리곤 배열(같은 시트), boundary: { width, height } (원점 기준 사각형)
export function checkPlacements2d(polys, boundary) {
  const issues = [];
  const boxes = polys.map(aabbOf);
  boxes.forEach((box, i) => {
    if (box.minX < -EPS || box.minY < -EPS ||
        box.maxX > boundary.width + EPS || box.maxY > boundary.height + EPS) {
      issues.push({ type: 'out_of_bounds', index: i });
    }
  });
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      if (aabbOverlap(boxes[i], boxes[j])) {
        issues.push({ type: 'overlap', index: i, other: j });
      }
    }
  }
  return issues;
}
