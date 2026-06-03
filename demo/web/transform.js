// 2D placement 변환: P' = R(θ)·(flip(P)) + (x, y)
// pivot = origin, 회전 먼저 → 평행이동. θ는 degrees.
// 라이브러리 일치: crates/d2/src/geometry.rs::aabb_at_rotation
export function transformPolygon2d(points, placement) {
  const { x = 0, y = 0, rotation = 0, flipped = false } = placement;
  const rad = (rotation * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  return points.map(([px, py]) => {
    const fx = flipped ? -px : px;
    const rx = fx * cos - py * sin;
    const ry = fx * sin + py * cos;
    return [rx + x, ry + y];
  });
}

// 3D orientation 문자열로 원본 dims를 치환. "xyz"→[d0,d1,d2], "xzy"→[d0,d2,d1] ...
// 라이브러리 일치: crates/d3/src/geometry.rs::orientation_label / dimensions_for_orientation
export function dimsForOrientation(dimensions, orientation) {
  const axis = { x: 0, y: 1, z: 2 };
  const o = (orientation || 'xyz').toLowerCase();
  return [
    dimensions[axis[o[0]] ?? 0],
    dimensions[axis[o[1]] ?? 1],
    dimensions[axis[o[2]] ?? 2],
  ];
}
