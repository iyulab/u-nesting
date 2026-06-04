import { transformPolygon2d } from './transform.js';
import { checkPlacements2d } from './selfcheck.js';

const PALETTE = ['#4f8cff', '#ff7a59', '#33c08d', '#c084fc', '#f5b301', '#ef5da8', '#5ad1e6', '#a3e635'];
let colorMap = new Map();
// Assigns a stable color to each geometry id by its position in the current part list.
// Rebuilt per render (call from render2d/render3d) so a part's color depends only on the
// active geometry set — not on which mode/preset was viewed earlier in the session. The
// previous global cache leaked across mode switches: 3D's ids (big/mid/small) seeded the
// palette first, shifting 2D's A/B/C to later colors so the same part changed color.
export function seedColors(geometries) {
  colorMap = new Map();
  geometries.forEach((g, i) => colorMap.set(g.id, PALETTE[i % PALETTE.length]));
}
export function colorFor(id) {
  if (!colorMap.has(id)) colorMap.set(id, PALETTE[colorMap.size % PALETTE.length]);
  return colorMap.get(id);
}

export function boundaryPolygon(boundary) {
  if (boundary.polygon) return boundary.polygon;
  return [[0, 0], [boundary.width, 0], [boundary.width, boundary.height], [0, boundary.height]];
}

// boundary AABB (polygon or rectangle)
export function boundaryExtent(boundary) {
  const pts = boundaryPolygon(boundary);
  let maxX = 0, maxY = 0;
  for (const [x, y] of pts) { if (x > maxX) maxX = x; if (y > maxY) maxY = y; }
  return { width: maxX, height: maxY };
}

// ctx: CanvasRenderingContext2D, response: SolveResponse, viewport: Viewport2D
// returns: { hitItems, issues, sheets }
export function render2d(ctx, geometries, response, viewport, boundary, activeSheet) {
  seedColors(geometries);
  const canvas = ctx.canvas;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const polyById = new Map(geometries.map((g) => [g.id, g.polygon]));
  const sheets = [...new Set(response.placements.map((p) => p.sheet_index))].sort((a, b) => a - b);

  // boundary outline
  const bpoly = boundaryPolygon(boundary);
  ctx.strokeStyle = '#3a4252';
  ctx.lineWidth = 1;
  ctx.beginPath();
  bpoly.forEach(([x, y], i) => { const [sx, sy] = viewport.toScreen(x, y); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
  ctx.closePath();
  ctx.stroke();

  const hitItems = [];
  const placed = response.placements.filter((p) => p.sheet_index === activeSheet);
  const checkPolys = [];
  placed.forEach((p) => {
    const base = polyById.get(p.id);
    if (!base) return;
    const world = transformPolygon2d(base, p);
    checkPolys.push(world);
    hitItems.push({ poly: world, id: p.id, instance: p.instance, rotation: p.rotation, sheet: p.sheet_index });
    ctx.beginPath();
    world.forEach(([x, y], i) => { const [sx, sy] = viewport.toScreen(x, y); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
    ctx.closePath();
    ctx.fillStyle = colorFor(p.id) + 'cc';
    ctx.fill();
    ctx.strokeStyle = '#0e1116';
    ctx.stroke();
    // Draw holes
    const baseHoles = (geometries.find((g) => g.id === p.id) || {}).holes || [];
    for (const hole of baseHoles) {
      const whole = transformPolygon2d(hole, p);
      ctx.beginPath();
      whole.forEach(([x, y], i) => { const [sx, sy] = viewport.toScreen(x, y); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
      ctx.closePath();
      ctx.fillStyle = '#0e1116';
      ctx.fill();
    }
  });

  const ext = boundaryExtent(boundary);
  const issues = checkPlacements2d(checkPolys, ext);
  return { hitItems, issues, sheets };
}
