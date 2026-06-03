import { transformPolygon2d } from './transform.js';
import { render2d } from './render2d.js';

// Draws the placement first, then overlays the cutting path.
// geometries: request geometries, solveResp: SolveResponse, cutResp: CuttingResponse.
// pierce_point / rapid_from are assumed to be in placement (world) coordinates
// (verified in plan Task 11).
export function renderCutting(ctx, geometries, solveResp, cutResp, viewport, boundary, activeSheet) {
  render2d(ctx, geometries, solveResp, viewport, boundary, activeSheet);

  const geomById = new Map(geometries.map((g) => [g.id, g]));
  // placement lookup: (geometry_id, instance) -> placement
  const placementOf = new Map();
  for (const p of solveResp.placements) placementOf.set(`${p.geometry_id ?? p.id}#${p.instance}`, p);

  cutResp.sequence.forEach((step, i) => {
    const p = placementOf.get(`${step.geometry_id}#${step.instance}`);
    if (!p || p.sheet_index !== activeSheet) return;

    // Highlight the contour being cut
    const geom = geomById.get(step.geometry_id);
    let contour = null;
    if (step.contour_type === 'exterior') contour = geom && geom.polygon;
    else if (geom && geom.holes && geom.holes.length) contour = geom.holes[0]; // demo: first hole
    if (contour) {
      const world = transformPolygon2d(contour, p);
      ctx.beginPath();
      world.forEach(([x, y], k) => { const [sx, sy] = viewport.toScreen(x, y); k ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
      ctx.closePath();
      ctx.strokeStyle = step.contour_type === 'exterior' ? '#33c08d' : '#f5b301';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Rapid traverse line (dashed)
    if (step.rapid_from) {
      const [fx, fy] = viewport.toScreen(step.rapid_from[0], step.rapid_from[1]);
      const [tx, ty] = viewport.toScreen(step.pierce_point[0], step.pierce_point[1]);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#ef5da8';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.restore();
    }

    // Pierce point marker + order index
    const [px, py] = viewport.toScreen(step.pierce_point[0], step.pierce_point[1]);
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#9aa7b8';
    ctx.font = '10px system-ui';
    ctx.fillText(String(i + 1), px + 5, py - 5);
  });
}
