import { transformPolygon2d } from './transform.js';
import { checkPlacements2d } from './selfcheck.js';

const COLORS = ['#4f8cff', '#ff7a59', '#33c08d', '#c084fc', '#f5b301', '#ef5da8', '#5ad1e6'];

// geometries: 요청의 geometries (id→polygon 조회용)
// response: SolveResponse, canvas: HTMLCanvasElement, boundary: { width, height }
export function render2d(geometries, response, canvas, boundary) {
  const ctx = canvas.getContext('2d');
  const polyById = new Map(geometries.map((g) => [g.id, g.polygon]));

  // 시트별 그룹화
  const sheets = new Map();
  for (const p of response.placements) {
    if (!sheets.has(p.sheet_index)) sheets.set(p.sheet_index, []);
    sheets.get(p.sheet_index).push(p);
  }
  const sheetIdx = [...sheets.keys()].sort((a, b) => a - b);
  const cols = Math.min(sheetIdx.length || 1, 3);
  const rows = Math.ceil((sheetIdx.length || 1) / cols);

  const pad = 16;
  const cellW = (canvas.width - pad * (cols + 1)) / cols;
  const cellH = (canvas.height - pad * (rows + 1)) / rows;
  const scale = Math.min(cellW / boundary.width, cellH / boundary.height);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const allIssues = [];

  sheetIdx.forEach((sIdx, gridPos) => {
    const col = gridPos % cols;
    const row = Math.floor(gridPos / cols);
    const ox = pad + col * (cellW + pad);
    const oy = pad + row * (cellH + pad);

    // 시트 boundary (좌상단 원점, y-down 화면)
    ctx.strokeStyle = '#3a4252';
    ctx.strokeRect(ox, oy, boundary.width * scale, boundary.height * scale);

    const placements = sheets.get(sIdx);
    const transformedForCheck = [];

    placements.forEach((p, k) => {
      const base = polyById.get(p.id);
      if (!base) return;
      const world = transformPolygon2d(base, p);
      transformedForCheck.push(world);

      ctx.beginPath();
      world.forEach(([wx, wy], vi) => {
        const sx = ox + wx * scale;
        const sy = oy + wy * scale; // y-down 화면; boundary 원점=좌상단
        if (vi === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
      });
      ctx.closePath();
      ctx.fillStyle = COLORS[k % COLORS.length] + 'cc';
      ctx.fill();
      ctx.strokeStyle = '#0e1116';
      ctx.stroke();
    });

    const issues = checkPlacements2d(transformedForCheck, boundary);
    if (issues.length) allIssues.push({ sheet: sIdx, issues });
  });

  return allIssues;
}
