// 2D zoom/pan viewport. screen = world * scale + (panX, panY). Zoom keeps the point under the cursor fixed.
export class Viewport2D {
  constructor() { this.scale = 1; this.panX = 0; this.panY = 0; }
  // Set base scale/pan (used to apply a fit result)
  setBase(scale, panX, panY) { this.scale = scale; this.panX = panX; this.panY = panY; }
  toScreen(wx, wy) { return [wx * this.scale + this.panX, wy * this.scale + this.panY]; }
  toWorld(sx, sy) { return [(sx - this.panX) / this.scale, (sy - this.panY) / this.scale]; }
  // Zoom by factor while keeping the world point under cursor (sx,sy) fixed
  zoomAt(sx, sy, factor) {
    const [wx, wy] = this.toWorld(sx, sy);
    this.scale *= factor;
    this.panX = sx - wx * this.scale;
    this.panY = sy - wy * this.scale;
  }
  panBy(dx, dy) { this.panX += dx; this.panY += dy; }
}
