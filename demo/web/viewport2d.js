// 2D 줌/팬 뷰포트. screen = world * scale + (panX, panY). 줌은 커서 기준 점 고정.
export class Viewport2D {
  constructor() { this.scale = 1; this.panX = 0; this.panY = 0; }
  // base scale/pan 설정(fit 결과 적용용)
  setBase(scale, panX, panY) { this.scale = scale; this.panX = panX; this.panY = panY; }
  toScreen(wx, wy) { return [wx * this.scale + this.panX, wy * this.scale + this.panY]; }
  toWorld(sx, sy) { return [(sx - this.panX) / this.scale, (sy - this.panY) / this.scale]; }
  // 커서(sx,sy) 아래 world점을 고정한 채 factor 배 줌
  zoomAt(sx, sy, factor) {
    const [wx, wy] = this.toWorld(sx, sy);
    this.scale *= factor;
    this.panX = sx - wx * this.scale;
    this.panY = sy - wy * this.scale;
  }
  panBy(dx, dy) { this.panX += dx; this.panY += dy; }
}
