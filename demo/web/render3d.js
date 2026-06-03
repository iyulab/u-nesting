import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { dimsForOrientation } from './transform.js';

const COLORS = [0x4f8cff, 0xff7a59, 0x33c08d, 0xc084fc, 0xf5b301, 0xef5da8, 0x5ad1e6];

let scene, camera, renderer, controls, group;

function ensureScene(canvas) {
  if (renderer) return;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  camera = new THREE.PerspectiveCamera(50, canvas.clientWidth / canvas.clientHeight, 0.1, 5000);
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  controls = new OrbitControls(camera, renderer.domElement);
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(1, 2, 1);
  scene.add(dir);
  group = new THREE.Group();
  scene.add(group);
  function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
  animate();
}

// 3D self-check: 박스가 bin 내부에 있고 서로 겹치지 않는지 (라이브러리 좌표계 기준).
// 라이브러리: dims=(width,depth,height)=(x,y,z), position=(x,y,z) min corner, z=up.
function check3d(boxes, boundary) {
  const EPS = 1e-6;
  const [bw, bd, bh] = boundary.dimensions;
  const issues = [];
  boxes.forEach((b, i) => {
    if (b.x < -EPS || b.y < -EPS || b.z < -EPS ||
        b.x + b.w > bw + EPS || b.y + b.d > bd + EPS || b.z + b.h > bh + EPS) {
      issues.push({ type: 'out_of_bounds', index: i });
    }
  });
  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i], c = boxes[j];
      if (a.x < c.x + c.w - EPS && a.x + a.w > c.x + EPS &&
          a.y < c.y + c.d - EPS && a.y + a.d > c.y + EPS &&
          a.z < c.z + c.h - EPS && a.z + a.h > c.z + EPS) {
        issues.push({ type: 'overlap', index: i, other: j });
      }
    }
  }
  return issues;
}

// geometries: 요청 geometries (id→dimensions), response: Pack3DResponse, boundary: { dimensions:[w,d,h] }
export function render3d(geometries, response, canvas, boundary) {
  ensureScene(canvas);
  group.clear();
  const dimsById = new Map(geometries.map((g) => [g.id, g.dimensions]));
  const [bw, bd, bh] = boundary.dimensions;

  // bin 컨테이너 와이어프레임 (THREE: x=width, y=height(lib z), z=depth(lib y))
  const binGeo = new THREE.BoxGeometry(bw, bh, bd);
  const binEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(binGeo),
    new THREE.LineBasicMaterial({ color: 0x3a4252 })
  );
  binEdges.position.set(bw / 2, bh / 2, bd / 2);
  group.add(binEdges);

  const checkBoxes = [];
  let k = 0;
  for (const p of response.placements) {
    if (p.bin_index !== 0) continue; // 데모: 첫 bin만
    const dims = dimsById.get(p.id);
    if (!dims) continue;
    const [w, d, h] = dimsForOrientation(dims, p.orientation); // [width(x), depth(y), height(z)]
    checkBoxes.push({ x: p.x, y: p.y, z: p.z, w, d, h });

    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w, h, d), // THREE(width, height, depth)
      new THREE.MeshLambertMaterial({ color: COLORS[k % COLORS.length] })
    );
    // 라이브러리 (x,y,z) min corner, z=up → THREE 중심 (x+w/2, z+h/2, y+d/2)
    mesh.position.set(p.x + w / 2, p.z + h / 2, p.y + d / 2);
    group.add(mesh);
    const edge = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0x0e1116 })
    );
    edge.position.copy(mesh.position);
    group.add(edge);
    k++;
  }

  // 카메라 배치
  const maxDim = Math.max(bw, bd, bh);
  camera.position.set(bw / 2 + maxDim, bh / 2 + maxDim, bd / 2 + maxDim);
  controls.target.set(bw / 2, bh / 2, bd / 2);
  controls.update();

  return check3d(checkBoxes, boundary);
}
