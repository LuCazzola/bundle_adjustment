/**
 * Camera frustum wireframe builder for Three.js.
 *
 * Uses the Rodrigues rotation formula (implemented in JS to avoid a math dep)
 * to convert the camera rvec into a rotation matrix, then computes the four
 * corner rays at near/far planes and connects them as a wireframe.
 *
 * World → Three.js mapping: three.x = world_x, three.y = world_z, three.z = -world_y
 */

import { w2t } from "./court.js";

// ── Rodrigues rotation in JS ──────────────────────────────────────────────────

function rodrigues(rvec) {
  // rvec: [rx, ry, rz]
  const angle = Math.sqrt(rvec[0] ** 2 + rvec[1] ** 2 + rvec[2] ** 2);
  if (angle < 1e-12) {
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  }
  const k = rvec.map(v => v / angle);   // unit axis
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  const [kx, ky, kz] = k;
  // Rodrigues formula: R = c*I + s*[k]× + t*(k⊗k)
  return [
    [t * kx * kx + c,      t * kx * ky - s * kz, t * kx * kz + s * ky],
    [t * kx * ky + s * kz, t * ky * ky + c,       t * ky * kz - s * kx],
    [t * kx * kz - s * ky, t * ky * kz + s * kx,  t * kz * kz + c     ],
  ];
}

function matVec(R, v) {
  return [
    R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
    R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
    R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
  ];
}

function matTransposeVec(R, v) {
  // R^T * v
  return [
    R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
    R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
    R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2],
  ];
}

function vecAdd(a, b)  { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function vecScale(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }
function vecNorm(v) {
  const n = Math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2);
  return n > 1e-12 ? v.map(x => x / n) : v;
}

// ── Frustum builder ───────────────────────────────────────────────────────────

/**
 * Build a wireframe frustum for one camera.
 *
 * camData: {
 *   rvec: [rx, ry, rz],
 *   tvec: [tx, ty, tz],
 *   center: [cx, cy, cz],    // optical center in world (from /api/calibration)
 *   fov_x_deg: float,
 *   fov_y_deg: float,
 * }
 *
 * near / far: frustum depth in world metres.
 *
 * Returns a THREE.Group containing:
 *   - LineSegments for the frustum wireframe
 *   - a Sprite label ("cam_N")
 */
export function buildFrustum(THREE, camData, label, near = 2, far = 18) {
  const R    = rodrigues(camData.rvec);
  const C    = camData.center;   // world optical center

  // Half-angles in each axis
  const hx = (camData.fov_x_deg / 2) * (Math.PI / 180);
  const hy = (camData.fov_y_deg / 2) * (Math.PI / 180);

  // Four corner directions in camera space (normalised)
  const cornersCam = [
    [ Math.tan(hx),  Math.tan(hy), 1],
    [-Math.tan(hx),  Math.tan(hy), 1],
    [-Math.tan(hx), -Math.tan(hy), 1],
    [ Math.tan(hx), -Math.tan(hy), 1],
  ].map(v => vecNorm(v));

  // Rotate to world space: dir_world = R^T * dir_cam
  const cornersWorld = cornersCam.map(d => vecNorm(matTransposeVec(R, d)));

  // Near and far corner points
  const nearPts = cornersWorld.map(d => vecAdd(C, vecScale(d, near)));
  const farPts  = cornersWorld.map(d => vecAdd(C, vecScale(d, far)));

  // Build line segments: 4 edges near→far, near quad, far quad
  const verts = [];
  const push = (wx, wy, wz) => verts.push(...w2t(wx, wy, wz));

  for (let i = 0; i < 4; i++) {
    push(...nearPts[i]); push(...farPts[i]);  // lateral edges
  }
  // Near quad
  for (let i = 0; i < 4; i++) {
    push(...nearPts[i]); push(...nearPts[(i + 1) % 4]);
  }
  // Far quad
  for (let i = 0; i < 4; i++) {
    push(...farPts[i]); push(...farPts[(i + 1) % 4]);
  }
  // Apex lines (camera centre to near corners)
  for (let i = 0; i < 4; i++) {
    push(...C); push(...nearPts[i]);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x44aaff, linewidth: 1 });
  const lines = new THREE.LineSegments(geo, mat);

  // Camera centre dot
  const dotGeo = new THREE.SphereGeometry(0.2, 8, 8);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0x44aaff });
  const dot = new THREE.Mesh(dotGeo, dotMat);
  dot.position.set(...w2t(...C));

  const group = new THREE.Group();
  group.add(lines);
  group.add(dot);
  group.userData.label = label;
  return group;
}

/**
 * Build a canvas-based text sprite for camera labels.
 */
export function makeLabel(THREE, text, color = "#44aaff") {
  const canvas = document.createElement("canvas");
  canvas.width  = 128;
  canvas.height = 48;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(0, 0, 128, 48);
  ctx.fillStyle = color;
  ctx.font = "bold 22px monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 64, 24);

  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(2, 0.75, 1);
  return sprite;
}
