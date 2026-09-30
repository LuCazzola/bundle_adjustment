/**
 * FIBA basketball court geometry for Three.js.
 *
 * World coordinate system (metres):
 *   X — longitudinal (baseline to baseline, ±14)
 *   Y — lateral (sideline to sideline, ±7.5)
 *   Z — height (floor = 0)
 *
 * Three.js mapping applied by the caller:
 *   three.x = world_x,  three.y = world_z,  three.z = -world_y
 */

export function w2t(x, y, z = 0) {
  // World → Three.js coordinate conversion
  return [x, z, -y];
}

// ── Arc helpers ────────────────────────────────────────────────────────────────

function arcPts(cx, cy, r, aStart, aEnd, n = 64) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = aStart + (aEnd - aStart) * (i / n);
    pts.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return pts;
}

/** Return a flat Float32Array of [x,y,z, x,y,z, ...] line-segment pairs. */
function ptsToSegments(pts) {
  const verts = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[i + 1];
    verts.push(...w2t(x1, y1), ...w2t(x2, y2));
  }
  return verts;
}

// ── Court line segments (as flat vertex arrays) ───────────────────────────────

function outerBoundary() {
  const corners = [
    [-14, -7.5], [14, -7.5], [14, 7.5], [-14, 7.5], [-14, -7.5],
  ];
  return ptsToSegments(corners);
}

function centerLine() {
  return [...w2t(0, -7.5), ...w2t(0, 7.5)];
}

function centerCircle() {
  return ptsToSegments(arcPts(0, 0, 1.8, 0, 2 * Math.PI, 64));
}

function paintBox(signX) {
  // X: from baseline (±14) inward to free-throw line (±8.325)
  const x1 = signX * 14;
  const x2 = signX * 8.325;
  const corners = [
    [x1, -2.45], [x2, -2.45], [x2, 2.45], [x1, 2.45], [x1, -2.45],
  ];
  return ptsToSegments(corners);
}

function freeThrowArc(signX) {
  // Facing-court semicircle at free-throw line
  const cx = signX * 8.325;
  const aStart = signX > 0 ?  Math.PI / 2 : -Math.PI / 2;
  const aEnd   = signX > 0 ? (3 * Math.PI) / 2 : Math.PI / 2;
  return ptsToSegments(arcPts(cx, 0, 1.8, aStart, aEnd, 32));
}

function threePointLine(signX) {
  const basketX = signX * 12.15;
  const r = 6.75;
  const halfW = 7.5;
  const sinClip = Math.min(halfW / r, 1.0);
  const aClip = Math.asin(sinClip);  // ≈63°

  const verts = [];

  if (signX < 0) {
    // Left side: arc goes from +aClip to π-aClip (rightward half circle)
    verts.push(...ptsToSegments(arcPts(basketX, 0, r, aClip, Math.PI - aClip, 64)));
    // Top straight: from arc top to baseline
    const ax = basketX + r * Math.cos(aClip);
    const bx = basketX + r * Math.cos(Math.PI - aClip);
    verts.push(...w2t(-14, halfW), ...w2t(ax,  halfW));
    verts.push(...w2t(-14, -halfW), ...w2t(bx, -halfW));
  } else {
    // Right side: arc goes from π+aClip to 2π-aClip
    verts.push(...ptsToSegments(arcPts(basketX, 0, r, Math.PI + aClip, 2 * Math.PI - aClip, 64)));
    const ax = basketX - r * Math.cos(aClip);
    verts.push(...w2t(14, halfW), ...w2t(ax,  halfW));
    verts.push(...w2t(14, -halfW), ...w2t(ax, -halfW));
  }

  return verts;
}

function restrictedArc(signX) {
  const cx = signX * 12.15;
  const aStart = signX > 0 ?  Math.PI / 2 : -Math.PI / 2;
  const aEnd   = signX > 0 ? (3 * Math.PI) / 2 : Math.PI / 2;
  return ptsToSegments(arcPts(cx, 0, 1.25, aStart, aEnd, 32));
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Build a THREE.LineSegments object for the full FIBA court.
 * Requires THREE to be in scope.
 */
export function buildCourtLines(THREE) {
  const allVerts = [
    ...outerBoundary(),
    ...centerLine(),
    ...centerCircle(),
    ...paintBox(-1), ...paintBox(1),
    ...freeThrowArc(-1), ...freeThrowArc(1),
    ...threePointLine(-1), ...threePointLine(1),
    ...restrictedArc(-1), ...restrictedArc(1),
  ];

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position",
    new THREE.Float32BufferAttribute(allVerts, 3));

  const mat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 1 });
  return new THREE.LineSegments(geo, mat);
}

/**
 * Build a semi-transparent court floor plane.
 */
export function buildCourtFloor(THREE) {
  const geo = new THREE.PlaneGeometry(28, 15);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x2a4a2a,
    transparent: true,
    opacity: 0.4,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geo, mat);
  // PlaneGeometry lies in XY plane; rotate so it lies in the XZ plane (Y=0 ground)
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

/**
 * Build sphere markers for the calibration keypoints.
 * pts: array of [x, y, 0] world positions from /api/court.
 */
export function buildKeypointSpheres(THREE, pts) {
  const group = new THREE.Group();
  const geo = new THREE.SphereGeometry(0.12, 8, 8);
  const mat = new THREE.MeshBasicMaterial({ color: 0xf0c040 });
  for (const [wx, wy, wz] of pts) {
    const mesh = new THREE.Mesh(geo, mat);
    const [tx, ty, tz] = w2t(wx, wy, wz);
    mesh.position.set(tx, ty, tz);
    group.add(mesh);
  }
  return group;
}
