/**
 * 3D skeleton builder for the 18-keypoint basketball HPE schema.
 *
 * Keypoints (0-indexed):
 *   0  Hips        1  RHip        2  RKnee       3  RAnkle     4  RFoot
 *   5  LHip        6  LKnee       7  LAnkle      8  LFoot      9  Spine
 *  10  Neck       11  Head       12  RShoulder   13  RElbow    14  RHand
 *  15  LShoulder  16  LElbow     17  LHand
 *
 * Skeleton edges (0-indexed pairs):
 *   Right leg: 0-1-2-3-4
 *   Left  leg: 0-5-6-7-8
 *   Spine/head: 0-9-10-11
 *   Right arm: 10-12-13-14
 *   Left  arm: 10-15-16-17
 */

import { w2t } from "./court.js";

export const KPT_NAMES = [
  "Hips", "RHip", "RKnee", "RAnkle", "RFoot",
  "LHip", "LKnee", "LAnkle", "LFoot", "Spine",
  "Neck", "Head", "RShoulder", "RElbow", "RHand",
  "LShoulder", "LElbow", "LHand",
];

export const SKELETON_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],    // right leg
  [0, 5], [5, 6], [6, 7], [7, 8],    // left leg
  [0, 9], [9, 10], [10, 11],          // spine + head
  [10, 12], [12, 13], [13, 14],       // right arm
  [10, 15], [15, 16], [16, 17],       // left arm
];

// Colour palette for multiple players (cycles)
const PLAYER_COLORS = [
  0x00ff88, 0xff6633, 0x33aaff, 0xffdd00, 0xff44cc,
  0x44ffee, 0xff8800, 0x9944ff, 0x88ff44, 0xff4444,
];

/**
 * Create a new THREE.Group representing one player skeleton.
 * keypoints3d: array of 18 entries — each is [x, y, z, conf] or null.
 * colorIdx: player index (drives colour selection).
 */
export function buildSkeleton(THREE, keypoints3d, colorIdx = 0) {
  const color = PLAYER_COLORS[colorIdx % PLAYER_COLORS.length];
  const group = new THREE.Group();

  // Bone lines
  const boneVerts = [];
  for (const [i, j] of SKELETON_EDGES) {
    const ki = keypoints3d[i];
    const kj = keypoints3d[j];
    if (!ki || !kj) continue;
    boneVerts.push(...w2t(ki[0], ki[1], ki[2]), ...w2t(kj[0], kj[1], kj[2]));
  }

  if (boneVerts.length > 0) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(boneVerts, 3));
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    group.add(new THREE.LineSegments(geo, mat));
  }

  // Joint spheres
  const jointGeo = new THREE.SphereGeometry(0.08, 8, 8);
  const jointMat = new THREE.MeshBasicMaterial({ color });
  for (let k = 0; k < keypoints3d.length; k++) {
    const kp = keypoints3d[k];
    if (!kp) continue;
    const mesh = new THREE.Mesh(jointGeo, jointMat);
    mesh.position.set(...w2t(kp[0], kp[1], kp[2]));
    group.add(mesh);
  }

  group.userData.type    = "skeleton";
  group.userData.colorIdx = colorIdx;
  return group;
}

/**
 * Update an existing skeleton group with new keypoints (avoids GC pressure
 * on each frame).  Disposes old geometry/meshes and replaces them in-place.
 */
export function updateSkeleton(THREE, group, keypoints3d) {
  // Remove old children
  while (group.children.length) {
    const child = group.children[0];
    if (child.geometry) child.geometry.dispose();
    group.remove(child);
  }

  const colorIdx = group.userData.colorIdx ?? 0;
  const color    = PLAYER_COLORS[colorIdx % PLAYER_COLORS.length];

  const boneVerts = [];
  for (const [i, j] of SKELETON_EDGES) {
    const ki = keypoints3d[i];
    const kj = keypoints3d[j];
    if (!ki || !kj) continue;
    boneVerts.push(...w2t(ki[0], ki[1], ki[2]), ...w2t(kj[0], kj[1], kj[2]));
  }

  if (boneVerts.length > 0) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(boneVerts, 3));
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    group.add(new THREE.LineSegments(geo, mat));
  }

  const jointGeo = new THREE.SphereGeometry(0.08, 8, 8);
  const jointMat = new THREE.MeshBasicMaterial({ color });
  for (const kp of keypoints3d) {
    if (!kp) continue;
    const mesh = new THREE.Mesh(jointGeo, jointMat);
    mesh.position.set(...w2t(kp[0], kp[1], kp[2]));
    group.add(mesh);
  }
}
