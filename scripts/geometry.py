"""
Geometric utilities: triangulation, back-projection, cross-view observation building.
"""

import numpy as np
import cv2


def unpack_intr(intr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 9-element intr block → (3×3 mtx, dist[5])."""
    mtx = np.array([[intr[0], 0, intr[2]],
                    [0, intr[1], intr[3]],
                    [0,       0,       1]], dtype=np.float64)
    return mtx, intr[4:9]


def backproject_to_z0(pixel: np.ndarray, ex: np.ndarray,
                      intr: np.ndarray) -> np.ndarray | None:
    """Back-project one pixel onto the Z=0 world plane. Returns (x, y) or None."""
    mtx, dist = unpack_intr(intr)
    undist = cv2.undistortPoints(pixel.reshape(1, 1, 2), mtx, dist)
    xn, yn = float(undist[0, 0, 0]), float(undist[0, 0, 1])
    R, _ = cv2.Rodrigues(ex[0:3].reshape(3, 1))
    t    = ex[3:6]
    C         = -(R.T @ t)
    ray_world = R.T @ np.array([xn, yn, 1.0])
    if abs(ray_world[2]) < 1e-12:
        return None
    lam   = -C[2] / ray_world[2]
    world = C + lam * ray_world
    return world[:2]


def triangulate_rays(
    views: dict[int, np.ndarray],
    cams: list[dict],
    use_zero_dist: bool = False,
) -> "np.ndarray | None":
    """
    Least-squares ray intersection from ≥2 views. Returns (3,) or None.

    views : dict mapping cam_list_index → observed pixel (2,)
    cams  : list of camera dicts with keys rvec, tvec, mtx, dist
    """
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for cidx, px in views.items():
        cam  = cams[cidx]
        mtx  = cam["mtx"]
        dist = cam["dist"] if not use_zero_dist else np.zeros(5, dtype=np.float64)
        rvec = cam["rvec"]
        tvec = cam["tvec"]
        undist = cv2.undistortPoints(px.reshape(1, 1, 2), mtx, dist)
        xn, yn = float(undist[0, 0, 0]), float(undist[0, 0, 1])
        R_cam, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        orig = (R_cam.T @ (-tvec)).ravel()
        dirn = R_cam.T @ np.array([xn, yn, 1.0])
        dirn /= max(np.linalg.norm(dirn), 1e-12)
        P    = np.eye(3) - np.outer(dirn, dirn)
        A   += P
        b   += P @ orig
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def build_cross_view_observations(cameras: list[dict]) -> list[tuple]:
    """
    Find all (point, cam_j, cam_l) triplets where both cameras observe the
    same world point.

    Returns list of (pt3d, obs_j, obs_l, cam_j_idx, cam_l_idx).
    """
    cam_obs = []
    for c in cameras:
        obs_map = {}
        for pt3d, pt2d in zip(c["pts3d"], c["pts2d"]):
            key = tuple(np.round(pt3d, 6))
            obs_map[key] = pt2d
        cam_obs.append(obs_map)

    triplets = []
    for j in range(len(cameras)):
        for l in range(j + 1, len(cameras)):
            shared = set(cam_obs[j]) & set(cam_obs[l])
            for key in shared:
                pt3d  = np.array(key, dtype=np.float64)
                obs_j = cam_obs[j][key]
                obs_l = cam_obs[l][key]
                triplets.append((pt3d, obs_j, obs_l, j, l))
    return triplets
