"""
Bundle Adjustment for multi-camera basketball court setup.

Uses known 3D court keypoints (img_points.json) and initial camera parameters
to jointly refine camera extrinsics and/or intrinsics via Ceres Solver (pyceres).

OBJECTIVE
---------
Minimise 2-D pixel reprojection error (classic BA). Both 2-D (px) and 3-D
world-plane (m) errors are reported before and after optimisation for
comparison, but only the 2-D objective is used during optimisation.

PARAMETER BLOCKS (all float64, contiguous)
------------------------------------------
  ex  (6): rx ry rz  tx ty tz
  in  (9): fx fy cx cy  k1 k2 p1 p2 k3

DATA SOURCES  (--calib-source)
-------------------------------
  checkerboard   Use camera_calib.json (default, checkerboard-based)
  [mocap]        Future: human 3-D MoCap observations as additional BA constraints

DATA LAYOUT
-----------
  <data-dir>/
    cam_<id>/
      calib/
        camera_calib.json   intrinsics + extrinsics (checkerboard)
        img_points.json     court 3-D ↔ 2-D correspondences
      ba/
        camera_calib_ba.json   output (written by this script)

USAGE
-----
  python ba.py [options]

  --cameras       2 5 8 13        cameras to include (default: all found)
  --optimize      ex|in|both      what to refine     (default: both)
  --calib-source  checkerboard    starting calibration (default: checkerboard)
  --data-dir      <path>          root folder with cam_*/ sub-dirs
  --suffix        _ba             output file suffix (default: _ba)
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pyceres

N_IN, N_EX = 9, 6  # parameter block sizes — used throughout
EPS        = 1e-4  # relative central-difference step: h = EPS * max(|param|, 1.0)

@dataclasses.dataclass
class LAMBD:
    """Weights for the three terms in the BA objective."""
    INTR_REG:   float
    CROSS_VIEW: float

def _unpack_intr(intr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 9-element intr block → (3×3 mtx, dist[5])."""
    mtx  = np.array([[intr[0], 0, intr[2]],
                     [0, intr[1], intr[3]],
                     [0,       0,       1]], dtype=np.float64)
    return mtx, intr[4:9]

def _backproject_to_z0(pixel: np.ndarray, ex: np.ndarray,
                       intr: np.ndarray) -> np.ndarray | None:
    """Back-project one pixel onto the Z=0 world plane. Returns (x, y) or None."""
    mtx, dist = _unpack_intr(intr)
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


class ReprojectionCost(pyceres.CostFunction):
    """
    2-residual cost for one observation — pixel-space formulation.

    Residuals: [u_proj - u_obs,  v_proj - v_obs]  in pixels.
    Jacobians: analytic for ex via cv2.projectPoints, FD for intr.
    """

    def __init__(self, pt3d: np.ndarray, pt2d: np.ndarray) -> None:
        super().__init__()
        self.pt3d = pt3d.reshape(1, 1, 3).astype(np.float64)
        self.obs  = pt2d.astype(np.float64)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([N_EX, N_IN])

    def _project(self, ex: np.ndarray, intr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mtx, dist = _unpack_intr(intr)
        proj, jac = cv2.projectPoints(self.pt3d, ex[0:3].reshape(3, 1), ex[3:6].reshape(3, 1), mtx, dist)
        return proj.reshape(2), jac[:, 0:3], jac[:, 3:6]

    def Evaluate(self, parameters: list, residuals: np.ndarray,
                 jacobians: list) -> bool:
        ex   = parameters[0]
        intr = parameters[1]
        proj, j_rvec, j_tvec = self._project(ex, intr)
        residuals[0] = proj[0] - self.obs[0]
        residuals[1] = proj[1] - self.obs[1]
        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][:] = np.hstack([j_rvec, j_tvec]).ravel()
            if jacobians[1] is not None:
                j_in = np.zeros((2, N_IN), dtype=np.float64)
                for k in range(N_IN):
                    h = EPS * max(abs(intr[k]), 1.0)
                    intr_p = intr.copy(); intr_p[k] += h
                    intr_m = intr.copy(); intr_m[k] -= h
                    pp, _, _ = self._project(ex, intr_p)
                    pm, _, _ = self._project(ex, intr_m)
                    j_in[:, k] = (pp - pm) / (2 * h)
                jacobians[1][:] = j_in.ravel()
        return True


class IntrinsicRegularisationCost(pyceres.CostFunction):
    """
    Soft prior that keeps each intrinsic parameter close to its checkerboard
    value.  One scalar residual per parameter:

        residual[k] = weight * (intr[k] - prior[k]) / prior[k]

    Dividing by prior[k] makes the penalty dimensionless (fractional drift),
    so fx ≈ 4000 and k1 ≈ 0.3 are penalised on the same scale.
    Parameters whose prior is zero (e.g. skew, tangential dist) are skipped
    by setting their scale to 1 to avoid division-by-zero.
    """

    def __init__(self, prior: np.ndarray, weight: float) -> None:
        super().__init__()
        self.prior  = prior.copy()
        self.weight = weight
        # Scale for each parameter: use |prior| if nonzero, else 1.0
        self.scale  = np.where(np.abs(prior) > 1e-9, np.abs(prior), 1.0)
        self.set_num_residuals(N_IN)
        self.set_parameter_block_sizes([N_IN])

    def Evaluate(self, parameters: list, residuals: np.ndarray,
                 jacobians: list) -> bool:
        intr = parameters[0]
        for k in range(N_IN):
            residuals[k] = self.weight * (intr[k] - self.prior[k]) / self.scale[k]
        if jacobians is not None and jacobians[0] is not None:
            jac = np.zeros((N_IN, N_IN), dtype=np.float64)
            for k in range(N_IN):
                jac[k, k] = self.weight / self.scale[k]
            jacobians[0][:] = jac.ravel()
        return True


class CrossViewConsistencyCost(pyceres.CostFunction):
    """
    Cross-camera consistency for one shared world point observed in two cameras.

    Given cameras j and l that both observe the same 3-D point P:
      1. Back-project obs_j (camera j's pixel) through camera j to the Z=0 plane → ŵ
      2. Forward-project ŵ through camera l → π_l(ŵ)
      3. Residual = π_l(ŵ) - obs_l   (2 values, in pixels)

    Parameter blocks: [ex_j (6), in_j (9), ex_l (6), in_l (9)]
    Jacobians: finite differences over all four blocks (back-projection has no
    closed-form Jacobian w.r.t. camera parameters, so FD throughout).
    """

    def __init__(self, pt3d: np.ndarray,
                 obs_j: np.ndarray, obs_l: np.ndarray,
                 weight: float = 1.0) -> None:
        super().__init__()
        self.pt3d  = pt3d.astype(np.float64)
        self.obs_j = obs_j.astype(np.float64)
        self.obs_l = obs_l.astype(np.float64)
        self.weight = weight
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([N_EX, N_IN, N_EX, N_IN])

    def _backproject(self, obs: np.ndarray,
                     ex: np.ndarray, intr: np.ndarray) -> np.ndarray | None:
        """Back-project pixel to Z=0. Returns (x, y, 0) or None."""
        xy = _backproject_to_z0(obs, ex, intr)
        return None if xy is None else np.array([xy[0], xy[1], 0.0])

    def _forward(self, pt: np.ndarray,
                 ex: np.ndarray, intr: np.ndarray) -> np.ndarray:
        """Project 3-D point through camera (ex, intr). Returns (u, v)."""
        mtx, dist = _unpack_intr(intr)
        proj, _ = cv2.projectPoints(pt.reshape(1, 1, 3),
                                    ex[0:3].reshape(3, 1), ex[3:6].reshape(3, 1),
                                    mtx, dist)
        return proj.reshape(2)

    def _residual(self, ex_j, in_j, ex_l, in_l) -> np.ndarray | None:
        w = self._backproject(self.obs_j, ex_j, in_j)
        if w is None:
            return None
        proj_l = self._forward(w, ex_l, in_l)
        return self.weight * (proj_l - self.obs_l)

    def Evaluate(self, parameters: list, residuals: np.ndarray,
                 jacobians: list) -> bool:
        ex_j, in_j, ex_l, in_l = parameters
        r = self._residual(ex_j, in_j, ex_l, in_l)
        if r is None:
            residuals[:] = 0.0
            if jacobians is not None:
                for jac in jacobians:
                    if jac is not None:
                        jac[:] = 0.0
            return True
        residuals[:] = r

        if jacobians is not None:
            blocks = [ex_j, in_j, ex_l, in_l]
            sizes  = [N_EX, N_IN, N_EX, N_IN]
            for b, (block, sz) in enumerate(zip(blocks, sizes)):
                if jacobians[b] is None:
                    continue
                jac = np.zeros((2, sz), dtype=np.float64)
                for k in range(sz):
                    h = EPS * max(abs(block[k]), 1.0)
                    def perturbed(delta, bi=b, ki=k):
                        blks = [ex_j.copy(), in_j.copy(), ex_l.copy(), in_l.copy()]
                        blks[bi][ki] += delta
                        return self._residual(*blks)
                    rp = perturbed(+h)
                    rm = perturbed(-h)
                    if rp is None or rm is None:
                        jac[:, k] = 0.0
                    else:
                        jac[:, k] = (rp - rm) / (2 * h)
                jacobians[b][:] = jac.ravel()
        return True


def build_cross_view_observations(cameras: list[dict]) -> list[tuple]:
    """
    Find all (point, cam_j, cam_l) triplets where both cameras observe the
    same world point.  Returns list of (pt3d, obs_j, obs_l, cam_j_idx, cam_l_idx).
    """
    # index each camera's observations by world point key
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


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

CALIB_SOURCE_FILES = {
    # future mocap source will extend this dict (e.g. "mocap": ["camera_calib_mocap.json"])
    "checkerboard": ["camera_calib.json"],
}


def discover_cameras(data_dir: Path, requested: list[int],
                     calib_source: str = "checkerboard") -> list[int]:
    file_candidates = CALIB_SOURCE_FILES[calib_source]
    available = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("cam_"):
            continue
        try:
            cid = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        calib_dir = d / "calib"
        has_calib = any((calib_dir / f).exists() for f in file_candidates)
        if (calib_dir / "img_points.json").exists() and has_calib:
            available.append(cid)

    if requested:
        missing = [i for i in requested if i not in available]
        if missing:
            print(f"[WARN] Cameras not found / missing data: {missing}")
        return [i for i in requested if i in available]
    return available


def _warm_start_rvec(pts3d: np.ndarray, pts2d: np.ndarray,
                     mtx: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Use solvePnP(ITERATIVE) to get a sensible initial rvec/tvec.
    Falls back to zero-vectors on failure.
    """
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts3d.astype(np.float32),
            pts2d.astype(np.float32),
            mtx, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            return rvec.ravel(), tvec.ravel()
    except Exception:
        pass
    return np.zeros(3), np.zeros(3)


def _find_calib_file(base: Path, calib_source: str) -> Path:
    """Return the calibration JSON path for the requested source."""
    for fname in CALIB_SOURCE_FILES[calib_source]:
        p = base / fname
        if p.exists():
            return p
    names = ", ".join(CALIB_SOURCE_FILES[calib_source])
    raise FileNotFoundError(
        f"No calibration file found for source '{calib_source}' in {base}. "
        f"Tried: {names}"
    )


def load_camera(data_dir: Path, cam_id: int, val_frac: float = 0.0,
                calib_source: str = "checkerboard") -> dict:

    base = data_dir / f"cam_{cam_id}" / "calib"
    assert base.exists(), f"Missing camera directory for cam_{cam_id}"

    calib_file = _find_calib_file(base, calib_source)

    with open(calib_file) as f:
        cal = json.load(f)
    with open(base / "img_points.json") as f:
        raw = json.load(f)

    mtx   = np.array(cal["mtx"],  dtype=np.float64)
    dist  = np.array(cal["dist"], dtype=np.float64).ravel()
    pts3d = np.array(raw["real_corners"], dtype=np.float64)
    pts2d = np.array(raw["img_corners"],  dtype=np.float64)

    # held-out split — reproducible via fixed seed
    rng  = np.random.default_rng(seed=42 + cam_id)
    idx  = rng.permutation(len(pts3d))
    n_val = max(1, round(len(pts3d) * val_frac)) if val_frac > 0 else 0
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    rvec_init, tvec_init = _warm_start_rvec(pts3d[train_idx], pts2d[train_idx], mtx, dist)

    # keep stored tvec if solvePnP gives a worse starting error
    stored_tvec = np.array(cal["tvecs"], dtype=np.float64).ravel()
    proj_init   = _project(pts3d[train_idx], rvec_init, tvec_init,   mtx, dist)
    proj_stored = _project(pts3d[train_idx], rvec_init, stored_tvec, mtx, dist)
    err_init   = float(np.mean(np.linalg.norm(proj_init   - pts2d[train_idx], axis=1)))
    err_stored = float(np.mean(np.linalg.norm(proj_stored - pts2d[train_idx], axis=1)))

    cam = {
        "cam_id":   cam_id,
        "mtx":      mtx,
        "dist":     dist,
        "rvec":     rvec_init,
        "tvec":     tvec_init if err_init <= err_stored else stored_tvec,
        "pts3d":    pts3d[train_idx],
        "pts2d":    pts2d[train_idx],
        "pts3d_val": pts3d[val_idx],
        "pts2d_val": pts2d[val_idx],
    }
    if "w" in cal:
        cam["w"] = int(cal["w"])
    if "h" in cal:
        cam["h"] = int(cal["h"])
    return cam


# ──────────────────────────────────────────────────────────────────────────────
# PROJECTION
# ──────────────────────────────────────────────────────────────────────────────

def _project(pts3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
             mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Project 3-D points to image plane. Returns (N, 2)."""
    proj, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        mtx,
        dist,
    )
    return proj.reshape(-1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER BLOCKS  (owned by the caller, mutated in-place by Ceres)
# ──────────────────────────────────────────────────────────────────────────────

def make_param_blocks(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (ex[6], intr[9]) as contiguous float64 arrays."""
    ex   = np.ascontiguousarray(
               np.concatenate([cam["rvec"], cam["tvec"]]), dtype=np.float64)
    m, d = cam["mtx"], cam["dist"]
    intr = np.ascontiguousarray(
               [m[0,0], m[1,1], m[0,2], m[1,2],
                d[0], d[1], d[2], d[3], d[4]], dtype=np.float64)
    return ex, intr


def apply_param_blocks(cam: dict, ex: np.ndarray, intr: np.ndarray) -> dict:
    """Write Ceres-optimized parameter blocks back into a camera dict."""
    cam = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in cam.items()}
    cam["rvec"] = ex[0:3].copy()
    cam["tvec"] = ex[3:6].copy()
    cam["mtx"]  = np.array([[intr[0], 0, intr[2]],
                              [0, intr[1], intr[3]],
                              [0,       0,       1]], dtype=np.float64)
    cam["dist"] = intr[4:9].copy()
    return cam


# ──────────────────────────────────────────────────────────────────────────────
# REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def compute_stats(cameras: list[dict]) -> dict:
    """Compute per-camera and global error stats (2-D pixels and 3-D metres)."""
    stats  = {}
    all_3d = []
    all_2d = []

    for c in cameras:
        pts3d, pts2d = c["pts3d"], c["pts2d"]
        ex, intr = make_param_blocks(c)
        mtx, dist = _unpack_intr(intr)

        proj    = _project(pts3d, ex[0:3], ex[3:6], mtx, dist)
        errs_2d = np.linalg.norm(proj - pts2d, axis=1)

        errs_3d = []
        for i in range(len(pts2d)):
            est = _backproject_to_z0(pts2d[i], ex, intr)
            if est is not None:
                errs_3d.append(np.linalg.norm(est - pts3d[i, :2]))
        errs_3d = np.array(errs_3d) if errs_3d else np.array([0.0])

        stats[f"cam_{c['cam_id']}"] = {
            "mean_m":  float(np.mean(errs_3d)),
            "std_m":   float(np.std(errs_3d)),
            "min_m":   float(np.min(errs_3d)),
            "max_m":   float(np.max(errs_3d)),
            "mean_px": float(np.mean(errs_2d)),
            "std_px":  float(np.std(errs_2d)),
            "min_px":  float(np.min(errs_2d)),
            "max_px":  float(np.max(errs_2d)),
            "n_pts":   int(len(errs_2d)),
        }
        all_3d.extend(errs_3d.tolist())
        all_2d.extend(errs_2d.tolist())

    stats["GLOBAL"] = {
        "mean_m":  float(np.mean(all_3d)),
        "std_m":   float(np.std(all_3d)),
        "min_m":   float(np.min(all_3d)),
        "max_m":   float(np.max(all_3d)),
        "mean_px": float(np.mean(all_2d)),
        "std_px":  float(np.std(all_2d)),
        "min_px":  float(np.min(all_2d)),
        "max_px":  float(np.max(all_2d)),
        "n_pts":   int(len(all_2d)),
    }
    return stats


def compute_val_stats(cameras: list[dict]) -> dict | None:
    """
    Same as compute_stats but evaluated on the held-out validation split.
    Returns None if no camera has validation points (val_frac=0).
    """
    all_3d, all_2d = [], []
    stats = {}
    any_val = False

    for c in cameras:
        pts3d = c.get("pts3d_val")
        pts2d = c.get("pts2d_val")
        if pts3d is None or len(pts3d) == 0:
            continue
        any_val = True
        ex, intr = make_param_blocks(c)
        mtx, dist = _unpack_intr(intr)

        proj    = _project(pts3d, ex[0:3], ex[3:6], mtx, dist)
        errs_2d = np.linalg.norm(proj - pts2d, axis=1)

        errs_3d = []
        for i in range(len(pts2d)):
            est = _backproject_to_z0(pts2d[i], ex, intr)
            if est is not None:
                errs_3d.append(np.linalg.norm(est - pts3d[i, :2]))
        errs_3d = np.array(errs_3d) if errs_3d else np.array([0.0])

        stats[f"cam_{c['cam_id']}"] = {
            "mean_m":  float(np.mean(errs_3d)),
            "std_m":   float(np.std(errs_3d)),
            "min_m":   float(np.min(errs_3d)),
            "max_m":   float(np.max(errs_3d)),
            "mean_px": float(np.mean(errs_2d)),
            "std_px":  float(np.std(errs_2d)),
            "min_px":  float(np.min(errs_2d)),
            "max_px":  float(np.max(errs_2d)),
            "n_pts":   int(len(errs_2d)),
        }
        all_3d.extend(errs_3d.tolist())
        all_2d.extend(errs_2d.tolist())

    if not any_val:
        return None

    stats["GLOBAL"] = {
        "mean_m":  float(np.mean(all_3d)),
        "std_m":   float(np.std(all_3d)),
        "min_m":   float(np.min(all_3d)),
        "max_m":   float(np.max(all_3d)),
        "mean_px": float(np.mean(all_2d)),
        "std_px":  float(np.std(all_2d)),
        "min_px":  float(np.min(all_2d)),
        "max_px":  float(np.max(all_2d)),
        "n_pts":   int(len(all_2d)),
    }
    return stats


def print_val_summary(train_stats: dict, val_stats: dict) -> None:
    """Print train vs val error side-by-side to spot overfitting."""
    W = 100
    print(f"\n{'─'*W}")
    print("  Train vs Val  (held-out generalisation check)")
    print(f"{'─'*W}")
    print(f"  {'':12s}  {'train mean_3d':>13s}  {'val mean_3d':>11s}  {'Δ3d':>8s}    "
          f"{'train mean_2d':>13s}  {'val mean_2d':>11s}  {'Δ2d':>8s}  n_val")
    for k in val_stats:
        t = train_stats.get(k)
        v = val_stats[k]
        if t is None:
            continue
        d3 = v["mean_m"]  - t["mean_m"]
        d2 = v["mean_px"] - t["mean_px"]
        flag3 = "⚠" if d3 > 0.05 else " "
        flag2 = "⚠" if d2 > 3.0  else " "
        print(f"  {k:12s}  {t['mean_m']:13.4f}  {v['mean_m']:11.4f}  {d3:+8.4f} m {flag3}  "
              f"{t['mean_px']:13.2f}  {v['mean_px']:11.2f}  {d2:+8.2f} px {flag2}  {v['n_pts']}")
    print(f"{'─'*W}")


def print_stats(label: str, stats: dict) -> None:
    W = 100
    print(f"\n{'─'*W}")
    print(f"  {label}")
    print(f"{'─'*W}")
    print(f"  {'':12s}  {'── 3-D world error (m) ──':^36s}    {'── 2-D reprojection error (px) ──':^42s}  n")
    print(f"  {'':12s}  {'mean':>8s} {'std':>8s} {'min':>8s} {'max':>8s}    {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for k, s in stats.items():
        print(f"  {k:12s}  {s['mean_m']:8.4f} {s['std_m']:8.4f} {s['min_m']:8.4f} {s['max_m']:8.4f} m"
              f"  {s['mean_px']:9.2f} {s['std_px']:9.2f} {s['min_px']:9.2f} {s['max_px']:9.2f} px"
              f"  {s['n_pts']}")


def print_improvement(before: dict, after: dict) -> None:
    W = 100
    print(f"\n{'─'*W}")
    print("  Improvement per camera  (mean ± std  before → after)")
    print(f"{'─'*W}")
    for k in before:
        if k == "GLOBAL":
            continue
        bm, bs, am, as_ = before[k]["mean_m"], before[k]["std_m"], after[k]["mean_m"], after[k]["std_m"]
        bp, bsp, ap, asp = before[k]["mean_px"], before[k]["std_px"], after[k]["mean_px"], after[k]["std_px"]
        tm = "▼" if am < bm else "▲"
        tp = "▼" if ap < bp else "▲"
        print(f"  {k:12s}  3D: {bm:.4f}±{bs:.4f} → {am:.4f}±{as_:.4f} m {tm} ({bm-am:+.4f})    "
              f"2D: {bp:.2f}±{bsp:.2f} → {ap:.2f}±{asp:.2f} px {tp} ({bp-ap:+.2f})")
    bm, bs = before["GLOBAL"]["mean_m"],  before["GLOBAL"]["std_m"]
    am, as_ = after["GLOBAL"]["mean_m"],  after["GLOBAL"]["std_m"]
    bp, bsp = before["GLOBAL"]["mean_px"], before["GLOBAL"]["std_px"]
    ap, asp = after["GLOBAL"]["mean_px"],  after["GLOBAL"]["std_px"]
    print(f"\n  {'GLOBAL':12s}  3D: {bm:.4f}±{bs:.4f} → {am:.4f}±{as_:.4f} m {'▼' if am<bm else '▲'} ({bm-am:+.4f})    "
          f"2D: {bp:.2f}±{bsp:.2f} → {ap:.2f}±{asp:.2f} px {'▼' if ap<bp else '▲'} ({bp-ap:+.2f})")


# ──────────────────────────────────────────────────────────────────────────────
# SAVING
# ──────────────────────────────────────────────────────────────────────────────

def save_results(cameras: list[dict], data_dir: Path,
                 suffix: str, opt: str, calib_source: str = "checkerboard") -> None:
    # Outputs land in  data/cameras/cam_<id>/ba/camera_calib_ba.json
    fname = f"camera_calib{suffix}.json"
    for c in cameras:
        calib_base = data_dir / f"cam_{c['cam_id']}" / "calib"
        src = _find_calib_file(calib_base, calib_source)
        with open(src) as f:
            doc = json.load(f)
        if opt in ("ex", "both"):
            doc["rvecs"] = [[float(v)] for v in c["rvec"]]
            doc["tvecs"] = [[float(v)] for v in c["tvec"]]
        if opt in ("in", "both"):
            doc["mtx"]  = [list(map(float, row)) for row in c["mtx"]]
            doc["dist"] = [list(map(float, c["dist"]))]
        out_dir = data_dir / f"cam_{c['cam_id']}" / "ba"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / fname
        with open(out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def _run_ceres(cameras: list[dict], optimize: str,
               lam_intr_reg: float, lam_cross: float,
               label: str, huber_thresh: float) -> list[dict]:
    """Build and solve one Ceres problem. Returns optimised camera list."""
    n_res = sum(2 * len(c["pts3d"]) for c in cameras)
    n_par = len(cameras) * (N_EX + N_IN)
    print(f"\nBuilding Ceres problem — {label}  (Huber loss, DENSE_SCHUR)")
    print(f"  Residual blocks : {n_res // 2}  ({n_res} scalar residuals)")
    print(f"  Parameter blocks: {len(cameras) * 2}  ({n_par} scalars)")

    param_blocks = [make_param_blocks(c) for c in cameras]

    loss    = pyceres.HuberLoss(huber_thresh)
    problem = pyceres.Problem()

    for i, cam in enumerate(cameras):
        ex, intr = param_blocks[i]
        for pt3d, pt2d in zip(cam["pts3d"], cam["pts2d"]):
            cost = ReprojectionCost(pt3d, pt2d)
            problem.add_residual_block(cost, loss, [ex, intr])

        if optimize == "in":
            problem.set_parameter_block_constant(ex)
        elif optimize == "ex":
            problem.set_parameter_block_constant(intr)
        elif optimize == "both" and lam_intr_reg > 0.0:                
            reg = IntrinsicRegularisationCost(intr.copy(), lam_intr_reg)
            problem.add_residual_block(reg, None, [intr])

    if lam_cross > 0.0:
        triplets = build_cross_view_observations(cameras)
        print(f"  Cross-view pairs: {len(triplets)}  (weight={lam_cross})")
        for pt3d, obs_j, obs_l, j, l in triplets:
            ex_j, in_j = param_blocks[j]
            ex_l, in_l = param_blocks[l]
            cost = CrossViewConsistencyCost(pt3d, obs_j, obs_l, weight=lam_cross)
            problem.add_residual_block(cost, loss, [ex_j, in_j, ex_l, in_l])
            cost2 = CrossViewConsistencyCost(pt3d, obs_l, obs_j, weight=lam_cross)
            problem.add_residual_block(cost2, loss, [ex_l, in_l, ex_j, in_j])

    options = pyceres.SolverOptions()
    options.linear_solver_type             = pyceres.LinearSolverType.DENSE_SCHUR
    options.trust_region_strategy_type     = pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    options.max_num_iterations             = 500
    options.function_tolerance             = 1e-8
    options.gradient_tolerance             = 1e-10
    options.parameter_tolerance            = 1e-8
    options.minimizer_progress_to_stdout   = True
    options.num_threads                    = os.cpu_count() or 1

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    print(summary.BriefReport())

    return [
        apply_param_blocks(cam, ex, intr)
        for cam, (ex, intr) in zip(cameras, param_blocks)
    ]


def print_physical_summary(cameras: list[dict]) -> None:
    W = 100
    print(f"\n{'─'*W}")
    print("  Physical plausibility of calibrated parameters")
    print(f"{'─'*W}")
    print(f"  {'':8s}  {'fov_h':>6s}  {'cx':>5s}  {'cy':>5s}  "
          f"{'k1':>7s}  {'‖t‖':>6s}  {'tilt':>6s}  notes")
    for c in cameras:
        m, d = c["mtx"], c["dist"]
        fx, fy = m[0, 0], m[1, 1]
        cx, cy = m[0, 2], m[1, 2]
        # Use stored image size when available; fall back to 2*principal-point estimate.
        iw = c.get("w", round(cx * 2))
        ih = c.get("h", round(cy * 2))
        fov_h  = 2 * np.degrees(np.arctan(iw / (2 * fx)))
        cx_pct = cx / iw * 100
        cy_pct = cy / ih * 100
        R, _ = cv2.Rodrigues(c["rvec"].reshape(3, 1))
        optical_axis = R.T @ np.array([0.0, 0.0, 1.0])
        tilt   = 90.0 - np.degrees(np.arccos(np.clip(abs(optical_axis[2]), 0, 1)))
        dist_m = np.linalg.norm(c["tvec"])
        notes = []
        if cx_pct < 30 or cx_pct > 70:   notes.append(f"cx off-centre ({cx_pct:.0f}%)")
        if cy_pct < 20 or cy_pct > 80:   notes.append(f"cy off-centre ({cy_pct:.0f}%)")
        if abs(fx / fy - 1) > 0.10:      notes.append(f"fx/fy={fx/fy:.2f}")
        if abs(d[0]) > 1.0:              notes.append(f"k1={d[0]:.3f} large")
        if dist_m < 2 or dist_m > 80:    notes.append(f"‖t‖={dist_m:.1f}m implausible")
        if tilt < 10:                    notes.append("near-horizontal view")
        note_str = "  ⚠ " + ", ".join(notes) if notes else ""
        print(f"  cam_{c['cam_id']:<4d}  {fov_h:6.1f}°  {cx_pct:4.1f}%  {cy_pct:4.1f}%  "
              f"{d[0]:7.4f}  {dist_m:6.1f}m  {tilt:5.1f}°{note_str}")
    print(f"{'─'*W}")


def main(
    camera_ids: list[int],
    optimize: str,
    data_dir: Path,
    output_suffix: str,
    lambd: LAMBD,
    two_pass: bool,
    huber_thresh: float,
    val_frac: float,
    calib_source: str = "checkerboard",
) -> None:

    assert optimize in ("ex", "in", "both"), \
        f"--optimize must be 'ex', 'in', or 'both', got '{optimize}'"
    assert calib_source in CALIB_SOURCE_FILES, \
        f"--calib-source must be one of {list(CALIB_SOURCE_FILES)}, got '{calib_source}'"

    cam_ids = discover_cameras(data_dir, camera_ids, calib_source=calib_source)
    if not cam_ids:
        sys.exit("[ERROR] No cameras with required data found.")

    print(f"\nCameras selected : {cam_ids}")
    print(f"Optimize         : {optimize}")
    print(f"Calib source     : {calib_source}")
    print(f"Two-pass         : {two_pass and optimize == 'both'}")
    print(f"λ reproj         : {huber_thresh} px  (Huber threshold)")
    if optimize == "both":
        print(f"λ intr_reg       : {lambd.INTR_REG}")
    print(f"λ cross_view     : {lambd.CROSS_VIEW}  ({'enabled' if lambd.CROSS_VIEW > 0 else 'disabled'})")
    print(f"Val fraction     : {val_frac:.0%}  ({'enabled' if val_frac > 0 else 'disabled'})")
    print(f"Data directory   : {data_dir}")

    cameras = [load_camera(data_dir, cid, val_frac=val_frac,
                           calib_source=calib_source) for cid in cam_ids]

    stats_before = compute_stats(cameras)
    print_stats("BEFORE bundle adjustment  (after solvePnP warm-start)", stats_before)

    if two_pass and optimize == "both":
        cameras = _run_ceres(cameras, "ex", lambd.INTR_REG, lambd.CROSS_VIEW,
                             "pass 1/2 — extrinsics only", huber_thresh=huber_thresh)
        stats_pass1 = compute_stats(cameras)
        print_stats("AFTER pass 1 (extrinsics only)", stats_pass1)
        cameras_opt = _run_ceres(cameras, "both", lambd.INTR_REG, lambd.CROSS_VIEW,
                                 "pass 2/2 — extrinsics + intrinsics", huber_thresh=huber_thresh)
    else:
        cameras_opt = _run_ceres(cameras, optimize, lambd.INTR_REG, lambd.CROSS_VIEW,
                                 "single pass", huber_thresh=huber_thresh)

    stats_after = compute_stats(cameras_opt)
    print_physical_summary(cameras_opt)
    print_stats("AFTER bundle adjustment  (train set)", stats_after)
    print_improvement(stats_before, stats_after)

    val_stats = compute_val_stats(cameras_opt)
    if val_stats is not None:
        print_stats("AFTER bundle adjustment  (val set — held-out)", val_stats)
        print_val_summary(stats_after, val_stats)

    print(f"\nSaving (suffix='{output_suffix}') …")
    save_results(cameras_opt, data_dir, output_suffix, optimize,
                 calib_source=calib_source)
    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Bundle adjustment for multi-camera basketball court setup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--cameras", nargs="*", type=int, default=[],
        metavar="ID",
        help="Camera IDs to include (e.g. 1 2 12). Empty = all cameras.",
    )
    parser.add_argument(
        "--optimize", choices=("ex", "in", "both"), default="both",
        help="What to refine: extrinsics, intrinsics, or both.",
    )
    parser.add_argument(
        "--calib-source", choices=tuple(CALIB_SOURCE_FILES), default="checkerboard",
        metavar="SOURCE",
        help="Starting calibration source (currently only 'checkerboard').",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).parent / "data" / "camera_data",
        metavar="PATH",
        help="Root folder containing cam_*/calib/ sub-folders.",
    )
    parser.add_argument(
        "--suffix", default="_ba",
        metavar="SUFFIX",
        help="Suffix appended to output JSON (e.g. '_ba' → camera_calib_ba.json).",
    )
    parser.add_argument(
        "--huber-thresh", type=float, default=5.0,
        metavar="PX",
        help="Huber loss threshold in pixels — models annotation click accuracy.",
    )
    parser.add_argument(
        "--lambda-intr-reg", type=float, default=100.0,
        metavar="λ",
        help="λ for intrinsic regularisation toward checkerboard prior (--optimize both only).",
    )
    parser.add_argument(
        "--lambda-cross", type=float, default=1.0,
        metavar="λ",
        help="λ for cross-view consistency. 0 = disabled. Try 0.5–2.0.",
    )
    parser.add_argument(
        "--two-pass", action="store_true",
        help="Two-pass strategy: extrinsics-only first, then both (--optimize both only).",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.0,
        metavar="F",
        help="Fraction of keypoints per camera held out for validation (0 = disabled). e.g. 0.2",
    )
    
    args = parser.parse_args()

    main(
        camera_ids=args.cameras,
        optimize=args.optimize,
        data_dir=args.data_dir,
        output_suffix=args.suffix,
        lambd=LAMBD(INTR_REG=args.lambda_intr_reg, CROSS_VIEW=args.lambda_cross),
        two_pass=args.two_pass,
        huber_thresh=args.huber_thresh,
        val_frac=args.val_frac,
        calib_source=args.calib_source,
    )