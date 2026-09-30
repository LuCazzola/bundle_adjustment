"""
Optimisation: cost functions, parameter block helpers, and the Ceres solver.

REPROJECTION COST MODES
-----------------------
  "2d"      Project known pt3d through the camera; penalise pixel error.
            Parameter blocks: [ex(6), in(9)]

  "3d_z0"   Back-project the observed pixel to the Z=0 world plane; penalise the
            3-D distance to the known pt3d[:2].
            Parameter blocks: [ex(6), in(9)]

  "3d_tri"  For every pair of cameras that share the same world point: triangulate
            their two pixel observations to a 3-D point; penalise the 3-D distance
            from that point to the known pt3d.
            Parameter blocks: [ex_j(6), in_j(9), ex_l(6), in_l(9)]

OBJECTIVE COMPOSITION
---------------------
Objectives are specified as a list of ObjectiveSpec.  Each spec independently
controls which data it applies to (court keypoints or mocap joints), which cost
mode is used, a scalar weight, and a Huber loss threshold.  This lets you stack
objectives freely — e.g. 2d court + 3d_tri court coupling + 2d mocap.

INTRINSIC REGULARISATION
-------------------------
Soft quadratic prior keeping each intrinsic close to its checkerboard value.
Used when optimize == "both" and lam_intr_reg > 0.
"""

import dataclasses
import os
from typing import Literal, Optional

import numpy as np
import cv2
import pyceres

from .geometry import unpack_intr, backproject_to_z0, build_cross_view_observations, triangulate_rays


N_IN = 9    # fx fy cx cy k1 k2 p1 p2 k3
N_EX = 6    # rx ry rz tx ty tz
EPS  = 1e-4  # relative FD step: h = EPS * max(|param|, 1.0)

Mode   = Literal["2d", "3d_z0", "3d_tri"]
Source = Literal["court", "mocap"]


@dataclasses.dataclass
class ObjectiveSpec:
    """
    One term in the BA objective.

    source      : "court" — uses pts3d/pts2d from each camera dict
                  "mocap" — uses pre-transformed world points from mocap_obs
    mode        : "2d" | "3d_z0" | "3d_tri"  (see module docstring)
    weight      : scalar multiplier on residuals (default 1.0)
    huber       : Huber loss threshold in the residual's native units
                  (pixels for "2d", metres for "3d_z0" / "3d_tri")
    """
    source: Source
    mode:   Mode
    weight: float = 1.0
    huber:  float = 5.0


@dataclasses.dataclass
class LAMBD:
    """Weights for the regularisation terms in the BA objective."""
    INTR_REG: float


@dataclasses.dataclass
class CeresOptions:
    """Ceres solver hyper-parameters exposed as BA configuration."""
    max_num_iterations: int   = 500
    function_tolerance: float = 1e-8
    gradient_tolerance: float = 1e-10
    parameter_tolerance: float = 1e-8
    # "dogleg" uses Powell's dogleg method instead of Levenberg-Marquardt
    trust_region_strategy: str = "lm"        # "lm" | "dogleg"
    # linear solver — "dense_schur" is the pyceres default for small-medium BA
    linear_solver_type: str    = "dense_schur"  # "dense_schur" | "sparse_schur" | "sparse_normal_cholesky" | "iterative_schur"


# ──────────────────────────────────────────────────────────────────────────────
# COST FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

class ReprojectionCost(pyceres.CostFunction):
    """
    Reprojection cost supporting three modes (see module docstring).

    mode="2d":
        Residuals (2): weight * [u_proj - u_obs, v_proj - v_obs]  pixels
        Param blocks:  [ex(6), in(9)]

    mode="3d_z0":
        Residuals (2): weight * (backproj_to_z0(pixel) - pt3d[:2])  metres
        Param blocks:  [ex(6), in(9)]

    mode="3d_tri":
        Residuals (3): weight * (triangulate(obs_j, obs_l) - pt3d)  metres
        Param blocks:  [ex_j(6), in_j(9), ex_l(6), in_l(9)]
        pt2d_other must be supplied.
    """

    def __init__(self,
                 pt3d: np.ndarray,
                 pt2d: np.ndarray,
                 mode: Mode = "2d",
                 weight: float = 1.0,
                 pt2d_other: np.ndarray | None = None) -> None:
        super().__init__()
        assert mode in ("2d", "3d_z0", "3d_tri"), f"Unknown mode '{mode}'"
        if mode == "3d_tri":
            assert pt2d_other is not None, "mode='3d_tri' requires pt2d_other"

        self.pt3d   = pt3d.astype(np.float64)
        self.obs    = pt2d.astype(np.float64)
        self.mode   = mode
        self.weight = float(weight)
        self.obs_l  = pt2d_other.astype(np.float64) if pt2d_other is not None else None

        if mode in ("2d", "3d_z0"):
            self.set_num_residuals(2)
            self.set_parameter_block_sizes([N_EX, N_IN])
        else:  # 3d_tri
            self.set_num_residuals(3)
            self.set_parameter_block_sizes([N_EX, N_IN, N_EX, N_IN])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _project(self, pt3d, ex, intr):
        mtx, dist = unpack_intr(intr)
        proj, jac = cv2.projectPoints(
            pt3d.reshape(1, 1, 3),
            ex[0:3].reshape(3, 1), ex[3:6].reshape(3, 1),
            mtx, dist,
        )
        return proj.reshape(2), jac[:, 0:3], jac[:, 3:6]

    def _triangulate(self, obs_j, ex_j, in_j, obs_l, ex_l, in_l):
        cam_j = {"rvec": ex_j[0:3], "tvec": ex_j[3:6],
                  "mtx": unpack_intr(in_j)[0], "dist": unpack_intr(in_j)[1]}
        cam_l = {"rvec": ex_l[0:3], "tvec": ex_l[3:6],
                  "mtx": unpack_intr(in_l)[0], "dist": unpack_intr(in_l)[1]}
        return triangulate_rays({0: obs_j, 1: obs_l}, [cam_j, cam_l])

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def Evaluate(self, parameters, residuals, jacobians):
        w = self.weight

        if self.mode == "2d":
            ex, intr = parameters[0], parameters[1]
            proj, j_rvec, j_tvec = self._project(self.pt3d, ex, intr)
            residuals[0] = w * (proj[0] - self.obs[0])
            residuals[1] = w * (proj[1] - self.obs[1])
            if jacobians is None:
                return True
            if jacobians[0] is not None:
                jacobians[0][:] = (w * np.hstack([j_rvec, j_tvec])).ravel()
            if jacobians[1] is not None:
                J = np.zeros((2, N_IN), dtype=np.float64)
                for k in range(N_IN):
                    h = EPS * max(abs(intr[k]), 1.0)
                    ip = intr.copy(); ip[k] += h
                    im = intr.copy(); im[k] -= h
                    pp, _, _ = self._project(self.pt3d, ex, ip)
                    pm, _, _ = self._project(self.pt3d, ex, im)
                    J[:, k] = (pp - pm) / (2 * h)
                jacobians[1][:] = (w * J).ravel()
            return True

        if self.mode == "3d_z0":
            ex, intr = parameters[0], parameters[1]
            bp = backproject_to_z0(self.obs, ex, intr)
            if bp is None:
                residuals[:] = 0.0
                if jacobians is not None:
                    for jac in jacobians:
                        if jac is not None:
                            jac[:] = 0.0
                return True
            residuals[0] = w * (bp[0] - self.pt3d[0])
            residuals[1] = w * (bp[1] - self.pt3d[1])
            if jacobians is None:
                return True
            for b, (block, sz) in enumerate([(ex, N_EX), (intr, N_IN)]):
                if jacobians[b] is None:
                    continue
                J = np.zeros((2, sz), dtype=np.float64)
                for k in range(sz):
                    h = EPS * max(abs(block[k]), 1.0)
                    blks_p = [ex.copy(), intr.copy()]
                    blks_p[b][k] += h
                    blks_m = [ex.copy(), intr.copy()]
                    blks_m[b][k] -= h
                    bpp = backproject_to_z0(self.obs, blks_p[0], blks_p[1])
                    bpm = backproject_to_z0(self.obs, blks_m[0], blks_m[1])
                    if bpp is not None and bpm is not None:
                        J[:, k] = (np.array(bpp) - np.array(bpm)) / (2 * h)
                jacobians[b][:] = (w * J).ravel()
            return True

        # mode == "3d_tri"
        ex_j, in_j, ex_l, in_l = parameters
        tri = self._triangulate(self.obs, ex_j, in_j, self.obs_l, ex_l, in_l)
        if tri is None:
            residuals[:] = 0.0
            if jacobians is not None:
                for jac in jacobians:
                    if jac is not None:
                        jac[:] = 0.0
            return True
        residuals[0] = w * (tri[0] - self.pt3d[0])
        residuals[1] = w * (tri[1] - self.pt3d[1])
        residuals[2] = w * (tri[2] - self.pt3d[2])
        if jacobians is None:
            return True
        for b, (block, sz) in enumerate([(ex_j, N_EX), (in_j, N_IN),
                                          (ex_l, N_EX), (in_l, N_IN)]):
            if jacobians[b] is None:
                continue
            J = np.zeros((3, sz), dtype=np.float64)
            for k in range(sz):
                h = EPS * max(abs(block[k]), 1.0)
                blks_p = [ex_j.copy(), in_j.copy(), ex_l.copy(), in_l.copy()]
                blks_p[b][k] += h
                blks_m = [ex_j.copy(), in_j.copy(), ex_l.copy(), in_l.copy()]
                blks_m[b][k] -= h
                tp = self._triangulate(self.obs, blks_p[0], blks_p[1], self.obs_l, blks_p[2], blks_p[3])
                tm = self._triangulate(self.obs, blks_m[0], blks_m[1], self.obs_l, blks_m[2], blks_m[3])
                if tp is not None and tm is not None:
                    J[:, k] = (tp - tm) / (2 * h)
            jacobians[b][:] = (w * J).ravel()
        return True


class IntrinsicRegularisationCost(pyceres.CostFunction):
    """
    Soft quadratic prior keeping each intrinsic close to its checkerboard value.

        residual[k] = weight * (intr[k] - prior[k]) / |prior[k]|

    Parameters with a zero prior use scale=1 to avoid division-by-zero.
    """

    def __init__(self, prior: np.ndarray, weight: float) -> None:
        super().__init__()
        self.prior  = prior.copy()
        self.weight = weight
        self.scale  = np.where(np.abs(prior) > 1e-9, np.abs(prior), 1.0)
        self.set_num_residuals(N_IN)
        self.set_parameter_block_sizes([N_IN])

    def Evaluate(self, parameters, residuals, jacobians):
        intr = parameters[0]
        for k in range(N_IN):
            residuals[k] = self.weight * (intr[k] - self.prior[k]) / self.scale[k]
        if jacobians is not None and jacobians[0] is not None:
            J = np.zeros((N_IN, N_IN), dtype=np.float64)
            for k in range(N_IN):
                J[k, k] = self.weight / self.scale[k]
            jacobians[0][:] = J.ravel()
        return True


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER BLOCK HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def make_param_blocks(cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (ex[6], intr[9]) as contiguous float64 arrays."""
    ex   = np.ascontiguousarray(
               np.concatenate([cam["rvec"], cam["tvec"]]), dtype=np.float64)
    m, d = cam["mtx"], cam["dist"]
    intr = np.ascontiguousarray(
               [m[0, 0], m[1, 1], m[0, 2], m[1, 2],
                d[0], d[1], d[2], d[3], d[4]], dtype=np.float64)
    return ex, intr


def apply_param_blocks(cam: dict, ex: np.ndarray, intr: np.ndarray) -> dict:
    """Write Ceres-optimised parameter blocks back into a camera dict."""
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
# SIMILARITY ALIGNMENT
# ──────────────────────────────────────────────────────────────────────────────

def umeyama(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama similarity alignment: find s, R, t  s.t.  Y ≈ s * R @ X + t.
    X, Y: (N, 3) float64.  Returns (scale, R 3×3, t 3-vec).
    """
    N = X.shape[0]
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc   = X - muX, Y - muY
    SXY      = (Yc.T @ Xc) / N
    U, D, Vt = np.linalg.svd(SXY)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R    = U @ S @ Vt
    varX = (Xc ** 2).sum() / N
    s    = float(np.trace(np.diag(D) @ S) / varX) if varX > 1e-12 else 1.0
    t    = muY - s * (R @ muX)
    return s, R, t


# ──────────────────────────────────────────────────────────────────────────────
# SOLVER
# ──────────────────────────────────────────────────────────────────────────────

def run_ceres(
    cameras: list[dict],
    optimize: str,
    lam_intr_reg: float,
    label: str,
    objectives: list[ObjectiveSpec],
    mocap_obs: list[tuple[np.ndarray, np.ndarray, int]] | None = None,
    ceres_opts: Optional["CeresOptions"] = None,
) -> list[dict]:
    """
    Build and solve one Ceres problem. Returns optimised camera list.

    objectives  : list of ObjectiveSpec, each independently contributing
                  residuals with its own mode, weight, and Huber threshold.
                  "court" specs use pts3d/pts2d from camera dicts.
                  "mocap" specs use pt3d_world from mocap_obs.
    mocap_obs   : list of (pt3d_world [m], pt2d_pixel, cam_list_index).
                  Required when any ObjectiveSpec has source="mocap".
    """
    param_blocks = [make_param_blocks(c) for c in cameras]
    problem      = pyceres.Problem()

    # ── build one Huber loss per unique threshold ─────────────────────────────
    _loss_cache: dict[float, pyceres.HuberLoss] = {}
    def get_loss(h: float) -> pyceres.HuberLoss:
        if h not in _loss_cache:
            _loss_cache[h] = pyceres.HuberLoss(h)
        return _loss_cache[h]

    n_residuals: dict[str, int] = {}

    for spec in objectives:
        loss = get_loss(spec.huber)
        key  = f"{spec.source}/{spec.mode}(w={spec.weight},h={spec.huber})"

        if spec.source == "court":
            if spec.mode in ("2d", "3d_z0"):
                count = 0
                for i, cam in enumerate(cameras):
                    ex, intr = param_blocks[i]
                    for pt3d, pt2d in zip(cam["pts3d"], cam["pts2d"]):
                        cost = ReprojectionCost(pt3d, pt2d, mode=spec.mode, weight=spec.weight)
                        problem.add_residual_block(cost, loss, [ex, intr])
                        count += 1
                n_residuals[key] = count

            elif spec.mode == "3d_tri":
                triplets = build_cross_view_observations(cameras)
                for pt3d, obs_j, obs_l, j, l in triplets:
                    ex_j, in_j = param_blocks[j]
                    ex_l, in_l = param_blocks[l]
                    cost = ReprojectionCost(pt3d, obs_j, mode="3d_tri",
                                           weight=spec.weight, pt2d_other=obs_l)
                    problem.add_residual_block(cost, loss, [ex_j, in_j, ex_l, in_l])
                n_residuals[key] = len(triplets)

        elif spec.source == "mocap":
            if not mocap_obs:
                continue
            if spec.mode in ("2d", "3d_z0"):
                for pt3d_world, pt2d, cidx in mocap_obs:
                    ex, intr = param_blocks[cidx]
                    cost = ReprojectionCost(pt3d_world, pt2d, mode=spec.mode, weight=spec.weight)
                    problem.add_residual_block(cost, loss, [ex, intr])
                n_residuals[key] = len(mocap_obs)

            elif spec.mode == "3d_tri":
                # group mocap observations by (world point, camera pair)
                from collections import defaultdict
                pt_views: dict[tuple, dict[int, np.ndarray]] = defaultdict(dict)
                pt_anchor: dict[tuple, np.ndarray] = {}
                for pt3d_world, pt2d, cidx in mocap_obs:
                    key_pt = tuple(np.round(pt3d_world, 6))
                    pt_views[key_pt][cidx] = pt2d
                    pt_anchor[key_pt] = pt3d_world
                count = 0
                for key_pt, views in pt_views.items():
                    cidxs = list(views.keys())
                    for a in range(len(cidxs)):
                        for b in range(a + 1, len(cidxs)):
                            j, l = cidxs[a], cidxs[b]
                            ex_j, in_j = param_blocks[j]
                            ex_l, in_l = param_blocks[l]
                            cost = ReprojectionCost(
                                pt_anchor[key_pt], views[j], mode="3d_tri",
                                weight=spec.weight, pt2d_other=views[l],
                            )
                            problem.add_residual_block(cost, loss, [ex_j, in_j, ex_l, in_l])
                            count += 1
                n_residuals[key] = count

    # ── intrinsic regularisation / freeze ─────────────────────────────────────
    for i, cam in enumerate(cameras):
        ex, intr = param_blocks[i]
        if optimize == "in":
            problem.set_parameter_block_constant(ex)
        elif optimize == "ex":
            problem.set_parameter_block_constant(intr)
        elif optimize == "both" and lam_intr_reg > 0.0:
            reg = IntrinsicRegularisationCost(intr.copy(), lam_intr_reg)
            problem.add_residual_block(reg, None, [intr])

    opts = ceres_opts or CeresOptions()

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\nBuilding Ceres problem — {label}  ({opts.linear_solver_type})")
    for k, n in n_residuals.items():
        print(f"  {k} : {n} blocks")
    n_par = len(cameras) * (N_EX + N_IN)
    print(f"  Parameter blocks: {len(cameras) * 2}  ({n_par} scalars)")

    _LINEAR_SOLVER_MAP = {
        "dense_schur":            pyceres.LinearSolverType.DENSE_SCHUR,
        "sparse_schur":           pyceres.LinearSolverType.SPARSE_SCHUR,
        "sparse_normal_cholesky": pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY,
        "iterative_schur":        pyceres.LinearSolverType.ITERATIVE_SCHUR,
    }
    options = pyceres.SolverOptions()
    options.linear_solver_type           = _LINEAR_SOLVER_MAP.get(
        opts.linear_solver_type, pyceres.LinearSolverType.DENSE_SCHUR
    )
    options.trust_region_strategy_type   = (
        pyceres.TrustRegionStrategyType.DOGLEG
        if opts.trust_region_strategy == "dogleg"
        else pyceres.TrustRegionStrategyType.LEVENBERG_MARQUARDT
    )
    options.max_num_iterations           = opts.max_num_iterations
    options.function_tolerance           = opts.function_tolerance
    options.gradient_tolerance           = opts.gradient_tolerance
    options.parameter_tolerance          = opts.parameter_tolerance
    options.minimizer_progress_to_stdout = True
    options.num_threads                  = os.cpu_count() or 1

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    print(summary.BriefReport())

    return [
        apply_param_blocks(cam, ex, intr)
        for cam, (ex, intr) in zip(cameras, param_blocks)
    ]
