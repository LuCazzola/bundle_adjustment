"""
Camera data loading, calibration file discovery, projection utilities, and result saving.
"""

import json
from pathlib import Path

import numpy as np
import cv2

from .geometry import unpack_intr


CALIB_SOURCE_FILES = {
    "checkerboard": ["static_calib.json"],
}


def project(pts3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
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


def cam_root(data_dir: Path) -> Path:
    """Return the canonical directory that contains out<N>/ folders."""
    return data_dir / "cameras"


def warm_start_rvec(pts3d: np.ndarray, pts2d: np.ndarray,
                    mtx: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use solvePnP(ITERATIVE) to get a sensible initial rvec/tvec. Falls back to zeros."""
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


def find_calib_file(base: Path, calib_source: str) -> Path:
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


def discover_cameras(data_dir: Path, requested: list[int],
                     calib_source: str = "checkerboard") -> list[int]:
    file_candidates = CALIB_SOURCE_FILES[calib_source]
    root = cam_root(data_dir)
    available = []
    if not root.is_dir():
        return available
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("out"):
            continue
        try:
            cid = int(d.name[3:])
        except ValueError:
            continue
        calib_dir = d / "calibration"
        has_calib = any((calib_dir / f).exists() for f in file_candidates)
        if (d / "annotations" / "court_points.json").exists() and has_calib:
            available.append(cid)

    available.sort()
    if requested:
        missing = [i for i in requested if i not in available]
        if missing:
            print(f"[WARN] Cameras not found / missing data: {missing}")
        return [i for i in requested if i in available]
    return available


def load_camera(data_dir: Path, cam_id: int, val_frac: float = 0.0,
                calib_source: str = "checkerboard", seed: int = 42) -> dict:
    camera_dir = cam_root(data_dir) / f"out{cam_id}"
    base = camera_dir / "calibration"
    assert base.exists(), f"Missing camera directory for out{cam_id}"

    calib_file = find_calib_file(base, calib_source)

    with open(calib_file) as f:
        cal = json.load(f)
    with open(camera_dir / "annotations" / "court_points.json") as f:
        raw = json.load(f)

    mtx   = np.array(cal["mtx"],  dtype=np.float64)
    dist  = np.array(cal["dist"], dtype=np.float64).ravel()
    pts3d = np.array(raw["real_corners"], dtype=np.float64)
    pts2d = np.array(raw["img_corners"],  dtype=np.float64)

    rng   = np.random.default_rng(seed=seed + cam_id)
    idx   = rng.permutation(len(pts3d))
    n_val = max(1, round(len(pts3d) * val_frac)) if val_frac > 0 else 0
    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    rvec_init, tvec_init = warm_start_rvec(pts3d[train_idx], pts2d[train_idx], mtx, dist)

    stored_tvec  = np.array(cal["tvecs"], dtype=np.float64).ravel()
    proj_init    = project(pts3d[train_idx], rvec_init, tvec_init,   mtx, dist)
    proj_stored  = project(pts3d[train_idx], rvec_init, stored_tvec, mtx, dist)
    err_init     = float(np.mean(np.linalg.norm(proj_init   - pts2d[train_idx], axis=1)))
    err_stored   = float(np.mean(np.linalg.norm(proj_stored - pts2d[train_idx], axis=1)))

    cam = {
        "cam_id":    cam_id,
        "mtx":       mtx,
        "dist":      dist,
        "rvec":      rvec_init,
        "tvec":      tvec_init if err_init <= err_stored else stored_tvec,
        "pts3d":     pts3d[train_idx],
        "pts2d":     pts2d[train_idx],
        "pts3d_val": pts3d[val_idx],
        "pts2d_val": pts2d[val_idx],
    }
    if "w" in cal:
        cam["w"] = int(cal["w"])
    if "h" in cal:
        cam["h"] = int(cal["h"])
    return cam


def save_results(cameras: list[dict], data_dir: Path,
                 suffix: str, opt: str, calib_source: str = "checkerboard") -> None:
    fname    = f"ba_calib{suffix}.json"
    root     = cam_root(data_dir)
    for c in cameras:
        camera_base = root / f"out{c['cam_id']}"
        calib_base = camera_base / "calibration"
        src = find_calib_file(calib_base, calib_source)
        with open(src) as f:
            doc = json.load(f)
        if opt in ("ex", "both"):
            doc["rvecs"] = [[float(v)] for v in c["rvec"]]
            doc["tvecs"] = [[float(v)] for v in c["tvec"]]
        if opt in ("in", "both"):
            doc["mtx"]  = [list(map(float, row)) for row in c["mtx"]]
            doc["dist"] = [list(map(float, c["dist"]))]
        out_dir = (
            calib_base
            if not suffix
            else camera_base / "artifacts" / "bundle_adjustment"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / fname
        with open(out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"  Saved → {out}")
