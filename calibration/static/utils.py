"""Shared paths, data loading, geometry, scoring, and printing utilities."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BOARDS = {
    "big": (19, 13),
    "small": (6, 8),
}
MODEL_FLAGS = {
    "standard_5_term": 0,
    "fix_k3": cv2.CALIB_FIX_K3,
    "zero_tangent": cv2.CALIB_ZERO_TANGENT_DIST,
    "zero_tangent_fix_k3": (
        cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    ),
}
CAMERA_DIR_RE = re.compile(r"out(\d+)$")


@dataclass(frozen=True)
class CameraPaths:
    root: Path

    @property
    def camera_id(self) -> int:
        match = CAMERA_DIR_RE.fullmatch(self.root.name)
        if match is None:
            raise ValueError(f"Invalid camera directory: {self.root}")
        return int(match.group(1))

    @property
    def video(self) -> Path:
        return self.root / "checkerboard" / "vid.mp4"

    @property
    def detections(self) -> Path:
        return self.root / "checkerboard" / "det_cache.npz"

    @property
    def court_points(self) -> Path:
        return self.root / "annotations" / "court_points.json"

    @property
    def calibration_dir(self) -> Path:
        return self.root / "calibration"

    @property
    def static_calib(self) -> Path:
        return self.calibration_dir / "static_calib.json"

    @property
    def reference(self) -> Path:
        return self.artifacts_dir / "reference_calib.json"

    @property
    def sweep_state(self) -> Path:
        return self.artifacts_dir / "state.json"

    @property
    def sweep_runs(self) -> Path:
        return self.artifacts_dir / "sweeps"

    @property
    def diagnostics(self) -> Path:
        return self.artifacts_dir / "diagnostics"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts" / "static_calibration"


class Detection:
    __slots__ = ("fid", "board", "corners", "sharp", "cx", "cy", "span")

    def __init__(self, fid, board, corners, sharp, cx, cy, span):
        self.fid = int(fid)
        self.board = str(board)
        self.corners = np.asarray(corners, dtype=np.float32)
        self.sharp = float(sharp)
        self.cx = float(cx)
        self.cy = float(cy)
        self.span = float(span)


def discover_cameras(data_dir: Path) -> list[CameraPaths]:
    cameras = []
    if data_dir.is_dir():
        for path in data_dir.iterdir():
            if path.is_dir() and CAMERA_DIR_RE.fullmatch(path.name):
                cameras.append(CameraPaths(path))
    return sorted(cameras, key=lambda camera: camera.camera_id)


def select_cameras(data_dir: Path, camera_ids) -> list[CameraPaths]:
    if camera_ids is None:
        return discover_cameras(data_dir)
    return [
        CameraPaths(data_dir / f"out{camera_id}")
        for camera_id in dict.fromkeys(camera_ids)
    ]


def load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"File not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    temporary.replace(path)


def load_intrinsics(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    data = load_json(path)
    try:
        mtx = np.asarray(data["mtx"], dtype=np.float64)
        dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1, 1)
    except KeyError as exc:
        raise ValueError(f"{path} is missing key {exc.args[0]!r}") from None
    if mtx.shape != (3, 3):
        raise ValueError(f"{path}: mtx must be 3x3, got {mtx.shape}")
    if dist.size not in (4, 5, 8, 12, 14):
        raise ValueError(f"{path}: unsupported distortion size {dist.size}")
    if not np.isfinite(mtx).all() or not np.isfinite(dist).all():
        raise ValueError(f"{path}: intrinsic values must be finite")
    if mtx[0, 0] <= 0 or mtx[1, 1] <= 0:
        raise ValueError(f"{path}: focal lengths must be positive")
    return data, mtx, dist


def load_court_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_json(path)
    try:
        points_3d = np.asarray(data["real_corners"], dtype=np.float64)
        points_2d = np.asarray(data["img_corners"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"{path} is missing key {exc.args[0]!r}") from None
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"{path}: real_corners must have shape (N, 3)")
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(f"{path}: img_corners must have shape (N, 2)")
    if len(points_3d) != len(points_2d) or len(points_3d) < 4:
        raise ValueError(f"{path}: invalid correspondence count")
    if not np.isfinite(points_3d).all() or not np.isfinite(points_2d).all():
        raise ValueError(f"{path}: point coordinates must be finite")
    if np.linalg.matrix_rank(points_3d - points_3d.mean(axis=0)) < 2:
        raise ValueError(f"{path}: court points must not be collinear")
    return points_3d, points_2d


def load_detection_cache(
    path: Path,
) -> tuple[dict[int, Detection], set[int], tuple[int, int] | None]:
    if not path.exists():
        return {}, set(), None
    data = np.load(path, allow_pickle=True)
    detections = {
        int(data["fids"][index]): Detection(
            data["fids"][index],
            data["boards"][index],
            data["corners"][index],
            data["sharp"][index],
            data["cx"][index],
            data["cy"][index],
            data["span"][index],
        )
        for index in range(len(data["fids"]))
    }
    scanned = {int(frame) for frame in data["scanned"]}
    width, height = data["image_size"]
    return detections, scanned, (int(width), int(height))


def save_detection_cache(
    path: Path,
    detections: dict[int, Detection],
    scanned: set[int],
    image_size: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(detections.values(), key=lambda item: item.fid)
    np.savez_compressed(
        path,
        fids=np.array([item.fid for item in ordered], dtype=np.int64),
        corners=np.array([item.corners for item in ordered], dtype=object),
        boards=np.array([item.board for item in ordered]),
        sharp=np.array([item.sharp for item in ordered], dtype=np.float64),
        cx=np.array([item.cx for item in ordered], dtype=np.float64),
        cy=np.array([item.cy for item in ordered], dtype=np.float64),
        span=np.array([item.span for item in ordered], dtype=np.float64),
        scanned=np.array(sorted(scanned), dtype=np.int64),
        image_size=np.array(image_size, dtype=np.int64),
        boards_def=np.array(
            [f"{name}:{cols}x{rows}" for name, (cols, rows) in BOARDS.items()]
        ),
    )


def make_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_mm
    return points


def object_points_by_board(square_mm: float) -> dict[str, np.ndarray]:
    return {
        name: make_object_points(cols, rows, square_mm)
        for name, (cols, rows) in BOARDS.items()
    }


def select_views(
    detections,
    image_size: tuple[int, int],
    config: dict,
) -> list[Detection]:
    detections = list(detections)
    if not detections:
        return []
    floor = np.quantile(
        [item.sharp for item in detections], 1.0 - config["sharp_keep"]
    )
    kept = [item for item in detections if item.sharp >= floor]
    width, height = image_size
    gx, gy, gs = config["pose_grid"]
    bins = {}
    for item in kept:
        key = (
            item.board,
            min(int(item.cx / width * gx), gx - 1),
            min(int(item.cy / height * gy), gy - 1),
            min(int(item.span / (width / gs)), gs - 1),
        )
        bins.setdefault(key, []).append(item)
    selected = []
    for key in sorted(bins):
        selected.extend(
            sorted(bins[key], key=lambda item: (-item.sharp, item.fid))[
                : config["per_bin"]
            ]
        )
    max_views = config.get("max_views")
    if max_views and len(selected) > max_views:
        selected.sort(key=lambda item: item.fid)
        indices = np.linspace(0, len(selected) - 1, max_views, dtype=int)
        selected = [selected[index] for index in indices]
    return selected


def per_view_errors(views, object_sets, mtx, dist, rvecs, tvecs):
    errors = np.empty(len(views))
    for index, view in enumerate(views):
        projected, _ = cv2.projectPoints(
            object_sets[index], rvecs[index], tvecs[index], mtx, dist
        )
        residual = view.corners.reshape(-1, 2) - projected.reshape(-1, 2)
        errors[index] = np.sqrt(np.mean(np.sum(residual**2, axis=1)))
    return errors


def overall_rms(views, object_sets, mtx, dist, rvecs, tvecs) -> float:
    squared_error = 0.0
    count = 0
    for index, view in enumerate(views):
        projected, _ = cv2.projectPoints(
            object_sets[index], rvecs[index], tvecs[index], mtx, dist
        )
        residual = view.corners.reshape(-1, 2) - projected.reshape(-1, 2)
        squared_error += float(np.sum(residual**2))
        count += len(residual)
    return float(np.sqrt(squared_error / count))


def fit_intrinsics(
    views,
    image_size,
    object_points,
    config,
    with_uncertainty=False,
) -> dict:
    views = list(views)
    if len(views) < 12:
        raise ValueError("At least 12 checkerboard views are required")
    flags = MODEL_FLAGS[config["calibration_model"]]
    for _ in range(config["reject_iters"]):
        objects = [object_points[item.board] for item in views]
        images = [item.corners for item in views]
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objects, images, image_size, None, None, flags=flags
        )
        errors = per_view_errors(
            views, objects, mtx, dist, rvecs, tvecs
        )
        threshold = np.median(errors) + config["reject_sigma"] * errors.std()
        keep = np.flatnonzero(errors < threshold)
        if len(keep) == len(views) or len(keep) < 12:
            break
        views = [views[index] for index in keep]
    objects = [object_points[item.board] for item in views]
    images = [item.corners for item in views]
    uncertainty = None
    if with_uncertainty:
        result = cv2.calibrateCameraExtended(
            objects, images, image_size, None, None, flags=flags
        )
        _, mtx, dist, rvecs, tvecs, std_intrinsics = result[:6]
        std = std_intrinsics.ravel()
        uncertainty = {
            "fx": float(std[0] / mtx[0, 0] * 100.0),
            "fy": float(std[1] / mtx[1, 1] * 100.0),
        }
    else:
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objects, images, image_size, None, None, flags=flags
        )
    return {
        "mtx": mtx,
        "dist": dist,
        "checkerboard_rms_px": overall_rms(
            views, objects, mtx, dist, rvecs, tvecs
        ),
        "n_views": len(views),
        "focal_uncertainty_pct": uncertainty,
    }


def benchmark_views(detections, count=60):
    ordered = sorted(detections, key=lambda item: (item.board, item.fid))
    selected = []
    for board in BOARDS:
        board_views = [item for item in ordered if item.board == board]
        n_views = min(len(board_views), max(1, count // len(BOARDS)))
        indices = np.linspace(0, len(board_views) - 1, n_views, dtype=int)
        selected.extend(board_views[index] for index in indices)
    return selected


def checkerboard_rms(views, object_points, mtx, dist) -> float:
    squared_error = 0.0
    count = 0
    for view in views:
        objects = object_points[view.board]
        ok, rvec, tvec = cv2.solvePnP(
            objects,
            view.corners,
            mtx,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError(f"solvePnP failed for frame {view.fid}")
        projected, _ = cv2.projectPoints(objects, rvec, tvec, mtx, dist)
        residual = projected.reshape(-1, 2) - view.corners.reshape(-1, 2)
        squared_error += float(np.sum(residual**2))
        count += len(residual)
    return float(np.sqrt(squared_error / count))


def estimate_extrinsics(points_3d, points_2d, mtx, dist):
    centred = points_3d - points_3d.mean(axis=0)
    planar = np.linalg.matrix_rank(centred, tol=1e-9) == 2
    candidates = []
    if planar:
        result = cv2.solvePnPGeneric(
            points_3d, points_2d, mtx, dist, flags=cv2.SOLVEPNP_IPPE
        )
        if result[0]:
            candidates.extend(zip(result[1], result[2]))
    else:
        ok, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            mtx,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            candidates.append((rvec, tvec))
    if not candidates:
        raise RuntimeError("OpenCV could not estimate a camera pose")
    ranked = []
    for rvec, tvec in candidates:
        projected, _ = cv2.projectPoints(
            points_3d, rvec, tvec, mtx, dist
        )
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - points_2d, axis=1
        )
        rotation, _ = cv2.Rodrigues(rvec)
        depths = (rotation @ points_3d.T + tvec.reshape(3, 1))[2]
        ranked.append(
            (not bool(np.all(depths > 0)), float(np.sqrt(np.mean(errors**2))),
             rvec, tvec)
        )
    _, _, rvec, tvec = min(ranked, key=lambda item: item[:2])
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            points_3d, points_2d, mtx, dist, rvec, tvec
        )
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, mtx, dist)
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - points_2d, axis=1
    )
    return rvec.reshape(3, 1), tvec.reshape(3, 1), errors


def court_score(points_path: Path, mtx, dist) -> dict:
    points_3d, points_2d = load_court_points(points_path)
    rvec, tvec, errors = estimate_extrinsics(
        points_3d, points_2d, mtx, dist
    )
    rotation, _ = cv2.Rodrigues(rvec)
    centre = (-rotation.T @ tvec).ravel()
    return {
        "rms_px": float(np.sqrt(np.mean(errors**2))),
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
        "rvec": rvec,
        "tvec": tvec,
        "camera_centre": centre,
    }


def calibration_document(mtx, dist, pose=None) -> dict:
    return {
        "mtx": np.asarray(mtx).tolist(),
        "dist": np.asarray(dist).reshape(1, -1).tolist(),
        "rvecs": None if pose is None else pose["rvec"].reshape(3, 1).tolist(),
        "tvecs": None if pose is None else pose["tvec"].reshape(3, 1).tolist(),
    }


def reference_comparison(paths, current_mtx, current_dist, benchmark) -> dict | None:
    if not paths.reference.exists():
        return None
    _, ref_mtx, ref_dist = load_intrinsics(paths.reference)
    result = {
        "intrinsic_rms_px": checkerboard_rms(
            benchmark["views"], benchmark["objects"], ref_mtx, ref_dist
        ),
        "fx_delta_pct": float((ref_mtx[0, 0] / current_mtx[0, 0] - 1) * 100),
        "fy_delta_pct": float((ref_mtx[1, 1] / current_mtx[1, 1] - 1) * 100),
        "cx_delta_px": float(ref_mtx[0, 2] - current_mtx[0, 2]),
        "cy_delta_px": float(ref_mtx[1, 2] - current_mtx[1, 2]),
    }
    if paths.court_points.exists():
        result["court"] = court_score(paths.court_points, ref_mtx, ref_dist)
    return result


def print_header(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def print_calibration_result(label, fit, intrinsic_rms, court, reference) -> None:
    mtx = fit["mtx"]
    print(f"{label}")
    print(
        f"  views={fit['n_views']} train RMS={fit['checkerboard_rms_px']:.3f}px "
        f"validation RMS={intrinsic_rms:.3f}px"
    )
    print(
        f"  fx={mtx[0,0]:.2f} fy={mtx[1,1]:.2f} "
        f"cx={mtx[0,2]:.2f} cy={mtx[1,2]:.2f}"
    )
    if court is not None:
        print(
            f"  court RMS={court['rms_px']:.3f}px "
            f"mean={court['mean_px']:.3f}px max={court['max_px']:.3f}px"
        )
        centre = court["camera_centre"]
        print(
            f"  camera centre=[{centre[0]:.3f}, {centre[1]:.3f}, "
            f"{centre[2]:.3f}]"
        )
    if reference is not None:
        print(
            f"  reference checkerboard RMS={reference['intrinsic_rms_px']:.3f}px "
            f"delta={intrinsic_rms-reference['intrinsic_rms_px']:+.3f}px"
        )
        if "court" in reference and court is not None:
            ref_court = reference["court"]["rms_px"]
            print(
                f"  reference court RMS={ref_court:.3f}px "
                f"delta={court['rms_px']-ref_court:+.3f}px"
            )
