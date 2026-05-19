"""
Multi-camera point picker — debug visualisation tool.

Click anywhere on one camera view:
  • the clicked point shows as a GREEN dot on that camera
  • the same 3-D world point projected through all other cameras shows as RED dots

Run:
    python app.py
then open http://localhost:5000
"""

import argparse
import io
import json
import os
import signal
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
CAM_DIR   = HERE / "data" / "cameras"
VIDEO_DIR = HERE / "data" / "video"

DISPLAY_W = 960
DISPLAY_H = 540
ORIG_W    = 3840
ORIG_H    = 2160

# All camera IDs with calibration data (order controls grid layout)
CAMERA_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _load_calib_file(cam_dir: Path, cam_id: int) -> dict | None:
    """
    Load order (best first):
      1. cameras/cam_<id>/ba/camera_calib_ba.json   — BA-refined, preferred
      2. cameras/cam_<id>/calib/camera_calib_real.json  — original fallback
      3. cameras/cam_<id>/calib/camera_calib.json       — original fallback
    """
    base = cam_dir / f"cam_{cam_id}"
    candidates = [
        (base / "ba"    / "camera_calib_ba.json",   "ba"),
        (base / "calib" / "camera_calib_real.json", "calib (original)"),
        (base / "calib" / "camera_calib.json",      "calib"),
    ]
    for path, label in candidates:
        if path.exists():
            with open(path) as f:
                cal = json.load(f)
            print(f"  cam_{cam_id}: loaded from {label}")
            return cal
    print(f"[WARN] No calibration file for cam_{cam_id}, skipping.")
    return None


def _load_moge_calib_file(cam_dir: Path, cam_id: int) -> dict | None:
    """Load MoGe-estimated calibration if present."""
    path = cam_dir / f"cam_{cam_id}" / "calib" / "camera_calib_moge.json"
    if not path.exists():
        return None
    with open(path) as f:
        cal = json.load(f)
    print(f"  cam_{cam_id}: loaded MoGe calib (reproj={cal.get('_reproj_px', '?')} px)")
    return cal


def _parse_cam(cal: dict) -> dict:
    return {
        "mtx":  np.array(cal["mtx"],  dtype=np.float64),
        "dist": np.array(cal["dist"], dtype=np.float64).ravel(),
        "rvec": np.array(cal["rvecs"], dtype=np.float64).ravel(),
        "tvec": np.array(cal["tvecs"], dtype=np.float64).ravel(),
    }


def load_calibration(cam_dir: Path, cam_ids: list[int]) -> dict:
    """Load BA-refined params (preferred) or original as fallback."""
    cameras = {}
    for cid in cam_ids:
        cal = _load_calib_file(cam_dir, cid)
        if cal is None:
            continue
        cameras[cid] = _parse_cam(cal)
    return cameras


def load_original_calibration(cam_dir: Path, cam_ids: list[int]) -> dict:
    """Load only the original (pre-BA) camera parameters.

    Raw files store tvec in mm; world points are in metres, so we tag each
    camera with world_scale=1000 and project accordingly.
    """
    cameras = {}
    for cid in cam_ids:
        base = cam_dir / f"cam_{cid}" / "calib"
        path = base / "camera_calib_real.json"
        if not path.exists():
            path = base / "camera_calib.json"
        if not path.exists():
            continue
        with open(path) as f:
            cal = json.load(f)
        parsed = _parse_cam(cal)
        parsed["world_scale"] = 1000.0 if np.max(np.abs(parsed["tvec"])) > 100 else 1.0
        cameras[cid] = parsed
    return cameras


def load_moge_calibration(cam_dir: Path, cam_ids: list[int]) -> dict:
    """Load MoGe-estimated camera parameters (zero distortion, solvePnP extrinsics)."""
    cameras = {}
    for cid in cam_ids:
        cal = _load_moge_calib_file(cam_dir, cid)
        if cal is None:
            continue
        cameras[cid] = _parse_cam(cal)
    return cameras


# ──────────────────────────────────────────────────────────────────────────────
# POINT DATA
# ──────────────────────────────────────────────────────────────────────────────

def _project_pts(pts3d: np.ndarray, cam: dict,
                 scale_x: float, scale_y: float) -> list[dict]:
    """Project (N,3) world points onto the distorted image; return {x,y} at display resolution."""
    world_scale = cam.get("world_scale", 1.0)
    p = pts3d * world_scale
    proj, _ = cv2.projectPoints(
        p.reshape(-1, 1, 3),
        cam["rvec"].reshape(3, 1),
        cam["tvec"].reshape(3, 1),
        cam["mtx"], cam["dist"],
    )
    proj = proj.reshape(-1, 2)
    return [
        {"x": round(float(ox) / scale_x, 1), "y": round(float(oy) / scale_y, 1)}
        for ox, oy in proj
    ]


def load_ground_truth(cam_dir: Path, cam_ids: list[int],
                      scale_x: float, scale_y: float) -> dict:
    """
    Load raw annotated img_points for each camera.
    Returns { cam_id: [ {x, y, label} ] } at display resolution.
    """
    gt = {}
    for cid in cam_ids:
        path = cam_dir / f"cam_{cid}" / "calib" / "img_points.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        points = []
        for (wx, wy, wz), (px, py) in zip(raw["real_corners"], raw["img_corners"]):
            points.append({
                "x":     round(px / scale_x, 1),
                "y":     round(py / scale_y, 1),
                "label": f"({wx:.1f},{wy:.1f})",
            })
        gt[cid] = points
    return gt


def compute_reproj(cam_dir: Path, cameras: dict, cam_ids: list[int],
                   scale_x: float, scale_y: float) -> dict:
    """
    Project each 3-D GT point through the given camera parameters.
    Returns { cam_id: [ {x, y, label, err} ] } at display resolution.
    err = reprojection error in original-image pixels.
    """
    result = {}
    for cid in cam_ids:
        if cid not in cameras:
            continue
        path = cam_dir / f"cam_{cid}" / "calib" / "img_points.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        cam   = cameras[cid]
        pts3d = np.array(raw["real_corners"], dtype=np.float64)
        pts2d = np.array(raw["img_corners"],  dtype=np.float64)
        proj  = _project_pts(pts3d, cam, scale_x, scale_y)
        # recompute in original pixels for error measurement
        world_scale = cam.get("world_scale", 1.0)
        p = pts3d * world_scale
        proj_orig, _ = cv2.projectPoints(
            p.reshape(-1, 1, 3),
            cam["rvec"].reshape(3, 1),
            cam["tvec"].reshape(3, 1),
            cam["mtx"], cam["dist"],
        )
        proj_orig = proj_orig.reshape(-1, 2)
        points = []
        for i, (wx, wy, wz) in enumerate(raw["real_corners"]):
            err = float(np.linalg.norm(proj_orig[i] - pts2d[i]))
            points.append({
                **proj[i],
                "label": f"({wx:.1f},{wy:.1f})",
                "err":   round(err, 1),
            })
        result[cid] = points
    return result


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY (click → world → project)
# ──────────────────────────────────────────────────────────────────────────────

def pixel_to_world(px: float, py: float, cam: dict,
                   scale_x: float, scale_y: float) -> np.ndarray | None:
    """Back-project a display pixel to the Z=0 world plane."""
    pts = np.array([[[px * scale_x, py * scale_y]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, cam["mtx"], cam["dist"])
    xn, yn = float(undist[0, 0, 0]), float(undist[0, 0, 1])

    R, _ = cv2.Rodrigues(cam["rvec"].reshape(3, 1))
    t    = cam["tvec"].reshape(3, 1)
    Rt   = R.T
    ray_world = Rt @ np.array([xn, yn, 1.0])
    origin    = (Rt @ (-t)).ravel()

    if abs(ray_world[2]) < 1e-9:
        return None
    lam = -origin[2] / ray_world[2]
    return origin + lam * ray_world


def world_to_pixel(world_pt: np.ndarray, cam: dict,
                   scale_x: float, scale_y: float) -> tuple[float, float] | None:
    """Project a world point onto the display image of a camera."""
    proj, _ = cv2.projectPoints(
        world_pt.reshape(1, 1, 3),
        cam["rvec"].reshape(3, 1),
        cam["tvec"].reshape(3, 1),
        cam["mtx"], cam["dist"],
    )
    ox, oy = float(proj[0, 0, 0]), float(proj[0, 0, 1])
    dx, dy = ox / scale_x, oy / scale_y
    if dx < 0 or dy < 0 or dx > DISPLAY_W or dy > DISPLAY_H:
        return None
    return dx, dy


def pixel_to_ray(px: float, py: float, cam: dict,
                 scale_x: float, scale_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, direction) of the ray through a display pixel, in world coords."""
    pts = np.array([[[px * scale_x, py * scale_y]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, cam["mtx"], cam["dist"])
    xn, yn = float(undist[0, 0, 0]), float(undist[0, 0, 1])

    R, _ = cv2.Rodrigues(cam["rvec"].reshape(3, 1))
    t    = cam["tvec"].reshape(3, 1)
    origin    = (R.T @ (-t)).ravel()
    direction = R.T @ np.array([xn, yn, 1.0])
    direction /= np.linalg.norm(direction)
    return origin, direction


def triangulate_rays(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Linear least-squares triangulation from N rays.
    Each ray: (origin o_i, unit direction d_i).
    Minimises sum of squared distances from the point to each ray.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)   # projection onto plane perpendicular to d
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


# ──────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────────────────────────────────────

def compute_mean_reproj(cam_dir: Path, cameras: dict, cam_ids: list[int]) -> dict:
    """Return {cam_id: mean_reproj_px} for cameras that have data."""
    result = {}
    for cid in cam_ids:
        if cid not in cameras:
            continue
        path = cam_dir / f"cam_{cid}" / "calib" / "img_points.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        cam   = cameras[cid]
        pts3d = np.array(raw["real_corners"], dtype=np.float64)
        pts2d = np.array(raw["img_corners"],  dtype=np.float64)
        world_scale = cam.get("world_scale", 1.0)
        p = pts3d * world_scale
        proj, _ = cv2.projectPoints(
            p.reshape(-1, 1, 3),
            cam["rvec"].reshape(3, 1),
            cam["tvec"].reshape(3, 1),
            cam["mtx"], cam["dist"],
        )
        errs = np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)
        result[cid] = round(float(np.mean(errs)), 2)
    return result


app = Flask(__name__)

CAMERAS      = load_calibration(CAM_DIR, CAMERA_ORDER)
CAMERAS_ORIG = load_original_calibration(CAM_DIR, CAMERA_ORDER)
CAMERAS_MOGE = load_moge_calibration(CAM_DIR, CAMERA_ORDER)
SCALE_X      = ORIG_W / DISPLAY_W
SCALE_Y      = ORIG_H / DISPLAY_H

ACTIVE_CAMS  = [cid for cid in CAMERA_ORDER if cid in CAMERAS]
GROUND_TRUTH = load_ground_truth(CAM_DIR, ACTIVE_CAMS, SCALE_X, SCALE_Y)
REPROJ_BA    = compute_reproj(CAM_DIR, CAMERAS,      ACTIVE_CAMS, SCALE_X, SCALE_Y)
REPROJ_ORIG  = compute_reproj(CAM_DIR, CAMERAS_ORIG, ACTIVE_CAMS, SCALE_X, SCALE_Y)
REPROJ_MOGE  = compute_reproj(CAM_DIR, CAMERAS_MOGE, ACTIVE_CAMS, SCALE_X, SCALE_Y)

# per-camera mean reprojection summary for the comparison table
MEAN_REPROJ  = {
    "ba":    compute_mean_reproj(CAM_DIR, CAMERAS,      ACTIVE_CAMS),
    "orig":  compute_mean_reproj(CAM_DIR, CAMERAS_ORIG, ACTIVE_CAMS),
    "moge":  compute_mean_reproj(CAM_DIR, CAMERAS_MOGE, ACTIVE_CAMS),
}


@app.get("/")
def index():
    return render_template(
        "index.html",
        cameras=ACTIVE_CAMS,
        display_w=DISPLAY_W,
        display_h=DISPLAY_H,
        ground_truth=GROUND_TRUTH,
        reproj_ba=REPROJ_BA,
        reproj_orig=REPROJ_ORIG,
        reproj_moge=REPROJ_MOGE,
        mean_reproj=MEAN_REPROJ,
        has_moge=bool(CAMERAS_MOGE),
    )


@app.post("/project")
def project():
    """
    Body: { "cam_id": int, "x": float, "y": float }
    Response: { "world": [x,y,z], "projections": { "cam_id": [x,y] | null, ... } }
    """
    body   = request.get_json()
    src_id = int(body["cam_id"])
    px     = float(body["x"])
    py     = float(body["y"])

    if src_id not in CAMERAS:
        return jsonify(error="unknown camera"), 400

    world_pt = pixel_to_world(px, py, CAMERAS[src_id], SCALE_X, SCALE_Y)
    if world_pt is None:
        return jsonify(error="ray parallel to court plane"), 400

    projections = {}
    for cid, cam in CAMERAS.items():
        if cid == src_id:
            continue
        result = world_to_pixel(world_pt, cam, SCALE_X, SCALE_Y)
        projections[str(cid)] = list(result) if result else None

    return jsonify(
        world=[round(float(v), 1) for v in world_pt],
        projections=projections,
    )


@app.post("/triangulate")
def triangulate():
    """
    Body: { "clicks": [ {"cam_id": int, "x": float, "y": float}, ... ] }
    At least 2 clicks from different cameras required.
    Response: { "world": [x,y,z], "projections": { "cam_id": [x,y] | null, ... } }
    """
    body   = request.get_json()
    clicks = body.get("clicks", [])

    if len(clicks) < 2:
        return jsonify(error="need at least 2 clicks"), 400

    rays = []
    for click in clicks:
        cid = int(click["cam_id"])
        if cid not in CAMERAS:
            return jsonify(error=f"unknown camera {cid}"), 400
        o, d = pixel_to_ray(float(click["x"]), float(click["y"]),
                            CAMERAS[cid], SCALE_X, SCALE_Y)
        rays.append((o, d))

    world_pt = triangulate_rays(rays)

    projections = {}
    for cid, cam in CAMERAS.items():
        result = world_to_pixel(world_pt, cam, SCALE_X, SCALE_Y)
        projections[str(cid)] = list(result) if result else None

    return jsonify(
        world=[round(float(v), 3) for v in world_pt],
        projections=projections,
    )


@app.get("/reload")
def reload_calib():
    """Re-read calibration files from disk and return updated overlay data."""
    global CAMERAS, CAMERAS_ORIG, CAMERAS_MOGE, \
           GROUND_TRUTH, REPROJ_BA, REPROJ_ORIG, REPROJ_MOGE, \
           ACTIVE_CAMS, MEAN_REPROJ

    CAMERAS      = load_calibration(CAM_DIR, CAMERA_ORDER)
    CAMERAS_ORIG = load_original_calibration(CAM_DIR, CAMERA_ORDER)
    CAMERAS_MOGE = load_moge_calibration(CAM_DIR, CAMERA_ORDER)
    ACTIVE_CAMS  = [cid for cid in CAMERA_ORDER if cid in CAMERAS]
    GROUND_TRUTH = load_ground_truth(CAM_DIR, ACTIVE_CAMS, SCALE_X, SCALE_Y)
    REPROJ_BA    = compute_reproj(CAM_DIR, CAMERAS,      ACTIVE_CAMS, SCALE_X, SCALE_Y)
    REPROJ_ORIG  = compute_reproj(CAM_DIR, CAMERAS_ORIG, ACTIVE_CAMS, SCALE_X, SCALE_Y)
    REPROJ_MOGE  = compute_reproj(CAM_DIR, CAMERAS_MOGE, ACTIVE_CAMS, SCALE_X, SCALE_Y)
    MEAN_REPROJ  = {
        "ba":   compute_mean_reproj(CAM_DIR, CAMERAS,      ACTIVE_CAMS),
        "orig": compute_mean_reproj(CAM_DIR, CAMERAS_ORIG, ACTIVE_CAMS),
        "moge": compute_mean_reproj(CAM_DIR, CAMERAS_MOGE, ACTIVE_CAMS),
    }

    return jsonify(
        ground_truth=GROUND_TRUTH,
        reproj_ba=REPROJ_BA,
        reproj_orig=REPROJ_ORIG,
        reproj_moge=REPROJ_MOGE,
        mean_reproj=MEAN_REPROJ,
        has_moge=bool(CAMERAS_MOGE),
    )


@app.get("/sessions")
def sessions():
    """Return sorted list of available session names (subdirs of VIDEO_DIR)."""
    if not VIDEO_DIR.exists():
        return jsonify([])
    names = sorted(d.name for d in VIDEO_DIR.iterdir() if d.is_dir())
    return jsonify(names)


@app.get("/frame/<session>/<int:cam_id>")
def frame(session: str, cam_id: int):
    """Extract frame 0 of out<cam_id>.mp4, undistort it, and return as JPEG."""
    video_path = VIDEO_DIR / session / f"out{cam_id}.mp4"
    if not video_path.exists():
        return "video not found", 404

    cap = cv2.VideoCapture(str(video_path))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return "could not read frame", 500

    img = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return "encode failed", 500

    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


def _free_port(port: int) -> None:
    try:
        result = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True,
        )
        pids = set()
        for line in result.stdout.splitlines():
            for part in line.split(","):
                if "pid=" in part:
                    pid = int(part.split("pid=")[1].split(",")[0].rstrip(")"))
                    pids.add(pid)
        for pid in pids:
            if pid == os.getpid():
                continue
            os.kill(pid, signal.SIGTERM)
            print(f"  Killed existing process on port {port} (PID {pid})")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-camera point picker viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to listen on.")
    parser.add_argument("--no-kill", action="store_true",
                        help="Don't kill existing process on the port.")
    args = parser.parse_args()

    if not args.no_kill:
        _free_port(args.port)

    print(f"  Starting viewer on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=True, use_reloader=False)
