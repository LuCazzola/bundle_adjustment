"""
Multi-camera point picker — debug visualisation tool.

Two independent controls drive everything:

  calib   checkerboard | ba
            Which camera parameters (K, dist, R, t) are used for ALL operations:
            on-the-fly frame rectification, GT overlay, reprojection overlay,
            click→world backprojection, Z=0 projection, triangulation.

  display  original | rectified
            original  — show the raw frame; points drawn in distorted pixel space.
            rectified — undistort the frame on the fly using the selected calib;
                        points drawn in undistorted pixel space (dist=0 projection).

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
CAM_DIR   = HERE / "data" / "camera_data"
VIDEO_DIR = HERE / "data" / "videos"

CAMERA_ORDER = [2, 5, 8, 13]
CALIB_SOURCES = ["checkerboard", "ba"]   # ba only shown when files exist

_ZEROS5 = np.zeros(5, dtype=np.float64)


def _effective_dist(cam: dict, rectified: bool) -> np.ndarray:
    """Distortion to use for projection/backprojection given the display mode."""
    return _ZEROS5 if rectified else cam["dist"]


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _read_image_size(cid: int) -> tuple[int, int]:
    """
    Return (width, height) for a camera.
    Priority: video file > image_size.json sidecar.
    Raises if neither is found.
    """
    video_path = VIDEO_DIR / "original" / f"out{cid}.mp4"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h

    sidecar = CAM_DIR / f"cam_{cid}" / "image_size.json"
    if sidecar.exists():
        with open(sidecar) as f:
            s = json.load(f)
        return int(s["width"]), int(s["height"])

    raise FileNotFoundError(
        f"cam_{cid}: no video and no image_size.json — cannot determine resolution"
    )


def _parse_cam(cal: dict, img_points_path: Path | None, image_size: tuple[int, int]) -> dict:
    mtx  = np.array(cal["mtx"],  dtype=np.float64)
    dist = np.array(cal["dist"], dtype=np.float64).ravel()
    rvec = np.array(cal["rvecs"], dtype=np.float64).ravel()
    tvec = np.array(cal["tvecs"], dtype=np.float64).ravel()

    # Stored tvec in mm (norm >> 100) — re-solve with solvePnP to get metres.
    if img_points_path is not None and img_points_path.exists() and np.linalg.norm(tvec) > 100:
        with open(img_points_path) as f:
            raw = json.load(f)
        pts3d = np.array(raw["real_corners"], dtype=np.float32)
        pts2d = np.array(raw["img_corners"],  dtype=np.float32)
        ok, rv, tv = cv2.solvePnP(pts3d, pts2d, mtx, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            rvec = rv.ravel()
            tvec = tv.ravel()

    return {
        "mtx": mtx, "dist": dist, "rvec": rvec, "tvec": tvec,
        "w": image_size[0], "h": image_size[1],
    }


def _load_cameras(calib_source: str) -> dict:
    """Load camera dicts for the given calib source ('checkerboard' or 'ba')."""
    cameras = {}
    for cid in CAMERA_ORDER:
        img_pts = CAM_DIR / f"cam_{cid}" / "calib" / "img_points.json"

        if calib_source == "ba":
            path = CAM_DIR / f"cam_{cid}" / "ba" / "camera_calib_ba.json"
            if not path.exists():
                path = CAM_DIR / f"cam_{cid}" / "calib" / "camera_calib.json"
        else:
            path = CAM_DIR / f"cam_{cid}" / "calib" / "camera_calib.json"

        if not path.exists():
            print(f"[WARN] No calib file for cam_{cid} (source={calib_source}), skipping.")
            continue

        try:
            image_size = _read_image_size(cid)
        except FileNotFoundError as e:
            print(f"[WARN] {e}, skipping.")
            continue

        with open(path) as f:
            cal = json.load(f)
        cameras[cid] = _parse_cam(cal, img_pts, image_size)
        print(f"  cam_{cid}: loaded {path.name}  [{image_size[0]}x{image_size[1]}]")

    return cameras


# ──────────────────────────────────────────────────────────────────────────────
# OVERLAY DATA  (ground truth + reprojection points)
# ──────────────────────────────────────────────────────────────────────────────

def _build_overlays(cameras: dict, cam_ids: list[int]) -> dict:
    """
    Build overlay point sets for both display modes using the given camera set.

    All x/y coordinates are normalised to [0, 1] relative to each camera's own
    resolution (cam["w"] × cam["h"]) so the JS can map them to any display size.

    Returns:
      {
        cam_id: {
          "gt": {
            "original":  [{x, y, label}, ...],
            "rectified": [{x, y, label}, ...],
          },
          "reproj": {
            "original":  [{x, y, label, err}, ...],
            "rectified": [{x, y, label, err}, ...],
          },
          "mean_err": float,
        }
      }
    """
    out = {}
    for cid in cam_ids:
        if cid not in cameras:
            continue
        path = CAM_DIR / f"cam_{cid}" / "calib" / "img_points.json"
        if not path.exists():
            continue

        with open(path) as f:
            raw = json.load(f)

        cam   = cameras[cid]
        pts3d = np.array(raw["real_corners"], dtype=np.float64)
        pts2d = np.array(raw["img_corners"],  dtype=np.float64)
        labels = [f"({wx:.1f},{wy:.1f})" for wx, wy, wz in raw["real_corners"]]

        iw, ih = cam["w"], cam["h"]

        def _norm(arr):
            """Pixel coords -> normalised [0,1] using this camera's resolution."""
            return [{"x": round(float(ox) / iw, 6),
                     "y": round(float(oy) / ih, 6)} for ox, oy in arr]

        # ── GT points ────────────────────────────────────────────────────────
        gt_orig = [{**p, "label": lbl} for p, lbl in zip(_norm(pts2d), labels)]

        pts2d_u = cv2.undistortPoints(
            pts2d.reshape(-1, 1, 2), cam["mtx"], cam["dist"], P=cam["mtx"]
        ).reshape(-1, 2)
        gt_rect = [{**p, "label": lbl} for p, lbl in zip(_norm(pts2d_u), labels)]

        # ── Reprojection points ───────────────────────────────────────────────
        proj_dist, _ = cv2.projectPoints(
            pts3d.reshape(-1, 1, 3),
            cam["rvec"].reshape(3, 1), cam["tvec"].reshape(3, 1),
            cam["mtx"], cam["dist"],
        )
        proj_dist = proj_dist.reshape(-1, 2)
        errs = [round(float(np.linalg.norm(proj_dist[i] - pts2d[i])), 1)
                for i in range(len(pts2d))]
        mean_err = round(float(np.mean(errs)), 2)

        reproj_orig = [{**p, "label": lbl, "err": e}
                       for p, lbl, e in zip(_norm(proj_dist), labels, errs)]

        proj_rect, _ = cv2.projectPoints(
            pts3d.reshape(-1, 1, 3),
            cam["rvec"].reshape(3, 1), cam["tvec"].reshape(3, 1),
            cam["mtx"], _ZEROS5,
        )
        reproj_rect = [{**p, "label": lbl, "err": e}
                       for p, lbl, e in zip(_norm(proj_rect.reshape(-1, 2)), labels, errs)]

        out[cid] = {
            "gt":     {"original": gt_orig,    "rectified": gt_rect},
            "reproj": {"original": reproj_orig, "rectified": reproj_rect},
            "mean_err": mean_err,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

def norm_to_world(nx: float, ny: float, cam: dict, rectified: bool) -> np.ndarray | None:
    """Backproject a normalised image coordinate to the Z=0 world plane."""
    pts    = np.array([[[nx * cam["w"], ny * cam["h"]]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, cam["mtx"], _effective_dist(cam, rectified))
    xn, yn = float(undist[0, 0, 0]), float(undist[0, 0, 1])

    R, _ = cv2.Rodrigues(cam["rvec"].reshape(3, 1))
    t    = cam["tvec"].reshape(3, 1)
    ray  = R.T @ np.array([xn, yn, 1.0])
    orig = (R.T @ (-t)).ravel()

    if abs(ray[2]) < 1e-9:
        return None
    lam = -orig[2] / ray[2]
    return orig + lam * ray


def world_to_norm(world_pt: np.ndarray, cam: dict,
                  rectified: bool) -> tuple[float, float] | None:
    """Project a 3D world point to normalised [0,1] image coords."""
    proj, _ = cv2.projectPoints(
        world_pt.reshape(1, 1, 3),
        cam["rvec"].reshape(3, 1), cam["tvec"].reshape(3, 1),
        cam["mtx"], _effective_dist(cam, rectified),
    )
    ox, oy = float(proj[0, 0, 0]), float(proj[0, 0, 1])
    nx, ny = ox / cam["w"], oy / cam["h"]
    if nx < 0 or ny < 0 or nx > 1 or ny > 1:
        return None
    return nx, ny


def norm_to_ray(nx: float, ny: float, cam: dict,
                rectified: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, unit_direction) ray for a normalised image coord."""
    pts    = np.array([[[nx * cam["w"], ny * cam["h"]]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, cam["mtx"], _effective_dist(cam, rectified))
    xn, yn = float(undist[0, 0, 0]), float(undist[0, 0, 1])

    R, _ = cv2.Rodrigues(cam["rvec"].reshape(3, 1))
    t    = cam["tvec"].reshape(3, 1)
    orig = (R.T @ (-t)).ravel()
    dirn = R.T @ np.array([xn, yn, 1.0])
    dirn /= np.linalg.norm(dirn)
    return orig, dirn


def triangulate_rays(rays: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b)


# ──────────────────────────────────────────────────────────────────────────────
# SERVER STATE  (mutated by /reload)
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Loaded once at startup; replaced atomically by /reload
CAMERAS_BY_SOURCE: dict[str, dict] = {}   # source -> {cam_id -> cam_dict}
OVERLAYS_BY_SOURCE: dict[str, dict] = {}  # source -> overlay data

HAS_BA = False


def _refresh_state():
    global CAMERAS_BY_SOURCE, OVERLAYS_BY_SOURCE, HAS_BA

    print("\nLoading calibration sources…")
    cams_cb = _load_cameras("checkerboard")
    cams_ba = _load_cameras("ba")

    active = [cid for cid in CAMERA_ORDER if cid in cams_cb]

    CAMERAS_BY_SOURCE = {
        "checkerboard": cams_cb,
        "ba":           cams_ba,
    }
    OVERLAYS_BY_SOURCE = {
        "checkerboard": _build_overlays(cams_cb, active),
        "ba":           _build_overlays(cams_ba, active),
    }
    HAS_BA = any(
        (CAM_DIR / f"cam_{cid}" / "ba" / "camera_calib_ba.json").exists()
        for cid in active
    )
    return active


ACTIVE_CAMS = _refresh_state()


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    # Per-camera image sizes so the JS can convert normalised -> pixel correctly.
    cam_sizes = {
        cid: {"w": cam["w"], "h": cam["h"]}
        for cid, cam in CAMERAS_BY_SOURCE.get("checkerboard", {}).items()
    }
    return render_template(
        "index.html",
        cameras=ACTIVE_CAMS,
        cam_sizes=cam_sizes,
        overlays=OVERLAYS_BY_SOURCE,
        calib_sources=CALIB_SOURCES,
        has_ba=HAS_BA,
    )


@app.post("/project")
def project():
    """
    Body: { cam_id, nx, ny, calib, rectified }   (nx/ny are normalised [0,1])
    Response: { world:[x,y,z], projections:{ cam_id:[nx,ny]|null } }
    """
    body      = request.get_json()
    src_id    = int(body["cam_id"])
    nx        = float(body["nx"])
    ny        = float(body["ny"])
    calib     = body.get("calib", "checkerboard")
    rectified = bool(body.get("rectified", False))

    cams = CAMERAS_BY_SOURCE.get(calib, {})
    if src_id not in cams:
        return jsonify(error="unknown camera"), 400

    world_pt = norm_to_world(nx, ny, cams[src_id], rectified)
    if world_pt is None:
        return jsonify(error="ray parallel to court plane"), 400

    projections = {}
    for cid, cam in cams.items():
        if cid == src_id:
            continue
        result = world_to_norm(world_pt, cam, rectified)
        projections[str(cid)] = list(result) if result else None

    return jsonify(
        world=[round(float(v), 3) for v in world_pt],
        projections=projections,
    )


@app.post("/triangulate")
def triangulate():
    """
    Body: { clicks:[{cam_id,nx,ny}], calib, rectified }   (nx/ny normalised [0,1])
    Response: { world:[x,y,z], projections:{ cam_id:[nx,ny]|null } }
    """
    body      = request.get_json()
    clicks    = body.get("clicks", [])
    calib     = body.get("calib", "checkerboard")
    rectified = bool(body.get("rectified", False))

    if len(clicks) < 2:
        return jsonify(error="need at least 2 clicks"), 400

    cams = CAMERAS_BY_SOURCE.get(calib, {})
    rays = []
    for click in clicks:
        cid = int(click["cam_id"])
        if cid not in cams:
            return jsonify(error=f"unknown camera {cid}"), 400
        o, d = norm_to_ray(float(click["nx"]), float(click["ny"]), cams[cid], rectified)
        rays.append((o, d))

    world_pt = triangulate_rays(rays)

    projections = {}
    for cid, cam in cams.items():
        result = world_to_norm(world_pt, cam, rectified)
        projections[str(cid)] = list(result) if result else None

    return jsonify(
        world=[round(float(v), 3) for v in world_pt],
        projections=projections,
    )


@app.get("/reload")
def reload_calib():
    """Re-read all calibration files from disk and return fresh overlay data."""
    global ACTIVE_CAMS
    ACTIVE_CAMS = _refresh_state()
    return jsonify(
        overlays=OVERLAYS_BY_SOURCE,
        has_ba=HAS_BA,
    )


@app.get("/frame/<calib>/<display>/<int:cam_id>")
def frame(calib: str, display: str, cam_id: int):
    """
    Return a full-resolution JPEG of frame 0 for cam_id.
    calib:   checkerboard | ba     — which K/dist to use for rectification
    display: original | rectified  — whether to undistort before returning
    ?t=<anything> is ignored; present only to bust the browser cache on reload.
    """
    if calib not in CALIB_SOURCES:
        return "unknown calib", 400
    if display not in ("original", "rectified"):
        return "unknown display", 400

    video_path = VIDEO_DIR / "original" / f"out{cam_id}.mp4"
    if not video_path.exists():
        return "video not found", 404

    cap = cv2.VideoCapture(str(video_path))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return "could not read frame", 500

    if display == "rectified":
        cam = CAMERAS_BY_SOURCE.get(calib, {}).get(cam_id)
        if cam is not None:
            mtx  = cam["mtx"].astype(np.float32)
            dist = cam["dist"].astype(np.float32)
            h, w = img.shape[:2]
            map1, map2 = cv2.initUndistortRectifyMap(
                mtx, dist, None, mtx, (w, h), cv2.CV_16SC2
            )
            img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return "encode failed", 500

    resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-kill", action="store_true")
    args = parser.parse_args()

    if not args.no_kill:
        _free_port(args.port)

    print(f"  Starting viewer on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=True, use_reloader=False)
