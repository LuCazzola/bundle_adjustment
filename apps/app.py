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
import re
import signal
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.geometry import triangulate_rays as _triangulate_rays
from scripts.mocap import (
    COCO_KP_NAMES as _COCO_KP_NAMES,
    CORR_JOINTS as _CORR_JOINTS,
    JOINT_MAP as _JOINT_MAP,
    MOCAP_FPS as _MOCAP_FPS,
    VIDEO_FPS as _VIDEO_FPS,
    compute_temporal_alignment,
)
from scripts.optim import umeyama as _umeyama

try:
    import scipy.io as sio
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
DATA_DIR  = PROJECT_ROOT / "data"
CAM_DIR   = DATA_DIR / "cameras"
VIDEO_DIR = DATA_DIR / "videos"

# Set at startup via --video; drives both frame serving and mocap/annotation loading.
VIDEO_ID = "26_03_26-cus_redfox"

CAMERA_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
CALIB_SOURCES = ["checkerboard", "ba"]   # ba only shown when files exist

_ZEROS5 = np.zeros(5, dtype=np.float64)

# Display image settings — tuned at startup via CLI, never affect math.
SERVE_SCALE   = 1.0   # fraction of native resolution sent to the browser
SERVE_QUALITY = 60    # JPEG quality [1–95]


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
    video_path = VIDEO_DIR / VIDEO_ID / f"out{cid}.mp4"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h

    sidecar = CAM_DIR / f"out{cid}" / "metadata" / "image_size.json"
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
        img_pts = CAM_DIR / f"out{cid}" / "annotations" / "court_points.json"

        if calib_source == "ba":
            path = CAM_DIR / f"out{cid}" / "calibration" / "ba_calib.json"
            if not path.exists():
                path = CAM_DIR / f"out{cid}" / "calibration" / "static_calib.json"
        else:
            path = CAM_DIR / f"out{cid}" / "calibration" / "static_calib.json"

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
# MOCAP DATA  (load once at startup / reload)
# ──────────────────────────────────────────────────────────────────────────────


def _load_mocap_world_data(cameras: dict) -> dict | None:
    """
    Build per-(frame, cam) overlay lists for the debug viewer.

    Always loads COCO 2-D keypoint annotations (gt_x/y + gt_rx/ry).
    Additionally loads 3-D mocap projections (x/y, rx/ry, err, world) when
    mocap_data.mat is present.  Returns None only if the COCO file is missing.
    """
    mat_path        = DATA_DIR / "mocap_data"  / VIDEO_ID / "mocap_data.mat"
    coco_path       = DATA_DIR / "annotations" / VIDEO_ID / "_annotations.coco.json"
    alignment_cache = DATA_DIR / "mocap_data"  / VIDEO_ID / "temporal_alignment.json"
    transform_cache = DATA_DIR / "mocap_data"  / VIDEO_ID / "mocap_transform.json"

    if not coco_path.exists():
        print(f"[WARN] No COCO annotations at {coco_path} — skipping mocap overlays.")
        return None

    # ── always: load COCO annotations ────────────────────────────────────────
    with open(coco_path) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    ann_index: dict[tuple[int, int], dict] = {}
    for ann in coco["annotations"]:
        img = img_info.get(ann["image_id"])
        if img is None:
            continue
        m = re.search(r"out(\d+)_frame_(\d+)", img["file_name"])
        if m is None:
            continue
        ann_index[(int(m.group(2)), int(m.group(1)))] = ann

    cam_id_list   = [cid for cid in CAMERA_ORDER if cid in cameras]
    cam_list      = [cameras[cid] for cid in cam_id_list]

    # ── optional: load mocap .mat + compute 3-D projections ──────────────────
    has_mocap_mat = mat_path.exists() and _SCIPY_OK
    pos_mm       = None
    coco_to_mocap: dict[int, int] = {}
    frame_map:    dict[int, int]  = {}
    s_mw = R_mw = t_mw = None

    if has_mocap_mat:
        mat          = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
        nick         = mat[[k for k in mat if not k.startswith("_")][0]]
        pos_mm       = nick.Skeletons.PositionData
        mocap_labels = list(np.asarray(nick.Skeletons.SegmentLabels).astype(str))

        for ki, kname in enumerate(_COCO_KP_NAMES):
            mname = _JOINT_MAP.get(kname)
            if mname and mname in mocap_labels:
                coco_to_mocap[ki] = mocap_labels.index(mname)

        print("\n[Mocap] Temporal alignment …")
        try:
            frame_map = compute_temporal_alignment(
                pos_mm, mocap_labels, coco_path, cam_list, cam_id_list,
                cache_path=alignment_cache,
            )
        except Exception as exc:
            print(f"[WARN] Temporal alignment failed ({exc}) — 3-D projections skipped.")
            frame_map = {}

        if frame_map:
            if transform_cache.exists():
                with open(transform_cache) as f:
                    tc = json.load(f)
                stage = "refined" if "refined" in tc else "seeded"
                entry = tc[stage]
                mt_vec = np.array(entry["rvec"] + entry["tvec"] + [entry["log_s"]], dtype=np.float64)
                R_mw, _ = cv2.Rodrigues(mt_vec[0:3].reshape(3, 1))
                t_mw    = mt_vec[3:6]
                s_mw    = float(np.exp(mt_vec[6]))
                print(f"  Mocap→world: loaded {stage} transform from cache  "
                      f"scale={s_mw:.5f}  |t|={np.linalg.norm(t_mw):.3f} m")
            else:
                cam_id_to_idx = {cid: i for i, cid in enumerate(cam_id_list)}
                tri_pts, moc_pts = [], []
                for fid, mocap_fid in frame_map.items():
                    if mocap_fid >= pos_mm.shape[2]:
                        continue
                    joint_views: dict[int, dict[int, np.ndarray]] = {ki: {} for ki in range(18)}
                    for cid in cam_id_list:
                        ann = ann_index.get((fid, cid))
                        if ann is None:
                            continue
                        kps  = ann["keypoints"]
                        cidx = cam_id_to_idx[cid]
                        for ki in range(18):
                            if kps[ki * 3 + 2] == 0:
                                continue
                            joint_views[ki][cidx] = np.array([kps[ki * 3], kps[ki * 3 + 1]])
                    for ki, views in joint_views.items():
                        if len(views) < 2 or ki not in coco_to_mocap:
                            continue
                        tri = _triangulate_rays(views, cam_list)
                        if tri is None:
                            continue
                        tri_pts.append(tri)
                        moc_pts.append(pos_mm[:, coco_to_mocap[ki], mocap_fid] / 1000.0)
                if len(tri_pts) >= 4:
                    s_mw, R_mw, t_mw = _umeyama(np.array(moc_pts), np.array(tri_pts))
                    print(f"  Mocap→world (Umeyama)  scale={s_mw:.5f}  "
                          f"|t|={np.linalg.norm(t_mw):.3f} m  ({len(tri_pts)} samples)")
                else:
                    print(f"[WARN] Only {len(tri_pts)} triangulated points — 3-D projections skipped.")
    else:
        print(f"  No mocap_data.mat for {VIDEO_ID} — loading COCO 2-D annotations only.")

    # ── build overlay entries ─────────────────────────────────────────────────
    # Collect every (fid, cid) pair that has a COCO annotation.
    all_fid_cid = set(ann_index.keys())
    # Also include fids from frame_map so 3-D entries are keyed consistently.
    fid_set = {fid for fid, _ in all_fid_cid}

    result: dict[tuple[int, int], list[dict]] = {}
    for fid in sorted(fid_set):
        mocap_fid = frame_map.get(fid)
        use_3d    = (mocap_fid is not None and pos_mm is not None
                     and mocap_fid < pos_mm.shape[2]
                     and s_mw is not None)

        for cid in cam_id_list:
            ann = ann_index.get((fid, cid))
            cam = cameras.get(cid)
            if ann is None or cam is None:
                continue

            iw, ih = cam["w"], cam["h"]
            kps    = ann["keypoints"]
            entries = []

            for ki in range(18):
                if kps[ki * 3 + 2] == 0:
                    continue

                gt_px = np.array([kps[ki * 3], kps[ki * 3 + 1]])
                gt_ud = cv2.undistortPoints(
                    gt_px.reshape(1, 1, 2).astype(np.float64),
                    cam["mtx"], cam["dist"], P=cam["mtx"],
                )[0, 0]

                entry: dict = {
                    "label": _COCO_KP_NAMES[ki],
                    "gt_x":  round(float(gt_px[0]) / iw, 6),
                    "gt_y":  round(float(gt_px[1]) / ih, 6),
                    "gt_rx": round(float(gt_ud[0]) / iw, 6),
                    "gt_ry": round(float(gt_ud[1]) / ih, 6),
                }

                if use_3d and ki in coco_to_mocap:
                    moc_m    = pos_mm[:, coco_to_mocap[ki], mocap_fid] / 1000.0
                    world_pt = s_mw * (R_mw @ moc_m) + t_mw
                    pt_cv    = world_pt.reshape(1, 1, 3)
                    rvec_cv  = cam["rvec"].reshape(3, 1)
                    tvec_cv  = cam["tvec"].reshape(3, 1)
                    proj_orig, _ = cv2.projectPoints(pt_cv, rvec_cv, tvec_cv, cam["mtx"], cam["dist"])
                    proj_rect, _ = cv2.projectPoints(pt_cv, rvec_cv, tvec_cv, cam["mtx"], _ZEROS5)
                    ox, oy = float(proj_orig[0, 0, 0]), float(proj_orig[0, 0, 1])
                    rx, ry = float(proj_rect[0, 0, 0]), float(proj_rect[0, 0, 1])
                    entry.update({
                        "x":     round(ox / iw, 6),
                        "y":     round(oy / ih, 6),
                        "rx":    round(rx / iw, 6),
                        "ry":    round(ry / ih, 6),
                        "err":   round(float(np.linalg.norm(np.array([ox, oy]) - gt_px)), 1),
                        "world": [round(float(v), 3) for v in world_pt],
                    })

                entries.append(entry)

            if entries:
                result[(fid, cid)] = entries

    print(f"  COCO overlays: {len(result)} (frame, cam) pairs loaded "
          f"({'with' if s_mw is not None else 'without'} 3-D mocap projections).")
    return result


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
        path = CAM_DIR / f"out{cid}" / "annotations" / "court_points.json"
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
CAMERAS_BY_SOURCE: dict[str, dict]       = {}   # source -> {cam_id -> cam_dict}
OVERLAYS_BY_SOURCE: dict[str, dict]      = {}   # source -> overlay data
# calib_source -> (video_frame, cam_id) -> list of mocap joint overlay dicts
MOCAP_OVERLAYS: dict[str, dict[tuple[int, int], list[dict]]] = {}

HAS_BA    = False
HAS_MOCAP = False


def _refresh_state():
    global CAMERAS_BY_SOURCE, OVERLAYS_BY_SOURCE, HAS_BA, MOCAP_OVERLAYS, HAS_MOCAP

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
        (CAM_DIR / f"out{cid}" / "calibration" / "ba_calib.json").exists()
        for cid in active
    )

    print("\nLoading mocap overlays (checkerboard calib)…")
    mo_cb = _load_mocap_world_data(cams_cb)
    MOCAP_OVERLAYS["checkerboard"] = mo_cb if mo_cb is not None else {}

    if HAS_BA and cams_ba:
        print("\nLoading mocap overlays (BA calib)…")
        mo_ba = _load_mocap_world_data(cams_ba)
        MOCAP_OVERLAYS["ba"] = mo_ba if mo_ba is not None else {}
    else:
        MOCAP_OVERLAYS["ba"] = {}

    HAS_MOCAP = bool(MOCAP_OVERLAYS["checkerboard"])

    return active


ACTIVE_CAMS: list[int] = []


def _discover_video_ids() -> list[str]:
    """Return sorted list of video IDs that exist under data/videos/."""
    vdir = VIDEO_DIR
    if not vdir.is_dir():
        return []
    return sorted(d.name for d in vdir.iterdir() if d.is_dir())


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

def _video_frame_count(cam_id: int) -> int:
    """Return total frame count for a camera's video, or 1 if unavailable."""
    p = VIDEO_DIR / VIDEO_ID / f"out{cam_id}.mp4"
    if not p.exists():
        return 1
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


@app.get("/")
def index():
    # Per-camera image sizes so the JS can convert normalised -> pixel correctly.
    cam_sizes = {
        cid: {"w": cam["w"], "h": cam["h"]}
        for cid, cam in CAMERAS_BY_SOURCE.get("checkerboard", {}).items()
    }
    # Per-camera frame count for the video slider.
    frame_counts = {cid: _video_frame_count(cid) for cid in ACTIVE_CAMS}
    # Serialise mocap overlays: { calib_source: { "fid_cid": [...] } }
    mocap_json: dict[str, dict[str, list]] = {}
    for src, src_overlays in MOCAP_OVERLAYS.items():
        src_json: dict[str, list] = {}
        for (fid, cid), entries in src_overlays.items():
            if cid in ACTIVE_CAMS:
                src_json[f"{fid}_{cid}"] = entries
        mocap_json[src] = src_json

    return render_template(
        "index.html",
        cameras=ACTIVE_CAMS,
        cam_sizes=cam_sizes,
        frame_counts=frame_counts,
        overlays=OVERLAYS_BY_SOURCE,
        calib_sources=CALIB_SOURCES,
        has_ba=HAS_BA,
        has_mocap=HAS_MOCAP,
        mocap_overlays=mocap_json,
        video_ids=_discover_video_ids(),
        current_video_id=VIDEO_ID,
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
    mocap_json: dict[str, dict[str, list]] = {}
    for src, src_overlays in MOCAP_OVERLAYS.items():
        src_json: dict[str, list] = {}
        for (fid, cid), entries in src_overlays.items():
            if cid in ACTIVE_CAMS:
                src_json[f"{fid}_{cid}"] = entries
        mocap_json[src] = src_json
    return jsonify(
        overlays=OVERLAYS_BY_SOURCE,
        has_ba=HAS_BA,
        has_mocap=HAS_MOCAP,
        mocap_overlays=mocap_json,
    )


@app.post("/switch")
def switch_video():
    """
    Body: { video_id }
    Switches VIDEO_ID (frames + mocap/annotations), reloads state, returns fresh data.
    """
    global VIDEO_ID, ACTIVE_CAMS
    body    = request.get_json()
    new_vid = body.get("video_id")

    if new_vid and new_vid != VIDEO_ID:
        VIDEO_ID    = new_vid
        ACTIVE_CAMS = _refresh_state()

    frame_counts = {cid: _video_frame_count(cid) for cid in ACTIVE_CAMS}
    mocap_json: dict[str, dict[str, list]] = {}
    for src, src_overlays in MOCAP_OVERLAYS.items():
        src_json: dict[str, list] = {}
        for (fid, cid), entries in src_overlays.items():
            if cid in ACTIVE_CAMS:
                src_json[f"{fid}_{cid}"] = entries
        mocap_json[src] = src_json

    return jsonify(
        video_id=VIDEO_ID,
        frame_counts=frame_counts,
        overlays=OVERLAYS_BY_SOURCE,
        has_ba=HAS_BA,
        has_mocap=HAS_MOCAP,
        mocap_overlays=mocap_json,
    )


@app.get("/mocap_frame/<int:frame_id>/<int:cam_id>")
def mocap_frame(frame_id: int, cam_id: int):
    """
    Return the mocap overlay entries for a specific (video_frame, cam_id, calib) triple.
    Query param: ?calib=checkerboard|ba  (default: checkerboard)
    Response: { entries: [{x, y, label, err, world, gt_x, gt_y}, ...] }
    """
    calib = request.args.get("calib", "checkerboard")
    src_overlays = MOCAP_OVERLAYS.get(calib, {})
    entries = src_overlays.get((frame_id, cam_id), [])
    return jsonify(entries=entries)


@app.get("/frame/<calib>/<display>/<int:cam_id>/<int:frame_idx>")
def frame(calib: str, display: str, cam_id: int, frame_idx: int):
    """
    Return a full-resolution JPEG for cam_id at frame_idx (0-based video frame).
    calib:     checkerboard | ba
    display:   original | rectified
    frame_idx: 0-based index into the video file
    ?t=<anything> is ignored; present only to bust the browser cache.
    """
    if calib not in CALIB_SOURCES:
        return "unknown calib", 400
    if display not in ("original", "rectified"):
        return "unknown display", 400

    video_path = VIDEO_DIR / VIDEO_ID / f"out{cam_id}.mp4"
    if not video_path.exists():
        return "video not found", 404

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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

    if SERVE_SCALE < 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, round(w * SERVE_SCALE)),
                               max(1, round(h * SERVE_SCALE))),
                         interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, SERVE_QUALITY])
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


def _main():
    global VIDEO_ID, SERVE_SCALE, SERVE_QUALITY, ACTIVE_CAMS

    parser = argparse.ArgumentParser(
        description="Multi-camera point picker viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-kill", action="store_true")
    parser.add_argument(
        "--video", default=VIDEO_ID,
        metavar="VIDEO_ID",
        help=(
            "Video identifier subfolder (e.g. 'mocap_4'). "
            "Selects data from videos/<VIDEO_ID>/, annotations/<VIDEO_ID>/, "
            "and mocap_data/<VIDEO_ID>/."
        ),
    )
    parser.add_argument(
        "--scale", type=float, default=SERVE_SCALE, metavar="F",
        help="Fraction of native resolution to serve (e.g. 0.33 for ~1/3 size). "
             "Does not affect overlay math — coordinates stay in normalised [0,1].",
    )
    parser.add_argument(
        "--quality", type=int, default=SERVE_QUALITY, metavar="Q",
        help="JPEG quality [1–95] for served frames.",
    )
    args = parser.parse_args()

    VIDEO_ID      = args.video
    SERVE_SCALE   = max(0.05, min(1.0, args.scale))
    SERVE_QUALITY = max(1,    min(95,  args.quality))

    print(f"  Video ID: {VIDEO_ID}")
    ACTIVE_CAMS = _refresh_state()

    if not args.no_kill:
        _free_port(args.port)

    print(f"  Starting viewer on http://0.0.0.0:{args.port}  "
          f"(scale={SERVE_SCALE:.2f}, quality={SERVE_QUALITY})")
    app.run(host="0.0.0.0", port=args.port, debug=True, use_reloader=False)


if __name__ == "__main__":
    _main()
