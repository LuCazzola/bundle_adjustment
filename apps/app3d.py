"""
3D Basketball Court Visualizer — bundle_adjustment edition.

Reads camera calibration from the bundle_adjustment data directory (same
cameras as app.py) and COCO keypoint annotations for the selected video ID.

For each frame it produces two skeleton sources:
  1. MOCAP 3D   — direct world-space joint positions from the mocap .mat file
                  (only available for video IDs that have mocap_data.mat).
  2. TRIANGULATED — joints back-projected and triangulated from COCO 2D
                    annotations using the top-k closest cameras to the player
                    (k >= 2, ranked by distance from camera centre to the
                    estimated floor position of the player).

USAGE
-----
    python app3d.py [--video mocap_4] [--port 5002] [--calib ba|checkerboard]
    Then open http://localhost:5002
"""

import argparse
import io
import json
import os
import re
import signal
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

try:
    import scipy.io as sio
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────

HERE      = Path(__file__).parent
PROJECT_ROOT = HERE.parent
DATA_DIR  = PROJECT_ROOT / "data"
CAM_DIR   = DATA_DIR / "cameras"
VIDEO_DIR = DATA_DIR / "videos"
ANN_DIR   = DATA_DIR / "annotations"
MOCAP_DIR = DATA_DIR / "mocap_data"

CAMERA_ORDER = [2, 5, 8, 13]
ORIG_W, ORIG_H       = 3840, 2160
DISPLAY_W, DISPLAY_H = 960,  540

# Set at startup
VIDEO_ID     = "mocap_4"
CALIB_SOURCE = "ba"          # "ba" or "checkerboard"

# ──────────────────────────────────────────────────────────────────────────────
# MOCAP / COCO CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

COCO_KP_NAMES = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand",
]

JOINT_MAP: dict[str, str] = {
    "Hips":      "Hips",
    "RHip":      "RightUpLeg",   "RKnee":  "RightLeg",
    "RAnkle":    "RightFoot",    "RFoot":  "RightToeBase",
    "LHip":      "LeftUpLeg",    "LKnee":  "LeftLeg",
    "LAnkle":    "LeftFoot",     "LFoot":  "LeftToeBase",
    "Spine":     "Spine2",       "Neck":   "Neck",
    "Head":      "Head",
    "RShoulder": "RightArm",     "RElbow": "RightForeArm",
    "RHand":     "RightHand",
    "LShoulder": "LeftArm",      "LElbow": "LeftForeArm",
    "LHand":     "LeftHand",
}

SKELETON_EDGES = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],
    [10,12],[12,13],[13,14],
    [10,15],[15,16],[16,17],
]

# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ──────────────────────────────────────────────────────────────────────────────

def _load_cameras(calib_source: str) -> dict[int, dict]:
    cameras: dict[int, dict] = {}
    for cid in CAMERA_ORDER:
        img_pts_path = CAM_DIR / f"out{cid}" / "annotations" / "court_points.json"

        if calib_source == "ba":
            path = CAM_DIR / f"out{cid}" / "calibration" / "ba_calib.json"
            if not path.exists():
                path = CAM_DIR / f"out{cid}" / "calibration" / "static_calib.json"
        else:
            path = CAM_DIR / f"out{cid}" / "calibration" / "static_calib.json"

        if not path.exists():
            print(f"[WARN] No calib for cam_{cid}, skipping.")
            continue

        with open(path) as f:
            cal = json.load(f)

        mtx  = np.array(cal["mtx"],   dtype=np.float64)
        dist = np.array(cal["dist"],  dtype=np.float64).ravel()
        rvec = np.array(cal["rvecs"], dtype=np.float64).ravel()
        tvec = np.array(cal["tvecs"], dtype=np.float64).ravel()

        # tvec in mm → re-solve with solvePnP to get metres
        if img_pts_path.exists() and np.linalg.norm(tvec) > 100:
            with open(img_pts_path) as f:
                raw = json.load(f)
            pts3d = np.array(raw["real_corners"], dtype=np.float32)
            pts2d = np.array(raw["img_corners"],  dtype=np.float32)
            ok, rv, tv = cv2.solvePnP(pts3d, pts2d, mtx, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rvec = rv.ravel()
                tvec = tv.ravel()

        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        center = (-R.T @ tvec).ravel()

        fx, fy = float(mtx[0, 0]), float(mtx[1, 1])
        fov_x  = 2 * float(np.degrees(np.arctan(ORIG_W / (2 * fx))))
        fov_y  = 2 * float(np.degrees(np.arctan(ORIG_H / (2 * fy))))

        cameras[cid] = {
            "mtx": mtx, "dist": dist, "rvec": rvec, "tvec": tvec,
            "center": center,
            "fov_x_deg": fov_x, "fov_y_deg": fov_y,
        }
        print(f"  cam_{cid}: loaded {path.name}  center={center.round(2)}")
    return cameras


def _cam_api_dict(cid: int, cam: dict) -> dict:
    return {
        "mtx":       [list(map(float, row)) for row in cam["mtx"]],
        "dist":      [float(v) for v in cam["dist"]],
        "rvec":      [float(v) for v in cam["rvec"]],
        "tvec":      [float(v) for v in cam["tvec"]],
        "center":    [round(float(v), 4) for v in cam["center"]],
        "fov_x_deg": round(cam["fov_x_deg"], 2),
        "fov_y_deg": round(cam["fov_y_deg"], 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# COURT GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

def _arc_pts(cx, cy, r, a_start, a_end, n=64):
    angles = np.linspace(a_start, a_end, n + 1)
    return [[round(cx + r * np.cos(a), 4), round(cy + r * np.sin(a), 4), 0.0]
            for a in angles]

def _arc_segs(cx, cy, r, a_start, a_end, n=64):
    pts = _arc_pts(cx, cy, r, a_start, a_end, n)
    return [{"pts": [pts[i], pts[i+1]]} for i in range(len(pts)-1)]

def _rect_segs(x1, y1, x2, y2):
    c = [[x1,y1,0.],[x2,y1,0.],[x2,y2,0.],[x1,y2,0.]]
    return [{"pts":[c[0],c[1]]},{"pts":[c[1],c[2]]},
            {"pts":[c[2],c[3]]},{"pts":[c[3],c[0]]}]

def build_fiba_court_lines():
    segs = []
    segs += _rect_segs(-14,-7.5,14,7.5)
    segs.append({"pts":[[0,-7.5,0.],[0,7.5,0.]]})
    segs += _arc_segs(0,0,1.8,0,2*np.pi,64)
    segs += _rect_segs(-14,-2.45,-8.325,2.45)
    segs += _rect_segs(8.325,-2.45,14,2.45)
    segs.append({"pts":[[-8.325,-2.45,0.],[-8.325,2.45,0.]]})
    segs.append({"pts":[[ 8.325,-2.45,0.],[ 8.325,2.45,0.]]})
    segs += _arc_segs(-8.325,0,1.8,-np.pi/2,np.pi/2,32)
    segs += _arc_segs( 8.325,0,1.8, np.pi/2,3*np.pi/2,32)
    for sx in (-1,1):
        bx = sx*12.15; r3 = 6.75
        sc = min(7.5/r3,1.0); ac = np.arcsin(sc)
        if sx < 0:
            segs += _arc_segs(bx,0,r3,ac,np.pi-ac,64)
            ax = bx+r3*np.cos(ac); bx2 = bx+r3*np.cos(np.pi-ac)
            segs.append({"pts":[[-14, 7.5,0.],[ax, 7.5,0.]]})
            segs.append({"pts":[[-14,-7.5,0.],[bx2,-7.5,0.]]})
        else:
            segs += _arc_segs(bx,0,r3,np.pi+ac,2*np.pi-ac,64)
            ax = bx-r3*np.cos(ac)
            segs.append({"pts":[[14, 7.5,0.],[ax, 7.5,0.]]})
            segs.append({"pts":[[14,-7.5,0.],[ax,-7.5,0.]]})
        segs += _arc_segs(sx*12.15,0,1.25,
                          np.pi/2 if sx>0 else -np.pi/2,
                          3*np.pi/2 if sx>0 else np.pi/2,32)
    return segs

def load_court_keypoints() -> list[list[float]]:
    seen: set[tuple] = set()
    kps = []
    for cid in CAMERA_ORDER:
        path = CAM_DIR / f"out{cid}" / "annotations" / "court_points.json"
        if not path.exists():
            continue
        with open(path) as f:
            raw = json.load(f)
        for pt in raw["real_corners"]:
            key = tuple(round(v,3) for v in pt)
            if key not in seen:
                seen.add(key)
                kps.append([float(pt[0]),float(pt[1]),float(pt[2])])
    return kps

COURT_LINES     = build_fiba_court_lines()
COURT_KEYPOINTS = load_court_keypoints()

# ──────────────────────────────────────────────────────────────────────────────
# VIDEO-SCOPED DATA (reloaded when VIDEO_ID changes)
# ──────────────────────────────────────────────────────────────────────────────

CAMERAS:   dict[int, dict]              = {}
ANN_INDEX: dict[tuple[int,int], dict]  = {}   # (fid, cam_id) → ann
ALL_FIDS:  list[int]                   = []

# mocap 3D (only when .mat present)
MOCAP_POS_MM:    np.ndarray | None = None   # (3, n_joints, n_frames)
COCO_TO_MOCAP:   dict[int, int]    = {}
FRAME_MAP:       dict[int, int]    = {}     # COCO fid → mocap frame idx

# Umeyama transform: world = scale * R @ (pos_mm / 1000) + t
MOCAP_SCALE:  float             = 1.0
MOCAP_R:      np.ndarray | None = None     # (3, 3)
MOCAP_T:      np.ndarray | None = None     # (3,)

# Precomputed per-frame skeleton cache: fid → {mocap, triangulated}
FRAME_CACHE: dict[int, dict] = {}


def _load_coco_annotations(video_id: str) -> tuple[dict, list[int]]:
    path = ANN_DIR / video_id / "_annotations.coco.json"
    if not path.exists():
        print(f"[WARN] No COCO annotations at {path}")
        return {}, []
    with open(path) as f:
        coco = json.load(f)
    img_info = {img["id"]: img for img in coco["images"]}
    ann_index: dict[tuple[int,int], dict] = {}
    for ann in coco["annotations"]:
        img = img_info.get(ann["image_id"])
        if img is None:
            continue
        m = re.search(r"out(\d+)_frame_(\d+)", img["file_name"])
        if m is None:
            continue
        ann_index[(int(m.group(2)), int(m.group(1)))] = ann
    all_fids = sorted({fid for fid, _ in ann_index})
    print(f"  COCO: {len(ann_index)} (frame, cam) pairs, {len(all_fids)} frames")
    return ann_index, all_fids


def _load_mocap(video_id: str, cameras: dict, ann_index: dict, all_fids: list):
    """Load mocap .mat, temporal alignment, and Umeyama world transform."""
    mat_path        = MOCAP_DIR / video_id / "mocap_data.mat"
    alignment_cache = MOCAP_DIR / video_id / "temporal_alignment.json"
    transform_cache = MOCAP_DIR / video_id / "mocap_transform.json"

    if not mat_path.exists() or not _SCIPY_OK:
        print(f"  No mocap_data.mat for {video_id} — skipping 3D mocap.")
        return None, {}, {}, 1.0, None, None

    mat          = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    nick         = mat[[k for k in mat if not k.startswith("_")][0]]
    pos_mm       = nick.Skeletons.PositionData
    mocap_labels = list(np.asarray(nick.Skeletons.SegmentLabels).astype(str))

    coco_to_mocap: dict[int, int] = {}
    for ki, kname in enumerate(COCO_KP_NAMES):
        mname = JOINT_MAP.get(kname)
        if mname and mname in mocap_labels:
            coco_to_mocap[ki] = mocap_labels.index(mname)

    # Load cached temporal alignment
    if not alignment_cache.exists():
        print(f"  [WARN] No temporal_alignment.json for {video_id} — mocap disabled.")
        return None, {}, {}, 1.0, None, None

    with open(alignment_cache) as f:
        cached = json.load(f)
    best_start = int(cached["mocap_start_frame"])
    video_fps  = float(cached.get("video_fps", 12.0))
    mocap_fps  = float(cached.get("mocap_fps", 100.0))

    frame_map: dict[int, int] = {}
    for i, fid in enumerate(all_fids):
        t_sec = i / video_fps
        frame_map[fid] = int(best_start + t_sec * mocap_fps)

    # Load Umeyama world transform  (world = scale * R @ pos_m + t)
    scale, R_mw, t_mw = 1.0, None, None
    if transform_cache.exists():
        with open(transform_cache) as f:
            tf = json.load(f)
        stage = tf.get("seeded") or tf.get(list(tf.keys())[0])
        scale = float(stage["scale"])
        rvec  = np.array(stage["rvec"], dtype=np.float64)
        R_mw, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        t_mw  = np.array(stage["tvec"], dtype=np.float64)
        print(f"  Mocap transform: scale={scale:.6f}  |t|={np.linalg.norm(t_mw):.3f} m")
    else:
        print(f"  [WARN] No mocap_transform.json for {video_id} — mocap shown in raw coords.")

    print(f"  Mocap: {pos_mm.shape[2]} frames, alignment offset={best_start}, "
          f"{len(coco_to_mocap)} joints mapped")
    return pos_mm, coco_to_mocap, frame_map, scale, R_mw, t_mw


def _precompute_frame_cache() -> dict[int, dict]:
    """Compute mocap + triangulated skeletons for every frame upfront."""
    cache: dict[int, dict] = {}
    print(f"  Precomputing {len(ALL_FIDS)} frames …", end="", flush=True)
    for fid in ALL_FIDS:
        cache[fid] = {
            "mocap":        _mocap_frame(fid),
            "triangulated": _triangulate_frame(fid),
        }
    print(" done")
    return cache


def _refresh_video_state():
    global CAMERAS, ANN_INDEX, ALL_FIDS, FRAME_CACHE
    global MOCAP_POS_MM, COCO_TO_MOCAP, FRAME_MAP, MOCAP_SCALE, MOCAP_R, MOCAP_T

    print(f"\nLoading state for video={VIDEO_ID}, calib={CALIB_SOURCE} …")
    CAMERAS = _load_cameras(CALIB_SOURCE)

    ANN_INDEX, ALL_FIDS = _load_coco_annotations(VIDEO_ID)

    cam_list = [CAMERAS[cid] for cid in CAMERA_ORDER if cid in CAMERAS]
    MOCAP_POS_MM, COCO_TO_MOCAP, FRAME_MAP, MOCAP_SCALE, MOCAP_R, MOCAP_T = _load_mocap(
        VIDEO_ID, cam_list, ANN_INDEX, ALL_FIDS,
    )

    FRAME_CACHE = _precompute_frame_cache()


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO SERVING
# ──────────────────────────────────────────────────────────────────────────────

_cap_cache: dict[tuple[str,int], cv2.VideoCapture] = {}
_cap_locks: dict[tuple[str,int], threading.Lock]   = {}
_cap_meta_lock = threading.Lock()


def _get_cap(video_id: str, cam_id: int):
    key = (video_id, cam_id)
    with _cap_meta_lock:
        if key not in _cap_cache or not _cap_cache[key].isOpened():
            path = VIDEO_DIR / video_id / f"out{cam_id}.mp4"
            if not path.exists():
                return None, None
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None, None
            _cap_cache[key]  = cap
            _cap_locks[key]  = threading.Lock()
    return _cap_cache[key], _cap_locks[key]


def _video_frame_count(video_id: str, cam_id: int) -> int:
    cap, lock = _get_cap(video_id, cam_id)
    if cap is None:
        return 0
    with lock:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _pixel_to_ray(px: float, py: float, cam: dict):
    pts    = np.array([[[px, py]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, cam["mtx"], cam["dist"])
    xn, yn = float(undist[0,0,0]), float(undist[0,0,1])
    R, _   = cv2.Rodrigues(cam["rvec"].reshape(3,1))
    origin = (-R.T @ cam["tvec"]).ravel()
    dirn   = R.T @ np.array([xn, yn, 1.0])
    dirn  /= np.linalg.norm(dirn)
    return origin, dirn


def _pixel_to_floor(px: float, py: float, cam: dict) -> np.ndarray | None:
    """Back-project pixel to Z=0 world plane."""
    pts    = np.array([[[px, py]]], dtype=np.float64)
    undist = cv2.undistortPoints(pts, cam["mtx"], cam["dist"])
    xn, yn = float(undist[0,0,0]), float(undist[0,0,1])
    R, _   = cv2.Rodrigues(cam["rvec"].reshape(3,1))
    origin = (-R.T @ cam["tvec"]).ravel()
    ray    = R.T @ np.array([xn, yn, 1.0])
    if abs(ray[2]) < 1e-9:
        return None
    lam = -origin[2] / ray[2]
    return origin + lam * ray


def _triangulate(rays: list[tuple]) -> np.ndarray | None:
    A = np.zeros((3,3)); b = np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# TRIANGULATION FROM COCO ANNOTATIONS
# ──────────────────────────────────────────────────────────────────────────────

_CONF_THRESH = 1   # COCO visibility flag: 0=absent, 1=occluded, 2=visible
_MIN_CAMS    = 2


def _rank_cameras_by_proximity(floor_pt: np.ndarray,
                                cam_ids: list[int]) -> list[int]:
    """Return cam_ids sorted by distance of camera centre to floor_pt (ascending)."""
    dists = []
    for cid in cam_ids:
        cam = CAMERAS.get(cid)
        if cam is None:
            continue
        d = float(np.linalg.norm(cam["center"][:2] - floor_pt[:2]))
        dists.append((d, cid))
    dists.sort()
    return [cid for _, cid in dists]


def _triangulate_frame(fid: int) -> list[dict]:
    """
    For a given COCO frame id, triangulate 3D keypoints using top-k cameras.

    Strategy:
      1. Estimate player floor position from the annotation with most visible
         keypoints (best single-camera anchor).
      2. Rank all cameras that have an annotation for this frame by distance
         to that floor position.
      3. Use the top-k (k >= 2) cameras, where k grows until either all
         available cameras are used or triangulation residual stops improving.
    """
    # Gather per-camera annotations for this frame
    cam_anns: dict[int, dict] = {}
    for cid in CAMERA_ORDER:
        ann = ANN_INDEX.get((fid, cid))
        if ann is not None:
            cam_anns[cid] = ann

    if len(cam_anns) < _MIN_CAMS:
        return []

    # Estimate floor position: use the camera with most visible keypoints,
    # project the hip (kp 0) or bbox bottom-centre to Z=0.
    best_cid = max(cam_anns, key=lambda c: sum(
        1 for i in range(0, len(cam_anns[c]["keypoints"]), 3)
        if cam_anns[c]["keypoints"][i+2] >= _CONF_THRESH
    ))
    best_ann = cam_anns[best_cid]
    best_cam = CAMERAS[best_cid]
    kps = best_ann["keypoints"]

    floor_pt = None
    # Try hip keypoint first
    hx, hy, hv = kps[0], kps[1], kps[2]
    if hv >= _CONF_THRESH:
        floor_pt = _pixel_to_floor(hx, hy, best_cam)
    # Fall back to bbox bottom-centre
    if floor_pt is None and best_ann.get("bbox"):
        bx, by, bw, bh = best_ann["bbox"]
        floor_pt = _pixel_to_floor(bx + bw/2, by + bh, best_cam)

    if floor_pt is None:
        return []

    # Rank cameras by distance to floor_pt
    ranked_cids = _rank_cameras_by_proximity(floor_pt, list(cam_anns.keys()))

    # Build per-keypoint rays from ranked cameras, then triangulate
    kpts_3d: list[list[float] | None] = []
    for ki in range(18):
        rays = []
        for cid in ranked_cids:
            ann = cam_anns[cid]
            cam = CAMERAS[cid]
            kp  = ann["keypoints"]
            x, y, v = kp[ki*3], kp[ki*3+1], kp[ki*3+2]
            if v < _CONF_THRESH:
                continue
            o, d = _pixel_to_ray(x, y, cam)
            rays.append((o, d))
            if len(rays) >= len(ranked_cids):  # use all available
                break

        if len(rays) < _MIN_CAMS:
            kpts_3d.append(None)
            continue

        pt = _triangulate(rays)
        if pt is None:
            kpts_3d.append(None)
        else:
            kpts_3d.append([round(float(pt[0]),4), round(float(pt[1]),4),
                            round(float(pt[2]),4), 1.0])

    if all(k is None for k in kpts_3d):
        return []

    return [{
        "global_id":    0,
        "source":       "triangulated",
        "cam_ids_used": ranked_cids,
        "keypoints_3d": kpts_3d,
        "edges":        SKELETON_EDGES,
    }]


# ──────────────────────────────────────────────────────────────────────────────
# MOCAP 3D SKELETON
# ──────────────────────────────────────────────────────────────────────────────

def _mocap_frame(fid: int) -> list[dict]:
    """Return mocap skeleton for COCO frame id transformed to world coords, or []."""
    if MOCAP_POS_MM is None or not FRAME_MAP:
        return []
    mocap_fid = FRAME_MAP.get(fid)
    if mocap_fid is None or mocap_fid >= MOCAP_POS_MM.shape[2]:
        return []

    has_transform = MOCAP_R is not None and MOCAP_T is not None

    kpts_3d: list[list[float] | None] = []
    for ki in range(18):
        mi = COCO_TO_MOCAP.get(ki)
        if mi is None:
            kpts_3d.append(None)
            continue
        pos_m = MOCAP_POS_MM[:, mi, mocap_fid] / 1000.0   # mm → m
        if has_transform:
            # world = scale * R @ pos_m + t
            pos_w = MOCAP_SCALE * (MOCAP_R @ pos_m) + MOCAP_T
        else:
            pos_w = pos_m
        kpts_3d.append([round(float(pos_w[0]), 4), round(float(pos_w[1]), 4),
                        round(float(pos_w[2]), 4), 1.0])

    if all(k is None for k in kpts_3d):
        return []

    return [{
        "global_id":    0,
        "source":       "mocap",
        "keypoints_3d": kpts_3d,
        "edges":        SKELETON_EDGES,
    }]


# ──────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.get("/")
def index():
    video_ids = sorted(d.name for d in VIDEO_DIR.iterdir() if d.is_dir())
    return render_template(
        "index3d.html",
        video_ids=video_ids,
        current_video_id=VIDEO_ID,
        current_calib=CALIB_SOURCE,
    )


@app.get("/api/calibration")
def api_calibration():
    return jsonify(cameras={str(cid): _cam_api_dict(cid, cam)
                             for cid, cam in CAMERAS.items()})


@app.get("/api/court")
def api_court():
    return jsonify(keypoints=COURT_KEYPOINTS, lines=COURT_LINES)


@app.get("/api/frames")
def api_frames():
    """Return sorted COCO frame ids for the current video."""
    return jsonify(frames=ALL_FIDS)


@app.get("/api/frame/<int:cam_id>")
def api_frame(cam_id: int):
    """Serve a JPEG frame at display resolution. Query param: ?frame=N (0-based video frame)."""
    frame_num = int(request.args.get("frame", 0))
    cap, lock = _get_cap(VIDEO_ID, cam_id)
    if cap is None:
        return "video not found", 404
    with lock:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, img = cap.read()
    if not ok or img is None:
        return "frame read failed", 500
    img = cv2.resize(img, (DISPLAY_W, DISPLAY_H))
    ok2, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok2:
        return "encode failed", 500
    resp = send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=60"
    return resp


@app.get("/api/frame_data/<int:fid>")
def api_frame_data(fid: int):
    """Single endpoint: precomputed skeletons for fid (served from cache)."""
    data = FRAME_CACHE.get(fid)
    if data is None:
        return jsonify(fid=fid, mocap=[], triangulated=[])
    return jsonify(fid=fid, mocap=data["mocap"], triangulated=data["triangulated"])


_switch_lock = threading.Lock()

@app.post("/api/switch")
def api_switch():
    """Switch video_id and/or calib_source, reload state. Serialised by lock."""
    global VIDEO_ID, CALIB_SOURCE
    body      = request.get_json()
    new_vid   = body.get("video_id")
    new_calib = body.get("calib_source")
    with _switch_lock:
        if new_vid:
            VIDEO_ID = new_vid
        if new_calib in ("ba", "checkerboard"):
            CALIB_SOURCE = new_calib
        _refresh_video_state()
        return jsonify(video_id=VIDEO_ID, calib_source=CALIB_SOURCE,
                       frames=ALL_FIDS, n_frames=len(ALL_FIDS))


# ──────────────────────────────────────────────────────────────────────────────
# PORT MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────

def _free_port(port: int) -> None:
    try:
        result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                                capture_output=True, text=True)
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
    global VIDEO_ID, CALIB_SOURCE

    parser = argparse.ArgumentParser(
        description="3D basketball court visualizer (bundle_adjustment edition).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--port",    type=int, default=5002)
    parser.add_argument("--no-kill", action="store_true")
    parser.add_argument(
        "--video", default=VIDEO_ID, metavar="VIDEO_ID",
        help="Video identifier (e.g. 'mocap_4'). Selects videos/, annotations/, mocap_data/.",
    )
    parser.add_argument(
        "--calib", choices=("ba", "checkerboard"), default=CALIB_SOURCE,
        help="Calibration source to use.",
    )
    args = parser.parse_args()

    VIDEO_ID     = args.video
    CALIB_SOURCE = args.calib

    _refresh_video_state()

    if not args.no_kill:
        _free_port(args.port)

    print(f"\n  Starting 3D viewer on http://0.0.0.0:{args.port}")
    print(f"  Video: {VIDEO_ID}  Calib: {CALIB_SOURCE}")
    print(f"  Court keypoints: {len(COURT_KEYPOINTS)}")
    app.run(host="0.0.0.0", port=args.port, debug=True,
            use_reloader=False, threaded=True)


if __name__ == "__main__":
    _main()
