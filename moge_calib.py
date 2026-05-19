"""
MoGe-2 camera calibration estimator.

For each camera, extracts one frame from the corresponding video, runs
MoGe-2 to predict normalised camera intrinsics, then uses those intrinsics
together with the known court keypoints (img_points.json) and solvePnP to
estimate extrinsics.  The result is written as:

    data/cameras/cam_<id>/calib/camera_calib_moge.json

The JSON schema is identical to camera_calib.json so ba.py and app.py can
consume it without changes.

USAGE
-----
  python moge_calib.py [options]

  --cameras  1 2 3        cameras to process (default: all)
  --session  hpe_1        video session subfolder under data/video/ (default: first available)
  --data-dir <path>       root calib folder
  --force                 overwrite existing camera_calib_moge.json files
  --resolution-level N    MoGe resolution level 0-9 (default: 7)
  --device   cuda|cpu     torch device (default: cuda if available, else cpu)

HOW IT WORKS
------------
1.  Read frame 0 of data/video/<session>/out<cam_id>.mp4.
2.  Run MoGe-2 (Ruicheng/moge-2-vitl) → normalised 3×3 intrinsic matrix K_norm.
3.  Denormalise:  fx = K_norm[0,0]*W,  fy = K_norm[1,1]*H,
                  cx = K_norm[0,2]*W,  cy = K_norm[1,2]*H.
    Assume zero distortion (MoGe has no distortion model).
4.  Call cv2.solvePnP with the court 2-D/3-D correspondences and the
    MoGe intrinsics to obtain rvec/tvec.
5.  Write camera_calib_moge.json.

NOTE: MoGe predicts the *geometric* intrinsics of the undistorted image.
      The court keypoints were annotated on the original (distorted) frames.
      The mismatch is small for typical lenses; if it is significant,
      running BA with --calib-source moge --optimize both will absorb it.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ── project paths ─────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
MOGE_DIR = HERE.parent.parent / "MoGe"   # cloned alongside basket_sports_journal
DEFAULT_DATA_DIR  = HERE / "data" / "cameras"
DEFAULT_VIDEO_DIR = HERE / "data" / "video"


def _ensure_moge_importable() -> None:
    if MOGE_DIR.exists():
        if str(MOGE_DIR) not in sys.path:
            sys.path.insert(0, str(MOGE_DIR))
    try:
        import moge  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            f"[ERROR] Cannot import 'moge'. Make sure MoGe is cloned at {MOGE_DIR} "
            "or installed via pip install git+https://github.com/microsoft/MoGe.git"
        )


def load_moge_model(device: torch.device, resolution_level: int):
    from moge.model.v2 import MoGeModel
    print("Loading MoGe-2 (Ruicheng/moge-2-vitl) …")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, resolution_level


def read_frame(video_path: Path) -> np.ndarray | None:
    """Return the first readable frame as uint8 RGB, or None."""
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def run_moge(model, frame_rgb: np.ndarray, device: torch.device,
             resolution_level: int) -> np.ndarray:
    """
    Run MoGe inference on a uint8 RGB frame.
    Returns the 3×3 **pixel-space** intrinsic matrix.
    """
    H, W = frame_rgb.shape[:2]
    tensor = torch.tensor(frame_rgb / 255.0, dtype=torch.float32,
                          device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(tensor, resolution_level=resolution_level,
                          use_fp16=(device.type == "cuda"))

    # out["intrinsics"] is normalised (values ≈ 0…1 for cx/cy, ≈ 0.5…2 for fx/fy)
    K_norm = out["intrinsics"].cpu().float().numpy()  # (3, 3)

    # Denormalise to pixel space
    K = K_norm.copy()
    K[0, 0] *= W   # fx
    K[1, 1] *= H   # fy
    K[0, 2] *= W   # cx
    K[1, 2] *= H   # cy
    return K


def estimate_extrinsics(pts3d: np.ndarray, pts2d: np.ndarray,
                        mtx: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Use solvePnP (ITERATIVE) to estimate rvec/tvec from known correspondences.
    Assumes zero distortion (MoGe has no distortion model).
    Returns (rvec, tvec) or None on failure.
    """
    dist_zero = np.zeros(5, dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            pts3d.astype(np.float32),
            pts2d.astype(np.float32),
            mtx, dist_zero,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            return rvec.ravel(), tvec.ravel()
    except Exception as e:
        print(f"    solvePnP failed: {e}")
    return None


def mean_reproj_error(pts3d, pts2d, rvec, tvec, mtx) -> float:
    dist_zero = np.zeros(5, dtype=np.float64)
    proj, _ = cv2.projectPoints(
        pts3d.reshape(-1, 1, 3).astype(np.float32),
        rvec.reshape(3, 1), tvec.reshape(3, 1),
        mtx, dist_zero,
    )
    return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)))


def discover_cameras(data_dir: Path, requested: list[int]) -> list[int]:
    available = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("cam_"):
            continue
        try:
            cid = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        calib = d / "calib"
        if (calib / "img_points.json").exists() and \
           ((calib / "camera_calib.json").exists() or (calib / "camera_calib_real.json").exists()):
            available.append(cid)
    if requested:
        missing = [i for i in requested if i not in available]
        if missing:
            print(f"[WARN] Cameras not found / missing data: {missing}")
        return [i for i in requested if i in available]
    return available


def pick_session(video_dir: Path, preferred: str | None) -> str | None:
    if not video_dir.exists():
        return None
    sessions = sorted(d.name for d in video_dir.iterdir() if d.is_dir())
    if not sessions:
        return None
    if preferred:
        if preferred in sessions:
            return preferred
        print(f"[WARN] Session '{preferred}' not found; using '{sessions[0]}'")
    return sessions[0]


def process_camera(cam_id: int, data_dir: Path, video_dir: Path,
                   session: str, model, device: torch.device,
                   resolution_level: int, force: bool) -> bool:
    out_path = data_dir / f"cam_{cam_id}" / "calib" / "camera_calib_moge.json"
    if out_path.exists() and not force:
        print(f"  cam_{cam_id}: skipping (already exists — use --force to overwrite)")
        return True

    # ── load keypoints ─────────────────────────────────────────────────────────
    kp_path = data_dir / f"cam_{cam_id}" / "calib" / "img_points.json"
    with open(kp_path) as f:
        raw = json.load(f)
    pts3d = np.array(raw["real_corners"], dtype=np.float64)
    pts2d = np.array(raw["img_corners"],  dtype=np.float64)

    # ── load original calibration (for JSON skeleton) ──────────────────────────
    base = data_dir / f"cam_{cam_id}" / "calib"
    src = base / "camera_calib_real.json"
    if not src.exists():
        src = base / "camera_calib.json"
    with open(src) as f:
        orig_doc = json.load(f)

    # ── read video frame ───────────────────────────────────────────────────────
    video_path = video_dir / session / f"out{cam_id}.mp4"
    if not video_path.exists():
        print(f"  cam_{cam_id}: [SKIP] video not found at {video_path}")
        return False
    frame_rgb = read_frame(video_path)
    if frame_rgb is None:
        print(f"  cam_{cam_id}: [SKIP] could not read frame from {video_path}")
        return False
    H, W = frame_rgb.shape[:2]
    print(f"  cam_{cam_id}: frame {W}×{H} from {video_path.name}")

    # ── MoGe inference ─────────────────────────────────────────────────────────
    print(f"  cam_{cam_id}: running MoGe-2 …", end=" ", flush=True)
    K = run_moge(model, frame_rgb, device, resolution_level)
    print(f"fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # ── solvePnP for extrinsics ────────────────────────────────────────────────
    result = estimate_extrinsics(pts3d, pts2d, K)
    if result is None:
        print(f"  cam_{cam_id}: [FAIL] solvePnP returned no solution")
        return False
    rvec, tvec = result
    err = mean_reproj_error(pts3d, pts2d, rvec, tvec, K)
    print(f"  cam_{cam_id}: reprojection error = {err:.2f} px")

    # ── write JSON ─────────────────────────────────────────────────────────────
    doc = dict(orig_doc)   # copy metadata fields (imsize etc.)
    doc["mtx"]   = [list(map(float, row)) for row in K]
    doc["dist"]  = [[0.0, 0.0, 0.0, 0.0, 0.0]]   # MoGe has no distortion model
    doc["rvecs"] = [[float(v)] for v in rvec]
    doc["tvecs"] = [[float(v)] for v in tvec]
    doc["_source"]  = "moge-2-vitl"
    doc["_reproj_px"] = round(err, 3)

    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  cam_{cam_id}: saved → {out_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate camera calibration from MoGe-2 monocular geometry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cameras", nargs="*", type=int, default=[],
                        metavar="ID",
                        help="Camera IDs to process (empty = all).")
    parser.add_argument("--session", default=None, metavar="NAME",
                        help="Video session folder under data/video/ (default: first available).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        metavar="PATH",
                        help="Root cameras folder containing cam_*/calib/.")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
                        metavar="PATH",
                        help="Root video folder containing <session>/out<id>.mp4.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing camera_calib_moge.json files.")
    parser.add_argument("--resolution-level", type=int, default=7, metavar="N",
                        help="MoGe resolution level 0-9 (higher = more accurate, slower).")
    parser.add_argument("--device", default=None, metavar="DEVICE",
                        help="Torch device string, e.g. 'cuda' or 'cpu' (default: auto).")
    args = parser.parse_args()

    _ensure_moge_importable()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    cam_ids = discover_cameras(args.data_dir, args.cameras)
    if not cam_ids:
        sys.exit("[ERROR] No cameras with required data found.")

    session = pick_session(args.video_dir, args.session)
    if session is None:
        sys.exit(f"[ERROR] No video sessions found under {args.video_dir}")

    print(f"Cameras  : {cam_ids}")
    print(f"Session  : {session}")
    print(f"Data dir : {args.data_dir}")
    print()

    model, res_level = load_moge_model(device, args.resolution_level)
    print()

    ok_count = 0
    for cid in cam_ids:
        ok = process_camera(cid, args.data_dir, args.video_dir, session,
                            model, device, res_level, args.force)
        ok_count += ok
        print()

    print(f"Done — {ok_count}/{len(cam_ids)} cameras calibrated.")


if __name__ == "__main__":
    main()
