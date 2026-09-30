"""
Mocap data loading, temporal alignment, and BA observation construction.
"""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import cv2

from .geometry import triangulate_rays
from .optim import umeyama


# ──────────────────────────────────────────────────────────────────────────────
# RETURN TYPE
# ──────────────────────────────────────────────────────────────────────────────

Obs = list[tuple[np.ndarray, np.ndarray, int]]   # (pt3d_world, pt2d_pixel, cam_idx)

@dataclasses.dataclass
class MocapObservations:
    """
    Split mocap observations for BA and reporting.

    train : frames selected for bundle adjustment (e.g. top-5 by reprojection error)
    val   : all remaining frames — held out, used only for validation stats
    """
    train: Obs
    val:   Obs

try:
    import scipy.io as sio
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# COCO keypoint names (index 0–17)
COCO_KP_NAMES = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand",
]

# Maps COCO keypoint name → mocap SegmentLabels name.
# Matches evaluate_3d_reprojection.py JOINT_MAPPING exactly.
JOINT_MAP: dict[str, str] = {
    "Hips":      "Hips",
    "RHip":      "RightUpLeg",
    "RKnee":     "RightLeg",
    "RAnkle":    "RightFoot",
    "RFoot":     "RightToeBase",
    "LHip":      "LeftUpLeg",
    "LKnee":     "LeftLeg",
    "LAnkle":    "LeftFoot",
    "LFoot":     "LeftToeBase",
    "Spine":     "Spine2",
    "Neck":      "Neck",
    "Head":      "Head",
    "RShoulder": "RightArm",
    "RElbow":    "RightForeArm",
    "RHand":     "RightHand",
    "LShoulder": "LeftArm",
    "LElbow":    "LeftForeArm",
    "LHand":     "LeftHand",
}

# Joints used for temporal cross-correlation (reliable, high-motion joints)
CORR_JOINTS = ["RHand", "LHand", "Head", "RAnkle", "LAnkle"]

# ── Joint presets for BA observation filtering ────────────────────────────────
# Each preset is a frozenset of COCO keypoint names (subset of COCO_KP_NAMES).
# Rationale: some joints are hard to annotate reliably in 2-D (e.g. Head when
# occluded, Spine which is often invisible), so using a cleaner subset can yield
# more stable BA convergence.

JOINT_PRESETS: dict[str, frozenset[str]] = {
    # Use every available joint — default / baseline.
    "all": frozenset(COCO_KP_NAMES),

    # Extremities only: easiest to localise visually and most kinematically
    # distinct.  Excludes trunk joints (Hips root, Spine, Neck) that are
    # ambiguous in close-up crops.
    "end_effectors": frozenset([
        "RHand", "LHand",
        "RFoot", "LFoot",
        "Head",
    ]),

    # Full limbs (arms + legs) without any trunk/spine joints.
    # Good balance between coverage and annotation reliability.
    "limbs": frozenset([
        "RShoulder", "RElbow", "RHand",
        "LShoulder", "LElbow", "LHand",
        "RHip",  "RKnee",  "RAnkle", "RFoot",
        "LHip",  "LKnee",  "LAnkle", "LFoot",
    ]),

    # Lower-body only: feet/ankles are on the court plane (Z≈0) and therefore
    # serve as the strongest geometric anchor for extrinsic refinement.
    "lower_body": frozenset([
        "RHip", "RKnee", "RAnkle", "RFoot",
        "LHip", "LKnee", "LAnkle", "LFoot",
    ]),

    # Large, easy-to-annotate joints — avoids small distal joints (hands/feet)
    # that can be hard to click precisely, and avoids head/spine ambiguity.
    "stable": frozenset([
        "RShoulder", "LShoulder",
        "RHip",      "LHip",
        "RKnee",     "LKnee",
        "RAnkle",    "LAnkle",
    ]),
}

VIDEO_FPS = 12.0    # effective sub-sampled RGB frame rate
MOCAP_FPS = 100.0   # mocap recording rate (Hz)


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL ALIGNMENT
# ──────────────────────────────────────────────────────────────────────────────

def compute_temporal_alignment(
    pos_mm: np.ndarray,
    mocap_labels: list[str],
    coco_path: Path,
    cameras: list[dict],
    cam_ids: list[int],
    mocap_fps: float = MOCAP_FPS,
    video_fps: float = VIDEO_FPS,
    cache_path: Path | None = None,
    force_recompute: bool = False,
) -> dict[int, int]:
    """
    Compute video→mocap frame mapping via multi-joint cross-correlation.

    Ports the algorithm from Computer_Vision/task3_alignment/align_mocap.py:
      1. Load original COCO annotations; pixels are undistorted on the fly
         using the camera parameters (no separate rectified file needed).
      2. Resample RGB Z-height signals to mocap rate.
      3. Sliding cross-correlation over all mocap frames (step 10, then refine ±50).
      4. Best combined start → frame_mapping[rgb_frame] = mocap_frame_index.

    If cache_path points to an existing JSON and force_recompute is False, the
    stored frame_mapping is returned immediately (skipping cross-correlation).
    Otherwise the alignment is computed and written to cache_path.

    Returns dict[rgb_frame_id (int, 1-based)] → mocap_frame_index (int, 0-based).
    """
    if cache_path is not None and cache_path.exists() and not force_recompute:
        with open(cache_path) as f:
            cached = json.load(f)
        frame_map = {int(k): int(v) for k, v in cached["frame_mapping"].items()}
        best_start = list(frame_map.values())[0]
        print(f"  Temporal alignment: loaded from cache ({cache_path.name}, "
              f"mocap start {best_start})")
        return frame_map
    with open(coco_path) as f:
        coco_data = json.load(f)

    img_info_r: dict[int, dict] = {img["id"]: img for img in coco_data["images"]}
    ann_index_r: dict[tuple[int, int], dict] = {}
    for ann in coco_data["annotations"]:
        img = img_info_r.get(ann["image_id"])
        if img is None:
            continue
        m = re.search(r"out(\d+)_frame_(\d+)", img["file_name"])
        if m is None:
            continue
        cid, fid = int(m.group(1)), int(m.group(2))
        ann_index_r[(fid, cid)] = ann

    cam_id_to_idx = {cid: i for i, cid in enumerate(cam_ids)}

    all_fids = sorted({fid for (fid, _) in ann_index_r.keys()})
    n_rgb = len(all_fids)
    if n_rgb == 0:
        raise RuntimeError("No frames found in COCO annotations.")

    # ── triangulate Z-height signals ─────────────────────────────────────────
    rgb_z: dict[str, list[float]] = {j: [] for j in CORR_JOINTS}

    for fid in all_fids:
        views_by_kp: dict[int, dict[int, np.ndarray]] = {}
        for cam_id in cam_ids:
            ann = ann_index_r.get((fid, cam_id))
            if ann is None:
                continue
            kps  = ann["keypoints"]
            cidx = cam_id_to_idx[cam_id]
            for ki in range(18):
                if kps[ki * 3 + 2] == 0:
                    continue
                views_by_kp.setdefault(ki, {})[cidx] = np.array(
                    [kps[ki * 3], kps[ki * 3 + 1]], dtype=np.float64
                )

        for joint_name in CORR_JOINTS:
            coco_idx = COCO_KP_NAMES.index(joint_name)
            views = views_by_kp.get(coco_idx, {})
            if len(views) >= 2:
                pt3d = triangulate_rays(views, cameras)
                rgb_z[joint_name].append(float(pt3d[2]) if pt3d is not None else float("nan"))
            else:
                rgb_z[joint_name].append(float("nan"))

    # ── extract mocap Z signals ───────────────────────────────────────────────
    mocap_z: dict[str, np.ndarray] = {}
    for joint_name in CORR_JOINTS:
        mocap_name = JOINT_MAP[joint_name]
        if mocap_name not in mocap_labels:
            continue
        mi = mocap_labels.index(mocap_name)
        sig = pos_mm[2, mi, :].astype(np.float64)
        mocap_z[joint_name] = sig

    active_joints = [j for j in CORR_JOINTS if j in mocap_z and not all(np.isnan(rgb_z[j]))]
    if not active_joints:
        raise RuntimeError("No valid joints for temporal cross-correlation.")

    n_mocap         = pos_mm.shape[2]
    rgb_duration_sec = n_rgb / video_fps
    n_mocap_for_rgb  = int(rgb_duration_sec * mocap_fps)
    max_start        = n_mocap - n_mocap_for_rgb
    if max_start <= 0:
        raise RuntimeError("Mocap recording shorter than RGB clip — cannot align.")

    # ── coarse sliding correlation (step 10) ─────────────────────────────────
    combined     = np.zeros(max_start)
    rgb_time_vec = np.arange(n_rgb) / video_fps
    mc_time_vec  = np.arange(n_mocap_for_rgb) / mocap_fps

    precomputed_rgb: dict[str, np.ndarray] = {}
    for j in active_joints:
        sig = np.array(rgb_z[j], dtype=np.float64)
        mask = np.isnan(sig)
        if mask.any() and not mask.all():
            sig[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), sig[~mask])
        resampled = np.interp(mc_time_vec, rgb_time_vec, sig)
        std = np.std(resampled)
        precomputed_rgb[j] = (resampled - np.mean(resampled)) / (std if std > 1e-8 else 1.0)

    for start in range(0, max_start, 10):
        for j in active_joints:
            win = mocap_z[j][start: start + n_mocap_for_rgb]
            std = np.std(win)
            if std < 5.0:
                continue
            win_norm = (win - np.mean(win)) / std
            combined[start] += np.corrcoef(precomputed_rgb[j], win_norm)[0, 1]

    # ── fine refinement ±50 frames around coarse best ────────────────────────
    coarse_best = int(np.argmax(combined))
    search      = range(max(0, coarse_best - 50), min(max_start, coarse_best + 50))
    refined     = np.zeros(len(search))
    for i, start in enumerate(search):
        for j in active_joints:
            win     = mocap_z[j][start: start + n_mocap_for_rgb]
            std     = np.std(win)
            win_norm = (win - np.mean(win)) / (std if std > 1e-8 else 1.0)
            refined[i] += np.corrcoef(precomputed_rgb[j], win_norm)[0, 1]

    best_start = list(search)[int(np.argmax(refined))]
    best_corr  = float(np.max(refined)) / len(active_joints)
    print(f"  Temporal alignment: mocap start frame {best_start} "
          f"({best_start / mocap_fps:.2f}s), avg corr={best_corr:.4f}")

    # ── build frame mapping ───────────────────────────────────────────────────
    frame_map: dict[int, int] = {}
    for i, fid in enumerate(all_fids):
        t_sec = i / video_fps
        frame_map[fid] = int(best_start + t_sec * mocap_fps)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mocap_start_frame":      best_start,
            "mocap_end_frame":        best_start + n_mocap_for_rgb,
            "mocap_time_start_sec":   round(best_start / mocap_fps, 4),
            "mocap_time_end_sec":     round((best_start + n_mocap_for_rgb) / mocap_fps, 4),
            "video_fps":              video_fps,
            "mocap_fps":              mocap_fps,
            "fps_ratio":              mocap_fps / video_fps,
            "n_rgb_frames":           n_rgb,
            "n_mocap_frames_aligned": n_mocap_for_rgb,
            "correlation_score":      round(best_corr, 6),
            "per_joint_correlations": {
                j: round(float(np.corrcoef(
                    precomputed_rgb[j],
                    (lambda w: (w - np.mean(w)) / (np.std(w) if np.std(w) > 1e-8 else 1.0))(
                        mocap_z[j][best_start: best_start + n_mocap_for_rgb]
                    ),
                )[0, 1]), 6)
                for j in active_joints
            },
            "alignment_method": f"multi-joint cross-correlation ({', '.join(active_joints)})",
            "frame_mapping":    {str(k): v for k, v in frame_map.items()},
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Temporal alignment cached → {cache_path}")

    return frame_map


# ──────────────────────────────────────────────────────────────────────────────
# OBSERVATION LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_mocap_observations(
    data_dir: Path,
    cam_ids: list[int],
    cameras: list[dict],
    video_id: str,
    mocap_frames: str | list[int] = "top5",
    force_recompute: bool = False,
    mocap_fps: float = MOCAP_FPS,
    video_fps: float = VIDEO_FPS,
    joint_preset: str = "all",
    val_frac: float = 0.0,
    seed: int = 42,
) -> MocapObservations | None:
    """
    Load mocap observations with world coordinates pre-computed via Umeyama.

    Parameters
    ----------
    mocap_frames : str or list[int]
        Which video frames to use for BA observations (training split).
        - "topK"  (e.g. "top5"): pick the K frames ranked by lowest mean 2-D
          reprojection error after the Umeyama fit.
        - list[int]: explicit 0-based video frame indices (COCO fid = index + 1).
    force_recompute : bool
        If True, ignore cached temporal_alignment.json and rerun cross-correlation.
    joint_preset : str
        Name of a preset from JOINT_PRESETS controlling which COCO joints are
        used for training.  Does not affect temporal alignment or the Umeyama fit.
        Choices: all | end_effectors | limbs | lower_body | stable.
    val_frac : float
        Fraction of allowed-joint observations from selected frames to hold out
        for validation (sampled randomly, default 0 = none held out).
    seed : int
        RNG seed for the val_frac random hold-out.

    Returns
    -------
    MocapObservations with:
      .train  — allowed-joint obs from selected frames (~1-val_frac kept)
      .val    — union of three pools:
                  A. excluded-joint obs from selected frames
                  B. random val_frac of allowed-joint obs from selected frames
                  C. all obs from non-selected frames
    Returns None on failure (missing files, too few triangulated points, etc.).

    Steps
    -----
    1. Temporal alignment via cross-correlation (cached in temporal_alignment.json).
    2. Triangulate joints from ≥2 cameras across all frames; fit Umeyama.
    3. Score each frame by mean 2-D reprojection error; select requested frames.
    4. Split observations into train / val as described above.
    """
    if joint_preset not in JOINT_PRESETS:
        raise ValueError(
            f"Unknown joint_preset '{joint_preset}'. "
            f"Choose from: {sorted(JOINT_PRESETS)}"
        )
    allowed_ki: frozenset[int] = frozenset(
        i for i, name in enumerate(COCO_KP_NAMES)
        if name in JOINT_PRESETS[joint_preset]
    )

    if not _SCIPY_OK:
        print("[WARN] scipy not available — skipping mocap observations.")
        return None

    mat_path        = data_dir / "mocap_data"   / video_id / "mocap_data.mat"
    coco_path       = data_dir / "annotations"  / video_id / "_annotations.coco.json"
    alignment_cache = data_dir / "mocap_data"   / video_id / "temporal_alignment.json"
    transform_cache = data_dir / "mocap_data"   / video_id / "mocap_transform.json"

    missing = [p for p in (mat_path, coco_path) if not p.exists()]
    if missing:
        print(f"[WARN] Mocap files not found — skipping: {missing}")
        return None

    # ── load mocap ────────────────────────────────────────────────────────────
    mat          = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    nick         = mat[[k for k in mat if not k.startswith("_")][0]]
    pos_mm       = nick.Skeletons.PositionData
    mocap_labels = list(np.asarray(nick.Skeletons.SegmentLabels).astype(str))

    coco_to_mocap: dict[int, int] = {}
    for ki, kname in enumerate(COCO_KP_NAMES):
        mname = JOINT_MAP.get(kname)
        if mname and mname in mocap_labels:
            coco_to_mocap[ki] = mocap_labels.index(mname)

    # ── temporal alignment (cached) ───────────────────────────────────────────
    print("\n[Mocap] Temporal alignment …")
    try:
        frame_map = compute_temporal_alignment(
            pos_mm, mocap_labels, coco_path, cameras, cam_ids,
            mocap_fps=mocap_fps, video_fps=video_fps,
            cache_path=alignment_cache, force_recompute=force_recompute,
        )
    except Exception as exc:
        print(f"[WARN] Temporal alignment failed ({exc}) — skipping mocap.")
        return None

    # ── load COCO annotations for pixel observations ──────────────────────────
    with open(coco_path) as f:
        coco = json.load(f)

    img_info: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    ann_index: dict[tuple[int, int], dict] = {}
    for ann in coco["annotations"]:
        img = img_info.get(ann["image_id"])
        if img is None:
            continue
        m = re.search(r"out(\d+)_frame_(\d+)", img["file_name"])
        if m is None:
            continue
        cid, fid = int(m.group(1)), int(m.group(2))
        ann_index[(fid, cid)] = ann

    cam_id_to_idx = {cid: i for i, cid in enumerate(cam_ids)}

    # ── collect raw mocap positions + pixel observations ──────────────────────
    frame_joints:  dict[int, dict[int, dict[int, np.ndarray]]] = {}
    frame_mocap3d: dict[int, np.ndarray] = {}

    for fid, mocap_fid in frame_map.items():
        if mocap_fid >= pos_mm.shape[2]:
            continue
        mocap_pts = np.zeros((18, 3), dtype=np.float64)
        for ki, mi in coco_to_mocap.items():
            mocap_pts[ki] = pos_mm[:, mi, mocap_fid] / 1000.0

        frame_mocap3d[fid] = mocap_pts
        frame_joints[fid]  = {ki: {} for ki in range(18)}

        for cam_id in cam_ids:
            if cam_id not in cam_id_to_idx:
                continue
            ann = ann_index.get((fid, cam_id))
            if ann is None:
                continue
            kps  = ann["keypoints"]
            cidx = cam_id_to_idx[cam_id]
            for ki in range(18):
                if kps[ki * 3 + 2] == 0:
                    continue
                frame_joints[fid][ki][cidx] = np.array(
                    [kps[ki * 3], kps[ki * 3 + 1]], dtype=np.float64
                )

    if not frame_joints:
        print("[WARN] No matched (frame, cam) pairs in mocap data.")
        return None

    # ── triangulate to seed Umeyama ───────────────────────────────────────────
    tri_pts: list[np.ndarray] = []
    moc_pts: list[np.ndarray] = []

    for fid, joint_views in frame_joints.items():
        mocap_pts = frame_mocap3d[fid]
        for ki, views in joint_views.items():
            if len(views) < 2:
                continue
            tri = triangulate_rays(views, cameras)
            if tri is None:
                continue
            tri_pts.append(tri)
            moc_pts.append(mocap_pts[ki])

    if len(tri_pts) < 4:
        print(f"[WARN] Only {len(tri_pts)} triangulated pairs — skipping mocap.")
        return None

    tri_arr = np.array(tri_pts, dtype=np.float64)
    moc_arr = np.array(moc_pts, dtype=np.float64)

    s_mw, R_mw, t_mw = umeyama(moc_arr, tri_arr)
    reg_errs = np.linalg.norm((s_mw * (R_mw @ moc_arr.T).T + t_mw) - tri_arr, axis=1)
    print(f"  Umeyama  scale={s_mw:.6f}  |t|={np.linalg.norm(t_mw):.3f} m  "
          f"err mean={reg_errs.mean():.4f} m  max={reg_errs.max():.4f} m  "
          f"(n={len(tri_pts)})")

    # ── write transform to cache ──────────────────────────────────────────────
    rvec_mw, _ = cv2.Rodrigues(R_mw)
    mt_vec = np.concatenate([rvec_mw.ravel(), t_mw, [np.log(s_mw)]]).astype(np.float64)
    _save_transform(transform_cache, "seeded", mt_vec)

    # ── score every frame by mean 2-D reprojection error ─────────────────────
    from .cameras import project
    from .geometry import unpack_intr
    from .optim import make_param_blocks

    all_fids = sorted(frame_joints.keys())
    frame_scores: list[tuple[float, int]] = []  # (mean_px_err, fid)

    for fid in all_fids:
        mocap_pts = frame_mocap3d[fid]
        errs: list[float] = []
        for ki, views in frame_joints[fid].items():
            pt3d_w = s_mw * (R_mw @ mocap_pts[ki]) + t_mw
            for cidx, px_obs in views.items():
                ex, intr = make_param_blocks(cameras[cidx])
                mtx, dist = unpack_intr(intr)
                px_proj = project(pt3d_w.reshape(1, 3), ex[0:3], ex[3:6], mtx, dist)[0]
                errs.append(float(np.linalg.norm(px_obs - px_proj)))
        mean_err = float(np.mean(errs)) if errs else float("inf")
        frame_scores.append((mean_err, fid))

    frame_scores.sort(key=lambda x: x[0])  # ascending error → best first

    # ── resolve frame selection ───────────────────────────────────────────────
    if isinstance(mocap_frames, str) and mocap_frames.lower().startswith("top"):
        k = int(mocap_frames[3:])
        selected_fids = [fid for _, fid in frame_scores[:k]]
        print(f"  Frame selection: top-{k} by 2-D reprojection error "
              f"→ COCO fids {sorted(selected_fids)} "
              f"(errors: {[round(e,1) for e,_ in frame_scores[:k]]} px)")
    else:
        # explicit 0-based video frame indices → COCO fids are 1-based
        requested = list(mocap_frames)
        selected_fids = []
        for vf in requested:
            fid = vf + 1
            if fid in frame_joints:
                selected_fids.append(fid)
            else:
                available = ", ".join(str(f - 1) for f in all_fids)
                raise ValueError(
                    f"--mocap-frames: video frame {vf} → COCO fid {fid} not found. "
                    f"Available (0-based): {available}"
                )
        print(f"  Frame selection: explicit video frames {requested} "
              f"→ COCO fids {selected_fids}")

    if not selected_fids:
        print("[WARN] No frames selected — returning empty mocap observations.")
        return None

    selected_set = set(selected_fids)
    val_fids     = [fid for fid in all_fids if fid not in selected_set]

    preset_names = sorted(JOINT_PRESETS[joint_preset] & set(COCO_KP_NAMES),
                          key=lambda n: COCO_KP_NAMES.index(n))
    print(f"  Joint preset: '{joint_preset}' "
          f"({len(allowed_ki)} joints: {', '.join(preset_names)})")

    # ── build observations ────────────────────────────────────────────────────
    # Val is the union of three pools:
    #   A. excluded-joint obs from selected frames   (preset didn't cover them)
    #   B. all obs from non-selected (val) frames
    #   C. a random val_frac of the allowed-joint obs from selected frames
    #
    # Train gets the remaining allowed-joint obs from selected frames (1-val_frac).
    # This means every annotation ends up in exactly one of train or val.

    rng = np.random.default_rng(seed=seed)

    train_obs: Obs = []
    val_obs:   Obs = []

    # Pools A + C: iterate over selected frames
    for fid in selected_fids:
        mocap_pts = frame_mocap3d[fid]
        for ki, views in frame_joints[fid].items():
            pt3d_world = s_mw * (R_mw @ mocap_pts[ki]) + t_mw
            # Draw once per (fid, ki) so all camera views of the same joint
            # land in the same split — required for 3D triangulation in val.
            hold_out = ki not in allowed_ki or (val_frac > 0 and rng.random() < val_frac)
            for cidx, px in views.items():
                entry = (pt3d_world.copy(), px.copy(), cidx)
                if hold_out:
                    val_obs.append(entry)
                else:
                    train_obs.append(entry)

    # Pool B: non-selected frames → all joints → val
    for fid in val_fids:
        mocap_pts = frame_mocap3d[fid]
        for ki, views in frame_joints[fid].items():
            pt3d_world = s_mw * (R_mw @ mocap_pts[ki]) + t_mw
            for cidx, px in views.items():
                val_obs.append((pt3d_world.copy(), px.copy(), cidx))

    n_sel = len(selected_fids)
    n_val_fids = len(val_fids)
    print(f"  Mocap observations — train: {len(train_obs)} "
          f"(from {n_sel} frame{'s' if n_sel != 1 else ''}, "
          f"preset joints, ~{100*(1-val_frac):.0f}% kept)  |  "
          f"val: {len(val_obs)} "
          f"(excluded joints + ~{100*val_frac:.0f}% preset hold-out "
          f"+ {n_val_fids} non-selected frame{'s' if n_val_fids != 1 else ''})")
    return MocapObservations(train=train_obs, val=val_obs)


def _save_transform(path: Path, stage: str, mt: np.ndarray) -> None:
    """Write or update mocap_transform.json with the given stage's mt vector."""
    doc: dict = {}
    if path.exists():
        with open(path) as f:
            doc = json.load(f)
    doc[stage] = {
        "rvec":  [round(float(v), 8) for v in mt[0:3]],
        "tvec":  [round(float(v), 6) for v in mt[3:6]],
        "log_s": round(float(mt[6]), 8),
        "scale": round(float(np.exp(mt[6])), 8),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Mocap transform ({stage}) cached → {path}")
