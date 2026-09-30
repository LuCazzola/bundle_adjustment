"""
Bundle Adjustment for multi-camera basketball court setup.

Uses known 3D court keypoints (court_points.json) and static best calibration
to jointly refine camera extrinsics and/or intrinsics via Ceres Solver (pyceres).

OBJECTIVE COMPOSITION
---------------------
Each objective is specified as:  <source>/<mode>[:<weight>[:<huber>]]

  source  : court | mocap
  mode    : 2d | 3d_z0 | 3d_tri
  weight  : scalar multiplier on residuals          (default 1.0)
  huber   : Huber loss threshold in residual units  (default 5.0 px / 0.05 m)

Examples:
  --objectives court/2d                         # classic pixel reprojection
  --objectives court/2d court/3d_tri:1.0:0.05  # pixel + 3D coupling
  --objectives court/2d mocap/2d:0.5            # court + weighted mocap

PARAMETER BLOCKS (all float64, contiguous)
------------------------------------------
  ex  (6): rx ry rz  tx ty tz
  in  (9): fx fy cx cy  k1 k2 p1 p2 k3

DATA SOURCES  (--calib-source)
-------------------------------
  checkerboard   Use calibration/static_calib.json

MOCAP CONSTRAINTS
-----------------
  Triggered automatically when any objective uses source=mocap.
  World coordinates are pre-computed via Umeyama alignment (applied once
  upfront, not optimised jointly).

DATA LAYOUT
-----------
  <data-dir>/
    cameras/out<id>/
      annotations/court_points.json
      calibration/static_calib.json
      calibration/ba_calib.json
    mocap_data/<video-id>/
    annotations/<video-id>/
    videos/<video-id>/out<cam_id>.mp4
"""

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cameras import CALIB_SOURCE_FILES, discover_cameras, load_camera, save_results
from scripts.mocap import JOINT_PRESETS, load_mocap_observations
from scripts.optim import CeresOptions, LAMBD, ObjectiveSpec, run_ceres
from scripts.stats import (
    compute_stats, compute_val_stats,
    compute_mocap_stats,
    print_stats_full,
    print_improvement_full,
    print_physical_summary,
)

_DEFAULT_HUBER = {"2d": 10.0, "3d_z0": 0.05, "3d_tri": 0.05}


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """Write every line to both the original stream and a file."""
    def __init__(self, stream: io.TextIOBase, path: Path) -> None:
        self._stream = stream
        self._file   = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> int:
        self._stream.write(s)
        self._stream.flush()
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class RunLogger:
    """
    Manages the two log files for one BA run.

    Usage
    -----
        logger = RunLogger(log_dir, config_dict)
        logger.start()          # redirects stdout → also written to <name>.log
        ...                     # run BA normally; all prints captured
        logger.finish(metrics)  # writes <name>.json and restores stdout
    """

    def __init__(self, log_dir: Path, config: dict) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        slug      = "_".join([
            config["optimize"],
            config["calib_source"],
            *[o.replace("/", "-") for o in config["objectives"]],
        ])
        self.name     = f"{ts}_{slug}"
        self.log_path  = log_dir / f"{self.name}.log"
        self.json_path = log_dir / f"{self.name}.json"
        self._config   = config
        self._tee: _Tee | None = None
        self._orig_stdout = sys.stdout

    def start(self) -> None:
        self._tee = _Tee(self._orig_stdout, self.log_path)
        sys.stdout = self._tee
        print(f"Run log  → {self.log_path}")
        print(f"Run JSON → {self.json_path}")

    def finish(self, metrics: dict) -> None:
        payload = {"config": self._config, "metrics": _serialise(metrics)}
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nJSON log → {self.json_path}")
        if self._tee is not None:
            sys.stdout = self._orig_stdout
            self._tee.close()


def _serialise(obj):
    """Recursively make stat dicts JSON-safe (round floats, drop None)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def parse_objectives(specs: list[str]) -> list[ObjectiveSpec]:
    """Parse objective strings into ObjectiveSpec instances."""
    result = []
    for s in specs:
        parts = s.split(":")
        src_mode = parts[0].split("/")
        if len(src_mode) != 2:
            raise argparse.ArgumentTypeError(
                f"Objective '{s}' must be <source>/<mode>[:<weight>[:<huber>]]"
            )
        source, mode = src_mode
        if source not in ("court", "mocap"):
            raise argparse.ArgumentTypeError(f"Unknown source '{source}' in '{s}'")
        if mode not in ("2d", "3d_z0", "3d_tri"):
            raise argparse.ArgumentTypeError(f"Unknown mode '{mode}' in '{s}'")
        weight = float(parts[1]) if len(parts) > 1 else 1.0
        huber  = float(parts[2]) if len(parts) > 2 else _DEFAULT_HUBER[mode]
        result.append(ObjectiveSpec(source=source, mode=mode, weight=weight, huber=huber))
    return result


def main(
    camera_ids: list[int],
    optimize: str,
    data_dir: Path,
    output_suffix: str,
    lambd: LAMBD,
    two_pass: bool,
    val_frac: float,
    objectives: list[ObjectiveSpec],
    calib_source: str = "checkerboard",
    video_id: str = "mocap_4",
    mocap_frames: str | list[int] = "top5",
    recompute_alignment: bool = False,
    seed: int = 42,
    joint_preset: str = "all",
    log_dir: Path | None = None,
    ceres_opts: CeresOptions | None = None,
) -> None:

    assert optimize in ("ex", "in", "both"), \
        f"--optimize must be 'ex', 'in', or 'both', got '{optimize}'"
    assert calib_source in CALIB_SOURCE_FILES, \
        f"--calib-source must be one of {list(CALIB_SOURCE_FILES)}, got '{calib_source}'"

    cam_ids = discover_cameras(data_dir, camera_ids, calib_source=calib_source)
    if not cam_ids:
        sys.exit("[ERROR] No cameras with required data found.")

    needs_mocap = any(o.source == "mocap" for o in objectives)

    _co = ceres_opts or CeresOptions()
    config = {
        "optimize":            optimize,
        "calib_source":        calib_source,
        "video_id":            video_id,
        "cameras":             cam_ids,
        "two_pass":            two_pass and optimize == "both",
        "lambda_intr_reg":     lambd.INTR_REG if optimize == "both" else None,
        "objectives":          [f"{o.source}/{o.mode}:w{o.weight}:h{o.huber}" for o in objectives],
        "mocap_frames":        mocap_frames if isinstance(mocap_frames, str) else list(mocap_frames),
        "joint_preset":        joint_preset,
        "val_frac":            val_frac,
        "seed":                seed,
        "output_suffix":       output_suffix,
        "data_dir":            str(data_dir),
        "ceres_max_iter":      _co.max_num_iterations,
        "ceres_fn_tol":        _co.function_tolerance,
        "ceres_grad_tol":      _co.gradient_tolerance,
        "ceres_param_tol":     _co.parameter_tolerance,
        "ceres_tr_strategy":   _co.trust_region_strategy,
        "ceres_linear_solver": _co.linear_solver_type,
    }

    logger = RunLogger(log_dir or data_dir.parent / "log", config)
    logger.start()

    print(f"\nCameras selected : {cam_ids}")
    print(f"Optimize         : {optimize}")
    print(f"Calib source     : {calib_source}")
    print(f"Video ID         : {video_id}")
    print(f"Two-pass         : {two_pass and optimize == 'both'}")
    if optimize == "both":
        print(f"λ intr_reg       : {lambd.INTR_REG}")
    print(f"Objectives       :")
    for o in objectives:
        print(f"  {o.source}/{o.mode}  weight={o.weight}  huber={o.huber}")
    print(f"Mocap frames     : {mocap_frames}")
    print(f"Joint preset     : {joint_preset}")
    print(f"Val fraction     : {val_frac:.0%}  ({'enabled' if val_frac > 0 else 'disabled'})")
    print(f"Val seed         : {seed}")
    print(f"Data directory   : {data_dir}")
    print(f"Ceres max_iter   : {_co.max_num_iterations}")
    print(f"Ceres fn_tol     : {_co.function_tolerance}")
    print(f"Ceres grad_tol   : {_co.gradient_tolerance}")
    print(f"Ceres param_tol  : {_co.parameter_tolerance}")
    print(f"Ceres TR strategy: {_co.trust_region_strategy}")

    cameras = [load_camera(data_dir, cid, val_frac=val_frac,
                           calib_source=calib_source, seed=seed) for cid in cam_ids]

    if needs_mocap:
        try:
            import scipy.io  # noqa: F401
        except ImportError:
            sys.exit("[ERROR] mocap objectives require scipy. Install with: pip install scipy")

    mocap_split = load_mocap_observations(
        data_dir, cam_ids, cameras,
        video_id=video_id,
        mocap_frames=mocap_frames,
        force_recompute=recompute_alignment,
        joint_preset=joint_preset,
        val_frac=val_frac,
        seed=seed,
    )
    if mocap_split is None:
        print("[WARN] No mocap observations loaded — mocap stats will be skipped.")

    # Unpack train / val splits for downstream use
    mocap_train = mocap_split.train if mocap_split is not None else None
    mocap_val   = mocap_split.val   if mocap_split is not None else None

    court_before     = compute_stats(cameras)
    mocap_before     = compute_mocap_stats(cameras, mocap_train)
    court_val_before = compute_val_stats(cameras)
    mocap_val_before = compute_mocap_stats(cameras, mocap_val)
    print_stats_full(
        "BEFORE bundle adjustment  (after solvePnP warm-start)",
        court_before, mocap_before,
    )

    run_kwargs = dict(objectives=objectives, mocap_obs=mocap_train, ceres_opts=_co)

    if two_pass and optimize == "both":
        cameras = run_ceres(cameras, "ex", lambd.INTR_REG,
                            "pass 1/2 — extrinsics only", **run_kwargs)
        court_pass1 = compute_stats(cameras)
        mocap_pass1 = compute_mocap_stats(cameras, mocap_train)
        print_stats_full("AFTER pass 1 (extrinsics only)",
                         court_pass1, mocap_pass1)
        cameras_opt = run_ceres(cameras, "both", lambd.INTR_REG,
                                "pass 2/2 — extrinsics + intrinsics", **run_kwargs)
    else:
        cameras_opt = run_ceres(cameras, optimize, lambd.INTR_REG,
                                "single pass", **run_kwargs)

    court_after      = compute_stats(cameras_opt)
    mocap_after      = compute_mocap_stats(cameras_opt, mocap_train)
    court_val_stats  = compute_val_stats(cameras_opt)
    mocap_val_stats  = compute_mocap_stats(cameras_opt, mocap_val)

    has_val = court_val_stats is not None or mocap_val_stats is not None
    print_physical_summary(cameras_opt)
    print_stats_full(
        "AFTER bundle adjustment  (train / val)" if has_val
        else "AFTER bundle adjustment  (train set)",
        court_after, mocap_after,
        court_val=court_val_stats,
        mocap_val=mocap_val_stats,
    )
    print_improvement_full(
        court_before, court_after,
        mocap_before, mocap_after,
        before_court_val=court_val_before,
        after_court_val=court_val_stats,
        before_mocap_val=mocap_val_before,
        after_mocap_val=mocap_val_stats,
    )

    print(f"\nSaving (suffix='{output_suffix}') …")
    save_results(cameras_opt, data_dir, output_suffix, optimize,
                 calib_source=calib_source)
    print("\nDone.")

    logger.finish({
        "court": {
            "before": {"train": court_before, "val": court_val_before},
            "after":  {"train": court_after,  "val": court_val_stats},
        },
        "mocap": {
            "before": {"train": mocap_before, "val": mocap_val_before},
            "after":  {"train": mocap_after,  "val": mocap_val_stats},
        },
    })


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
        "--objectives", nargs="+", default=["court/2d"],
        metavar="SPEC",
        help=(
            "One or more objective specs: <source>/<mode>[:<weight>[:<huber>]]. "
            "source: court|mocap. mode: 2d|3d_z0|3d_tri. "
            "Default huber: 5.0 px (2d) or 0.05 m (3d_*). "
            "Example: court/2d court/3d_tri:1.0:0.05 mocap/2d:0.5"
        ),
    )
    parser.add_argument(
        "--calib-source", choices=tuple(CALIB_SOURCE_FILES), default="checkerboard",
        metavar="SOURCE",
        help="Starting calibration source (currently only 'checkerboard').",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data",
        metavar="PATH",
        help="Root data folder (contains cameras/, mocap_data/, annotations/).",
    )
    parser.add_argument(
        "--suffix", default="",
        metavar="SUFFIX",
        help=(
            "Optional suffix for an artifact output. Empty publishes "
            "calibration/ba_calib.json; '_trial' writes "
            "artifacts/bundle_adjustment/ba_calib_trial.json."
        ),
    )
    parser.add_argument(
        "--lambda-intr-reg", type=float, default=1.0,
        metavar="λ",
        help="λ for intrinsic regularisation toward checkerboard prior (--optimize both only).",
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
    parser.add_argument(
        "--mocap-frames", nargs="+", default=["top5"],
        metavar="SPEC",
        help=(
            "Which video frames to use for mocap observations. "
            "'topK' selects the K frames with lowest 2-D reprojection error after Umeyama "
            "(e.g. top5, top1). Explicit 0-based indices also accepted (e.g. 0 3 7). "
            "Default: top5."
        ),
    )
    parser.add_argument(
        "--video", default="mocap_4",
        metavar="VIDEO_ID",
        help=(
            "Video identifier subfolder (e.g. 'mocap_4'). "
            "Selects data from annotations/<VIDEO_ID>/, mocap_data/<VIDEO_ID>/, "
            "and videos/<VIDEO_ID>/."
        ),
    )
    parser.add_argument(
        "--mocap-joints", choices=tuple(sorted(JOINT_PRESETS)), default="all",
        metavar="PRESET",
        help=(
            "Which mocap joints to include in BA observations. "
            "Choices: " + " | ".join(sorted(JOINT_PRESETS)) + ". "
            "Does not affect temporal alignment or the Umeyama fit. "
            "Subsets can improve stability when some joints are hard to annotate reliably. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--recompute-alignment", action="store_true",
        help="Force recomputation of temporal alignment even if temporal_alignment.json exists.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        metavar="S",
        help="Base RNG seed for val/train split (seed + cam_id per camera). Default: 42.",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=PROJECT_ROOT / "logs" / "bundle_adjustment",
        metavar="PATH",
        help="Directory for run log files (<name>.log and <name>.json). Default: logs/bundle_adjustment/.",
    )

    # ── Ceres solver options ──────────────────────────────────────────────────
    parser.add_argument(
        "--ceres-max-iter", type=int, default=500,
        metavar="N",
        help="Ceres max_num_iterations. Default: 500.",
    )
    parser.add_argument(
        "--ceres-fn-tol", type=float, default=1e-8,
        metavar="TOL",
        help="Ceres function_tolerance (relative cost change). Default: 1e-8.",
    )
    parser.add_argument(
        "--ceres-grad-tol", type=float, default=1e-10,
        metavar="TOL",
        help="Ceres gradient_tolerance (gradient norm). Default: 1e-10.",
    )
    parser.add_argument(
        "--ceres-param-tol", type=float, default=1e-8,
        metavar="TOL",
        help="Ceres parameter_tolerance (step size). Default: 1e-8.",
    )
    parser.add_argument(
        "--ceres-tr-strategy", choices=("lm", "dogleg"), default="lm",
        metavar="STRAT",
        help="Trust-region strategy: 'lm' (Levenberg-Marquardt) or 'dogleg'. Default: lm.",
    )
    parser.add_argument(
        "--ceres-linear-solver",
        choices=("dense_schur", "sparse_schur", "sparse_normal_cholesky", "iterative_schur"),
        default="dense_schur",
        metavar="SOLVER",
        help="Linear solver type. Default: dense_schur.",
    )

    args = parser.parse_args()

    try:
        objectives = parse_objectives(args.objectives)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    # Parse --mocap-frames: single "topK" token or list of ints
    raw_frames = args.mocap_frames
    if len(raw_frames) == 1 and raw_frames[0].lower().startswith("top"):
        mocap_frames: str | list[int] = raw_frames[0]
    else:
        try:
            mocap_frames = [int(v) for v in raw_frames]
        except ValueError:
            parser.error(
                f"--mocap-frames: expected 'topK' or integer frame indices, got {raw_frames}"
            )

    data_dir = args.data_dir

    main(
        camera_ids=args.cameras,
        optimize=args.optimize,
        data_dir=data_dir,
        output_suffix=args.suffix,
        lambd=LAMBD(INTR_REG=args.lambda_intr_reg),
        two_pass=args.two_pass,
        val_frac=args.val_frac,
        objectives=objectives,
        calib_source=args.calib_source,
        video_id=args.video,
        mocap_frames=mocap_frames,
        recompute_alignment=args.recompute_alignment,
        seed=args.seed,
        joint_preset=args.mocap_joints,
        log_dir=args.log_dir,
        ceres_opts=CeresOptions(
            max_num_iterations=args.ceres_max_iter,
            function_tolerance=args.ceres_fn_tol,
            gradient_tolerance=args.ceres_grad_tol,
            parameter_tolerance=args.ceres_param_tol,
            trust_region_strategy=args.ceres_tr_strategy,
            linear_solver_type=args.ceres_linear_solver,
        ),
    )

    # python ba.py --objectives court/2d:5:5 mocap/2d:1:5
