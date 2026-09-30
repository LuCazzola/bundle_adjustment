"""
Fit camera intrinsics, solve court extrinsics, compare references, and sweep.

Commands:
  run       Fit one configured calibration per camera, then solve extrinsics.
  sweep     Search quick, standard, or deep hyperparameter profiles.
  diagnose  Report holdout, temporal, and board-specific consistency.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import cv2
import numpy as np

try:
    from .utils import (
        BOARDS,
        MODEL_FLAGS,
        benchmark_views,
        calibration_document,
        checkerboard_rms,
        court_score,
        fit_intrinsics,
        load_detection_cache,
        load_intrinsics,
        load_json,
        object_points_by_board,
        print_calibration_result,
        print_header,
        reference_comparison,
        select_cameras,
        select_views,
        write_json,
    )
except ImportError:
    from utils import (
        BOARDS,
        MODEL_FLAGS,
        benchmark_views,
        calibration_document,
        checkerboard_rms,
        court_score,
        fit_intrinsics,
        load_detection_cache,
        load_intrinsics,
        load_json,
        object_points_by_board,
        print_calibration_result,
        print_header,
        reference_comparison,
        select_cameras,
        select_views,
        write_json,
    )


DEFAULT_CONFIG = {
    "square_mm": 28.0,
    "sharp_keep": 0.50,
    "pose_grid": [10, 6, 4],
    "per_bin": 8,
    "reject_iters": 6,
    "reject_sigma": 2.0,
    "max_views": None,
    "calibration_model": "standard_5_term",
}
COURT_WEIGHT = 0.5
CHECKERBOARD_WEIGHT = 0.5
BALANCED_GUARD = 1.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_id(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


def sweep_configs(profile: str) -> list[dict]:
    if profile == "quick":
        axes = product(
            (0.25, 0.35),
            ((6, 4, 3), (8, 5, 3)),
            (2, 3),
            (0, 1),
            (20, 25),
            ("standard_5_term", "fix_k3"),
        )
        configs = [
            {
                "sharp_keep": sharp,
                "pose_grid": list(grid),
                "per_bin": per_bin,
                "reject_iters": reject,
                "reject_sigma": 2.0,
                "max_views": max_views,
                "calibration_model": model,
            }
            for sharp, grid, per_bin, reject, max_views, model in axes
        ]
    elif profile == "standard":
        axes = product(
            (0.25, 0.30, 0.35),
            ((6, 4, 3), (8, 5, 3)),
            (2, 3),
            (0, 1),
            (1.5, 2.0),
            (20, 25),
            ("standard_5_term", "fix_k3"),
        )
        configs = [
            {
                "sharp_keep": sharp,
                "pose_grid": list(grid),
                "per_bin": per_bin,
                "reject_iters": reject,
                "reject_sigma": sigma,
                "max_views": max_views,
                "calibration_model": model,
            }
            for sharp, grid, per_bin, reject, sigma, max_views, model in axes
        ]
    else:
        axes = product(
            (0.10, 0.20, 0.30, 0.50, 0.75, 1.0),
            ((4, 3, 2), (6, 4, 3), (8, 5, 3), (10, 6, 4)),
            (1, 2, 3),
            (0, 1, 2),
            (1.5, 2.0, 2.5),
            (20, 30, 50, None),
            tuple(MODEL_FLAGS),
        )
        configs = [
            {
                "sharp_keep": sharp,
                "pose_grid": list(grid),
                "per_bin": per_bin,
                "reject_iters": reject,
                "reject_sigma": sigma,
                "max_views": max_views,
                "calibration_model": model,
            }
            for sharp, grid, per_bin, reject, sigma, max_views, model in axes
        ]
        # The full Cartesian product is excessive. Keep deterministic coverage
        # while retaining every distortion model and all axis extremes.
        configs = configs[::19]
    unique = {}
    for config in configs:
        unique[json.dumps(config, sort_keys=True)] = config
    return list(unique.values())


def load_camera_inputs(paths, square_mm):
    detections, _, image_size = load_detection_cache(paths.detections)
    if image_size is None or not detections:
        raise ValueError(f"No checkerboard cache at {paths.detections}")
    objects = object_points_by_board(square_mm)
    validation = benchmark_views(list(detections.values()), 60)
    return detections, image_size, objects, validation


def attach_extrinsics(paths, mtx, dist):
    if not paths.court_points.exists():
        return None
    return court_score(paths.court_points, mtx, dist)


def evaluate_reference(paths, mtx, dist, validation, objects):
    return reference_comparison(
        paths,
        mtx,
        dist,
        {"views": validation, "objects": objects},
    )


def active_config(paths, override=None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if paths.sweep_state.exists():
        config.update(load_json(paths.sweep_state).get("best_config", {}))
    if override:
        config.update(override)
    return config


def fit_camera(paths, config, write=True) -> dict:
    detections, image_size, objects, validation = load_camera_inputs(
        paths, config["square_mm"]
    )
    selected = select_views(detections.values(), image_size, config)
    fit = fit_intrinsics(
        selected,
        image_size,
        objects,
        config,
        with_uncertainty=True,
    )
    intrinsic_rms = checkerboard_rms(
        validation, objects, fit["mtx"], fit["dist"]
    )
    court = attach_extrinsics(paths, fit["mtx"], fit["dist"])
    reference = evaluate_reference(
        paths, fit["mtx"], fit["dist"], validation, objects
    )
    document = calibration_document(
        fit["mtx"], fit["dist"], court
    )
    if write:
        write_json(paths.static_calib, document)
    return {
        "config": config,
        "fit": fit,
        "intrinsic_rms_px": intrinsic_rms,
        "court": court,
        "reference": reference,
        "document": document,
    }


def run_command(args) -> int:
    failures = 0
    for paths in select_cameras(args.data_dir, args.cameras):
        print_header(f"out{paths.camera_id}: intrinsics then extrinsics")
        try:
            result = fit_camera(paths, active_config(paths), not args.no_write)
        except (ValueError, RuntimeError, cv2.error) as exc:
            failures += 1
            print(f"  [ERROR] {exc}")
            continue
        print_calibration_result(
            "  result",
            result["fit"],
            result["intrinsic_rms_px"],
            result["court"],
            None if args.no_reference else result["reference"],
        )
        if result["fit"]["focal_uncertainty_pct"]:
            print(
                "  focal uncertainty="
                f"{result['fit']['focal_uncertainty_pct']['fx']:.3f}%"
            )
        if not args.no_write:
            print(f"  written={paths.static_calib}")
    return int(bool(failures))


def score_existing(paths, validation, objects):
    source = paths.static_calib
    _, mtx, dist = load_intrinsics(source)
    intrinsic = checkerboard_rms(validation, objects, mtx, dist)
    court = attach_extrinsics(paths, mtx, dist)
    return source, mtx, dist, intrinsic, court


def combined_score(court, intrinsic, baseline_court, baseline_intrinsic):
    court_term = (
        court / baseline_court
        if baseline_court is not None and court is not None
        else 1.0
    )
    return (
        COURT_WEIGHT * court_term
        + CHECKERBOARD_WEIGHT * intrinsic / baseline_intrinsic
    )


def seed_state(paths, source, config, intrinsic, court, reference):
    state = load_json(paths.sweep_state) if paths.sweep_state.exists() else {}
    state.setdefault("created_at", utc_now())
    state.setdefault("best_config", config)
    state["best_score_px"] = None if court is None else court["rms_px"]
    state["best_intrinsic_rms_px"] = intrinsic
    state["reference_score_px"] = (
        None
        if reference is None or "court" not in reference
        else reference["court"]["rms_px"]
    )
    state["reference_intrinsic_rms_px"] = (
        None if reference is None else reference["intrinsic_rms_px"]
    )
    state["updated_at"] = utc_now()
    baseline = paths.sweep_runs / "baseline"
    write_json(baseline / "config.json", state["best_config"])
    shutil.copy2(paths.static_calib, baseline / "calibration.json")
    write_json(
        baseline / "metrics.json",
        {
            "court_rms_px": state["best_score_px"],
            "checkerboard_rms_px": intrinsic,
            "reference_court_rms_px": state["reference_score_px"],
            "reference_checkerboard_rms_px": state[
                "reference_intrinsic_rms_px"
            ],
        },
    )
    write_json(paths.sweep_state, state)
    return state


def sweep_camera(paths, profile) -> None:
    detections, image_size, objects, validation = load_camera_inputs(
        paths, DEFAULT_CONFIG["square_mm"]
    )
    if not paths.static_calib.exists():
        fit_camera(paths, DEFAULT_CONFIG)
    source, _, _, baseline_intrinsic, baseline_court_result = score_existing(
        paths, validation, objects
    )
    baseline_court = (
        None if baseline_court_result is None else baseline_court_result["rms_px"]
    )
    reference = evaluate_reference(
        paths,
        load_intrinsics(source)[1],
        load_intrinsics(source)[2],
        validation,
        objects,
    )
    state = seed_state(
        paths,
        source,
        active_config(paths),
        baseline_intrinsic,
        baseline_court_result,
        reference,
    )
    best_combined = 1.0
    configs = sweep_configs(profile)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = paths.sweep_runs / run_id
    write_json(
        run_dir / "run.json",
        {
            "created_at": utc_now(),
            "profile": profile,
            "configurations": len(configs),
            "baseline_court_rms_px": baseline_court,
            "baseline_checkerboard_rms_px": baseline_intrinsic,
        },
    )
    print_header(
        f"out{paths.camera_id}: {profile} sweep ({len(configs)} configs)"
    )
    print(
        f"  baseline court={baseline_court} "
        f"checkerboard={baseline_intrinsic:.4f}px"
    )
    results = []
    for index, partial in enumerate(configs, start=1):
        config = {**DEFAULT_CONFIG, **partial}
        selected = select_views(
            detections.values(), image_size, config
        )
        try:
            fit = fit_intrinsics(
                selected, image_size, objects, config
            )
            intrinsic = checkerboard_rms(
                validation, objects, fit["mtx"], fit["dist"]
            )
            court_result = attach_extrinsics(
                paths, fit["mtx"], fit["dist"]
            )
        except (ValueError, RuntimeError, cv2.error) as exc:
            print(f"  [{index:04d}] failed: {exc}")
            continue
        court = None if court_result is None else court_result["rms_px"]
        joint = combined_score(
            court, intrinsic, baseline_court, baseline_intrinsic
        )
        balanced = (
            intrinsic <= baseline_intrinsic * BALANCED_GUARD
            and (
                baseline_court is None
                or court <= baseline_court * BALANCED_GUARD
            )
        )
        improved = balanced and joint < best_combined
        metrics = {
            "config": config,
            "court_rms_px": court,
            "checkerboard_rms_px": intrinsic,
            "training_rms_px": fit["checkerboard_rms_px"],
            "combined_score": joint,
            "balanced_guard_passed": balanced,
            "n_views": fit["n_views"],
        }
        results.append(metrics)
        candidate = run_dir / (
            f"candidate_{index:04d}_{config_id(config)}"
        )
        write_json(candidate / "config.json", config)
        write_json(candidate / "metrics.json", metrics)
        document = calibration_document(
            fit["mtx"], fit["dist"], court_result
        )
        write_json(candidate / "calibration.json", document)
        print(
            f"  [{index:04d}/{len(configs)}] court={court} "
            f"check={intrinsic:.3f}px train={fit['checkerboard_rms_px']:.3f}px "
            f"joint={joint:.4f}{' NEW BEST' if improved else ''}"
        )
        if not improved:
            continue
        best_combined = joint
        write_json(paths.static_calib, document)
        reference_intrinsic = (
            None if reference is None else reference["intrinsic_rms_px"]
        )
        state.update(
            {
                "best_config": config,
                "best_score_px": court,
                "best_intrinsic_rms_px": intrinsic,
                "best_combined_score": joint,
                "best_training_rms_px": fit["checkerboard_rms_px"],
                "best_n_views": fit["n_views"],
                "best_run": str(candidate.relative_to(paths.root)),
                "reference_intrinsic_rms_px": reference_intrinsic,
                "intrinsic_delta_vs_reference_px": (
                    None
                    if reference_intrinsic is None
                    else intrinsic - reference_intrinsic
                ),
                "updated_at": utc_now(),
            }
        )
        write_json(paths.sweep_state, state)
    top = sorted(results, key=lambda item: item["combined_score"])[:20]
    summary = {
        "completed_at": utc_now(),
        "profile": profile,
        "configurations_generated": len(configs),
        "configurations_tested": len(results),
        "run_dir": str(run_dir.relative_to(paths.root)),
        "top_results": top,
    }
    history = state.setdefault("sweep_history", [])
    history.append({key: value for key, value in summary.items() if key != "top_results"})
    state["last_sweep"] = summary
    state["updated_at"] = utc_now()
    write_json(paths.sweep_state, state)
    print(
        f"  best court={state.get('best_score_px')} "
        f"checkerboard={state.get('best_intrinsic_rms_px')}"
    )


def sweep_command(args) -> int:
    failures = 0
    for paths in select_cameras(args.data_dir, args.cameras):
        try:
            sweep_camera(paths, args.profile)
        except (ValueError, RuntimeError, cv2.error) as exc:
            failures += 1
            print(f"out{paths.camera_id}: [ERROR] {exc}")
    return int(bool(failures))


def stratified_split(detections):
    train, holdout = [], []
    for board in BOARDS:
        views = sorted(
            (item for item in detections if item.board == board),
            key=lambda item: item.fid,
        )
        for index, item in enumerate(views):
            (holdout if index % 4 == 0 else train).append(item)
    return train, holdout


def even_sample(views, count):
    views = sorted(views, key=lambda item: item.fid)
    if len(views) <= count:
        return views
    indices = np.linspace(0, len(views) - 1, count, dtype=int)
    return [views[index] for index in indices]


def diagnose_camera(paths) -> dict:
    detections, image_size, objects, _ = load_camera_inputs(
        paths, DEFAULT_CONFIG["square_mm"]
    )
    all_views = list(detections.values())
    train, holdout = stratified_split(all_views)
    holdout = even_sample(holdout, 120)
    report = {
        "camera": f"out{paths.camera_id}",
        "detection_count": len(all_views),
        "holdout_count": len(holdout),
        "tests": {},
    }
    if paths.static_calib.exists():
        _, mtx, dist = load_intrinsics(paths.static_calib)
        report["static_calibration"] = {
            "checkerboard_rms_px": checkerboard_rms(
                holdout, objects, mtx, dist
            ),
            "court": (
                attach_extrinsics(paths, mtx, dist)["rms_px"]
                if paths.court_points.exists()
                else None
            ),
        }
    board_results = {}
    for board in BOARDS:
        board_train = even_sample(
            [item for item in train if item.board == board], 120
        )
        fit = fit_intrinsics(
            board_train,
            image_size,
            objects,
            {**DEFAULT_CONFIG, "reject_iters": 1},
        )
        board_results[board] = {
            "same_board_rms_px": checkerboard_rms(
                [item for item in holdout if item.board == board],
                objects,
                fit["mtx"],
                fit["dist"],
            ),
            "other_board_rms_px": checkerboard_rms(
                [item for item in holdout if item.board != board],
                objects,
                fit["mtx"],
                fit["dist"],
            ),
        }
    report["tests"]["board_only"] = board_results
    temporal = {}
    edges = np.linspace(
        min(item.fid for item in all_views),
        max(item.fid for item in all_views) + 1,
        4,
    )
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        segment = even_sample(
            [item for item in train if low <= item.fid < high], 100
        )
        fit = fit_intrinsics(
            segment,
            image_size,
            objects,
            {**DEFAULT_CONFIG, "reject_iters": 1},
            with_uncertainty=True,
        )
        temporal[f"segment_{index}"] = {
            "frame_range": [int(low), int(high - 1)],
            "fx": float(fit["mtx"][0, 0]),
            "fy": float(fit["mtx"][1, 1]),
            "holdout_rms_px": checkerboard_rms(
                holdout, objects, fit["mtx"], fit["dist"]
            ),
            "focal_uncertainty_pct": fit["focal_uncertainty_pct"],
        }
    report["tests"]["temporal"] = temporal
    report_path = paths.diagnostics / "latest.json"
    write_json(report_path, report)
    return report


def diagnose_command(args) -> int:
    failures = 0
    for paths in select_cameras(args.data_dir, args.cameras):
        print_header(f"out{paths.camera_id}: diagnostics")
        try:
            report = diagnose_camera(paths)
        except (ValueError, RuntimeError, cv2.error) as exc:
            failures += 1
            print(f"  [ERROR] {exc}")
            continue
        print(f"  static={report.get('static_calibration')}")
        print(f"  board-only={report['tests']['board_only']}")
        print(f"  temporal={report['tests']['temporal']}")
        print(f"  written={paths.diagnostics / 'latest.json'}")
    return int(bool(failures))


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Camera intrinsic/extrinsic calibration pipeline."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "cameras",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Fit intrinsics then extrinsics.")
    run.add_argument("--cameras", nargs="*", type=int, default=None)
    run.add_argument("--no-reference", action="store_true")
    run.add_argument("--no-write", action="store_true")
    run.set_defaults(handler=run_command)

    sweep = subparsers.add_parser("sweep", help="Search calibration settings.")
    sweep.add_argument("--cameras", nargs="+", type=int, required=True)
    sweep.add_argument(
        "--profile",
        choices=("quick", "standard", "deep"),
        default="standard",
    )
    sweep.set_defaults(handler=sweep_command)

    diagnose = subparsers.add_parser(
        "diagnose", help="Run temporal and board consistency checks."
    )
    diagnose.add_argument("--cameras", nargs="+", type=int, required=True)
    diagnose.set_defaults(handler=diagnose_command)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
