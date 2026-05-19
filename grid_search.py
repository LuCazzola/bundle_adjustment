"""
Grid search over BA hyperparameters.

Runs ba.main() for every combination of --huber-thresh, --lambda-intr-reg,
and --lambda-cross.  Each run's full output is saved to log/<config>.txt.
A ranked summary table is printed at the end.

Usage
-----
  python grid_search.py [--cameras 1 2 5] [--optimize both] [--data-dir PATH]
"""

import argparse
import io
import itertools
import sys
from contextlib import redirect_stdout
from pathlib import Path

# ── grid to sweep ─────────────────────────────────────────────────────────────
HUBER_THRESH_VALUES    = [1.0, 2.0, 5.0]
LAMBDA_INTR_REG_VALUES = [10.0, 100.0]
LAMBDA_CROSS_VALUES    = [0.0, 1.0, 2.0]
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
import ba as ba_module
from ba import LAMBD

LOG_DIR = Path(__file__).parent / "log"


def _config_name(optimize, huber, intr_reg, cross):
    """Filesystem-safe name for one grid point."""
    return f"opt-{optimize}_huber-{huber:g}_intr-{intr_reg:g}_cross-{cross:g}"


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _extract_after_global(log_text: str) -> tuple[float, float]:
    """
    Pull mean_3d and mean_2d from the GLOBAL line in the AFTER stats block.
    Returns (inf, inf) on parse failure.
    """
    in_after = False
    for line in log_text.splitlines():
        if "AFTER bundle adjustment" in line:
            in_after = True
        if not in_after:
            continue
        # The improvement summary line contains "→" — skip it
        if "GLOBAL" in line and "→" not in line:
            nums = [float(p) for p in line.split() if _is_float(p)]
            if len(nums) >= 5:
                return nums[0], nums[4]   # mean_3d, mean_2d
    return float("inf"), float("inf")


def _run_one(camera_ids, optimize, data_dir, huber, intr_reg, cross, log_path: Path,
             val_frac: float = 0.0):
    """
    Run ba.main(), capture all output, write it to log_path.
    Returns (mean_3d, mean_2d) parsed from the log.
    """
    buf = io.StringIO()
    orig_save = ba_module.save_results
    ba_module.save_results = lambda *a, **kw: None   # no file writes during search
    try:
        with redirect_stdout(buf):
            ba_module.main(
                camera_ids=camera_ids,
                optimize=optimize,
                data_dir=data_dir,
                output_suffix="_gs_tmp",
                lambd=LAMBD(INTR_REG=intr_reg, CROSS_VIEW=cross),
                two_pass=False,
                huber_thresh=huber,
                val_frac=val_frac,
            )
        failed = False
    except SystemExit:
        failed = True
    finally:
        ba_module.save_results = orig_save

    log_text = buf.getvalue()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    if failed:
        return None, None
    return _extract_after_global(log_text)


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over BA hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cameras", nargs="*", type=int, default=[])
    parser.add_argument("--optimize", choices=("ex", "in", "both"), default="both")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).parent / "data" / "cameras",
    )
    parser.add_argument(
        "--sort-by", choices=("3d", "2d"), default="3d",
        help="Metric to rank results by.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.0,
        metavar="F",
        help="Fraction of keypoints held out for validation per run (0 = disabled).",
    )
    args = parser.parse_args()

    grid = list(itertools.product(
        HUBER_THRESH_VALUES,
        LAMBDA_INTR_REG_VALUES,
        LAMBDA_CROSS_VALUES,
    ))

    print(f"Grid search: {len(grid)} combinations  →  logs in {LOG_DIR}/")
    print(f"  huber_thresh     : {HUBER_THRESH_VALUES}")
    print(f"  lambda_intr_reg  : {LAMBDA_INTR_REG_VALUES}")
    print(f"  lambda_cross     : {LAMBDA_CROSS_VALUES}")
    print(f"  optimize         : {args.optimize}")
    print()

    results = []
    for i, (huber, intr_reg, cross) in enumerate(grid, 1):
        name     = _config_name(args.optimize, huber, intr_reg, cross)
        log_path = LOG_DIR / f"{name}.txt"
        tag      = f"[{i:2d}/{len(grid)}]  huber={huber:<4g}  intr_reg={intr_reg:<6g}  cross={cross:<4g}"
        print(tag, end="  ", flush=True)

        m3, m2 = _run_one(
            camera_ids=args.cameras,
            optimize=args.optimize,
            data_dir=args.data_dir,
            huber=huber,
            intr_reg=intr_reg,
            cross=cross,
            log_path=log_path,
            val_frac=args.val_frac,
        )

        if m3 is None:
            print(f"FAILED  (see {log_path})")
            continue

        print(f"3D={m3:.4f}m  2D={m2:.2f}px  →  {log_path.name}")
        results.append((huber, intr_reg, cross, m3, m2, log_path))

    if not results:
        print("No successful runs.")
        return

    key = 3 if args.sort_by == "3d" else 4
    results.sort(key=lambda r: r[key])

    W = 80
    print(f"\n{'─'*W}")
    print(f"  Results sorted by mean {'3D' if args.sort_by == '3d' else '2D'} error  ({args.optimize})")
    print(f"{'─'*W}")
    print(f"  {'huber':>5}  {'intr_reg':>8}  {'cross':>5}  {'mean_3d (m)':>12}  {'mean_2d (px)':>12}  log")
    for j, (huber, intr_reg, cross, m3, m2, log_path) in enumerate(results):
        marker = "  ◀ best" if j == 0 else ""
        print(f"  {huber:5g}  {intr_reg:8g}  {cross:5g}  {m3:12.4f}  {m2:12.2f}  {log_path.name}{marker}")
    print(f"{'─'*W}")

    best = results[0]
    print(
        f"\nBest: --huber-thresh {best[0]} --lambda-intr-reg {best[1]} --lambda-cross {best[2]}"
        f"\n      full log: {best[5]}"
    )


if __name__ == "__main__":
    main()
