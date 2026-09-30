"""
Grid search over BA hyperparameter configurations.

Runs each configuration N times with different random val/train splits (seeds),
reads the per-run JSON log, aggregates the held-out validation metrics across
seeds, and prints a ranked summary table.

The primary ranking metric is  mocap val 2-D global mean  (px).
Court val 2-D global mean is shown as a secondary column.

USAGE
-----
  python grid_search.py              # run all CONFIGS below, 8 seeds each
  python grid_search.py --n-seeds 4  # fewer seeds
  python grid_search.py --dry-run    # print commands without executing
  python grid_search.py --out results.json

ADDING / REMOVING CONFIGURATIONS
---------------------------------
Edit the CONFIGS list below.  Each entry is a dict of ba.py CLI args
(keys = argparse dest names, values = the argument value).  What you
see in CONFIGS is exactly what gets passed to ba.py — no hidden defaults,
no merging.
"""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATIONS TO TEST
# Each dict is one complete ba.py invocation (minus --seed and --log-dir,
# which are injected automatically).  Add / remove entries freely.
#
# Fixed across ALL configs:
#   optimize      = both          (extrinsics + intrinsics)
#   joint_preset  = all           (all mocap joints)
#   val_frac      = 0.2
#   losses        = 2D only       (court/2d + mocap/2d)
#
# Axes swept:
#   A  court/mocap weight ratio
#   B  Huber threshold — court
#   C  Huber threshold — mocap
#   D  λ (intrinsic regularisation)
#   E  two_pass on/off  ×  λ
#   F  mocap frame count
#   G  Ceres max_num_iterations
#   H  Ceres tolerances
#   I  Ceres trust-region strategy
# ──────────────────────────────────────────────────────────────────────────────

_BASE = dict(
    joint_preset="all",
    val_frac=0.2,
    optimize="both",
    mocap_frames="top5",
)

CONFIGS: list[dict] = [

    # ── A: COURT / MOCAP WEIGHT RATIO ─────────────────────────────────────────
    # λ=10 throughout A so intrinsics are well-anchored while we probe weighting

    # A1: court only — no mocap (baseline)
    {**_BASE, "objectives": ["court/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # A2: court dominant  (5:1)
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # A3: equal weight  (1:1)
    {**_BASE, "objectives": ["court/2d:1:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # A4: mocap dominant  (1:5)
    {**_BASE, "objectives": ["court/2d:1:10", "mocap/2d:5:10"],
     "lambda_intr_reg": 10.0},

    # A5: strongly court-dominant  (10:1)
    {**_BASE, "objectives": ["court/2d:10:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # A6: strongly mocap-dominant  (1:10)
    {**_BASE, "objectives": ["court/2d:1:10", "mocap/2d:10:10"],
     "lambda_intr_reg": 10.0},

    # ── B: COURT HUBER THRESHOLD  (mocap huber fixed at 10, weight 5:1) ───────

    # B1: very tight court Huber — strongly rejects court outliers
    {**_BASE, "objectives": ["court/2d:5:2", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # B2: moderate
    {**_BASE, "objectives": ["court/2d:5:5", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # B3: standard (reference for B)
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # B4: loose
    {**_BASE, "objectives": ["court/2d:5:20", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # B5: very loose — essentially L2
    {**_BASE, "objectives": ["court/2d:5:50", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # ── C: MOCAP HUBER THRESHOLD  (court huber fixed at 10, weight 5:1) ───────

    # C1: very tight mocap Huber
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:2"],
     "lambda_intr_reg": 10.0},

    # C2: moderate
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:5"],
     "lambda_intr_reg": 10.0},

    # C3: standard (reference for C, same as B3)
    # (duplicate of B3 — intentionally omitted)

    # C4: loose
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:20"],
     "lambda_intr_reg": 10.0},

    # C5: very loose
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:50"],
     "lambda_intr_reg": 10.0},

    # ── D: INTRINSIC REGULARISATION λ  (fixed weights 5:1, hubers 10/10) ──────

    # D1: nearly free intrinsics
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 0.01},

    # D2
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 0.1},

    # D3
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 1.0},

    # D4 (reference)
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0},

    # D5: tight prior
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 100.0},

    # D6: nearly frozen intrinsics
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 1000.0},

    # ── E: TWO-PASS STRATEGY  (weights 5:1, hubers 10/10) ────────────────────

    # E1: single-pass, λ=1
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 1.0},

    # E2: two-pass, λ=1
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 1.0, "two_pass": True},

    # E3: single-pass, λ=10
    # (same as D4 — reference point for two-pass comparison)

    # E4: two-pass, λ=10
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "two_pass": True},

    # E5: two-pass, λ=100
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 100.0, "two_pass": True},

    # ── F: MOCAP FRAME COUNT  (weights 5:1, hubers 10/10, λ=10) ──────────────

    # F1: single best frame
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "mocap_frames": "top1"},

    # F2
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "mocap_frames": "top3"},

    # F3 (reference — same as D4)

    # F4
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "mocap_frames": "top10"},

    # F5
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "mocap_frames": "top20"},

    # ── G: CERES MAX ITERATIONS  (weights 5:1, hubers 10/10, λ=10) ───────────

    # G1: very few — check if it converges early
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "ceres_max_iter": 50},

    # G2
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "ceres_max_iter": 100},

    # G3
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "ceres_max_iter": 200},

    # G4: default (500) — reference for G
    # (same as D4)

    # G5
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "ceres_max_iter": 1000},

    # ── H: CERES TOLERANCES  (weights 5:1, hubers 10/10, λ=10) ──────────────

    # H1: loose — faster, may under-converge
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0,
     "ceres_fn_tol": 1e-6, "ceres_grad_tol": 1e-6, "ceres_param_tol": 1e-6},

    # H2: default tolerances — reference for H
    # (same as D4)

    # H3: tight
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0,
     "ceres_fn_tol": 1e-10, "ceres_grad_tol": 1e-12, "ceres_param_tol": 1e-10},

    # ── I: TRUST-REGION STRATEGY  (weights 5:1, hubers 10/10, λ=10) ──────────

    # I1: Levenberg-Marquardt (default) — reference for I
    # (same as D4)

    # I2: Powell's dogleg
    {**_BASE, "objectives": ["court/2d:5:10", "mocap/2d:1:10"],
     "lambda_intr_reg": 10.0, "ceres_tr_strategy": "dogleg"},

]


# ──────────────────────────────────────────────────────────────────────────────
# JOINT-PRESET SWEEP
# Fixed config matching:  python ba.py --objectives court/2d:5:5 mocap/2d:1:5
# (weights 5:1, Huber thresholds 5 px court / 5 px mocap, all other settings
#  at their ba.py defaults: λ=1.0, frames=top5, single-pass, LM solver)
#
# Axes swept:
#   J  joint preset  (all 5 presets from JOINT_PRESETS)
# ──────────────────────────────────────────────────────────────────────────────

_JOINT_BASE = dict(
    val_frac=0.2,
    optimize="both",
    mocap_frames="top5",
    objectives=["court/2d:5:5", "mocap/2d:1:5"],
    lambda_intr_reg=0.0,
)

JOINT_CONFIGS: list[dict] = [
    # J1: all joints (default baseline)
    {**_JOINT_BASE, "joint_preset": "all"},

    # J2: end-effectors only (hands, feet, head — easiest to localise)
    {**_JOINT_BASE, "joint_preset": "end_effectors"},

    # J3: full limbs (arms + legs, no trunk)
    {**_JOINT_BASE, "joint_preset": "limbs"},

    # J4: lower body only (strongest geometric anchor — near court plane)
    {**_JOINT_BASE, "joint_preset": "lower_body"},

    # J5: stable large joints (shoulders, hips, knees, ankles)
    {**_JOINT_BASE, "joint_preset": "stable"},
]


# ──────────────────────────────────────────────────────────────────────────────
# LINEAR SOLVER SWEEP
# Fixed config matching the rank-3 equilibrium result:
#   obj=court/2d:1:10+mocap/2d:1:10  opt=both  frames=top5  λ=10.0
# All solver flags left at their ba.py defaults except linear_solver_type.
#
# Axis swept:
#   K  linear_solver_type  (dense_schur | sparse_schur |
#                           sparse_normal_cholesky | iterative_schur)
# ──────────────────────────────────────────────────────────────────────────────

_SOLVER_BASE = dict(
    val_frac=0.2,
    optimize="both",
    mocap_frames="top5",
    objectives=["court/2d:1:10", "mocap/2d:1:10"],
    lambda_intr_reg=10.0,
    joint_preset="all",
)

SOLVER_CONFIGS: list[dict] = [
    # K1: dense Schur complement — pyceres default, fast for small camera counts
    {**_SOLVER_BASE, "ceres_linear_solver": "dense_schur"},

    # K2: sparse Schur complement — textbook BA choice for larger problems
    {**_SOLVER_BASE, "ceres_linear_solver": "sparse_schur"},

    # K3: sparse normal equations — no Schur elimination, simpler structure
    {**_SOLVER_BASE, "ceres_linear_solver": "sparse_normal_cholesky"},

    # K4: iterative Schur (PCG) — scales best to large problems, less exact
    {**_SOLVER_BASE, "ceres_linear_solver": "iterative_schur"},
]


# ──────────────────────────────────────────────────────────────────────────────
# METRIC EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────

def _global_mean(metrics: dict, source: str, phase: str, split: str,
                 dim: str = "2d") -> float | None:
    """
    Extract metrics[source][phase][split][dim]["GLOBAL"]["mean"].
    Returns None if any key is missing (e.g. mocap not available).
    """
    try:
        return metrics[source][phase][split][dim]["GLOBAL"]["mean"]
    except (KeyError, TypeError):
        return None


def extract_scalars(json_path: Path) -> dict[str, float | None]:
    """Return the eight headline scalars we care about from one run JSON."""
    with open(json_path) as f:
        doc = json.load(f)
    m = doc["metrics"]
    return {
        "court_train":    _global_mean(m, "court", "after", "train"),
        "court_val":      _global_mean(m, "court", "after", "val"),
        "mocap_train":    _global_mean(m, "mocap",  "after", "train"),
        "mocap_val":      _global_mean(m, "mocap",  "after", "val"),
        "court_train_3d": _global_mean(m, "court", "after", "train", "3d"),
        "court_val_3d":   _global_mean(m, "court", "after", "val",   "3d"),
        "mocap_train_3d": _global_mean(m, "mocap",  "after", "train", "3d"),
        "mocap_val_3d":   _global_mean(m, "mocap",  "after", "val",   "3d"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AggStats:
    values: list[float] = field(default_factory=list)

    def add(self, v: float | None) -> None:
        if v is not None:
            self.values.append(v)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else float("nan")

    @property
    def median(self) -> float:
        return float(np.median(self.values)) if self.values else float("nan")

    @property
    def stderr(self) -> float:
        if len(self.values) < 2:
            return float("nan")
        return float(np.std(self.values, ddof=1) / math.sqrt(len(self.values)))

    @property
    def best(self) -> float:
        return float(np.min(self.values)) if self.values else float("nan")

    def fmt(self) -> str:
        return f"{self.mean:6.2f} ± {self.stderr:4.2f}  (med {self.median:5.2f})"


@dataclass
class ConfigResult:
    label: str
    config: dict
    court_val:      AggStats = field(default_factory=AggStats)
    court_train:    AggStats = field(default_factory=AggStats)
    mocap_val:      AggStats = field(default_factory=AggStats)
    mocap_train:    AggStats = field(default_factory=AggStats)
    court_val_3d:   AggStats = field(default_factory=AggStats)
    court_train_3d: AggStats = field(default_factory=AggStats)
    mocap_val_3d:   AggStats = field(default_factory=AggStats)
    mocap_train_3d: AggStats = field(default_factory=AggStats)
    failed_seeds: list[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# CLI ARG BUILDING
# ──────────────────────────────────────────────────────────────────────────────

# Maps CONFIGS key names → actual ba.py flag names (where they differ).
_FLAG_MAP = {
    "joint_preset":      "mocap-joints",
    "lambda_intr_reg":   "lambda-intr-reg",
    "two_pass":          "two-pass",
    "mocap_frames":      "mocap-frames",
    "ceres_max_iter":    "ceres-max-iter",
    "ceres_fn_tol":        "ceres-fn-tol",
    "ceres_grad_tol":      "ceres-grad-tol",
    "ceres_param_tol":     "ceres-param-tol",
    "ceres_tr_strategy":   "ceres-tr-strategy",
    "ceres_linear_solver": "ceres-linear-solver",
}

def _build_argv(cfg: dict, seed: int, log_dir: Path) -> list[str]:
    """Convert a config dict + seed into a ba.py argv list."""
    argv = [sys.executable, str(Path(__file__).with_name("ba.py"))]

    for key, val in cfg.items():
        flag = "--" + _FLAG_MAP.get(key, key.replace("_", "-"))
        if isinstance(val, list):
            argv += [flag] + [str(v) for v in val]
        elif isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]

    argv += ["--seed", str(seed), "--log-dir", str(log_dir)]
    return argv


_LABEL_SKIP = {"val_frac"}   # fixed across all configs — skip in label

_LABEL_EXPLICIT = ("objectives", "optimize", "mocap_frames",
                   "lambda_intr_reg", "two_pass", "joint_preset",
                   "ceres_linear_solver")

def _config_label(cfg: dict, show_joint: bool = False,
                  show_solver: bool = False) -> str:
    """Human-readable one-line label for a config dict."""
    parts = []
    if "objectives" in cfg:
        parts.append("obj=" + "+".join(cfg["objectives"]))
    if "optimize" in cfg:
        parts.append(f"opt={cfg['optimize']}")
    if "mocap_frames" in cfg:
        parts.append(f"frames={cfg['mocap_frames']}")
    if "lambda_intr_reg" in cfg:
        parts.append(f"λ={cfg['lambda_intr_reg']}")
    if cfg.get("two_pass"):
        parts.append("2pass")
    if show_joint and "joint_preset" in cfg:
        parts.append(f"joints={cfg['joint_preset']}")
    if show_solver and "ceres_linear_solver" in cfg:
        parts.append(f"solver={cfg['ceres_linear_solver']}")
    for k, v in cfg.items():
        if k in _LABEL_SKIP or k in _LABEL_EXPLICIT:
            continue
        parts.append(f"{k}={v}")
    return "  ".join(parts) if parts else json.dumps(cfg, separators=(",", ":"))


# ──────────────────────────────────────────────────────────────────────────────
# PRINTING
# ──────────────────────────────────────────────────────────────────────────────

_W = 118

def _print_results(results: list[ConfigResult]) -> None:
    def _rank_key(r: ConfigResult) -> float:
        if r.mocap_val.n > 0:
            return r.mocap_val.mean
        return r.court_val.mean if r.court_val.n > 0 else float("inf")

    ranked = sorted(results, key=_rank_key)

    print(f"\n{'═'*_W}")
    print(f"  Grid search results")
    print(f"  Primary sort: mocap val 2-D global mean px  (lower = better)")
    print(f"{'═'*_W}")

    lw = max(len(r.label) for r in ranked) + 2
    print(f"  {'rank':>4}  {'configuration':<{lw}}  "
          f"{'mocap val (px)':>28}  {'court val (px)':>28}  {'n':>3}  {'failed':>6}")
    print(f"  {'─'*(_W-2)}")

    for rank, r in enumerate(ranked, 1):
        mocap_s  = r.mocap_val.fmt() if r.mocap_val.n  > 0 else f"{'—':>28}"
        court_s  = r.court_val.fmt() if r.court_val.n  > 0 else f"{'—':>28}"
        failed_s = str(len(r.failed_seeds)) if r.failed_seeds else "—"
        marker   = "  ◀ best" if rank == 1 else ""
        print(f"  {rank:>4}  {r.label:<{lw}}  {mocap_s}  {court_s}  "
              f"{r.mocap_val.n or r.court_val.n:>3}  {failed_s:>6}{marker}")

    print(f"\n{'─'*_W}")
    print(f"  Detail 2-D  (train shown for reference; val is the ranking signal)")
    print(f"{'─'*_W}")
    print(f"  {'rank':>4}  {'configuration':<{lw}}  "
          f"{'mocap train':>18}  {'mocap val':>18}  "
          f"{'court train':>18}  {'court val':>18}")
    print(f"  {'─'*(_W-2)}")
    for rank, r in enumerate(ranked, 1):
        def _s(a: AggStats) -> str:
            return f"{a.mean:6.2f} ±{a.stderr:4.2f}" if a.n > 0 else f"{'—':>11}"
        print(f"  {rank:>4}  {r.label:<{lw}}  "
              f"{_s(r.mocap_train):>18}  {_s(r.mocap_val):>18}  "
              f"{_s(r.court_train):>18}  {_s(r.court_val):>18}")

    # ── 3-D section (metres) ──────────────────────────────────────────────────
    has_3d = any(r.mocap_val_3d.n > 0 or r.court_val_3d.n > 0 for r in ranked)
    if has_3d:
        def _s3(a: AggStats) -> str:
            return f"{a.mean*100:6.2f} ±{a.stderr*100:4.2f}" if a.n > 0 else f"{'—':>11}"

        print(f"\n{'─'*_W}")
        print(f"  Detail 3-D triangulation error  (cm, val pairs only)")
        print(f"  Note: court 3-D = pairwise triangulation residual (cm);")
        print(f"        mocap 3-D = world-space reprojection error (cm)")
        print(f"{'─'*_W}")
        print(f"  {'rank':>4}  {'configuration':<{lw}}  "
              f"{'mocap train (cm)':>18}  {'mocap val (cm)':>18}  "
              f"{'court train (cm)':>18}  {'court val (cm)':>18}")
        print(f"  {'─'*(_W-2)}")
        for rank, r in enumerate(ranked, 1):
            print(f"  {rank:>4}  {r.label:<{lw}}  "
                  f"{_s3(r.mocap_train_3d):>18}  {_s3(r.mocap_val_3d):>18}  "
                  f"{_s3(r.court_train_3d):>18}  {_s3(r.court_val_3d):>18}")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY WRITING
# ──────────────────────────────────────────────────────────────────────────────

def _results_payload(results: list[ConfigResult], n_seeds: int) -> dict:
    def _agg_dict(a: AggStats) -> dict:
        return {"mean": round(a.mean, 4), "stderr": round(a.stderr, 4),
                "median": round(a.median, 4), "n": a.n}

    def _rank_key(r: ConfigResult) -> float:
        if r.mocap_val.n > 0:
            return r.mocap_val.mean
        return r.court_val.mean if r.court_val.n > 0 else float("inf")

    ranked = sorted(results, key=_rank_key)
    return {
        "n_seeds": n_seeds,
        "ranked_results": [
            {
                "rank":           rank,
                "label":          r.label,
                "config":         r.config,
                "failed_seeds":   r.failed_seeds,
                "mocap_val":      _agg_dict(r.mocap_val),
                "mocap_train":    _agg_dict(r.mocap_train),
                "court_val":      _agg_dict(r.court_val),
                "court_train":    _agg_dict(r.court_train),
                "mocap_val_3d":   _agg_dict(r.mocap_val_3d),
                "mocap_train_3d": _agg_dict(r.mocap_train_3d),
                "court_val_3d":   _agg_dict(r.court_val_3d),
                "court_train_3d": _agg_dict(r.court_train_3d),
            }
            for rank, r in enumerate(ranked, 1)
        ],
    }


def _write_summary_json(results: list[ConfigResult], path: Path, n_seeds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_results_payload(results, n_seeds), f, indent=2)


def _write_summary(results: list[ConfigResult], log_dir: Path, n_seeds: int,
                   seeds: list[int]) -> None:
    """Write <log_dir>/summary_<timestamp>.json and summary_<timestamp>.txt."""
    import io as _io
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"summary_{ts}.json"
    txt_path  = log_dir / f"summary_{ts}.txt"

    # JSON
    _write_summary_json(results, json_path, n_seeds)

    # TXT — capture the same table that was printed to stdout
    buf = _io.StringIO()
    _old = sys.stdout
    sys.stdout = buf
    _print_results(results)
    sys.stdout = _old
    table = buf.getvalue()

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Grid search summary  —  {ts}  UTC\n")
        f.write(f"Seeds: {seeds[0]}…{seeds[-1]}  ({n_seeds} per config)\n")
        f.write(f"Configs: {len(results)}\n\n")
        f.write(table)

    print(f"\nSummary JSON → {json_path}")
    print(f"Summary TXT  → {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search over BA hyperparameter configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=("hparam", "joints", "solvers"), default="hparam",
        help=(
            "'hparam' runs the full hyperparameter sweep (CONFIGS, default log/gs/). "
            "'joints' runs the joint-preset sweep (JOINT_CONFIGS, default log/gs/joints/). "
            "'solvers' sweeps linear solver types (SOLVER_CONFIGS, default log/gs/solvers/)."
        ),
    )
    parser.add_argument(
        "--n-seeds", type=int, default=None,
        metavar="N",
        help=(
            "Number of random seeds (val/train splits) per configuration. "
            "Defaults to 5 (joints mode) or 8 (hparam mode)."
        ),
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        metavar="S",
        help="First seed; subsequent seeds are seed_start+1, seed_start+2, …",
    )
    parser.add_argument(
        "--log-dir", type=Path, default=None,
        metavar="PATH",
        help=(
            "Directory for run logs produced by ba.py. "
            "Defaults to log/gs/ (hparam mode) or log/gs/joints/ (joints mode)."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        metavar="PATH",
        help="Write aggregated results to this JSON file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    args = parser.parse_args()

    if args.mode == "joints":
        active_configs  = JOINT_CONFIGS
        show_joint      = True
        show_solver     = False
        default_log     = Path(__file__).resolve().parents[2] / "logs" / "bundle_adjustment" / "gs" / "joints"
        n_seeds_default = 5
    elif args.mode == "solvers":
        active_configs  = SOLVER_CONFIGS
        show_joint      = False
        show_solver     = True
        default_log     = Path(__file__).resolve().parents[2] / "logs" / "bundle_adjustment" / "gs" / "solvers"
        n_seeds_default = 5
    else:
        active_configs  = CONFIGS
        show_joint      = False
        show_solver     = False
        default_log     = Path(__file__).resolve().parents[2] / "logs" / "bundle_adjustment" / "gs"
        n_seeds_default = 8

    n_seeds = args.n_seeds if args.n_seeds is not None else n_seeds_default
    seeds   = list(range(args.seed_start, args.seed_start + n_seeds))
    log_dir = args.log_dir if args.log_dir is not None else default_log
    log_dir.mkdir(parents=True, exist_ok=True)
    total   = len(active_configs) * len(seeds)

    print(f"Mode       : {args.mode}")
    print(f"Grid search: {len(active_configs)} configs × {len(seeds)} seeds = {total} runs")
    print(f"Seeds      : {seeds[0]} … {seeds[-1]}  (val/train split varies per seed)")
    print(f"Log dir    : {log_dir}")
    print(f"\nConfigurations:")
    for i, cfg in enumerate(active_configs, 1):
        print(f"  [{i}] {_config_label(cfg, show_joint=show_joint, show_solver=show_solver)}")
    print()

    results: list[ConfigResult] = []

    for ci, cfg in enumerate(active_configs):
        label = _config_label(cfg, show_joint=show_joint, show_solver=show_solver)
        cr = ConfigResult(label=label, config=cfg)
        results.append(cr)

        for si, seed in enumerate(seeds):
            run_num = ci * len(seeds) + si + 1
            argv = _build_argv(cfg, seed, log_dir)

            print(f"[{run_num:>{len(str(total))}}/{total}]  {label}  seed={seed}")

            if args.dry_run:
                print("  " + " ".join(argv))
                continue

            proc = subprocess.run(argv, cwd=Path(__file__).resolve().parents[2])

            if proc.returncode != 0:
                print(f"  [WARN] Run failed (returncode={proc.returncode})")
                cr.failed_seeds.append(seed)
                continue

            json_files = sorted(log_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
            if not json_files:
                print("  [WARN] No JSON log found after run.")
                cr.failed_seeds.append(seed)
                continue

            scalars = extract_scalars(json_files[-1])
            cr.court_train.add(scalars["court_train"])
            cr.court_val.add(scalars["court_val"])
            cr.mocap_train.add(scalars["mocap_train"])
            cr.mocap_val.add(scalars["mocap_val"])
            cr.court_train_3d.add(scalars["court_train_3d"])
            cr.court_val_3d.add(scalars["court_val_3d"])
            cr.mocap_train_3d.add(scalars["mocap_train_3d"])
            cr.mocap_val_3d.add(scalars["mocap_val_3d"])

    if args.dry_run:
        print("\n(dry run — no results to aggregate)")
        return

    _print_results(results)
    _write_summary(results, log_dir, n_seeds, seeds)

    if args.out is not None:
        _write_summary_json(results, args.out, n_seeds)
        print(f"\nAggregated results → {args.out}")


if __name__ == "__main__":
    main()
