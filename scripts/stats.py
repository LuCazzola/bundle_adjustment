"""
Error statistics: per-camera 2-D and per-pair 3-D triangulation errors.

Public API
----------
compute_stats(cameras)
    Court keypoints: 2-D reprojection per camera + 3-D triangulation per camera pair.

compute_val_stats(cameras)
    Same structure as compute_stats but using held-out pts3d_val / pts2d_val.
    Returns None when no val points exist.

compute_mocap_stats(cameras, mocap_obs)
    Mocap joints: same structure as compute_stats.

print_stats_full(label, court, mocap, court_val=None, mocap_val=None)
    Print both sections under a banner.
    When *_val dicts are supplied the tables gain train / val / Δ columns.

print_improvement_full(before_court, after_court, before_mocap, after_mocap)
    Before → after deltas for court and mocap (2-D and 3-D).

Stat dict format
----------------
{
  "2d": {
      "cam_<id>":  {"mean", "std", "min", "max", "n"},
      "GLOBAL":    {...},
  },
  "3d": {
      "cam_<j>×cam_<l>": {"mean", "std", "min", "max", "n"},
      "GLOBAL":           {...},
  },
}
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .cameras import project
from .geometry import triangulate_rays, unpack_intr
from .optim import make_param_blocks


# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _summary(values: list[float]) -> dict:
    """Return mean/std/min/max/n dict from a list of floats."""
    a = np.array(values, dtype=np.float64)
    if len(a) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "min":  float(np.min(a)),
        "max":  float(np.max(a)),
        "n":    int(len(a)),
    }


def _add_global(per_entry: dict[str, dict]) -> dict:
    """Pool _raw lists across all entries into GLOBAL, then strip _raw."""
    all_vals: list[float] = []
    for v in per_entry.values():
        all_vals.extend(v.pop("_raw", []))
    out = dict(per_entry)
    out["GLOBAL"] = _summary(all_vals)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2-D ERROR (per camera)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_2d(
    pts3d_by_cam: dict[int, np.ndarray],
    pts2d_by_cam: dict[int, np.ndarray],
    cameras: list[dict],
) -> dict:
    """
    Compute 2-D reprojection errors.

    pts3d_by_cam / pts2d_by_cam : cam_list_index → array of shape (N, 3/2)
    Returns stat dict with per-camera and GLOBAL entries.
    """
    per_cam: dict[str, dict] = {}

    for cidx, pts3d in pts3d_by_cam.items():
        pts2d = pts2d_by_cam[cidx]
        c = cameras[cidx]
        ex, intr = make_param_blocks(c)
        mtx, dist = unpack_intr(intr)
        proj  = project(pts3d, ex[0:3], ex[3:6], mtx, dist)
        errs  = np.linalg.norm(proj - pts2d, axis=1).tolist()
        label = f"cam_{c['cam_id']}"
        s = _summary(errs)
        s["_raw"] = errs
        per_cam[label] = s

    return _add_global(per_cam)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D ERROR VIA TRIANGULATION (per camera pair)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_3d_tri(
    point_views: dict[tuple, dict[int, np.ndarray]],
    point_world: dict[tuple, np.ndarray],
    cameras: list[dict],
) -> dict:
    """
    For every shared world-point and every pair of cameras that observe it,
    triangulate the two pixel observations with the current camera parameters
    and compare to the known world point.

    point_views : world_key → {cam_list_index: pixel (2,)}
    point_world : world_key → pt3d_world (3,)
    Returns stat dict with per-pair and GLOBAL entries.
    """
    pair_errors: dict[tuple[int, int], list[float]] = defaultdict(list)

    for key, views in point_views.items():
        pt3d = point_world[key]
        cidxs = sorted(views.keys())
        for a in range(len(cidxs)):
            for b in range(a + 1, len(cidxs)):
                j, l = cidxs[a], cidxs[b]
                tri = triangulate_rays({0: views[j], 1: views[l]},
                                       [cameras[j], cameras[l]])
                if tri is None:
                    continue
                err = float(np.linalg.norm(tri - pt3d))
                pair_errors[(j, l)].append(err)

    per_pair: dict[str, dict] = {}
    for (j, l), errs in sorted(pair_errors.items()):
        cj = cameras[j]["cam_id"]
        cl = cameras[l]["cam_id"]
        label = f"cam_{cj}×cam_{cl}"
        s = _summary(errs)
        s["_raw"] = errs
        per_pair[label] = s

    return _add_global(per_pair)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC COMPUTE FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def _court_view_dicts(cameras: list[dict], use_val: bool) -> tuple[
    dict[int, np.ndarray], dict[int, np.ndarray],
    dict[tuple, dict[int, np.ndarray]], dict[tuple, np.ndarray],
]:
    """Build the four dicts needed by _compute_2d and _compute_3d_tri.

    use_val=False → pts3d/pts2d (training split)
    use_val=True  → pts3d_val/pts2d_val (validation split)
    """
    pts3d_key = "pts3d_val" if use_val else "pts3d"
    pts2d_key = "pts2d_val" if use_val else "pts2d"

    pts3d_by_cam: dict[int, np.ndarray] = {}
    pts2d_by_cam: dict[int, np.ndarray] = {}
    point_views: dict[tuple, dict[int, np.ndarray]] = defaultdict(dict)
    point_world: dict[tuple, np.ndarray] = {}

    for cidx, c in enumerate(cameras):
        pts3d = c.get(pts3d_key)
        pts2d = c.get(pts2d_key)
        if pts3d is None or len(pts3d) == 0:
            continue
        pts3d = np.array(pts3d, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)
        pts3d_by_cam[cidx] = pts3d
        pts2d_by_cam[cidx] = pts2d
        for pt3d, pt2d in zip(pts3d, pts2d):
            key = tuple(np.round(pt3d, 6))
            point_views[key][cidx] = pt2d
            point_world[key] = pt3d

    point_views = {k: v for k, v in point_views.items() if len(v) >= 2}
    return pts3d_by_cam, pts2d_by_cam, point_views, point_world


def compute_stats(cameras: list[dict]) -> dict:
    """
    Court keypoints (training split) — 2-D per camera + 3-D triangulation per pair.
    Returns {"2d": {...}, "3d": {...}}.
    """
    pts3d_by_cam, pts2d_by_cam, point_views, point_world = _court_view_dicts(
        cameras, use_val=False
    )
    return {
        "2d": _compute_2d(pts3d_by_cam, pts2d_by_cam, cameras),
        "3d": _compute_3d_tri(point_views, point_world, cameras),
    }


def compute_val_stats(cameras: list[dict]) -> dict | None:
    """
    Court keypoints (validation split) — same {"2d", "3d"} structure as compute_stats.
    Returns None when no camera has held-out points.
    """
    pts3d_by_cam, pts2d_by_cam, point_views, point_world = _court_view_dicts(
        cameras, use_val=True
    )
    if not pts3d_by_cam:
        return None
    return {
        "2d": _compute_2d(pts3d_by_cam, pts2d_by_cam, cameras),
        "3d": _compute_3d_tri(point_views, point_world, cameras),
    }


def compute_mocap_stats(
    cameras: list[dict],
    mocap_obs: list[tuple[np.ndarray, np.ndarray, int]] | None,
) -> dict | None:
    """
    Mocap joints — 2-D per camera + 3-D triangulation per camera pair.

    mocap_obs: list of (pt3d_world [m], pt2d_pixel, cam_list_index)
    Returns {"2d": {...}, "3d": {...}}, or None if mocap_obs is empty.
    """
    if not mocap_obs:
        return None

    pts3d_by_cam: dict[int, list] = defaultdict(list)
    pts2d_by_cam: dict[int, list] = defaultdict(list)
    for pt3d, pt2d, cidx in mocap_obs:
        pts3d_by_cam[cidx].append(pt3d)
        pts2d_by_cam[cidx].append(pt2d)

    pts3d_by_cam_arr = {k: np.array(v, dtype=np.float64) for k, v in pts3d_by_cam.items()}
    pts2d_by_cam_arr = {k: np.array(v, dtype=np.float64) for k, v in pts2d_by_cam.items()}

    point_views: dict[tuple, dict[int, np.ndarray]] = defaultdict(dict)
    point_world: dict[tuple, np.ndarray] = {}
    for pt3d, pt2d, cidx in mocap_obs:
        key = tuple(np.round(pt3d, 6))
        point_views[key][cidx] = pt2d
        point_world[key] = pt3d

    point_views = {k: v for k, v in point_views.items() if len(v) >= 2}

    return {
        "2d": _compute_2d(pts3d_by_cam_arr, pts2d_by_cam_arr, cameras),
        "3d": _compute_3d_tri(point_views, point_world, cameras),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PRINTING PRIMITIVES
# ──────────────────────────────────────────────────────────────────────────────

_W = 108

# Column widths
_KEY_W  = 20   # row label
_STAT_W = 7    # mean/std/min/max cells


def _fmt_px(s: dict) -> str:
    return (f"{s['mean']:{_STAT_W}.2f}  {s['std']:{_STAT_W}.2f}  "
            f"{s['min']:{_STAT_W}.2f}  {s['max']:{_STAT_W}.2f} px  {s['n']:3d}")

def _fmt_m(s: dict) -> str:
    return (f"{s['mean']:{_STAT_W}.4f}  {s['std']:{_STAT_W}.4f}  "
            f"{s['min']:{_STAT_W}.4f}  {s['max']:{_STAT_W}.4f} m   {s['n']:3d}")

_STAT_HDR = f"{'mean':>{_STAT_W}}  {'std':>{_STAT_W}}  {'min':>{_STAT_W}}  {'max':>{_STAT_W}}      n"


def _print_2d_section(
    train: dict,
    val: dict | None = None,
    title: str = "2-D reprojection  (per camera)",
) -> None:
    print(f"\n  {'─'*(_W-4)}")
    print(f"  {title}")
    print(f"  {'─'*(_W-4)}")
    print(f"  {'':>{_KEY_W}}  {_STAT_HDR}")

    # ── train rows ────────────────────────────────────────────────────────────
    for k, s in train.items():
        if k == "GLOBAL":
            continue
        print(f"  {k:>{_KEY_W}}  {_fmt_px(s)}")
    if "GLOBAL" in train:
        print(f"  {'GLOBAL':>{_KEY_W}}  {_fmt_px(train['GLOBAL'])}")

    if val is None:
        return

    # ── val sub-table ─────────────────────────────────────────────────────────
    val_rows = {k: s for k, s in val.items() if k != "GLOBAL" and s["n"] > 0}
    val_global = val.get("GLOBAL")
    has_val_data = bool(val_rows) or (val_global and val_global["n"] > 0)
    if not has_val_data:
        return

    print(f"\n  {'(val)':>{_KEY_W}}  {_STAT_HDR}")
    for k in train:           # same row order as train
        if k == "GLOBAL":
            continue
        va = val.get(k)
        if va and va["n"] > 0:
            print(f"  {k:>{_KEY_W}}  {_fmt_px(va)}")
        else:
            print(f"  {k:>{_KEY_W}}  {'—':>{_STAT_W}}  {'—':>{_STAT_W}}  {'—':>{_STAT_W}}  {'—':>{_STAT_W}}      —")
    if val_global and val_global["n"] > 0:
        tr_g = train.get("GLOBAL")
        delta = val_global["mean"] - tr_g["mean"] if tr_g else float("nan")
        flag  = "  ⚠ val mean > train mean by more than 3 px" if delta > 3.0 else ""
        print(f"  {'GLOBAL':>{_KEY_W}}  {_fmt_px(val_global)}{flag}")


def _print_3d_section(
    train: dict,
    val: dict | None = None,
    title: str = "3-D triangulation error  (per camera pair)",
) -> None:
    if not train or all(k == "GLOBAL" for k in train):
        return
    print(f"\n  {'─'*(_W-4)}")
    print(f"  {title}")
    print(f"  {'─'*(_W-4)}")
    print(f"  {'':>{_KEY_W}}  {_STAT_HDR}")

    # ── train rows ────────────────────────────────────────────────────────────
    for k, s in train.items():
        if k == "GLOBAL":
            continue
        print(f"  {k:>{_KEY_W}}  {_fmt_m(s)}")
    if "GLOBAL" in train:
        print(f"  {'GLOBAL':>{_KEY_W}}  {_fmt_m(train['GLOBAL'])}")

    if val is None:
        return

    # ── val sub-table ─────────────────────────────────────────────────────────
    val_rows = {k: s for k, s in val.items() if k != "GLOBAL" and s["n"] > 0}
    val_global = val.get("GLOBAL")
    has_val_data = bool(val_rows) or (val_global and val_global["n"] > 0)
    if not has_val_data:
        return

    print(f"\n  {'(val)':>{_KEY_W}}  {_STAT_HDR}")
    for k in train:
        if k == "GLOBAL":
            continue
        va = val.get(k)
        if va and va["n"] > 0:
            print(f"  {k:>{_KEY_W}}  {_fmt_m(va)}")
        else:
            print(f"  {k:>{_KEY_W}}  {'—':>{_STAT_W}}  {'—':>{_STAT_W}}  {'—':>{_STAT_W}}  {'—':>{_STAT_W}}      —")
    if val_global and val_global["n"] > 0:
        tr_g = train.get("GLOBAL")
        delta = val_global["mean"] - tr_g["mean"] if tr_g else float("nan")
        flag  = "  ⚠ val mean > train mean by more than 0.05 m" if delta > 0.05 else ""
        print(f"  {'GLOBAL':>{_KEY_W}}  {_fmt_m(val_global)}{flag}")


def _print_block(
    source_label: str,
    train: dict,
    val: dict | None = None,
) -> None:
    """Print one source block (court or mocap) with 2D and 3D sub-sections."""
    has_3d_pairs = bool(train["3d"]) and any(k != "GLOBAL" for k in train["3d"])
    val_2d = val["2d"] if val else None
    val_3d = val["3d"] if val else None

    _print_2d_section(train["2d"], val_2d)

    if has_3d_pairs:
        _print_3d_section(train["3d"], val_3d)
    else:
        print(f"\n  (no shared observations between ≥2 cameras — 3-D triangulation unavailable)")


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC PRINT FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def print_stats_full(
    label: str,
    court_stats: dict,
    mocap_stats: dict | None,
    court_val: dict | None = None,
    mocap_val: dict | None = None,
) -> None:
    """
    Print court and (optionally) mocap stats under a banner.

    When court_val / mocap_val are supplied the tables show
    train | val | Δ columns side by side.
    """
    print(f"\n{'═'*_W}")
    print(f"  {label}")

    print(f"\n{'─'*_W}")
    print(f"  court keypoints")
    _print_block("court", court_stats, court_val)

    if mocap_stats is not None:
        print(f"\n{'─'*_W}")
        print(f"  mocap joints")
        _print_block("mocap", mocap_stats, mocap_val)


# ──────────────────────────────────────────────────────────────────────────────
# IMPROVEMENT PRINTING
# ──────────────────────────────────────────────────────────────────────────────

_PFX_W = 8    # "(train) " / "(val)   " / ""

def _imp_row_px(prefix: str, name: str, b: dict, a: dict) -> str:
    """Format one before→after improvement row in pixels."""
    delta = b["mean"] - a["mean"]
    arrow = "▼" if a["mean"] < b["mean"] else "▲"
    key = f"{prefix:{_PFX_W}}{name}"
    return (f"  {key:>{_KEY_W + _PFX_W}}  "
            f"{b['mean']:7.2f}±{b['std']:6.2f} → {a['mean']:7.2f}±{a['std']:6.2f} px  "
            f"{arrow} ({delta:+.2f})")


def _imp_row_m(prefix: str, name: str, b: dict, a: dict) -> str:
    """Format one before→after improvement row in metres."""
    delta = b["mean"] - a["mean"]
    arrow = "▼" if a["mean"] < b["mean"] else "▲"
    key = f"{prefix:{_PFX_W}}{name}"
    return (f"  {key:>{_KEY_W + _PFX_W}}  "
            f"{b['mean']:8.4f}±{b['std']:7.4f} → {a['mean']:8.4f}±{a['std']:7.4f} m  "
            f"{arrow} ({delta:+.4f})")


def _print_improvement_2d(
    before_tr: dict, after_tr: dict,
    before_val: dict | None, after_val: dict | None,
    source: str,
) -> None:
    print(f"\n  {'─'*(_W-4)}")
    print(f"  Improvement — {source}  [2-D reprojection per camera]")
    print(f"  {'─'*(_W-4)}")

    has_val = before_val is not None and after_val is not None
    tr_pfx  = "(train)" if has_val else ""

    for k in before_tr:
        if k == "GLOBAL" or k not in after_tr:
            continue
        print(_imp_row_px(tr_pfx, k, before_tr[k], after_tr[k]))
        if has_val:
            bv, av = before_val.get(k), after_val.get(k)
            if bv and av and bv["n"] > 0 and av["n"] > 0:
                print(_imp_row_px("(val)", k, bv, av))

    if "GLOBAL" in before_tr and "GLOBAL" in after_tr:
        print()
        print(_imp_row_px(tr_pfx, "GLOBAL", before_tr["GLOBAL"], after_tr["GLOBAL"]))
        if has_val:
            bv = before_val.get("GLOBAL") if before_val else None
            av = after_val.get("GLOBAL")  if after_val  else None
            if bv and av and bv["n"] > 0 and av["n"] > 0:
                print(_imp_row_px("(val)", "GLOBAL", bv, av))


def _print_improvement_3d(
    before_tr: dict, after_tr: dict,
    before_val: dict | None, after_val: dict | None,
    source: str,
) -> None:
    if not before_tr or not after_tr:
        return
    if all(k == "GLOBAL" for k in before_tr):
        return
    print(f"\n  {'─'*(_W-4)}")
    print(f"  Improvement — {source}  [3-D triangulation error per camera pair]")
    print(f"  {'─'*(_W-4)}")

    has_val = before_val is not None and after_val is not None
    tr_pfx  = "(train)" if has_val else ""

    for k in before_tr:
        if k == "GLOBAL" or k not in after_tr:
            continue
        print(_imp_row_m(tr_pfx, k, before_tr[k], after_tr[k]))
        if has_val:
            bv, av = before_val.get(k), after_val.get(k)
            if bv and av and bv["n"] > 0 and av["n"] > 0:
                print(_imp_row_m("(val)", k, bv, av))

    if "GLOBAL" in before_tr and "GLOBAL" in after_tr:
        print()
        print(_imp_row_m(tr_pfx, "GLOBAL", before_tr["GLOBAL"], after_tr["GLOBAL"]))
        if has_val:
            bv = before_val.get("GLOBAL") if before_val else None
            av = after_val.get("GLOBAL")  if after_val  else None
            if bv and av and bv["n"] > 0 and av["n"] > 0:
                print(_imp_row_m("(val)", "GLOBAL", bv, av))


def print_improvement_full(
    before_court:     dict,
    after_court:      dict,
    before_mocap:     dict | None,
    after_mocap:      dict | None,
    before_court_val: dict | None = None,
    after_court_val:  dict | None = None,
    before_mocap_val: dict | None = None,
    after_mocap_val:  dict | None = None,
) -> None:
    """
    Print before→after improvement for court and (optionally) mocap.

    When *_val dicts are supplied each per-camera row is doubled:
      (train) cam_X   before → after
      (val)   cam_X   before → after   ← held-out; shows generalisation
    """
    print(f"\n{'═'*_W}")
    print(f"  Improvement summary")

    _print_improvement_2d(
        before_court["2d"], after_court["2d"],
        before_court_val["2d"] if before_court_val else None,
        after_court_val["2d"]  if after_court_val  else None,
        "court keypoints",
    )
    _print_improvement_3d(
        before_court["3d"], after_court["3d"],
        before_court_val["3d"] if before_court_val else None,
        after_court_val["3d"]  if after_court_val  else None,
        "court keypoints",
    )

    if before_mocap is not None and after_mocap is not None:
        _print_improvement_2d(
            before_mocap["2d"], after_mocap["2d"],
            before_mocap_val["2d"] if before_mocap_val else None,
            after_mocap_val["2d"]  if after_mocap_val  else None,
            "mocap joints",
        )
        _print_improvement_3d(
            before_mocap["3d"], after_mocap["3d"],
            before_mocap_val["3d"] if before_mocap_val else None,
            after_mocap_val["3d"]  if after_mocap_val  else None,
            "mocap joints",
        )


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICAL PLAUSIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def print_physical_summary(cameras: list[dict]) -> None:
    print(f"\n{'─'*_W}")
    print("  Physical plausibility of calibrated parameters")
    print(f"{'─'*_W}")
    print(f"  {'':8s}  {'fov_h':>6s}  {'cx':>5s}  {'cy':>5s}  "
          f"{'k1':>7s}  {'‖t‖':>6s}  {'tilt':>6s}  notes")
    for c in cameras:
        m, d = c["mtx"], c["dist"]
        fx, fy = m[0, 0], m[1, 1]
        cx, cy = m[0, 2], m[1, 2]
        iw = c.get("w", round(cx * 2))
        ih = c.get("h", round(cy * 2))
        fov_h  = 2 * np.degrees(np.arctan(iw / (2 * fx)))
        cx_pct = cx / iw * 100
        cy_pct = cy / ih * 100
        R, _ = cv2.Rodrigues(c["rvec"].reshape(3, 1))
        optical_axis = R.T @ np.array([0.0, 0.0, 1.0])
        tilt   = 90.0 - np.degrees(np.arccos(np.clip(abs(optical_axis[2]), 0, 1)))
        dist_m = np.linalg.norm(c["tvec"])
        notes = []
        if cx_pct < 30 or cx_pct > 70:  notes.append(f"cx off-centre ({cx_pct:.0f}%)")
        if cy_pct < 20 or cy_pct > 80:  notes.append(f"cy off-centre ({cy_pct:.0f}%)")
        if abs(fx / fy - 1) > 0.10:     notes.append(f"fx/fy={fx/fy:.2f}")
        if abs(d[0]) > 1.0:             notes.append(f"k1={d[0]:.3f} large")
        if dist_m < 2 or dist_m > 80:   notes.append(f"‖t‖={dist_m:.1f}m implausible")
        if tilt < 10:                   notes.append("near-horizontal view")
        note_str = "  ⚠ " + ", ".join(notes) if notes else ""
        print(f"  cam_{c['cam_id']:<4d}  {fov_h:6.1f}°  {cx_pct:4.1f}%  {cy_pct:4.1f}%  "
              f"{d[0]:7.4f}  {dist_m:6.1f}m  {tilt:5.1f}°{note_str}")
    print(f"{'─'*_W}")
