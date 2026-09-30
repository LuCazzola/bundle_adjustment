# Bundle Adjustment — `ba.py`

Joint refinement of camera extrinsics and/or intrinsics for a multi-camera basketball court setup. Observations are manually annotated 2-D court keypoints matched to known 3-D world positions on the Z=0 court plane. The solver is Ceres Solver via `pyceres`, using Levenberg–Marquardt with a DENSE_SCHUR linear solver.

---

## Problem formulation

The goal is to jointly minimise pixel reprojection error across all cameras, subject to three terms: the per-observation reprojection error, a regularisation penalty keeping intrinsics near their checkerboard values, and a cross-view consistency penalty coupling cameras that observe the same court point. All residuals are wrapped in a Huber loss (threshold 5 px) to limit the influence of gross annotation errors.

### Parameter blocks

Each camera contributes two contiguous `float64` arrays to the problem:

| Block | Size | Contents |
|-------|------|----------|
| Extrinsics | 6 | `rx ry rz` (Rodrigues rotation) + `tx ty tz` (translation) |
| Intrinsics | 9 | `fx fy cx cy k1 k2 p1 p2 k3` |

A run on 6 cameras gives 12 parameter blocks, 90 scalar unknowns total.

---

## Cost functions and Jacobians

Ceres is a non-linear least-squares solver: at each iteration it linearises the residuals around the current parameter values and solves a linear system to find the best parameter update. The Jacobian of each cost function — i.e. how much each residual changes when each parameter is nudged slightly — is what makes that linearisation possible and what the solver uses to decide which direction to step. Providing accurate Jacobians is therefore critical for both convergence speed and the quality of the final solution.

### 1. `ReprojectionCost` — per-observation pixel error

Measures how far the projected 3-D point lands from the annotated pixel in the image. One residual block per observation, two scalar residuals (u error, v error).

**Jacobian w.r.t. extrinsics**: analytic. `cv2.projectPoints` returns the derivative of the projected pixel w.r.t. the rotation vector and translation vector as part of its normal output, so these are used directly. This is the cheapest and most accurate option.

**Jacobian w.r.t. intrinsics**: numerical central finite differences — each of the 9 intrinsic parameters is perturbed by a small step and the projection is re-evaluated. Analytic derivatives through the distortion model would work but are fragile to maintain; the finite-difference cost is negligible here because the intrinsic block is small (9 parameters).

### 2. `IntrinsicRegularisationCost` — per-camera prior on intrinsics

Penalises intrinsic parameters drifting away from the checkerboard-calibrated values. This is necessary because the court keypoints are all on the same flat plane, which is a poor configuration for constraining intrinsics — without this term the solver would be free to trade focal length against distortion coefficients in ways that reduce reprojection error but produce physically implausible cameras.

The penalty is expressed as fractional drift (change relative to the prior value), so `fx ≈ 4000` and `k1 ≈ 0.35` are treated on the same scale. The default weight of 100 is intentionally large, acting as a strong prior that keeps the intrinsics close to the checkerboard values unless the reprojection evidence is overwhelming.

**Jacobian**: analytic diagonal — each residual depends only on its own parameter, so the derivative is immediate.

### 3. `CrossViewConsistencyCost` — per shared-point camera pair

For every world point that is visible in two cameras, this term checks that the two cameras agree geometrically: the observation in camera A is back-projected onto the court plane, then re-projected into camera B and compared to camera B's annotation. The same check is also done in the opposite direction (B → A). With 6 cameras and overlapping fields of view, 131 such pairs were found.

This term couples the extrinsics of different cameras to each other, which the per-camera reprojection cost cannot do. It helps ensure that a point that both cameras see ends up in a consistent 3-D position rather than each camera overfitting its own annotations independently.

**Jacobian**: numerical finite differences throughout. The back-projection step (finding where a pixel ray hits the Z=0 plane) involves undistortion followed by a ray–plane intersection, which does not have a convenient closed-form derivative with respect to camera parameters, so all four parameter blocks (extrinsics and intrinsics of both cameras) are differentiated numerically.

---

## Solver configuration

| Setting | Value |
|---------|-------|
| Algorithm | Levenberg–Marquardt |
| Linear solver | `DENSE_SCHUR` |
| Max iterations | 500 |
| Function tolerance | $10^{-8}$ |
| Gradient tolerance | $10^{-10}$ |
| Parameter tolerance | $10^{-8}$ |
| Loss function | Huber, $\delta = 5$ px |
| Threads | all available |

`DENSE_SCHUR` is appropriate here: the problem is small (90 unknowns, 88 + 131 residual blocks) so the dense Schur complement factorisation is faster than sparse alternatives and avoids fill-in concerns.

The Huber threshold of 5 px is chosen to match expected annotation click accuracy — observations further than 5 px from the projection contribute linearly rather than quadratically, limiting the influence of gross mis-clicks.

---

## Warm start

Before calling Ceres, each camera's initial extrinsics are computed with `cv2.solvePnP(ITERATIVE)` on the training keypoints. The resulting $({\bf r}_{\rm vec}, {\bf t}_{\rm vec})$ is compared against the stored tvec from the calibration JSON: whichever gives lower mean 2-D reprojection error is passed to the optimiser. This avoids handing Ceres a starting point that is worse than what was already in the file.

---

## Two-pass strategy (`--two-pass`, `--optimize both`)

When both extrinsics and intrinsics are to be refined, an optional first pass freezes intrinsics and refines only extrinsics, then a second pass jointly refines both with intrinsic regularisation active. This helps when the checkerboard intrinsics are reliable and the dominant error is in the pose — stabilising the pose first prevents the intrinsics from absorbing pose error in the second pass.

---

## Reported metrics

Two error metrics are computed before and after optimisation:

- **2-D reprojection error (px)**: distance in pixels between the projected 3-D point and the annotated pixel — directly measures fit to observations.
- **3-D world error (m)**: each annotated pixel is back-projected to the Z=0 court plane and compared to the known world position. This is a unit-meaningful diagnostic that reveals whether errors are due to a mis-scaled or mis-oriented camera.

### Convergence example (default run, 6 cameras)

```
Before BA:  global mean  8.81 ± 5.47 px   /   0.106 ± 0.095 m
After  BA:  global mean  5.51 ± 3.79 px   /   0.045 ± 0.023 m

53 iterations,  initial cost 1.99e+04  →  final cost 4.09e+03  (5× reduction)
```

The cost dropped sharply in the first two iterations (gradient-dominated phase of LM), then slowed from iteration 10 onward as the trust radius hit the upper bound (`1e+16`) and the gradient magnitude decayed — consistent with convergence to a local minimum near the checkerboard prior.

The large initial cost for cam_7 (14.2 px / 0.18 m) relative to cam_1 (3.7 px / 0.05 m) indicates that the checkerboard calibration quality varied across cameras; BA reduced cam_7's error by 10 px (73%) while leaving cam_1 roughly unchanged, as expected.

---

## Usage

```bash
python ba.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--cameras 1 2 3` | all | Camera IDs to include |
| `--optimize ex\|in\|both` | `both` | What to refine |
| `--calib-source checkerboard\|moge` | `checkerboard` | Starting calibration file |
| `--data-dir PATH` | `./data` | Root shared data folder |
| `--suffix SUFFIX` | empty | Suffix for output JSON (e.g. `_trial` gives `bundle_adjustment_trial.json`) |
| `--huber-thresh PX` | `5.0` | Huber loss threshold in pixels |
| `--lambda-intr-reg λ` | `100.0` | Intrinsic regularisation weight (`--optimize both` only) |
| `--lambda-cross λ` | `1.0` | Cross-view consistency weight (`0` = disabled) |
| `--two-pass` | off | Extrinsics-only first pass before joint refinement |
| `--val-frac F` | `0.0` | Fraction of keypoints held out for validation (e.g. `0.2`) |

### Examples

```bash
# Extrinsics only, all cameras
python ba.py --optimize ex

# Both, two-pass, cameras 1 2 3
python ba.py --optimize both --two-pass --cameras 1 2 3

# Start from MoGe monocular estimates
python ba.py --calib-source moge --optimize ex
```

---

## Outputs

Refined parameters are written to
`<data-dir>/cameras/out<id>/calibration/ba_calib.json`, preserving all original
fields and overwriting only the refined blocks (`rvecs`/`tvecs` for
extrinsics, `mtx`/`dist` for intrinsics). Runs with `--suffix` are retained
under `artifacts/bundle_adjustment/` instead of being published.

---

## Data layout

```text
data/
  cameras/
    out2/
      annotations/court_points.json
      calibration/static_calib.json
      calibration/ba_calib.json
  annotations/<video-id>/
  mocap_data/<video-id>/
  videos/<video-id>/out2.mp4
```

Bundle adjustment reads `static_calib.json` directly; no calibration export or
copy is required.

## Dependencies

- `opencv-python`
- `numpy`
- `pyceres`
