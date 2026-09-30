# Camera calibration

The calibration workspace has three scripts with distinct responsibilities:

- `detect.py`: decode videos and cache checkerboard detections.
- `calibrate.py`: fit intrinsics, solve extrinsics, compare references, sweep
  configurations, and run consistency diagnostics.
- `utils.py`: paths, schemas, loading, geometry, scoring, and printing helpers.

## Data layout

Each camera separates usable products, checkerboard inputs, and execution
artifacts:

```text
data/cameras/out<N>/
├── checkerboard/
│   ├── vid.mp4
│   └── det_cache.npz
├── annotations/
│   └── court_points.json
├── calibration/
│   ├── static_calib.json
│   └── ba_calib.json
└── artifacts/
    └── static_calibration/
        ├── state.json
        ├── reference_calib.json
        ├── sweeps/
        │   ├── baseline/
        │   └── <timestamp>/
        └── diagnostics/
```

Files are optional where the corresponding data is unavailable. For example,
cameras without court annotations omit `annotations/court_points.json`.

### Published calibration

`calibration/static_calib.json` is the only static-calibration product consumed
by bundle adjustment and applications. A normal run writes it directly; a
sweep replaces it only when a candidate passes the acceptance criteria.

Reference data, sweep state, diagnostics, and candidate results are execution
artifacts. Every sweep candidate is retained under
`artifacts/static_calibration/sweeps/<timestamp>/` with:

```text
candidate_<index>_<hash>/
├── config.json
├── calibration.json
└── metrics.json
```

## Detection

`detect.py` only manages checkerboard caches. Detection is frame-indexed, so a
finer stride reuses frames already scanned by earlier runs.

```bash
python calibration/static/detect.py
python calibration/static/detect.py --cameras 2 5 8
python calibration/static/detect.py --cameras 2 --stride 1
python calibration/static/detect.py --cameras 2 --downsample 1 --rescan
```

Default detection uses a half-resolution seed followed by full-resolution
`cornerSubPix` refinement. `--downsample 1` performs direct full-resolution
detection, which is slower but useful for difficult footage such as out2.

## Calibration

### Normal pipeline

`run` performs the complete camera pipeline in order:

1. Load cached checkerboard detections.
2. Select sharp, pose-diverse views using the camera's best configuration.
3. Fit intrinsics.
4. Evaluate a fixed checkerboard validation set.
5. If court annotations exist, independently solve court extrinsics.
6. If a reference exists, independently solve its checkerboard poses and court
   extrinsics and print the comparison.

```bash
python calibration/static/calibrate.py run
python calibration/static/calibrate.py run --cameras 2 5 8 13
python calibration/static/calibrate.py run --cameras 2 --no-write
python calibration/static/calibrate.py run --no-reference
```

The calibration schema remains:

```json
{
  "mtx": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist": [[k1, k2, p1, p2, k3]],
  "rvecs": [[rx], [ry], [rz]],
  "tvecs": [[tx], [ty], [tz]]
}
```

`rvecs` and `tvecs` are `null` when court annotations are unavailable.

### Sweeps

Sweeps score both:

- court reprojection RMS after independently re-solving extrinsics;
- checkerboard validation RMS after independently re-solving every board pose.

The combined score gives each normalized metric 50% weight. Candidates are
accepted only if neither individual metric regresses by more than 5%.

```bash
python calibration/static/calibrate.py sweep --cameras 2 --profile quick
python calibration/static/calibrate.py sweep --cameras 2 5 8 13 --profile standard
python calibration/static/calibrate.py sweep --cameras 2 --profile deep
```

Profiles:

- `quick`: 64 configurations for smoke tests and coarse refinement.
- `standard`: 192 configurations covering the proven useful region.
- `deep`: 546 configurations, including very sharp and all-frame selections,
  coarse/fine pose grids, multiple rejection strengths, view counts, and all
  supported distortion constraints.

The deep profile includes the observations from the out2 investigation:
blur sensitivity, pose coverage, view-count conditioning, `k3` constraints,
zero-tangential models, and all-frame fits.

### Diagnostics

The integrated diagnostic command checks board consistency and temporal
stability using deterministic holdouts:

```bash
python calibration/static/calibrate.py diagnose --cameras 2
```

It writes `artifacts/static_calibration/diagnostics/latest.json`.

## Court annotations

`annotations/court_points.json` contains:

```json
{
  "real_corners": [[x, y, z], "..."],
  "img_corners": [[u, v], "..."]
}
```

Court coordinates and output translations share the same unit. The stored pose
is OpenCV world-to-camera:

```text
X_camera = R @ X_world + t
camera_center_world = -R.T @ t
```

Planar courts use `SOLVEPNP_IPPE`, choose the valid lowest-error solution, and
then refine it with Levenberg-Marquardt.

## Requirements

Python 3, NumPy, and OpenCV.
