# Basketball camera system

This project has two user-facing areas: calibration and apps. The scripts
remain separate; they share one canonical `data/` tree instead of copying
camera calibration and court annotation files between projects.

## Layout

```text
.
├── calibration/
│   ├── static/                 # checkerboard detection and static calibration
│   └── bundle_adjustment/      # multi-camera refinement and grid search
├── apps/                       # 2-D and 3-D visualization applications
├── scripts/                    # helpers shared by BA and the applications
├── data/
│   ├── cameras/
│   │   └── out<N>/
│   │       ├── checkerboard/
│   │       │   ├── vid.mp4
│   │       │   └── det_cache.npz
│   │       ├── annotations/court_points.json
│   │       ├── calibration/
│   │       │   ├── static_calib.json
│   │       │   └── ba_calib.json
│   │       └── artifacts/static_calibration/
│   │           ├── sweeps/
│   │           ├── diagnostics/
│   │           ├── state.json
│   │           └── reference_calib.json
│   ├── videos/<video-id>/
│   ├── annotations/<video-id>/
│   └── mocap_data/<video-id>/
└── logs/bundle_adjustment/
```

## Data flow

1. Static calibration reads each `data/cameras/out<N>/checkerboard/` folder and
   publishes its selected result at `calibration/static_calib.json`.
2. Bundle adjustment discovers only cameras that have both `static_calib.json`
   and `annotations/court_points.json`. It uses `static_calib.json` as its
   starting point.
3. Bundle adjustment writes the refined result beside the static result as
   `calibration/ba_calib.json`.
4. Apps read the same camera folders and can display either
   `static_calib.json` or `ba_calib.json`.

There is no synchronization or export step between calibration stages.

## Commands

Run from the project root.

```bash
# Static calibration
python calibration/static/detect.py --cameras 2 5 8 13
python calibration/static/calibrate.py sweep --cameras 2 5 8 13
python calibration/static/calibrate.py run --cameras 2 5 8 13

# Bundle adjustment (reads static_calib.json files)
python calibration/bundle_adjustment/ba.py --cameras 2 5 8 13

# Applications
python apps/app.py --video mocap_4
python apps/app3d.py --video mocap_4 --calib ba
python apps/court_points_editor.py --video 26_03_26-cus_redfox
```

Every entry point accepts its existing functional options. Static calibration
also accepts `--data-dir` for a camera-folder root; bundle adjustment accepts
`--data-dir` for a complete data root.

## Calibration contract

The calibration JSON schema remains unchanged:

```json
{
  "mtx": [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
  "dist": [[0, 0, 0, 0, 0]],
  "rvecs": [[0], [0], [0]],
  "tvecs": [[0], [0], [0]]
}
```

Court coordinates and camera translations use meters in the shared annotated camera set.
