# Apps

- `app.py` is the multi-camera 2-D point picker and overlay viewer.
- `app3d.py` is the 3-D court and skeleton viewer.
- `court_points_editor.py` creates and edits per-camera
  `annotations/court_points.json` files from recording frames.

The apps read the top-level `data/` tree. Static calibration is loaded from
`data/cameras/out<N>/calibration/static_calib.json`; refined calibration is
loaded from `data/cameras/out<N>/calibration/ba_calib.json` with a static fallback
when no refined result exists.

Templates and browser assets remain local to this folder under `templates/`
and `static/`.

## Court-point editor

Run from the project root:

```bash
python apps/court_points_editor.py --video 26_03_26-cus_redfox
```

Open `http://localhost:5003`. The editor discovers `out<N>.mp4` files in the
selected video folder and creates an empty annotation file for cameras that do
not have one. Selectable 3D coordinates are stored in
`apps/court_point_catalog.json`. The editor can add new X/Y/Z coordinates to
this catalog or remove existing ones.

Select a camera and 3D point, scroll to a useful frame, and click the image to
place or move that point. Saving writes original-resolution pixel coordinates
atomically to:

```text
data/cameras/out<N>/annotations/court_points.json
```

Use `--scale` to control served frame resolution without changing stored pixel
coordinates. Zoom with the mouse wheel or the `+`/`-` controls. Wheel zoom is
centered on the cursor. Pan with the middle or right mouse button, or hold
`Space` while dragging with the left mouse button. The `fit` button resets the
view.

Deleting a catalog point that is already used identifies the affected cameras
and requests confirmation before removing that coordinate from their
`court_points.json` files.
