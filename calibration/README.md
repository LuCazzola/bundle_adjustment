# Calibration

Calibration is intentionally split into two stages.

- `static/` detects checkerboards, fits each camera independently, and owns the
  selected `data/cameras/out<N>/calibration/static_calib.json`.
- `bundle_adjustment/` jointly refines the selected static calibrations and
  writes `data/cameras/out<N>/calibration/ba_calib.json`.

The stages do not share implementation scripts. Their integration boundary is
the camera data layout, so bundle adjustment consumes the published static
calibration directly rather than an exported copy. Sweep state, candidate
calibrations, diagnostics, and references live under
`artifacts/static_calibration/`, never under `calibration/`.
