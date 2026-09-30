"""Detect checkerboards in calibration videos and maintain frame caches."""

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from .utils import (
        BOARDS,
        Detection,
        load_detection_cache,
        save_detection_cache,
        select_cameras,
    )
except ImportError:
    from utils import (
        BOARDS,
        Detection,
        load_detection_cache,
        save_detection_cache,
        select_cameras,
    )


DETECT_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
)
REFINE_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    100,
    1e-6,
)


def sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
    points = corners.reshape(-1, 2)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    roi = gray[max(y0, 0):y1, max(x0, 0):x1]
    if not roi.size:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def scan_frames(
    video_path: Path,
    frame_ids: list[int],
    downsample: int,
    show: bool,
) -> tuple[list[Detection], tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detection_size = (width // downsample, height // downsample)
    wanted = set(frame_ids)
    detections = []
    counts = {name: 0 for name in BOARDS}
    frame_id = -1
    processed = 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        frame_id += 1
        if frame_id not in wanted:
            continue
        wanted.remove(frame_id)
        processed += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection_gray = (
            gray
            if downsample == 1
            else cv2.resize(gray, detection_size)
        )
        for board, size in BOARDS.items():
            found, seed = cv2.findChessboardCorners(
                detection_gray, size, DETECT_FLAGS
            )
            if not found:
                continue
            full_seed = (seed * downsample).astype(np.float32)
            corners = cv2.cornerSubPix(
                gray, full_seed, (11, 11), (-1, -1), REFINE_CRITERIA
            )
            points = corners.reshape(-1, 2)
            span = float(np.max(np.ptp(points, axis=0)))
            detections.append(
                Detection(
                    frame_id,
                    board,
                    corners,
                    sharpness(gray, corners),
                    points[:, 0].mean(),
                    points[:, 1].mean(),
                    span,
                )
            )
            counts[board] += 1
            if show:
                preview = cv2.resize(frame, (width // 2, height // 2))
                cv2.drawChessboardCorners(
                    preview,
                    size,
                    corners * 0.5,
                    True,
                )
                cv2.imshow("checkerboard detections", preview)
                cv2.waitKey(1)
            break
        tally = " ".join(f"{name}={counts[name]}" for name in BOARDS)
        print(
            f"\r  scanned {processed}/{len(frame_ids)} "
            f"(frame {frame_id}/{total}) {tally}",
            end="",
            flush=True,
        )
    print()
    capture.release()
    if show:
        cv2.destroyAllWindows()
    return detections, (width, height)


def detect_camera(paths, stride, downsample, rescan, show) -> None:
    print(f"\n{'=' * 76}\nout{paths.camera_id}: detection cache\n{'=' * 76}")
    detections, scanned, image_size = (
        ({}, set(), None)
        if rescan
        else load_detection_cache(paths.detections)
    )
    if paths.video.exists():
        capture = cv2.VideoCapture(str(paths.video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    elif image_size is not None:
        total = max(scanned) + 1 if scanned else 0
        print("  video missing; using existing cache only")
    else:
        print(f"  [SKIP] missing {paths.video}")
        return
    target = list(range(0, total, stride))
    missing = [frame for frame in target if frame not in scanned]
    if missing and not paths.video.exists():
        print(f"  [WARN] {len(missing)} uncached frames cannot be scanned")
        return
    if missing:
        print(
            f"  scanning {len(missing)} frames at {downsample}x "
            f"detection downsample"
        )
        new_detections, image_size = scan_frames(
            paths.video, missing, downsample, show
        )
        detections.update(
            {item.fid: item for item in new_detections}
        )
        scanned.update(missing)
        if not show:
            save_detection_cache(
                paths.detections, detections, scanned, image_size
            )
    else:
        print(f"  all {len(target)} stride-{stride} frames are cached")
    target_set = set(target)
    usable = sum(frame in target_set for frame in detections)
    print(
        f"  cache={paths.detections} detections={len(detections)} "
        f"scanned={len(scanned)} usable_at_stride={usable}"
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Cache checkerboard detections from calibration videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data" / "cameras",
    )
    parser.add_argument("--cameras", nargs="*", type=int, default=None)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--downsample",
        type=int,
        choices=(1, 2),
        default=2,
        help="Use 1 for direct full-resolution detection.",
    )
    parser.add_argument("--rescan", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    cameras = select_cameras(args.data_dir, args.cameras)
    if not cameras:
        parser.error("no camera directories found")
    for paths in cameras:
        detect_camera(
            paths,
            args.stride,
            args.downsample,
            args.rescan,
            args.show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
