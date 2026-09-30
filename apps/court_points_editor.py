"""Browser editor for per-camera court_points.json annotations."""

import argparse
import io
import json
import math
import re
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_file


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CAMERA_DIR = DATA_DIR / "cameras"
VIDEO_DIR = DATA_DIR / "videos"
CATALOG_PATH = Path(__file__).resolve().parent / "court_point_catalog.json"
VIDEO_FILE_RE = re.compile(r"out(\d+)\.mp4$")

VIDEO_ID = "26_03_26-cus_redfox"
SERVE_SCALE = 0.5
SERVE_QUALITY = 80
CAMERAS: dict[int, dict] = {}
POINT_CATALOG: list[list[float]] = []

app = Flask(__name__)


def annotation_path(camera_id: int) -> Path:
    return CAMERA_DIR / f"out{camera_id}" / "annotations" / "court_points.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_annotation(camera_id: int) -> dict:
    path = annotation_path(camera_id)
    if not path.exists():
        return {"real_corners": [], "img_corners": []}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc

    real = payload.get("real_corners")
    image = payload.get("img_corners")
    if not isinstance(real, list) or not isinstance(image, list):
        raise ValueError(f"{path} must contain real_corners and img_corners arrays")
    if len(real) != len(image):
        raise ValueError(f"{path} has mismatched real_corners and img_corners")
    return {"real_corners": real, "img_corners": image}


def normalize_world_point(point) -> tuple[float, float, float]:
    if (
        not isinstance(point, list)
        or len(point) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in point
        )
    ):
        raise ValueError("world must contain three finite numbers")
    return tuple(float(value) for value in point)


def load_point_catalog() -> list[list[float]]:
    try:
        with open(CATALOG_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"Point catalog not found: {CATALOG_PATH}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read point catalog {CATALOG_PATH}: {exc}") from exc
    raw_points = payload.get("points") if isinstance(payload, dict) else None
    if not isinstance(raw_points, list):
        raise ValueError(f"{CATALOG_PATH} must contain a points array")
    points = {normalize_world_point(point) for point in raw_points}
    return [list(point) for point in sorted(points)]


def save_point_catalog(points: list[list[float]]) -> None:
    normalized = sorted({normalize_world_point(point) for point in points})
    write_json_atomic(
        CATALOG_PATH,
        {"points": [list(point) for point in normalized]},
    )


def point_usage(world: tuple[float, float, float]) -> list[int]:
    cameras = []
    for path in CAMERA_DIR.glob("out*/annotations/court_points.json"):
        match = re.fullmatch(r"out(\d+)", path.parents[1].name)
        if match is None:
            continue
        try:
            payload = load_annotation(int(match.group(1)))
        except ValueError:
            continue
        if world in {
            normalize_world_point(point)
            for point in payload["real_corners"]
        }:
            cameras.append(int(match.group(1)))
    return sorted(cameras)


def remove_point_from_annotation(
    camera_id: int,
    world: tuple[float, float, float],
) -> None:
    payload = load_annotation(camera_id)
    kept = [
        (real, image)
        for real, image in zip(payload["real_corners"], payload["img_corners"])
        if normalize_world_point(real) != world
    ]
    write_json_atomic(
        annotation_path(camera_id),
        {
            "real_corners": [real for real, _ in kept],
            "img_corners": [image for _, image in kept],
        },
    )


def inspect_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    metadata = {
        "path": path,
        "frame_count": max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    capture.release()
    if metadata["width"] <= 0 or metadata["height"] <= 0:
        raise RuntimeError(f"Cannot determine dimensions for {path}")
    return metadata


def initialize(video_id: str) -> None:
    global VIDEO_ID, CAMERAS, POINT_CATALOG

    video_dir = VIDEO_DIR / video_id
    if not video_dir.is_dir():
        raise ValueError(f"Video set not found: {video_dir}")

    POINT_CATALOG = load_point_catalog()

    cameras = {}
    for path in sorted(video_dir.glob("out*.mp4")):
        match = VIDEO_FILE_RE.fullmatch(path.name)
        if match is None:
            continue
        camera_id = int(match.group(1))
        cameras[camera_id] = inspect_video(path)
        output = annotation_path(camera_id)
        if not output.exists():
            write_json_atomic(output, {"real_corners": [], "img_corners": []})

    if not cameras:
        raise ValueError(f"No out<N>.mp4 files found in {video_dir}")

    VIDEO_ID = video_id
    CAMERAS = dict(sorted(cameras.items()))


def serialize_annotation(camera_id: int) -> list[dict]:
    payload = load_annotation(camera_id)
    return [
        {
            "world": [float(value) for value in world],
            "image": [float(value) for value in image],
        }
        for world, image in zip(
            payload["real_corners"],
            payload["img_corners"],
        )
    ]


def validate_points(camera_id: int, points) -> dict:
    if not isinstance(points, list):
        raise ValueError("points must be an array")

    catalog = {tuple(point) for point in POINT_CATALOG}
    seen = set()
    normalized = []
    camera = CAMERAS[camera_id]
    for index, item in enumerate(points):
        if not isinstance(item, dict):
            raise ValueError(f"points[{index}] must be an object")
        world = item.get("world")
        image = item.get("image")
        try:
            world_key = normalize_world_point(world)
        except ValueError as exc:
            raise ValueError(f"points[{index}].{exc}") from exc
        if (
            not isinstance(image, list)
            or len(image) != 2
            or not all(isinstance(value, (int, float)) for value in image)
        ):
            raise ValueError(f"points[{index}].image must contain two numbers")

        if world_key not in catalog:
            raise ValueError(f"Unknown court point: {world}")
        if world_key in seen:
            raise ValueError(f"Duplicate court point: {world}")
        seen.add(world_key)

        x, y = (float(image[0]), float(image[1]))
        if not (0 <= x < camera["width"] and 0 <= y < camera["height"]):
            raise ValueError(
                f"Image point {[x, y]} is outside "
                f"{camera['width']}x{camera['height']}"
            )
        normalized.append((world_key, [round(x, 3), round(y, 3)]))

    normalized.sort(key=lambda item: item[0])
    return {
        "real_corners": [list(world) for world, _ in normalized],
        "img_corners": [image for _, image in normalized],
    }


@app.get("/api/catalog")
def get_catalog():
    return jsonify(points=POINT_CATALOG, path=str(CATALOG_PATH))


@app.post("/api/catalog")
def add_catalog_point():
    global POINT_CATALOG

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Expected a JSON object"), 400
    try:
        world = normalize_world_point(body.get("world"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if world in {tuple(point) for point in POINT_CATALOG}:
        return jsonify(error=f"Point {list(world)} already exists"), 409
    try:
        updated = sorted(POINT_CATALOG + [list(world)])
        save_point_catalog(updated)
        POINT_CATALOG = updated
    except OSError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(points=POINT_CATALOG, added=list(world)), 201


@app.delete("/api/catalog")
def delete_catalog_point():
    global POINT_CATALOG

    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Expected a JSON object"), 400
    try:
        world = normalize_world_point(body.get("world"))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if world not in {tuple(point) for point in POINT_CATALOG}:
        return jsonify(error=f"Unknown point {list(world)}"), 404

    usage = point_usage(world)
    if usage and not body.get("force", False):
        return jsonify(
            error="Point is used by camera annotations",
            cameras=usage,
        ), 409
    try:
        for camera_id in usage:
            remove_point_from_annotation(camera_id, world)
        updated = [
            point for point in POINT_CATALOG if tuple(point) != world
        ]
        save_point_catalog(updated)
        POINT_CATALOG = updated
    except (OSError, ValueError) as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(points=POINT_CATALOG, removed=list(world), cameras=usage)


@app.get("/")
def index():
    return render_template(
        "court_points_editor.html",
        video_id=VIDEO_ID,
        cameras=[
            {
                "id": camera_id,
                "frame_count": data["frame_count"],
                "width": data["width"],
                "height": data["height"],
                "fps": round(data["fps"], 3),
            }
            for camera_id, data in CAMERAS.items()
        ],
        point_catalog=POINT_CATALOG,
    )


@app.get("/api/cameras/<int:camera_id>/points")
def get_points(camera_id: int):
    if camera_id not in CAMERAS:
        return jsonify(error="Unknown camera"), 404
    try:
        points = serialize_annotation(camera_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(points=points, path=str(annotation_path(camera_id)))


@app.put("/api/cameras/<int:camera_id>/points")
def save_points(camera_id: int):
    if camera_id not in CAMERAS:
        return jsonify(error="Unknown camera"), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Expected a JSON object"), 400
    try:
        payload = validate_points(camera_id, body.get("points"))
        write_json_atomic(annotation_path(camera_id), payload)
    except (OSError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(
        saved=len(payload["real_corners"]),
        path=str(annotation_path(camera_id)),
    )


@app.get("/api/cameras/<int:camera_id>/frames/<int:frame_index>")
def frame(camera_id: int, frame_index: int):
    camera = CAMERAS.get(camera_id)
    if camera is None:
        return "Unknown camera", 404
    if frame_index < 0 or frame_index >= camera["frame_count"]:
        return "Frame index out of range", 400

    capture = cv2.VideoCapture(str(camera["path"]))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    capture.release()
    if not ok:
        return "Could not decode frame", 500

    if SERVE_SCALE < 1:
        image = cv2.resize(
            image,
            (
                max(1, round(camera["width"] * SERVE_SCALE)),
                max(1, round(camera["height"] * SERVE_SCALE)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, SERVE_QUALITY],
    )
    if not ok:
        return "Could not encode frame", 500

    response = send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


def main() -> None:
    global SERVE_SCALE, SERVE_QUALITY

    parser = argparse.ArgumentParser(
        description="Edit per-camera court_points.json annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=VIDEO_ID, metavar="VIDEO_ID")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--scale", type=float, default=SERVE_SCALE)
    parser.add_argument("--quality", type=int, default=SERVE_QUALITY)
    args = parser.parse_args()

    SERVE_SCALE = max(0.1, min(1.0, args.scale))
    SERVE_QUALITY = max(1, min(95, args.quality))
    initialize(args.video)

    print(f"  Video set: {VIDEO_ID}")
    print(f"  Cameras: {list(CAMERAS)}")
    print(f"  Court-point catalog: {len(POINT_CATALOG)} points")
    print(f"  Starting editor on http://0.0.0.0:{args.port}")
    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
