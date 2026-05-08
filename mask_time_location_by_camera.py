#!/usr/bin/env python3
"""
Mask fixed timestamp/location regions in videos using camera LabelMe JSON boxes.

Matching strategy:
  1. Extract one frame from the video (fast: reads only the first keyframe).
  2. Filter camera candidates to those whose JSON imageWidth/imageHeight matches
     the video resolution (huge speedup: from 64 candidates to 3-5).
  3. For each remaining candidate, compute NCC (Normalized Cross-Correlation)
     similarity between the video frame and the camera reference JPG at every
     annotated box position, then take the area-weighted average as the score.
  4. Accept the best match if score >= min_score and margin over 2nd-best >= min_margin.

Blurring: FFmpeg filter_complex with crop+boxblur+overlay chain (fast, preserves audio).
          OpenCV GaussianBlur fallback when ffmpeg is unavailable.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
NCC_CROP_W = 128
NCC_CROP_H = 64
PROJ_ROWS = 16
PROJ_COLS = 32


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass(frozen=True)
class CameraBoxes:
    camera_id: str
    json_path: Path
    image_width: int
    image_height: int
    boxes: tuple[Box, ...]


@dataclass(frozen=True)
class Job:
    video_path: Path
    output_path: Path
    camera: CameraBoxes
    boxes: tuple[Box, ...]
    width: int
    height: int
    match_score: float | None = None
    match_method: str = "exact"


@dataclass(frozen=True)
class VisualMatch:
    camera: CameraBoxes
    score: float
    margin: float


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Blur timestamp/location regions in videos using camera JSON boxes."
    )
    parser.add_argument(
        "--input-dir",
        default=str(repo_root / "blurred_videos" / "to_mask_time_location"),
        help="Directory containing videos to process.",
    )
    parser.add_argument(
        "--camera-dir",
        default=str(repo_root / "camera"),
        help="Directory containing LabelMe camera JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "blurred_videos" / "time_location_masked"),
        help="Directory where masked videos will be written.",
    )
    parser.add_argument(
        "--mapping",
        help=(
            'Optional JSON mapping from video stem/name to camera id, e.g. '
            '{"12-step1": "11", "436-step1.mp4": "433"}. '
            "Overrides automatic matching for the specified videos."
        ),
    )
    parser.add_argument(
        "--mapping-fuzzy-min-chars",
        type=int,
        default=3,
        help=(
            "When exact mapping misses, allow fuzzy mapping if the video name and a mapping key "
            "share at least this many consecutive normalized characters."
        ),
    )
    parser.add_argument(
        "--match-mode",
        choices=("exact", "exact-or-ncc", "ncc"),
        default="exact-or-ncc",
        help=(
            "How to match videos to camera JSON. "
            "'exact': stem/number prefix only; "
            "'ncc': NCC visual matching with resolution filter; "
            "'exact-or-ncc': try exact first, fall back to NCC."
        ),
    )
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=0.50,
        help="Minimum NCC match score (0-1) required to accept an automatic camera match.",
    )
    parser.add_argument(
        "--min-score-margin",
        type=float,
        default=0.02,
        help="Minimum score gap between best and second-best NCC matches.",
    )
    parser.add_argument(
        "--min-fast-match-score",
        type=float,
        default=0.72,
        help="Minimum fast text-layout match score (0-1) before accepting a candidate.",
    )
    parser.add_argument(
        "--min-fast-score-margin",
        type=float,
        default=0.015,
        help="Minimum fast text-layout score gap over second-best candidate.",
    )
    parser.add_argument(
        "--suffix",
        default="_time_location_masked",
        help="Suffix added before the extension for output videos.",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=24,
        help="FFmpeg boxblur radius. Higher is stronger.",
    )
    parser.add_argument(
        "--blur-power",
        type=int,
        default=4,
        help="How many times to apply boxblur in FFmpeg. Higher is blurrier.",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=4,
        help="Extra pixels added around every scaled box.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of videos to process in parallel.",
    )
    parser.add_argument(
        "--max-output-width",
        type=int,
        default=1920,
        help=(
            "If positive, downscale wider FFmpeg outputs to this width for easier "
            "VS Code/browser preview. Use 0 to preserve original width."
        ),
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "ffmpeg", "opencv"),
        default="auto",
        help="Processing engine. FFmpeg is fastest and preserves audio; OpenCV is a fallback.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default=None,
        help="Path to ffmpeg.",
    )
    parser.add_argument(
        "--ffprobe-bin",
        default=None,
        help="Path to ffprobe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matches and output paths; do not process videos.",
    )
    parser.add_argument(
        "--force-best-match",
        action="store_true",
        help=(
            "For videos that fail NCC/exact matching, accept the best-scoring camera "
            "regardless of score threshold. Ensures no video is left unmasked."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# FFmpeg / FFprobe helpers
# ---------------------------------------------------------------------------

def find_ffmpeg(explicit: str | None, required: bool = True) -> str | None:
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if not found and required:
        raise RuntimeError("Cannot find ffmpeg. Install ffmpeg or pass --ffmpeg-bin.")
    return found


def find_ffprobe(explicit: str | None, ffmpeg_bin: str | None, required: bool = True) -> str | None:
    if explicit:
        return explicit
    if ffmpeg_bin:
        sibling = Path(ffmpeg_bin).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)
    found = shutil.which("ffprobe")
    if not found and required:
        raise RuntimeError("Cannot find ffprobe. Install ffprobe or pass --ffprobe-bin.")
    return found


def probe_video_size(ffprobe_bin: str | None, video: Path) -> tuple[int, int]:
    if ffprobe_bin:
        cmd = [
            ffprobe_bin, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(video),
        ]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        streams = json.loads(result.stdout).get("streams") or []
        if not streams:
            raise RuntimeError(f"No video stream found: {video}")
        return int(streams[0]["width"]), int(streams[0]["height"])

    import cv2  # type: ignore
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Cannot read video dimensions: {video}")
        return w, h
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# JSON / mapping helpers
# ---------------------------------------------------------------------------

def load_mapping(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Mapping file must contain a JSON object: {path}")
    return {str(k): str(v).removesuffix(".json") for k, v in data.items()}


def _normalize_mapping_text(text: str) -> str:
    text = Path(text).stem
    text = text.lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def fuzzy_match_mapping_key(
    video: Path,
    mapping: dict[str, str],
    min_shared_chars: int,
) -> str | None:
    if min_shared_chars <= 0 or not mapping:
        return None

    video_candidates = [_normalize_mapping_text(video.name), _normalize_mapping_text(video.stem)]
    video_candidates = [item for item in video_candidates if item]
    if not video_candidates:
        return None

    scored: list[tuple[int, str]] = []
    for raw_key, camera_id in mapping.items():
        norm_key = _normalize_mapping_text(raw_key)
        if not norm_key or norm_key.isdigit():
            continue
        best_score = max(_longest_common_substring_len(v, norm_key) for v in video_candidates)
        if best_score >= min_shared_chars:
            scored.append((best_score, camera_id))

    if not scored:
        return None

    scored.sort(reverse=True)
    best_score, best_camera_id = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1
    if best_score == second_score:
        return None
    return best_camera_id


def box_from_points(points: list[Any]) -> Box:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Box(
        x1=int(round(min(xs))),
        y1=int(round(min(ys))),
        x2=int(round(max(xs))),
        y2=int(round(max(ys))),
    )


def load_camera_json(path: Path) -> CameraBoxes:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"Missing imageWidth/imageHeight in {path}")
    boxes: list[Box] = []
    for shape in data.get("shapes", []):
        points = shape.get("points")
        if points:
            box = box_from_points(points)
            if box.width > 0 and box.height > 0:
                boxes.append(box)
    if not boxes:
        raise ValueError(f"No valid boxes found in {path}")
    return CameraBoxes(
        camera_id=path.stem,
        json_path=path,
        image_width=width,
        image_height=height,
        boxes=tuple(boxes),
    )


def load_cameras(camera_dir: Path) -> dict[str, CameraBoxes]:
    cameras: dict[str, CameraBoxes] = {}
    for path in sorted(camera_dir.glob("*.json"), key=lambda p: p.stem):
        try:
            cam = load_camera_json(path)
            cameras[cam.camera_id] = cam
        except (ValueError, KeyError):
            pass  # skip non-LabelMe JSONs
    return cameras


def list_videos(input_dir: Path) -> list[Path]:
    return sorted(
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS),
        key=lambda p: p.name,
    )


def is_already_osd_masked_video(video: Path) -> bool:
    """Detect videos that already went through step3 once."""
    stem = video.stem.lower()
    return "_time_location_masked" in stem


# ---------------------------------------------------------------------------
# Exact matching (video stem / number prefix)
# ---------------------------------------------------------------------------

def exact_match_camera(
    video: Path,
    cameras: dict[str, CameraBoxes],
    mapping: dict[str, str],
    mapping_fuzzy_min_chars: int = 3,
) -> CameraBoxes | None:
    mapped = mapping.get(video.name) or mapping.get(video.stem)
    if mapped:
        return cameras.get(mapped)

    fuzzy_mapped = fuzzy_match_mapping_key(video, mapping, mapping_fuzzy_min_chars)
    if fuzzy_mapped:
        cam = cameras.get(fuzzy_mapped)
        if cam:
            return cam

    for candidate_id in [video.stem] + (
        [re.match(r"^(\d+)", video.stem).group(1)]
        if re.match(r"^(\d+)", video.stem)
        else []
    ):
        cam = cameras.get(candidate_id)
        if cam:
            return cam
    return None


# ---------------------------------------------------------------------------
# NCC visual matching
# ---------------------------------------------------------------------------

def _extract_region(img: Any, x1: int, y1: int, x2: int, y2: int) -> Any | None:
    """Crop a region from a cv2 BGR image and resize to fixed NCC size."""
    import cv2  # type: ignore
    h, w = img.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    roi = img[y1c:y2c, x1c:x2c]
    if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return None
    return cv2.resize(roi, (NCC_CROP_W, NCC_CROP_H))


def _ncc(a: Any, b: Any) -> float:
    """Normalized cross-correlation between two same-shape BGR crops."""
    import numpy as np
    import cv2  # type: ignore
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 -= g1.mean()
    g2 -= g2.mean()
    n1 = np.linalg.norm(g1)
    n2 = np.linalg.norm(g2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return float(np.dot(g1.flatten(), g2.flatten()) / (n1 * n2))


def _camera_image(camera: CameraBoxes) -> Any | None:
    """Load the camera reference image as a cv2 BGR array, or None if missing."""
    import cv2  # type: ignore
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        path = camera.json_path.with_suffix(suffix)
        if path.exists():
            img = cv2.imread(str(path))
            if img is not None:
                return img
    return None


def _resize_vector(values: Any, target_len: int) -> Any:
    import numpy as np
    if len(values) == target_len:
        return values.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(x_new, x_old, values).astype(np.float32)


def _text_projection_signature(img: Any, boxes: list[tuple[int, int, int, int]]) -> Any | None:
    import cv2  # type: ignore
    import numpy as np

    parts: list[np.ndarray] = []
    for x1, y1, x2, y2 in boxes:
        roi = _extract_region(img, x1, y1, x2, y2)
        if roi is None:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
        detail = cv2.absdiff(gray, blur)
        _, detail_mask = cv2.threshold(detail, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        dark = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)[1]
        mask = cv2.bitwise_and(cv2.bitwise_or(bright, dark), detail_mask)
        row_signal = (mask.sum(axis=1) / 255.0).astype(np.float32)
        col_signal = (mask.sum(axis=0) / 255.0).astype(np.float32)
        row_signal = _resize_vector(row_signal, PROJ_ROWS)
        col_signal = _resize_vector(col_signal, PROJ_COLS)
        feature = np.concatenate([row_signal, col_signal]).astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 1e-6:
            feature /= norm
            parts.append(feature)

    if not parts:
        return None
    signature = np.concatenate(parts).astype(np.float32)
    norm = np.linalg.norm(signature)
    if norm <= 1e-6:
        return None
    return signature / norm


def _scaled_box_coords(camera: CameraBoxes, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    coords: list[tuple[int, int, int, int]] = []
    for box in camera.boxes:
        x1 = int(box.x1 * img_w / camera.image_width)
        y1 = int(box.y1 * img_h / camera.image_height)
        x2 = int(box.x2 * img_w / camera.image_width)
        y2 = int(box.y2 * img_h / camera.image_height)
        coords.append((x1, y1, x2, y2))
    return coords


def _fast_camera_score(vid_frame: Any, camera: CameraBoxes) -> float | None:
    import numpy as np

    ref_img = _camera_image(camera)
    if ref_img is None:
        return None

    vh, vw = vid_frame.shape[:2]
    rh, rw = ref_img.shape[:2]
    v_boxes = _scaled_box_coords(camera, vw, vh)
    r_boxes = _scaled_box_coords(camera, rw, rh)
    v_sig = _text_projection_signature(vid_frame, v_boxes)
    r_sig = _text_projection_signature(ref_img, r_boxes)
    if v_sig is None or r_sig is None or len(v_sig) != len(r_sig):
        return None
    return float(np.dot(v_sig, r_sig))


def _ncc_camera_score(vid_frame: Any, camera: CameraBoxes) -> float | None:
    """
    Area-weighted average NCC score across all boxes in the camera JSON.
    vid_frame is a cv2 BGR array of the video frame.
    Returns None if the reference image is unavailable.
    """
    ref_img = _camera_image(camera)
    if ref_img is None:
        return None

    vh, vw = vid_frame.shape[:2]
    rh, rw = ref_img.shape[:2]
    jw, jh = camera.image_width, camera.image_height

    box_scores: list[tuple[float, int]] = []
    for box in camera.boxes:
        # Scale box from JSON coords to actual image coords
        vx1 = int(box.x1 * vw / jw); vy1 = int(box.y1 * vh / jh)
        vx2 = int(box.x2 * vw / jw); vy2 = int(box.y2 * vh / jh)
        rx1 = int(box.x1 * rw / jw); ry1 = int(box.y1 * rh / jh)
        rx2 = int(box.x2 * rw / jw); ry2 = int(box.y2 * rh / jh)

        v_roi = _extract_region(vid_frame, vx1, vy1, vx2, vy2)
        r_roi = _extract_region(ref_img, rx1, ry1, rx2, ry2)
        if v_roi is not None and r_roi is not None:
            s = _ncc(v_roi, r_roi)
            area = (box.x2 - box.x1) * (box.y2 - box.y1)
            box_scores.append((s, area))

    if not box_scores:
        return None
    total_w = sum(a for _, a in box_scores)
    return sum(s * a for s, a in box_scores) / total_w


def _extract_first_frame(video: Path, ffmpeg_bin: str | None) -> Any:
    """Extract the first frame of a video as a cv2 BGR array."""
    import cv2  # type: ignore
    import numpy as np

    # Try FFmpeg pipe first (fastest: reads only one I-frame)
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error",
            "-i", str(video),
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw = result.stdout
        if raw:
            # We don't know the size yet; probe it quickly
            import re as _re
            probe_cmd = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                "-i", str(video),
                "-frames:v", "1",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ]
            # Actually probe via ffprobe-style output; use cv2 as fallback
            pass

    # OpenCV fallback (also fast for the first frame)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for NCC matching: {video}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Cannot read first frame from video: {video}")
        return frame
    finally:
        cap.release()


def ncc_match_camera(
    video: Path,
    cameras: dict[str, CameraBoxes],
    video_width: int,
    video_height: int,
    ffmpeg_bin: str | None,
    min_score: float,
    min_margin: float,
) -> VisualMatch | None:
    """
    Match video to the best camera using NCC similarity.
    Only considers cameras whose JSON resolution matches the video resolution
    (exact pixel match or within 1 px rounding tolerance).
    """
    # Resolution-based filtering: only compare same-resolution cameras
    target_res = (video_width, video_height)
    candidates = [
        cam for cam in cameras.values()
        if (cam.image_width, cam.image_height) == target_res
    ]

    # Fallback: if no exact-resolution match, try all cameras (unusual case)
    if not candidates:
        candidates = list(cameras.values())

    if not candidates:
        return None

    vid_frame = _extract_first_frame(video, ffmpeg_bin)

    scored: list[tuple[float, CameraBoxes]] = []
    for cam in candidates:
        score = _ncc_camera_score(vid_frame, cam)
        if score is not None:
            scored.append((score, cam))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cam = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - second_score

    if best_score < min_score or margin < min_margin:
        return None

    return VisualMatch(camera=best_cam, score=best_score, margin=margin)


def force_best_match_camera(
    video: Path,
    cameras: dict[str, CameraBoxes],
    video_width: int,
    video_height: int,
    ffmpeg_bin: str | None,
) -> VisualMatch | None:
    """Return the highest-scoring camera with no threshold gatekeeping."""
    target_res = (video_width, video_height)
    candidates = [c for c in cameras.values() if (c.image_width, c.image_height) == target_res]
    if not candidates:
        candidates = list(cameras.values())
    if not candidates:
        return None

    vid_frame = _extract_first_frame(video, ffmpeg_bin)
    scored: list[tuple[float, CameraBoxes]] = []
    for cam in candidates:
        score = _fast_camera_score(vid_frame, cam)
        if score is None:
            score = _ncc_camera_score(vid_frame, cam) or 0.0
        scored.append((score, cam))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cam = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    return VisualMatch(camera=best_cam, score=best_score, margin=best_score - second_score)


def fast_text_match_camera(
    video: Path,
    cameras: dict[str, CameraBoxes],
    video_width: int,
    video_height: int,
    ffmpeg_bin: str | None,
    min_score: float,
    min_margin: float,
) -> VisualMatch | None:
    target_res = (video_width, video_height)
    candidates = [
        cam for cam in cameras.values()
        if (cam.image_width, cam.image_height) == target_res
    ]
    if not candidates:
        candidates = list(cameras.values())
    if not candidates:
        return None

    vid_frame = _extract_first_frame(video, ffmpeg_bin)
    scored: list[tuple[float, CameraBoxes]] = []
    for cam in candidates:
        score = _fast_camera_score(vid_frame, cam)
        if score is not None:
            scored.append((score, cam))

    if not scored:
        return None

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_cam = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - second_score
    if best_score < min_score or margin < min_margin:
        return None
    return VisualMatch(camera=best_cam, score=best_score, margin=margin)


# ---------------------------------------------------------------------------
# Box scaling and FFmpeg filter
# ---------------------------------------------------------------------------

def scale_boxes(camera: CameraBoxes, video_width: int, video_height: int, pad: int) -> tuple[Box, ...]:
    sx = video_width / camera.image_width
    sy = video_height / camera.image_height
    scaled: list[Box] = []
    for box in camera.boxes:
        x1 = max(0, int(round(box.x1 * sx)) - pad)
        y1 = max(0, int(round(box.y1 * sy)) - pad)
        x2 = min(video_width, int(round(box.x2 * sx)) + pad)
        y2 = min(video_height, int(round(box.y2 * sy)) + pad)
        if x2 > x1 and y2 > y1:
            scaled.append(Box(x1, y1, x2, y2))
    return tuple(scaled)


def ffmpeg_filter(boxes: tuple[Box, ...], blur_radius: int, blur_power: int) -> str:
    if not boxes:
        raise ValueError("At least one box is required")
    n = len(boxes)
    split_labels = ["base"] + [f"crop{i}" for i in range(n)]
    parts = [f"[0:v]split={len(split_labels)}" + "".join(f"[{lbl}]" for lbl in split_labels)]
    current = "base"
    for i, box in enumerate(boxes):
        luma_r = max(1, min(blur_radius, max(1, min(box.width, box.height) // 2)))
        # chroma planes are subsampled (yuv420p: half width/height), so max chroma radius is floor(luma_r/2)
        chroma_r = max(1, luma_r // 2)
        blurred = f"blur{i}"
        output = "vout" if i == n - 1 else f"masked{i}"
        parts.append(
            f"[crop{i}]crop={box.width}:{box.height}:{box.x1}:{box.y1},"
            f"boxblur=luma_radius={luma_r}:luma_power={max(1, blur_power)}:"
            f"chroma_radius={chroma_r}:chroma_power={max(1, blur_power)}"
            f"[{blurred}]"
        )
        parts.append(f"[{current}][{blurred}]overlay={box.x1}:{box.y1}[{output}]")
        current = output
    return ";".join(parts)


# ---------------------------------------------------------------------------
# Job building and execution
# ---------------------------------------------------------------------------

def build_job(
    video: Path,
    output_dir: Path,
    suffix: str,
    camera: CameraBoxes,
    ffprobe_bin: str | None,
    pad: int,
    match_score: float | None = None,
    match_method: str = "exact",
) -> Job:
    width, height = probe_video_size(ffprobe_bin, video)
    boxes = scale_boxes(camera, width, height, pad)
    output_path = output_dir / f"{video.stem}{suffix}{video.suffix}"
    return Job(
        video_path=video,
        output_path=output_path,
        camera=camera,
        boxes=boxes,
        width=width,
        height=height,
        match_score=match_score,
        match_method=match_method,
    )


def passthrough_output_path(video: Path, output_dir: Path, suffix: str) -> Path:
    return output_dir / f"{video.stem}{suffix}{video.suffix}"


def run_job_ffmpeg(
    job: Job,
    ffmpeg_bin: str,
    blur_radius: int,
    blur_power: int,
    overwrite: bool,
    max_output_width: int,
) -> None:
    if job.output_path.exists() and job.output_path.stat().st_size > 0 and not overwrite:
        print(f"SKIP exists: {job.output_path}", flush=True)
        return
    filter_complex = ffmpeg_filter(job.boxes, blur_radius, blur_power)
    map_label = "[vout]"
    if max_output_width > 0 and job.width > max_output_width:
        filter_complex += f";[vout]scale={max_output_width}:-2[vpreview]"
        map_label = "[vpreview]"
    cmd = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error",
        "-y" if overwrite else "-n",
        "-i", str(job.video_path),
        "-filter_complex", filter_complex,
        "-map", map_label,
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(job.output_path),
    ]
    subprocess.run(cmd, check=True)


def _get_ffmpeg_bin() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_job_opencv(job: Job, blur_radius: int, overwrite: bool) -> Path:
    out_path = job.output_path.with_suffix(".mp4")
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"SKIP exists: {out_path}", flush=True)
        return out_path
    import cv2  # type: ignore
    import tempfile as _tmp

    cap = cv2.VideoCapture(str(job.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {job.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Write blurred frames to a temporary MJPG AVI, then transcode to H.264 mp4
    tmp_fd, tmp_avi = _tmp.mkstemp(suffix=".avi")
    import os as _os; _os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp_avi, fourcc, fps, (job.width, job.height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open temp writer: {tmp_avi}")
    kernel = blur_radius * 2 + 1
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, kernel)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for box in job.boxes:
                roi = frame[box.y1:box.y2, box.x1:box.x2]
                if roi.size:
                    frame[box.y1:box.y2, box.x1:box.x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    # Transcode AVI → H.264 mp4 using bundled ffmpeg
    ffmpeg_bin = _get_ffmpeg_bin()
    if ffmpeg_bin:
        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_avi,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        Path(tmp_avi).unlink(missing_ok=True)
    else:
        # No ffmpeg at all — keep as AVI
        out_path = out_path.with_suffix(".avi")
        Path(tmp_avi).rename(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    camera_dir = Path(args.camera_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not camera_dir.is_dir():
        raise SystemExit(f"Camera directory does not exist: {camera_dir}")

    use_ffmpeg = args.engine in {"auto", "ffmpeg"}
    ffmpeg_bin = find_ffmpeg(args.ffmpeg_bin, required=(args.engine == "ffmpeg" and not args.dry_run))
    ffprobe_bin = find_ffprobe(args.ffprobe_bin, ffmpeg_bin, required=False)
    mapping = load_mapping(args.mapping)
    cameras = load_cameras(camera_dir)
    videos = list_videos(input_dir)

    if not videos:
        print(f"No videos found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    unmatched: list[Path] = []
    total_videos = len(videos)

    for index, video in enumerate(videos, start=1):
        camera: CameraBoxes | None = None
        match_score: float | None = None
        match_method = "exact"
        vid_w: int | None = None
        vid_h: int | None = None

        print(f"[match] {index}/{total_videos} {video.name}: start", flush=True)

        # Step 1: exact match (mapping / stem / number prefix)
        if args.match_mode in {"exact", "exact-or-ncc"}:
            camera = exact_match_camera(
                video,
                cameras,
                mapping,
                mapping_fuzzy_min_chars=args.mapping_fuzzy_min_chars,
            )

        already_osd_masked = is_already_osd_masked_video(video)

        # Step 2: NCC visual match
        if camera is None and already_osd_masked and args.match_mode in {"exact-or-ncc", "ncc"}:
            try:
                print(
                    f"[match] {index}/{total_videos} {video.name}: exact miss on already-masked video, trying guarded fast/ncc",
                    flush=True,
                )
                vid_w, vid_h = probe_video_size(ffprobe_bin, video)
                visual_fast = fast_text_match_camera(
                    video,
                    cameras,
                    vid_w,
                    vid_h,
                    ffmpeg_bin,
                    args.min_fast_match_score,
                    args.min_fast_score_margin,
                )
                visual_ncc = ncc_match_camera(
                    video,
                    cameras,
                    vid_w,
                    vid_h,
                    ffmpeg_bin,
                    args.min_match_score,
                    args.min_score_margin,
                )
                if (
                    visual_fast is not None
                    and visual_ncc is not None
                    and visual_fast.camera.camera_id == visual_ncc.camera.camera_id
                ):
                    camera = visual_fast.camera
                    match_score = min(visual_fast.score, visual_ncc.score)
                    match_method = (
                        f"guarded fast={visual_fast.score:.4f}/{visual_fast.margin:.4f} "
                        f"ncc={visual_ncc.score:.4f}/{visual_ncc.margin:.4f}"
                    )
                else:
                    print(
                        f"[match] {index}/{total_videos} {video.name}: guarded visual rematch rejected",
                        flush=True,
                    )
            except Exception as exc:
                print(f"WARNING: guarded matching failed for {video.name}: {exc}", file=sys.stderr)
        elif camera is None and args.match_mode in {"exact-or-ncc", "ncc"}:
            try:
                print(
                    f"[match] {index}/{total_videos} {video.name}: exact miss, trying fast/ncc",
                    flush=True,
                )
                # Probe video size once — reused in build_job too
                vid_w, vid_h = probe_video_size(ffprobe_bin, video)
                visual = fast_text_match_camera(
                    video,
                    cameras,
                    vid_w,
                    vid_h,
                    ffmpeg_bin,
                    args.min_fast_match_score,
                    args.min_fast_score_margin,
                )
                if visual:
                    camera = visual.camera
                    match_score = visual.score
                    match_method = f"fast score={visual.score:.4f} margin={visual.margin:.4f}"

                if camera is not None:
                    pass
                else:
                    visual = ncc_match_camera(
                        video,
                        cameras,
                        vid_w,
                        vid_h,
                        ffmpeg_bin,
                        args.min_match_score,
                        args.min_score_margin,
                    )
                    if visual:
                        camera = visual.camera
                        match_score = visual.score
                        match_method = f"ncc score={visual.score:.4f} margin={visual.margin:.4f}"
            except Exception as exc:
                print(f"WARNING: NCC matching failed for {video.name}: {exc}", file=sys.stderr)

        if camera is None and already_osd_masked and args.force_best_match:
            print(
                f"[match] {index}/{total_videos} {video.name}: skipping force-best on already-masked video",
                flush=True,
            )
        elif camera is None and args.force_best_match:
            try:
                print(
                    f"[match] {index}/{total_videos} {video.name}: trying force-best match",
                    flush=True,
                )
                if vid_w is None or vid_h is None:
                    vid_w, vid_h = probe_video_size(ffprobe_bin, video)
                visual = force_best_match_camera(video, cameras, vid_w, vid_h, ffmpeg_bin)
                if visual:
                    camera = visual.camera
                    match_score = visual.score
                    match_method = f"force-best score={visual.score:.4f} margin={visual.margin:.4f}"
            except Exception as exc:
                print(f"WARNING: force-best-match failed for {video.name}: {exc}", file=sys.stderr)

        if camera is None:
            unmatched.append(video)
            print(f"UNMATCHED {video.name}", flush=True)
            continue

        score_text = f", {match_method}" if match_method != "exact" else ""
        if args.dry_run:
            output_path = output_dir / f"{video.stem}{args.suffix}{video.suffix}"
            print(
                f"MATCH {video.name} -> camera/{camera.camera_id}.json "
                f"({len(camera.boxes)} boxes{score_text}) -> {output_path.name}",
                flush=True,
            )
        else:
            job = build_job(
                video, output_dir, args.suffix, camera,
                ffprobe_bin, args.pad, match_score, match_method,
            )
            jobs.append(job)
            print(
                f"MATCH {job.video_path.name} -> camera/{job.camera.camera_id}.json "
                f"({len(job.boxes)} boxes, {job.width}x{job.height}{score_text}) -> "
                f"{job.output_path.name}",
                flush=True,
            )

    if unmatched:
        print("\nUnmatched videos (not processed):", file=sys.stderr)
        for v in unmatched:
            print(f"  {v.name}", file=sys.stderr)
        print(
            "\nAdd a --mapping JSON or lower --min-match-score / --min-score-margin.",
            file=sys.stderr,
        )

    if args.dry_run:
        return 1 if unmatched else 0
    if unmatched:
        print(
            "\nPassing unmatched videos through without OSD masking so downstream steps can continue.",
            file=sys.stderr,
        )
        for video in unmatched:
            passthrough_path = passthrough_output_path(video, output_dir, args.suffix)
            if passthrough_path.exists() and passthrough_path.stat().st_size > 0 and not args.overwrite:
                print(f"SKIP passthrough exists: {passthrough_path}", flush=True)
                continue
            shutil.copy2(video, passthrough_path)
            print(f"PASSTHROUGH {video.name} -> {passthrough_path.name}", flush=True)

    if not jobs:
        return 0

    if args.engine == "ffmpeg" and not ffmpeg_bin:
        raise RuntimeError("FFmpeg engine requested but ffmpeg is unavailable.")
    if args.engine == "auto" and not ffmpeg_bin:
        print(
            "WARNING: ffmpeg unavailable; using OpenCV fallback. Audio will not be preserved.",
            file=sys.stderr,
        )

    def _run(job: Job) -> None:
        if ffmpeg_bin and args.engine != "opencv":
            run_job_ffmpeg(
                job,
                ffmpeg_bin,
                args.blur_radius,
                args.blur_power,
                args.overwrite,
                args.max_output_width,
            )
            print(f"DONE {job.output_path}", flush=True)
        else:
            out = run_job_opencv(job, args.blur_radius, args.overwrite)
            print(f"DONE {out}", flush=True)

    workers = max(1, int(args.workers))
    failed_jobs: list[tuple[Path, str]] = []
    if workers == 1:
        for job in jobs:
            try:
                _run(job)
            except Exception as exc:
                failed_jobs.append((job.video_path, str(exc)))
                print(f"WARNING: failed OSD masking for {job.video_path.name}: {exc}", file=sys.stderr, flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failed_jobs.append((job.video_path, str(exc)))
                    print(f"WARNING: failed OSD masking for {job.video_path.name}: {exc}", file=sys.stderr, flush=True)

    if failed_jobs:
        print(
            f"WARNING: {len(failed_jobs)} video(s) failed OSD masking and were skipped instead of aborting the whole pipeline.",
            file=sys.stderr,
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
