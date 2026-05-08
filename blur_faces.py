#!/usr/bin/env python3
"""
blur_faces.py  —  Batch face detection and blurring for dashcam videos.

Pipeline:
  1. YOLO inference on all input videos (GPU batched, one pass)
  2. Parse per-frame detections into JSON
  3. Gap-fill short misses to reduce flicker
  4. Re-read each video, blur detected face regions, write output

Usage:
  python blur_faces.py                          # uses configs/face_blur.yaml
  python blur_faces.py --config my.yaml
  python blur_faces.py --videos path/to/dir     # override videos_path from CLI
  python blur_faces.py --videos single.mp4 --conf 0.45 --workers 4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Disable Ultralytics auto-install of optional packages (e.g. pi-heif) which
# can block execution when the network is unavailable or the package is
# incompatible with the current Python / OS environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import numpy as np
import yaml
from natsort import natsorted
from rich.console import Console
from rich.progress import track
from ultralytics import YOLO

try:
    import pybboxes as pbx  # type: ignore
    HAS_PYBBOXES = True
except ImportError:
    HAS_PYBBOXES = False

console = Console()

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
REPO_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = "."
ARTIFACT_RUNS_DIR = "runs"
ARTIFACT_DETECT_DIR = os.path.join("runs", "detect")
ARTIFACT_JSON_DIR = "annot_jsons"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description="Detect and blur faces in dashcam videos using YOLOv8."
    )
    p.add_argument(
        "--config",
        default=str(repo_root / "configs" / "face_blur.yaml"),
        help="Path to YAML config file.",
    )
    p.add_argument("--videos", default=None, help="Override videos_path from config.")
    p.add_argument("--output-dir", default=None, help="Override output_dir from config.")
    p.add_argument("--artifact-root", default=None, help="Directory for temporary runs/ and annot_jsons/ artifacts.")
    p.add_argument("--model", default=None, help="Override model_path from config.")
    p.add_argument("--conf", type=float, default=None, help="Override detection_conf_thresh.")
    p.add_argument("--imgsz", type=int, default=None, help="Override YOLO imgsz.")
    p.add_argument("--blur-radius", type=int, default=None, help="Override blur_radius.")
    p.add_argument("--workers", type=int, default=None, help="Number of parallel video writers.")
    p.add_argument("--gpu-workers", type=int, default=None, help="Number of parallel GPU detection workers.")
    p.add_argument("--gpu-ids", default=None, help="Comma-separated physical GPU ids for detection shards.")
    p.add_argument("--detect-batch-size", type=int, default=None, help="YOLO face detection batch size.")
    p.add_argument("--no-gpu", action="store_true", help="Force CPU inference.")
    p.add_argument("--skip-detection", action="store_true",
                   help="Skip YOLO inference; use existing annot_jsons/.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output videos.")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, "r") as fh:
        cfg = yaml.safe_load(fh)
    # CLI overrides
    if args.videos:
        cfg["videos_path"] = args.videos
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.artifact_root:
        cfg["artifact_root"] = args.artifact_root
    if args.model:
        cfg["model_path"] = args.model
    if args.conf is not None:
        cfg["detection_conf_thresh"] = args.conf
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.blur_radius is not None:
        cfg["blur_radius"] = args.blur_radius
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.gpu_workers is not None:
        cfg["gpu_workers"] = args.gpu_workers
    if args.gpu_ids is not None:
        cfg["gpu_ids"] = args.gpu_ids
    if args.detect_batch_size is not None:
        cfg["detect_batch_size"] = args.detect_batch_size
    if args.no_gpu:
        cfg["gpu_avail"] = False
    cfg.setdefault("workers", 2)
    cfg.setdefault("gpu_workers", 1)
    cfg.setdefault("gpu_ids", "")
    cfg.setdefault("gap_fill_window", 5)
    cfg.setdefault("gap_fill_mode", "interpolate")
    cfg.setdefault("blur_radius", 31)
    cfg.setdefault("imgsz", 1280)
    cfg.setdefault("detection_conf_thresh", 0.40)
    cfg.setdefault("detect_batch_size", 4)
    cfg.setdefault("save_frames", False)
    cfg.setdefault("time_limit", None)
    cfg.setdefault("debug_keep_artifacts", False)
    return cfg


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def list_videos(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)] if p.suffix.lower() in VIDEO_EXTS else []
    videos = [str(v) for v in p.rglob("*") if v.is_file() and v.suffix.lower() in VIDEO_EXTS]
    return natsorted(videos)


def get_fps(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        val = result.stdout.strip()
        if "/" in val:
            n, d = val.split("/")
            return int(n) / int(d)
        return float(val)
    except Exception:
        pass
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def get_video_size(video_path: str) -> tuple[int, int]:
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def parse_gpu_ids(raw_value: str | None) -> list[int]:
    if not raw_value:
        return []
    gpu_ids: list[int] = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if part:
            gpu_ids.append(int(part))
    return gpu_ids


def get_video_frame_count_estimate(video_path: str) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration",
                "-of", "default=noprint_wrappers=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        nb_frames: int | None = None
        duration: float | None = None
        for line in result.stdout.splitlines():
            if line.startswith("nb_frames="):
                value = line.split("=", 1)[1].strip()
                if value and value.upper() != "N/A":
                    nb_frames = int(float(value))
            elif line.startswith("duration="):
                value = line.split("=", 1)[1].strip()
                if value and value.upper() != "N/A":
                    duration = float(value)
        if nb_frames and nb_frames > 0:
            return nb_frames
        if duration and duration > 0:
            fps = get_fps(video_path)
            return max(1, int(round(duration * fps)))
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames if frames > 0 else 1


def split_balanced_by_weight(items: list[dict], num_shards: int) -> list[tuple[list[dict], int]]:
    num_shards = max(1, num_shards)
    shards = [[] for _ in range(num_shards)]
    shard_weights = [0 for _ in range(num_shards)]
    weighted_items = sorted(items, key=lambda item: item.get("weight", 1), reverse=True)
    for item in weighted_items:
        shard_idx = min(range(num_shards), key=lambda idx: shard_weights[idx])
        shards[shard_idx].append(item)
        shard_weights[shard_idx] += int(item.get("weight", 1))
    return [(shard, shard_weights[idx]) for idx, shard in enumerate(shards) if shard]


def build_detection_shard_source(shard_items: list[dict]) -> tuple[str, str | None]:
    if len(shard_items) == 1:
        return shard_items[0]["source_path"], None

    shard_dir = tempfile.mkdtemp(prefix="face_yolo_shard_")
    try:
        for item in shard_items:
            src = os.path.abspath(item["source_path"])
            dst = os.path.join(shard_dir, os.path.basename(src))
            if not os.path.exists(dst):
                os.symlink(src, dst)
    except Exception:
        shutil.rmtree(shard_dir, ignore_errors=True)
        raise
    return shard_dir, shard_dir


def run_detection_shard(
    shard_items: list[dict],
    physical_gpu_id: int,
    model_path: str,
    imgsz: int,
    conf: float,
    batch_size: int,
) -> int:
    run_name = f"face_pred_gpu{physical_gpu_id}"
    run_dir = os.path.join(ARTIFACT_DETECT_DIR, run_name)

    def reset_run_dir() -> None:
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)

    def run_cmd_or_raise(source_path: str, current_batch_size: int) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        cmd = [
            sys.executable,
            str(REPO_DIR / "yolo_detect_shard.py"),
            "--model", model_path,
            "--source", source_path,
            "--imgsz", str(imgsz),
            "--conf", str(conf),
            "--project", ARTIFACT_DETECT_DIR,
            "--name", run_name,
            "--device", "0",
            "--physical-gpu-id", str(physical_gpu_id),
            "--batch", str(current_batch_size),
        ]
        tail_lines: list[str] = []
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            tail_lines.append(line.rstrip("\n"))
            if len(tail_lines) > 80:
                tail_lines = tail_lines[-80:]
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"Face shard failed on GPU {physical_gpu_id} batch={current_batch_size} source={source_path}\n"
                + "\n".join(tail_lines).strip()
            )

    source = None
    cleanup_dir = None
    try:
        source, cleanup_dir = build_detection_shard_source(shard_items)
        batch_attempts = [max(1, int(batch_size))]
        if batch_attempts[0] > 1:
            batch_attempts.append(1)

        last_error: Exception | None = None
        for attempt_idx, current_batch_size in enumerate(batch_attempts, start=1):
            reset_run_dir()
            try:
                if current_batch_size != batch_attempts[0]:
                    console.print(
                        f"[face shard] GPU {physical_gpu_id}: retrying full shard with batch={current_batch_size}",
                        style="bold yellow",
                    )
                run_cmd_or_raise(source, current_batch_size)
                return len(shard_items)
            except Exception as exc:
                last_error = exc
                console.print(
                    f"[face shard] GPU {physical_gpu_id}: full-shard attempt {attempt_idx}/{len(batch_attempts)} failed",
                    style="bold yellow",
                )
                console.print(str(exc), style="red")

        if len(shard_items) > 1:
            console.print(
                f"[face shard] GPU {physical_gpu_id}: falling back to per-video isolation with batch=1",
                style="bold yellow",
            )
            reset_run_dir()
            success_count = 0
            failed_sources: list[tuple[str, str]] = []
            for item in shard_items:
                item_source = item["source_path"]
                try:
                    run_cmd_or_raise(item_source, 1)
                    success_count += 1
                except Exception as item_exc:
                    failed_sources.append((item_source, str(item_exc)))
                    console.print(
                        f"[face shard] GPU {physical_gpu_id}: isolated failure on {Path(item_source).name}",
                        style="bold yellow",
                    )
            if failed_sources:
                console.print(
                    f"[face shard] GPU {physical_gpu_id}: {len(failed_sources)} video(s) failed detection; "
                    f"they will continue without face detections",
                    style="bold yellow",
                )
            return success_count

        console.print(
            f"[face shard] GPU {physical_gpu_id}: skipping {Path(source).name if source else '<unknown>'} after GPU retries",
            style="bold yellow",
        )
        if last_error is not None:
            console.print(str(last_error), style="red")
        return 0
    finally:
        if cleanup_dir and os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------

def run_yolo(cfg: dict, videos: list[str]) -> None:
    """Run YOLO face detection on all videos, save txt labels to runs/."""
    import torch
    from ultralytics.utils import SETTINGS
    SETTINGS["sync"] = False  # disable telemetry / auto-update checks

    console.print("Loading face detection pipeline...", style="bold green")
    console.print(
        f"Model: {Path(cfg['model_path']).name} | imgsz={cfg['imgsz']} conf={cfg['detection_conf_thresh']} batch={cfg['detect_batch_size']}",
        style="bold green",
    )

    if cfg.get("gpu_avail", True) and torch.cuda.is_available():
        requested_gpu_ids = parse_gpu_ids(cfg.get("gpu_ids", ""))
        gpu_id_pool = requested_gpu_ids[:] if requested_gpu_ids else list(range(torch.cuda.device_count()))
        gpu_worker_count = min(
            max(1, int(cfg.get("gpu_workers", 1))),
            len(gpu_id_pool),
            len(videos),
        ) if gpu_id_pool else 1

        if gpu_worker_count > 1:
            items = [
                {"source_path": vp, "weight": get_video_frame_count_estimate(vp)}
                for vp in videos
            ]
            device_ids = gpu_id_pool[:gpu_worker_count]
            console.print(
                f"Running face detection across physical GPUs: {','.join(str(x) for x in device_ids)}",
                style="bold green",
            )
            shard_infos = split_balanced_by_weight(items, gpu_worker_count)
            for idx, (shard, total_weight) in enumerate(shard_infos):
                console.print(
                    f"[face shard] GPU {device_ids[idx]} <- {len(shard)} video(s), weight={total_weight}",
                    style="cyan",
                )
            with ThreadPoolExecutor(max_workers=len(shard_infos)) as executor:
                futures = {
                    executor.submit(
                        run_detection_shard,
                        shard,
                        device_ids[idx],
                        cfg["model_path"],
                        int(cfg["imgsz"]),
                        float(cfg["detection_conf_thresh"]),
                        int(cfg["detect_batch_size"]),
                    ): device_ids[idx]
                    for idx, (shard, _) in enumerate(shard_infos)
                }
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    processed_count = future.result()
                    console.print(
                        f"✓ GPU {gpu_id} finished face detection shard ({processed_count} video(s))",
                        style="bold green",
                    )
            return

    console.print(
        "Running face detection in single-process mode",
        style="bold yellow" if not (cfg.get("gpu_avail", True) and torch.cuda.is_available()) else "bold green",
    )
    model = YOLO(cfg["model_path"])
    if cfg.get("gpu_avail", True) and torch.cuda.is_available():
        n = torch.cuda.device_count()
        device = list(range(n)) if n > 1 else 0
    else:
        device = "cpu"

    for src in videos:
        console.print(f"  Detecting: {Path(src).name}", style="cyan")
        try:
            results = model(
                source=src,
                stream=True,
                save=False,
                save_txt=True,
                conf=cfg["detection_conf_thresh"],
                imgsz=cfg["imgsz"],
                batch=int(cfg["detect_batch_size"]),
                device=device,
                project=ARTIFACT_DETECT_DIR,
                name="face_pred",
                exist_ok=True,
            )
            for _ in results:
                pass
        except Exception as exc:
            console.print(
                f"  Warning: GPU face detection failed for {Path(src).name}: {exc}. Retrying on CPU.",
                style="bold yellow"
            )
            try:
                results = model(
                    source=src,
                    stream=True,
                    save=False,
                    save_txt=True,
                    conf=cfg["detection_conf_thresh"],
                    imgsz=cfg["imgsz"],
                    batch=1,
                    device="cpu",
                    project=ARTIFACT_DETECT_DIR,
                    name="face_pred",
                    exist_ok=True,
                )
                for _ in results:
                    pass
            except Exception as cpu_exc:
                console.print(
                    f"  Warning: CPU face detection also failed for {Path(src).name}: {cpu_exc}. Continuing without face detections for this video.",
                    style="bold yellow"
                )


# ---------------------------------------------------------------------------
# Detection → JSON
# ---------------------------------------------------------------------------

def _yolo_to_voc(cx: float, cy: float, bw: float, bh: float,
                 img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert YOLO normalized xywh to VOC pixel x1y1x2y2."""
    x1 = (cx - bw / 2) * img_w
    y1 = (cy - bh / 2) * img_h
    x2 = (cx + bw / 2) * img_w
    y2 = (cy + bh / 2) * img_h
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(img_w, int(round(x2))),
        min(img_h, int(round(y2))),
    )


def build_json_for_video(vid_path: str, cfg: dict) -> str:
    """Parse YOLO txt labels for one video into an annot_jsons/<name>.json."""
    vid_name = Path(vid_path).stem
    w, h = get_video_size(vid_path)

    # Find label files: runs/detect/face_pred*/labels/<vid_name>_*.txt
    patterns = [
        os.path.join(ARTIFACT_DETECT_DIR, f"face_pred*/labels/{vid_name}_*.txt"),
        os.path.join(ARTIFACT_DETECT_DIR, f"face_pred*/labels/{vid_name}.txt"),
    ]
    label_files: list[str] = []
    for pat in patterns:
        label_files += glob.glob(pat)
    label_files = natsorted(label_files)

    data: dict[str, list] = {}

    for lf in label_files:
        stem = Path(lf).stem  # e.g. "myvideo_000042"
        parts = stem.split("_")
        try:
            frame_num = int(parts[-1])
        except ValueError:
            continue

        with open(lf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                if len(tokens) < 5:
                    continue
                try:
                    cx, cy, bw, bh = map(float, tokens[1:5])
                    bbox = _yolo_to_voc(cx, cy, bw, bh, w, h)
                except ValueError:
                    continue
                key = str(frame_num)
                if key not in data:
                    data[key] = []
                data[key].append(list(bbox))

    # Gap filling
    gap_win = cfg.get("gap_fill_window", 5)
    gap_mode = cfg.get("gap_fill_mode", "interpolate")
    if gap_win > 0 and data:
        _fill_gaps(data, gap_win, gap_mode)

    os.makedirs(ARTIFACT_JSON_DIR, exist_ok=True)
    json_path = os.path.join(ARTIFACT_JSON_DIR, f"{vid_name}.json")
    with open(json_path, "w") as fh:
        json.dump(data, fh)

    console.print(
        f"  {vid_name}: {len(data)} frames with detections -> {json_path}",
        style="green",
    )
    return json_path


def _interp_box(a: list, b: list, t: float) -> list:
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def _fill_gaps(data: dict[str, list], window: int, mode: str) -> None:
    keys = sorted(int(k) for k in data)
    for i in range(len(keys) - 1):
        f0, f1 = keys[i], keys[i + 1]
        gap = f1 - f0 - 1
        if gap <= 0 or gap > window:
            continue
        boxes0 = data[str(f0)]
        boxes1 = data[str(f1)]
        n = min(len(boxes0), len(boxes1))
        for missing in range(f0 + 1, f1):
            t = (missing - f0) / (f1 - f0)
            filled = []
            for j in range(n):
                if mode == "interpolate":
                    filled.append(_interp_box(boxes0[j], boxes1[j], t))
                else:
                    src = boxes0[j] if t < 0.5 else boxes1[j]
                    filled.append(src)
            if filled:
                data[str(missing)] = filled


# ---------------------------------------------------------------------------
# Blurring
# ---------------------------------------------------------------------------

def blur_frame(frame: np.ndarray, regions: list, radius: int) -> np.ndarray:
    k = radius if radius % 2 == 1 else radius + 1
    k = max(3, k)
    for x1, y1, x2, y2 in regions:
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(frame.shape[1], int(round(x2))), min(frame.shape[0], int(round(y2)))
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return frame


def _get_ffmpeg_bin() -> str | None:
    import shutil as _shutil
    found = _shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def process_video(vid_path: str, json_path: str, output_dir: str, cfg: dict,
                  overwrite: bool = False) -> str | None:
    """Read video, apply face blur per-frame, write output as H.264 mp4."""
    import tempfile as _tmp
    vid_name = Path(vid_path).stem
    out_path = os.path.join(output_dir, f"{vid_name}_face_blurred.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and not overwrite:
        console.print(f"SKIP (exists): {out_path}", style="yellow")
        return out_path

    with open(json_path) as fh:
        data: dict[str, list] = json.load(fh)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        console.print(f"Cannot open video: {vid_path}", style="bold red")
        return None

    fps = get_fps(vid_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_limit = cfg.get("time_limit")
    max_frames = int(time_limit * fps) if time_limit and time_limit > 0 else None

    # Write blurred frames to temp MJPG AVI, then transcode to H.264 mp4
    tmp_fd, tmp_avi = _tmp.mkstemp(suffix=".avi")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp_avi, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        console.print(f"Cannot open writer: {tmp_avi}", style="bold red")
        return None

    blur_r = cfg.get("blur_radius", 31)
    save_frames = cfg.get("save_frames", False)
    frames_dir = os.path.join(output_dir, "debug_frames", vid_name) if save_frames else None
    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)

    frame_idx = 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_idx > max_frames:
                break
            regions = data.get(str(frame_idx), [])
            if regions:
                frame = blur_frame(frame, regions, blur_r)
            if save_frames and regions:
                cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"), frame)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    # Transcode to H.264 mp4
    ffmpeg_bin = _get_ffmpeg_bin()
    if ffmpeg_bin:
        subprocess.run([
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_avi,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ], check=True)
        Path(tmp_avi).unlink(missing_ok=True)
    else:
        # No ffmpeg — keep as AVI
        out_path = out_path.replace(".mp4", ".avi")
        os.rename(tmp_avi, out_path)

    console.print(f"DONE [{frame_idx-1} frames, {len(data)} detected] -> {out_path}", style="bold green")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    global ARTIFACT_ROOT, ARTIFACT_RUNS_DIR, ARTIFACT_DETECT_DIR, ARTIFACT_JSON_DIR
    ARTIFACT_ROOT = os.path.abspath(cfg.get("artifact_root", "."))
    ARTIFACT_RUNS_DIR = os.path.join(ARTIFACT_ROOT, "runs")
    ARTIFACT_DETECT_DIR = os.path.join(ARTIFACT_RUNS_DIR, "detect")
    ARTIFACT_JSON_DIR = os.path.join(ARTIFACT_ROOT, "annot_jsons")
    os.makedirs(ARTIFACT_ROOT, exist_ok=True)
    console.print(f"Artifact root: {ARTIFACT_ROOT}", style="bold cyan")

    videos_path = cfg.get("videos_path", "videos")
    output_dir = cfg.get("output_dir", "blurred_videos/face_blurred")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ARTIFACT_JSON_DIR, exist_ok=True)

    videos = list_videos(videos_path)
    if not videos:
        console.print(f"No videos found in: {videos_path}", style="bold red")
        return 1
    console.print(f"Found {len(videos)} video(s) in {videos_path}", style="bold green")

    # Step 1: YOLO detection
    if not args.skip_detection:
        run_yolo(cfg, videos)
    else:
        console.print("Skipping YOLO detection (--skip-detection)", style="bold yellow")

    # Step 2: Build per-video JSON
    console.print("\nParsing detections into JSON...", style="bold green")
    json_map: dict[str, str] = {}
    for vp in videos:
        jp = build_json_for_video(vp, cfg)
        json_map[vp] = jp

    # Step 3: Blur videos (parallel)
    console.print(f"\nBlurring videos (workers={cfg['workers']})...", style="bold green")
    workers = max(1, int(cfg.get("workers", 2)))

    def _process(vp: str) -> str | None:
        return process_video(vp, json_map[vp], output_dir, cfg, overwrite=args.overwrite)

    if workers == 1:
        for vp in track(videos, description="Blurring..."):
            try:
                _process(vp)
            except Exception as exc:
                console.print(f"Warning: failed face blur for {Path(vp).name}: {exc}", style="bold yellow")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_process, vp): vp for vp in videos}
            for fut in track(as_completed(futs), total=len(futs), description="Blurring..."):
                vp = futs[fut]
                try:
                    fut.result()
                except Exception as exc:
                    console.print(f"Warning: failed face blur for {Path(vp).name}: {exc}", style="bold yellow")

    # Step 4: Cleanup
    if not cfg.get("debug_keep_artifacts", False):
        for d in [ARTIFACT_RUNS_DIR, ARTIFACT_JSON_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
        console.print(f"Cleaned up {ARTIFACT_RUNS_DIR} and {ARTIFACT_JSON_DIR}", style="dim")

    console.print(f"\nAll done. Outputs in: {output_dir}", style="bold green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
