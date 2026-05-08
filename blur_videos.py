# 逐行中文注释版副本见同目录下的 blur_videos_annotated.py。
import os
import glob
import json
import cv2
import pybboxes as pbx
import yaml
import argparse
from ultralytics import YOLO
import shutil
from rich.console import Console
from rich.progress import track
from natsort import natsorted
from os.path import join as osj
import subprocess
import tempfile
import imageio
import numpy as np
import torch
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

repo_dir = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = "."
ARTIFACT_RUNS_DIR = "runs"
ARTIFACT_DETECT_DIR = os.path.join("runs", "detect")
ARTIFACT_JSON_DIR = "annot_jsons"


def find_video_files(directory):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'}
    matches = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in video_extensions:
                matches.append(os.path.join(root, filename))
    return natsorted(matches)


def split_balanced_by_weight(items, num_shards):
    num_shards = max(1, num_shards)
    shards = [[] for _ in range(num_shards)]
    shard_weights = [0 for _ in range(num_shards)]
    weighted_items = sorted(items, key=lambda item: item.get("weight", 1), reverse=True)
    for item in weighted_items:
        shard_idx = min(range(num_shards), key=lambda idx: shard_weights[idx])
        shards[shard_idx].append(item)
        shard_weights[shard_idx] += item.get("weight", 1)
    return [(shard, shard_weights[idx]) for idx, shard in enumerate(shards) if shard]


def parse_gpu_ids(raw_value):
    if not raw_value:
        return []
    gpu_ids = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        gpu_ids.append(int(part))
    return gpu_ids


def get_video_frame_count_estimate(video_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,duration',
            '-of', 'default=noprint_wrappers=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        nb_frames = None
        duration = None
        for line in result.stdout.splitlines():
            if line.startswith('nb_frames='):
                value = line.split('=', 1)[1].strip()
                if value and value.upper() != 'N/A':
                    nb_frames = int(float(value))
            elif line.startswith('duration='):
                value = line.split('=', 1)[1].strip()
                if value and value.upper() != 'N/A':
                    duration = float(value)
        if nb_frames and nb_frames > 0:
            return nb_frames
        if duration and duration > 0:
            fps = get_video_fps_ffprobe(video_path) or 25.0
            return max(1, int(round(duration * fps)))
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(video_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            return frames
    except Exception:
        pass
    return 1


def get_video_resolution_estimate(video_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'default=noprint_wrappers=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width = None
        height = None
        for line in result.stdout.splitlines():
            if line.startswith('width='):
                value = line.split('=', 1)[1].strip()
                if value and value.upper() != 'N/A':
                    width = int(float(value))
            elif line.startswith('height='):
                value = line.split('=', 1)[1].strip()
                if value and value.upper() != 'N/A':
                    height = int(float(value))
        if width and height and width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return 1, 1


def get_video_shard_weight_estimate(video_path):
    frames = max(1, get_video_frame_count_estimate(video_path))
    width, height = get_video_resolution_estimate(video_path)
    pixels = max(1, int(width) * int(height))
    return frames * pixels


def choose_detect_batch_size(config, inference_size):
    explicit = config.get("detect_batch_size")
    if explicit is not None:
        return max(1, int(explicit))
    if inference_size >= 5120:
        return 2
    if inference_size >= 3840:
        return 3
    if inference_size >= 2560:
        return 4
    if inference_size >= 1920:
        return 8
    return 16


def allow_cpu_fallback(config):
    return bool(config.get("allow_cpu_fallback_on_detection_failure", False))


def build_detection_passes(config):
    primary_size = int(config.get('imgsz', 640))
    primary_batch = choose_detect_batch_size(config, primary_size)
    passes = [{
        "run_name": "yolo_videos_pred",
        "imgsz": primary_size,
        "conf": float(config['detection_conf_thresh']),
        "augment": bool(config.get('tta_augment', False)),
        "batch": primary_batch,
        "label": "primary",
    }]

    if config.get("secondary_detection_enabled", False):
        secondary_size = int(config.get("secondary_imgsz", primary_size))
        secondary_batch = config.get("secondary_detect_batch_size")
        if secondary_batch is None:
            secondary_batch = choose_detect_batch_size(config, secondary_size)
        passes.append({
            "run_name": "yolo_videos_pred_pass2",
            "imgsz": secondary_size,
            "conf": float(config.get("secondary_detection_conf_thresh", config['detection_conf_thresh'])),
            "augment": bool(config.get("secondary_tta_augment", True)),
            "batch": max(1, int(secondary_batch)),
            "label": "secondary",
        })

    return passes


def build_detection_shard_source(shard_items):
    kind = shard_items[0]["kind"]
    if kind == "video" and len(shard_items) == 1:
        return shard_items[0]["source_path"], None

    shard_dir = tempfile.mkdtemp(prefix="yolo_shard_")
    try:
        for item in shard_items:
            if item["kind"] == "video":
                src = os.path.abspath(item["source_path"])
                dst = os.path.join(shard_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    os.symlink(src, dst)
            else:
                pattern = os.path.join(item["source_root"], f"{item['vid_name']}_*.jpg")
                for frame_path in glob.glob(pattern):
                    dst = os.path.join(shard_dir, os.path.basename(frame_path))
                    if not os.path.exists(dst):
                        os.symlink(os.path.abspath(frame_path), dst)
    except Exception:
        shutil.rmtree(shard_dir, ignore_errors=True)
        raise
    return shard_dir, shard_dir


def run_detection_shard(shard_items, device_id, inference_size, conf_thresh, augment, model_path, batch_size, run_name="yolo_videos_pred", cpu_fallback_enabled=False):
    project_dir = ARTIFACT_DETECT_DIR
    run_name = f"{run_name}_gpu{device_id}"
    run_dir = os.path.join(project_dir, run_name)

    def reset_run_dir():
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)

    def build_cmd(source_path, current_batch_size, device_override=None, physical_gpu_label=None):
        resolved_device = "0" if device_override is None else str(device_override)
        resolved_physical_gpu = str(device_id) if physical_gpu_label is None else str(physical_gpu_label)
        cmd = [
            sys.executable,
            os.path.join(repo_dir, "yolo_detect_shard.py"),
            "--model", model_path,
            "--source", source_path,
            "--imgsz", str(inference_size),
            "--conf", str(conf_thresh),
            "--project", project_dir,
            "--name", run_name,
            "--device", resolved_device,
            "--physical-gpu-id", resolved_physical_gpu,
            "--batch", str(current_batch_size),
        ]
        if augment:
            cmd.append("--augment")
        return cmd

    def run_cmd_or_raise(source_path, current_batch_size, device_override=None, physical_gpu_label=None):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if device_override is None:
            env["CUDA_VISIBLE_DEVICES"] = str(device_id)
        elif str(device_override).lower() == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = build_cmd(source_path, current_batch_size, device_override=device_override, physical_gpu_label=physical_gpu_label)
        tail_lines = []
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
            if len(tail_lines) > 60:
                tail_lines = tail_lines[-60:]
        returncode = process.wait()
        if returncode != 0:
            tail_text = "\n".join(tail_lines).strip()
            raise RuntimeError(
                f"YOLO shard failed on GPU {device_id} with batch={current_batch_size} "
                f"source={source_path}\n{tail_text}"
            )

    source = None
    cleanup_dir = None
    try:
        source, cleanup_dir = build_detection_shard_source(shard_items)
        batch_attempts = [max(1, int(batch_size))]
        if batch_attempts[0] > 1:
            batch_attempts.append(1)

        last_error = None
        for attempt_idx, current_batch_size in enumerate(batch_attempts, start=1):
            reset_run_dir()
            try:
                if current_batch_size != batch_attempts[0]:
                    console.print(
                        f"[shard] GPU {device_id}: retrying full shard with smaller batch={current_batch_size}",
                        style="bold yellow"
                    )
                run_cmd_or_raise(source, current_batch_size)
                return len(shard_items), device_id
            except Exception as exc:
                last_error = exc
                console.print(
                    f"[shard] GPU {device_id}: full-shard attempt {attempt_idx}/{len(batch_attempts)} failed",
                    style="bold yellow"
                )
                console.print(str(exc), style="bold red")

        if len(shard_items) > 1:
            console.print(
                f"[shard] GPU {device_id}: falling back to per-video isolation with batch=1",
                style="bold yellow"
            )
            reset_run_dir()
            failed_sources = []
            success_count = 0

            for item in shard_items:
                item_source = None
                item_cleanup_dir = None
                try:
                    item_source, item_cleanup_dir = build_detection_shard_source([item])
                    try:
                        run_cmd_or_raise(item_source, 1)
                    except Exception as gpu_exc:
                        if not cpu_fallback_enabled:
                            raise
                        console.print(
                            f"[shard] GPU {device_id}: retrying {item.get('source_path', item.get('vid_name', '<unknown>'))} on CPU",
                            style="bold yellow"
                        )
                        run_cmd_or_raise(
                            item_source,
                            1,
                            device_override="cpu",
                            physical_gpu_label=f"{device_id}->cpu"
                        )
                    success_count += 1
                except Exception as item_exc:
                    label = item.get("source_path", item.get("vid_name", "<unknown>"))
                    failed_sources.append((label, str(item_exc)))
                    console.print(
                        f"[shard] GPU {device_id}: isolated failure on {label}",
                        style="bold red"
                    )
                finally:
                    if item_cleanup_dir and os.path.exists(item_cleanup_dir):
                        shutil.rmtree(item_cleanup_dir, ignore_errors=True)

            if failed_sources:
                console.print(
                    f"[shard] GPU {device_id}: {len(failed_sources)} input item(s) still failed after GPU/CPU fallback; "
                    f"they will continue with empty detections instead of aborting the pipeline",
                    style="bold yellow"
                )
                for label, err_text in failed_sources[:3]:
                    console.print(f"[shard] failed item: {label}", style="yellow")
                    console.print(err_text, style="red")
            return success_count, device_id

        single_item_label = shard_items[0].get("source_path", shard_items[0].get("vid_name", "<unknown>")) if shard_items else "<unknown>"
        if cpu_fallback_enabled:
            try:
                reset_run_dir()
                console.print(
                    f"[shard] GPU {device_id}: retrying single failing item on CPU -> {single_item_label}",
                    style="bold yellow"
                )
                run_cmd_or_raise(
                    source,
                    1,
                    device_override="cpu",
                    physical_gpu_label=f"{device_id}->cpu"
                )
                return len(shard_items), device_id
            except Exception as cpu_exc:
                console.print(
                    f"[shard] GPU {device_id}: {single_item_label} still failed on CPU; continuing with empty detections",
                    style="bold yellow"
                )
                console.print(str(cpu_exc), style="red")
                return 0, device_id

        console.print(
            f"[shard] GPU {device_id}: skipping {single_item_label} after GPU retries; CPU fallback disabled",
            style="bold yellow"
        )
        if last_error is not None:
            console.print(str(last_error), style="red")
        return 0, device_id
    finally:
        if cleanup_dir and os.path.exists(cleanup_dir):
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def run_detection_plan_for_shard(shard_items, device_id, model_path, detection_passes, cpu_fallback_enabled=False):
    processed_count = len(shard_items)
    for pass_index, pass_cfg in enumerate(detection_passes, start=1):
        console.print(
            f"[shard] GPU {device_id}: detection pass {pass_index}/{len(detection_passes)} "
            f"({pass_cfg['label']}, imgsz={pass_cfg['imgsz']}, conf={pass_cfg['conf']}, batch={pass_cfg['batch']}, augment={pass_cfg['augment']})",
            style="cyan"
        )
        processed_count, _ = run_detection_shard(
            shard_items,
            device_id,
            pass_cfg["imgsz"],
            pass_cfg["conf"],
            pass_cfg["augment"],
            model_path,
            pass_cfg["batch"],
            run_name=pass_cfg["run_name"],
            cpu_fallback_enabled=cpu_fallback_enabled,
        )
    return processed_count, device_id

def get_video_fps_ffprobe(video_path):
    """Get exact FPS from video using ffprobe (most accurate method)"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        r_frame_rate = result.stdout.strip()
        # Parse fraction like "25/1" or "30000/1001"
        if '/' in r_frame_rate:
            num, den = map(int, r_frame_rate.split('/'))
            fps = num / den
        else:
            fps = float(r_frame_rate)
        return fps
    except Exception as e:
        # Note: console may not be initialized yet, so we can't use it here
        # Return None to indicate failure, caller will handle fallback
        return None

def get_video_dimensions_ffprobe(video_path):
    """Get video width and height using ffprobe as a codec-agnostic fallback."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0:s=x',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dims = result.stdout.strip()
        if not dims or 'x' not in dims:
            return None
        width_str, height_str = dims.split('x', 1)
        width = int(width_str)
        height = int(height_str)
        return width, height
    except Exception:
        return None

def extract_frames_ffmpeg(video_path, vid_name, temp_dir, time_limit_sec=None):
    """Extract frames using ffmpeg at original video FPS."""
    try:
        original_fps = get_video_fps_ffprobe(video_path)
        if original_fps is None:
            test_cap = cv2.VideoCapture(video_path)
            original_fps = test_cap.get(cv2.CAP_PROP_FPS)
            test_cap.release()
            if original_fps <= 0:
                original_fps = 25.0

        output_pattern = os.path.join(temp_dir, f"{vid_name}_%06d.jpg")
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={original_fps}',
            '-q:v', '2',
        ]
        if time_limit_sec and time_limit_sec > 0:
            cmd.extend(['-t', str(time_limit_sec)])
        cmd.extend(['-y', output_pattern])

        subprocess.run(cmd, capture_output=True, text=True, check=True)

        extracted_frames = len(glob.glob(os.path.join(temp_dir, f"{vid_name}_*.jpg")))
        return extracted_frames, original_fps
    except subprocess.CalledProcessError as e:
        console.print(f"Error extracting frames from {os.path.basename(video_path)}: {e.stderr}", style="bold red")
        return 0, 25.0
    except Exception as e:
        console.print(f"Error extracting frames from {os.path.basename(video_path)}: {e}", style="bold red")
        return 0, 25.0

def bbox_center(region):
    x1, y1, x2, y2 = region
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def bbox_size(region):
    x1, y1, x2, y2 = region
    return max(0.0, x2 - x1), max(0.0, y2 - y1)

def bbox_sort_key(region):
    cx, cy = bbox_center(region)
    return (round(cx, 3), round(cy, 3))

def bbox_iou(region_a, region_b):
    ax1, ay1, ax2, ay2 = region_a
    bx1, by1, bx2, by2 = region_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def interpolate_bbox(prev_bbox, next_bbox, ratio):
    return [
        prev_bbox[0] + (next_bbox[0] - prev_bbox[0]) * ratio,
        prev_bbox[1] + (next_bbox[1] - prev_bbox[1]) * ratio,
        prev_bbox[2] + (next_bbox[2] - prev_bbox[2]) * ratio,
        prev_bbox[3] + (next_bbox[3] - prev_bbox[3]) * ratio,
    ]

def region_already_present(regions, candidate_region, min_iou=0.35, max_center_distance_ratio=0.6):
    candidate_cx, candidate_cy = bbox_center(candidate_region)
    candidate_w, candidate_h = bbox_size(candidate_region)
    candidate_scale = max(1.0, candidate_w, candidate_h)

    for existing_region in regions:
        iou = bbox_iou(existing_region, candidate_region)
        if iou >= min_iou:
            return True

        existing_cx, existing_cy = bbox_center(existing_region)
        center_distance = float(np.hypot(existing_cx - candidate_cx, existing_cy - candidate_cy))
        if center_distance / candidate_scale <= max_center_distance_ratio:
            return True
    return False

def append_region_if_missing(data_dict, frame_id, region, min_iou=0.35, max_center_distance_ratio=0.6):
    regions = [list(existing) for existing in data_dict.get(frame_id, [])]
    if region_already_present(regions, region, min_iou=min_iou, max_center_distance_ratio=max_center_distance_ratio):
        return False
    regions.append(list(region))
    data_dict[frame_id] = regions
    return True


def dedupe_regions(regions, min_iou=0.45, max_center_distance_ratio=0.7):
    deduped = []
    for region in sorted([list(region) for region in regions], key=bbox_sort_key):
        if not region_already_present(
            deduped,
            region,
            min_iou=min_iou,
            max_center_distance_ratio=max_center_distance_ratio,
        ):
            deduped.append(region)
    return deduped

def match_tracks_to_regions(active_tracks, current_regions, frame_id, max_center_distance_ratio=1.8, min_match_iou=0.0, distance_growth_per_frame=0.6):
    candidates = []

    for track_idx, track in enumerate(active_tracks):
        prev_region = track["bbox"]
        prev_cx, prev_cy = bbox_center(prev_region)
        prev_w, prev_h = bbox_size(prev_region)
        prev_scale = max(1.0, prev_w, prev_h)
        gap_frames = max(1, frame_id - track["last_seen"])

        for curr_idx, curr_region in enumerate(current_regions):
            curr_cx, curr_cy = bbox_center(curr_region)
            curr_w, curr_h = bbox_size(curr_region)
            curr_scale = max(1.0, curr_w, curr_h)
            scale = max(prev_scale, curr_scale)

            center_distance = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
            normalized_distance = center_distance / scale
            iou = bbox_iou(prev_region, curr_region)
            allowed_distance = max_center_distance_ratio + max(0, gap_frames - 1) * distance_growth_per_frame
            if iou < min_match_iou and normalized_distance > allowed_distance:
                continue

            candidates.append((iou, -normalized_distance, -gap_frames, track_idx, curr_idx))

    candidates.sort(reverse=True)
    matches = []
    used_tracks = set()
    used_regions = set()
    for _, _, _, track_idx, curr_idx in candidates:
        if track_idx in used_tracks or curr_idx in used_regions:
            continue
        used_tracks.add(track_idx)
        used_regions.add(curr_idx)
        matches.append((track_idx, curr_idx))

    return matches

def fill_detection_gaps(data_dict, gap_fill_window, gap_fill_mode="interpolate", config=None):
    """Fill short per-track detection gaps to reduce plate flicker."""
    if gap_fill_window <= 0 or not data_dict:
        return 0

    config = config or {}
    max_center_distance_ratio = float(config.get("gap_fill_match_distance_ratio", config.get("stabilize_match_distance_ratio", 1.8)))
    min_match_iou = float(config.get("gap_fill_match_iou", 0.0))
    distance_growth_per_frame = float(config.get("gap_fill_distance_growth_per_frame", 0.6))
    dedupe_iou = float(config.get("temporal_dedupe_iou", 0.35))
    dedupe_distance_ratio = float(config.get("temporal_dedupe_distance_ratio", 0.6))

    detected_frames = sorted(data_dict.keys())
    active_tracks = []
    filled = 0

    for frame_id in detected_frames:
        current_regions = [list(region) for region in data_dict.get(frame_id, [])]
        if not current_regions:
            continue

        matches = match_tracks_to_regions(
            active_tracks,
            current_regions,
            frame_id,
            max_center_distance_ratio=max_center_distance_ratio,
            min_match_iou=min_match_iou,
            distance_growth_per_frame=distance_growth_per_frame,
        )
        matched_track_indices = set()
        matched_region_indices = set()

        for track_idx, region_idx in matches:
            track = active_tracks[track_idx]
            prev_frame = track["last_seen"]
            prev_bbox = track["bbox"]
            next_bbox = current_regions[region_idx]
            gap = frame_id - prev_frame - 1
            if gap <= 0:
                matched_track_indices.add(track_idx)
                matched_region_indices.add(region_idx)
                track["bbox"] = list(next_bbox)
                track["last_seen"] = frame_id
                continue

            if gap <= gap_fill_window:
                for missing in range(prev_frame + 1, frame_id):
                    if gap_fill_mode == "interpolate":
                        ratio = (missing - prev_frame) / (frame_id - prev_frame)
                        filled_region = interpolate_bbox(prev_bbox, next_bbox, ratio)
                    else:
                        if missing - prev_frame <= frame_id - missing:
                            filled_region = prev_bbox
                        else:
                            filled_region = next_bbox

                    if append_region_if_missing(
                        data_dict,
                        missing,
                        filled_region,
                        min_iou=dedupe_iou,
                        max_center_distance_ratio=dedupe_distance_ratio,
                    ):
                        filled += 1

            matched_track_indices.add(track_idx)
            matched_region_indices.add(region_idx)
            track["bbox"] = list(next_bbox)
            track["last_seen"] = frame_id

        next_active_tracks = []
        for track_idx, track in enumerate(active_tracks):
            if frame_id - track["last_seen"] <= gap_fill_window and track_idx not in matched_track_indices:
                next_active_tracks.append(track)

        for track_idx in matched_track_indices:
            next_active_tracks.append(active_tracks[track_idx])

        for region_idx, region in enumerate(current_regions):
            if region_idx in matched_region_indices:
                continue
            next_active_tracks.append({"bbox": list(region), "last_seen": frame_id})

        active_tracks = next_active_tracks

    return filled

def carry_forward_recent_detections(data_dict, carry_window, config=None):
    """Carry unmatched tracks forward even if the current frame still contains other detections."""
    if carry_window <= 0 or not data_dict:
        return 0

    config = config or {}
    max_center_distance_ratio = float(config.get("carry_match_distance_ratio", config.get("stabilize_match_distance_ratio", 1.8)))
    min_match_iou = float(config.get("carry_match_iou", 0.0))
    distance_growth_per_frame = float(config.get("carry_distance_growth_per_frame", 0.5))
    dedupe_iou = float(config.get("temporal_dedupe_iou", 0.35))
    dedupe_distance_ratio = float(config.get("temporal_dedupe_distance_ratio", 0.6))

    first_frame = min(data_dict.keys())
    last_frame = max(data_dict.keys())
    carried = 0
    active_tracks = []

    for frame_id in range(first_frame, last_frame + carry_window + 1):
        detected_regions = [list(region) for region in data_dict.get(frame_id, [])]
        output_regions = [list(region) for region in detected_regions]
        matches = match_tracks_to_regions(
            active_tracks,
            detected_regions,
            frame_id,
            max_center_distance_ratio=max_center_distance_ratio,
            min_match_iou=min_match_iou,
            distance_growth_per_frame=distance_growth_per_frame,
        ) if active_tracks and detected_regions else []

        matched_track_indices = {track_idx for track_idx, _ in matches}
        matched_region_indices = {region_idx for _, region_idx in matches}
        next_active_tracks = []

        for track_idx, region_idx in matches:
            matched_region = list(detected_regions[region_idx])
            next_active_tracks.append({"bbox": matched_region, "last_seen": frame_id})

        for track_idx, track in enumerate(active_tracks):
            if track_idx in matched_track_indices:
                continue
            gap = frame_id - track["last_seen"]
            if gap <= 0 or gap > carry_window:
                continue
            if not region_already_present(
                output_regions,
                track["bbox"],
                min_iou=dedupe_iou,
                max_center_distance_ratio=dedupe_distance_ratio,
            ):
                output_regions.append(list(track["bbox"]))
                carried += 1
            next_active_tracks.append(track)

        for region_idx, region in enumerate(detected_regions):
            if region_idx in matched_region_indices:
                continue
            next_active_tracks.append({"bbox": list(region), "last_seen": frame_id})

        if output_regions:
            data_dict[frame_id] = sorted(output_regions, key=bbox_sort_key)

        active_tracks = next_active_tracks

    return carried

def expand_bbox(region, frame_width, frame_height, pad_px=0, pad_ratio=0.0, min_width=0.0, min_height=0.0):
    x1, y1, x2, y2 = [float(v) for v in region]
    cx, cy = bbox_center(region)
    width, height = bbox_size(region)
    target_width = max(width + 2.0 * pad_px, width * (1.0 + 2.0 * pad_ratio), float(min_width))
    target_height = max(height + 2.0 * pad_px, height * (1.0 + 2.0 * pad_ratio), float(min_height))
    half_w = target_width / 2.0
    half_h = target_height / 2.0
    return [
        max(0.0, cx - half_w),
        max(0.0, cy - half_h),
        min(float(frame_width), cx + half_w),
        min(float(frame_height), cy + half_h),
    ]

def smooth_bbox(prev_bbox, curr_bbox, alpha):
    return [
        prev_bbox[0] * (1.0 - alpha) + curr_bbox[0] * alpha,
        prev_bbox[1] * (1.0 - alpha) + curr_bbox[1] * alpha,
        prev_bbox[2] * (1.0 - alpha) + curr_bbox[2] * alpha,
        prev_bbox[3] * (1.0 - alpha) + curr_bbox[3] * alpha,
    ]

def match_regions(prev_regions, curr_regions, max_center_distance_ratio=1.8, min_match_iou=0.0):
    candidates = []
    for prev_idx, prev_region in enumerate(prev_regions):
        prev_cx, prev_cy = bbox_center(prev_region)
        prev_w, prev_h = bbox_size(prev_region)
        prev_scale = max(1.0, prev_w, prev_h)

        for curr_idx, curr_region in enumerate(curr_regions):
            curr_cx, curr_cy = bbox_center(curr_region)
            curr_w, curr_h = bbox_size(curr_region)
            curr_scale = max(1.0, curr_w, curr_h)
            scale = max(prev_scale, curr_scale)

            center_distance = float(np.hypot(curr_cx - prev_cx, curr_cy - prev_cy))
            normalized_distance = center_distance / scale
            iou = bbox_iou(prev_region, curr_region)
            if iou < min_match_iou and normalized_distance > max_center_distance_ratio:
                continue
            candidates.append((iou, -normalized_distance, prev_idx, curr_idx))

    candidates.sort(reverse=True)
    matches = []
    used_prev = set()
    used_curr = set()
    for _, _, prev_idx, curr_idx in candidates:
        if prev_idx in used_prev or curr_idx in used_curr:
            continue
        used_prev.add(prev_idx)
        used_curr.add(curr_idx)
        matches.append((prev_idx, curr_idx))

    return matches

def stabilize_detection_tracks(data_dict, frame_width, frame_height, config):
    if not data_dict or not config.get("temporal_smoothing", True):
        return 0

    alpha = min(max(float(config.get("temporal_smoothing_alpha", 0.55)), 0.0), 1.0)
    pad_px = int(config.get("stabilize_pad_px", 3))
    pad_ratio = float(config.get("stabilize_pad_ratio", 0.12))
    min_width = float(config.get("stabilize_min_width", 28))
    min_height = float(config.get("stabilize_min_height", 14))
    max_center_distance_ratio = float(config.get("stabilize_match_distance_ratio", 1.8))
    min_match_iou = float(config.get("stabilize_match_iou", 0.0))

    previous_regions = []
    updated = 0

    for frame_id in sorted(data_dict.keys()):
        current_regions = [list(region) for region in data_dict.get(frame_id, [])]
        if not current_regions:
            previous_regions = []
            continue

        current_regions = [
            expand_bbox(
                region,
                frame_width,
                frame_height,
                pad_px=pad_px,
                pad_ratio=pad_ratio,
                min_width=min_width,
                min_height=min_height,
            )
            for region in current_regions
        ]

        if previous_regions:
            matches = match_regions(
                previous_regions,
                current_regions,
                max_center_distance_ratio=max_center_distance_ratio,
                min_match_iou=min_match_iou,
            )
            for prev_idx, curr_idx in matches:
                current_regions[curr_idx] = smooth_bbox(previous_regions[prev_idx], current_regions[curr_idx], alpha)

        current_regions = sorted(current_regions, key=bbox_sort_key)
        data_dict[frame_id] = current_regions
        previous_regions = current_regions
        updated += 1

    return updated

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/vid_blur.yaml', help = "path of the training configuartion file", required = False)
parser.add_argument("--write-workers", type=int, default=4, help="parallel threads for video write-out")
parser.add_argument("--gpu-workers", type=int, default=4, help="parallel GPU workers for YOLO detection")
parser.add_argument("--gpu-ids", type=str, default="", help="comma-separated physical GPU ids to use for YOLO detection, e.g. 0,1,2,3")
args = parser.parse_args()
console = Console()

console.print(f"Reading the Configuration file from {args.config}", style="bold green")
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

debug_keep_artifacts = config.get("debug_keep_artifacts", True)
debug_log_details = config.get("debug_log_details", True)
ARTIFACT_ROOT = os.path.abspath(config.get("artifact_root", "."))
ARTIFACT_RUNS_DIR = os.path.join(ARTIFACT_ROOT, "runs")
ARTIFACT_DETECT_DIR = os.path.join(ARTIFACT_RUNS_DIR, "detect")
ARTIFACT_JSON_DIR = os.path.join(ARTIFACT_ROOT, "annot_jsons")
os.makedirs(ARTIFACT_ROOT, exist_ok=True)

console.print(
    f"Debug settings | keep_artifacts={debug_keep_artifacts} | detailed_logs={debug_log_details}",
    style="bold cyan"
)
console.print(
    f"Artifact root: {ARTIFACT_ROOT}",
    style="bold cyan"
)

console.print("Loading YOLO Model...", style="bold green")
try:
    model = YOLO(config["model_path"])
    # Display model information (works for YOLOv8, YOLOv11, etc.)
    model_info = f"Model: {os.path.basename(config['model_path'])}"
    if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
        model_info += f" | Architecture: {model.model.yaml.get('yaml_file', 'Unknown')}"
    console.print(f"✓ {model_info}", style="bold green")
except Exception as e:
    console.print(f"Error loading model from {config['model_path']}: {e}", style="bold red")
    raise

if(config["generate_detections"]):
    console.print("Generating YOLO Detections for the Videos", style="bold green")
    # Note: Supports YOLOv8, YOLOv11, and other Ultralytics YOLO models
    # The YOLO model() call is generally capable of finding all supported video formats in a directory.
    
    # Get time_limit early for use in detection phase
    time_limit = config.get("time_limit", None)  # None means process entire video
    if time_limit and time_limit > 0:
        console.print(f"Time limit: Processing only first {time_limit} seconds for detection", style="bold yellow")
    
    def reencode_video_with_ffmpeg(input_path, output_path):
        """Re-encode video using ffmpeg to ensure OpenCV compatibility"""
        console.print(f"Re-encoding video with ffmpeg: {os.path.basename(input_path)}", style="bold yellow")
        try:
            # Use more compatible encoding settings for OpenCV
            cmd = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',  # Most compatible pixel format
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',  # Web optimized
                '-c:a', 'aac',  # Re-encode audio for compatibility
                '-b:a', '128k',
                '-y',  # Overwrite output file
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print(f"✓ Video re-encoded successfully", style="bold green")
            return True
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else str(e))
            console.print(f"Error re-encoding video: {error_msg}", style="bold red")
            return False
        except FileNotFoundError:
            console.print("Error: ffmpeg not found. Please install ffmpeg.", style="bold red")
            return False
    
    # Process videos_path - handle single file or directory
    videos_to_process = []
    
    if os.path.isfile(config['videos_path']):
        videos_to_process = [config['videos_path']]
    elif os.path.isdir(config['videos_path']):
        videos_to_process = find_video_files(config['videos_path'])
    
    # Check if OpenCV can open videos, if not, use imageio to extract frames
    console.print("Checking video compatibility...", style="bold yellow")
    temp_frame_dir = None
    trimmed_videos_dir = None  # Track trimmed videos directory for cleanup
    trimmed_videos = []
    yolo_source = None
    detection_items = []
    
    # Test if first video can be opened by OpenCV
    if videos_to_process:
        test_video = videos_to_process[0]
        test_cap = cv2.VideoCapture(test_video)
        can_open = test_cap.isOpened()
        if can_open:
            ret, _ = test_cap.read()
            can_open = ret
        test_cap.release()
        
        if not can_open:
            # OpenCV can't open videos, use ffmpeg to extract frames (much faster than imageio)
            console.print("OpenCV cannot open videos, extracting frames with ffmpeg...", style="bold yellow")
            temp_frame_dir = tempfile.mkdtemp(prefix='yolo_frames_')
            
            # Extract frames from all videos in parallel
            # Store FPS for later use in video writing
            video_fps_dict = {}
            max_workers = min(len(videos_to_process), os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for video_path in videos_to_process:
                    vid_name = os.path.splitext(os.path.basename(video_path))[0]
                    console.print(f"Extracting frames from {os.path.basename(video_path)}...", style="yellow")
                    future = executor.submit(extract_frames_ffmpeg, video_path, vid_name, temp_frame_dir, time_limit)
                    futures[future] = (vid_name, video_path)
                
                # Wait for all extractions to complete
                for future in as_completed(futures):
                    vid_name, video_path = futures[future]
                    try:
                        frame_count, fps = future.result()
                        video_fps_dict[vid_name] = fps
                        console.print(f"✓ Extracted {frame_count} frames from {vid_name} (FPS: {fps:.2f})", style="bold green")
                    except Exception as e:
                        console.print(f"✗ Failed to extract frames from {vid_name}: {e}", style="bold red")
                        # Try to get FPS anyway
                        try:
                            fps = get_video_fps_ffprobe(video_path)
                            video_fps_dict[vid_name] = fps
                        except:
                            video_fps_dict[vid_name] = 25.0
            
            yolo_source = temp_frame_dir
            detection_items = []
            for video_path in videos_to_process:
                vid_name = os.path.splitext(os.path.basename(video_path))[0]
                pattern = os.path.join(temp_frame_dir, f"{vid_name}_*.jpg")
                frame_count = len(glob.glob(pattern))
                width, height = get_video_resolution_estimate(video_path)
                detection_items.append({
                    "kind": "frames",
                    "vid_name": vid_name,
                    "source_root": temp_frame_dir,
                    "weight": max(1, frame_count) * max(1, int(width) * int(height)),
                })
        else:
            # OpenCV can open videos, but if time_limit is set, we need to create a trimmed version
            if time_limit and time_limit > 0:
                # Create trimmed videos for YOLO processing
                console.print(f"Creating trimmed videos (first {time_limit} seconds) for YOLO detection...", style="bold yellow")
                trimmed_videos_dir = tempfile.mkdtemp(prefix='trimmed_videos_')
                trimmed_videos = []
                
                for video_path in videos_to_process:
                    vid_name = os.path.splitext(os.path.basename(video_path))[0]
                    trimmed_video_path = os.path.join(trimmed_videos_dir, f"{vid_name}_trimmed.mp4")
                    
                    # Use ffmpeg to trim video
                    cmd = [
                        'ffmpeg', '-i', video_path,
                        '-t', str(time_limit),  # Duration limit
                        '-c', 'copy',  # Copy codec (fast, no re-encoding)
                        '-y',  # Overwrite
                        trimmed_video_path
                    ]
                    try:
                        subprocess.run(cmd, capture_output=True, text=True, check=True)
                        trimmed_videos.append(trimmed_video_path)
                        console.print(f"✓ Created trimmed video: {os.path.basename(trimmed_video_path)}", style="green")
                    except subprocess.CalledProcessError as e:
                        console.print(f"Error trimming {os.path.basename(video_path)}: {e.stderr}", style="bold red")
                        # Fallback to original video
                        trimmed_videos.append(video_path)
                
                if len(trimmed_videos) == 1:
                    yolo_source = trimmed_videos[0]
                else:
                    yolo_source = trimmed_videos_dir
                detection_items = [
                    {"kind": "video", "source_path": path, "weight": get_video_shard_weight_estimate(path)}
                    for path in trimmed_videos
                ]
            else:
                # No time limit, use videos directly
                if len(videos_to_process) == 1:
                    yolo_source = videos_to_process[0]
                elif len(videos_to_process) > 1:
                    if os.path.isfile(config['videos_path']):
                        yolo_source = os.path.dirname(config['videos_path'])
                    else:
                        yolo_source = config['videos_path']
                else:
                    yolo_source = config['videos_path']
                detection_items = [
                    {"kind": "video", "source_path": path, "weight": get_video_shard_weight_estimate(path)}
                    for path in videos_to_process
                ]
    else:
        yolo_source = config['videos_path']
        detection_items = []
    
    if(config["gpu_avail"]):
        # Detect all available GPUs
        requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device = 0
            console.print(
                f"GPU Available: detected {num_gpus} GPU(s); requested detection workers={args.gpu_workers}",
                style="bold green"
            )
            if requested_gpu_ids:
                console.print(
                    f"Using explicit GPU ids for detection shards: {','.join(str(x) for x in requested_gpu_ids)}",
                    style="bold green"
                )
        else:
            device = 'cpu'
            if requested_gpu_ids:
                console.print(
                    f"Parent process cannot see CUDA, but explicit GPU ids were provided: {','.join(str(x) for x in requested_gpu_ids)}",
                    style="bold yellow"
                )
            else:
                console.print("GPU configured but not available, Running on CPU", style="bold yellow")

        detection_passes = build_detection_passes(config)
        cpu_fallback_enabled = allow_cpu_fallback(config)
        for pass_cfg in detection_passes:
            console.print(
                f"Detection pass [{pass_cfg['label']}]: imgsz={pass_cfg['imgsz']} "
                f"conf={pass_cfg['conf']} batch={pass_cfg['batch']} augment={pass_cfg['augment']}",
                style="bold cyan"
            )
        console.print(
            f"Detection failure CPU fallback: {'enabled' if cpu_fallback_enabled else 'disabled'}",
            style="bold cyan"
        )

        gpu_id_pool = requested_gpu_ids[:]
        if not gpu_id_pool and torch.cuda.is_available():
            gpu_id_pool = list(range(torch.cuda.device_count()))

        gpu_worker_count = 1
        if gpu_id_pool:
            gpu_worker_count = min(
                max(1, args.gpu_workers),
                len(gpu_id_pool),
                max(1, len(detection_items) if detection_items else 1),
            )

        if gpu_worker_count > 1 and detection_items and gpu_id_pool:
            device_ids = gpu_id_pool[:gpu_worker_count]
            console.print(
                f"Launching YOLO detection shards across physical GPUs: {','.join(str(x) for x in device_ids)}",
                style="bold green"
            )
            shard_infos = split_balanced_by_weight(detection_items, gpu_worker_count)
            for idx, (shard, total_weight) in enumerate(shard_infos):
                console.print(
                    f"[shard] GPU {device_ids[idx]} <- {len(shard)} input item(s), weight={total_weight}",
                    style="cyan"
                )
            with ThreadPoolExecutor(max_workers=len(shard_infos)) as executor:
                futures = {
                    executor.submit(
                        run_detection_plan_for_shard,
                        shard,
                        device_ids[idx],
                        config["model_path"],
                        detection_passes,
                        cpu_fallback_enabled,
                    ): device_ids[idx]
                    for idx, (shard, _) in enumerate(shard_infos)
                }
                for future in as_completed(futures):
                    gpu_id = futures[future]
                    processed_count, _ = future.result()
                    console.print(
                        f"✓ GPU {gpu_id} finished detection shard ({processed_count} input item(s))",
                        style="bold green"
                    )
            merged_labels = os.path.join(ARTIFACT_DETECT_DIR, "yolo_videos_pred", "labels")
            os.makedirs(merged_labels, exist_ok=True)
            for pass_cfg in detection_passes:
                for gpu_id in device_ids:
                    shard_labels = os.path.join(ARTIFACT_DETECT_DIR, f"{pass_cfg['run_name']}_gpu{gpu_id}", "labels")
                    if os.path.isdir(shard_labels):
                        for f in os.listdir(shard_labels):
                            src = os.path.join(shard_labels, f)
                            dst = os.path.join(merged_labels, f)
                            if not os.path.exists(dst):
                                shutil.copy2(src, dst)
                            else:
                                with open(src, "r") as fin:
                                    new_lines = [line for line in fin.readlines() if line.strip()]
                                if not new_lines:
                                    continue
                                existing_lines = set()
                                with open(dst, "r") as fin:
                                    existing_lines = {line.strip() for line in fin.readlines() if line.strip()}
                                append_lines = [line for line in new_lines if line.strip() not in existing_lines]
                                if append_lines:
                                    with open(dst, "a") as fout:
                                        for line in append_lines:
                                            if not line.endswith("\n"):
                                                line = line + "\n"
                                            fout.write(line)
            console.print(f"✓ Merged shard labels into {merged_labels}", style="bold green")
        else:
            for pass_cfg in detection_passes:
                _ = model(source=yolo_source,
                        save=False,
                        save_txt=True,
                        conf=pass_cfg['conf'],
                        imgsz=pass_cfg['imgsz'],
                        batch=pass_cfg['batch'],
                        augment=pass_cfg['augment'],
                        device=device,
                        project=ARTIFACT_DETECT_DIR,
                        name=pass_cfg["run_name"],
                        exist_ok=True)
    else:
        console.print("GPU Not Available, Running on CPU", style="bold yellow")
        detection_passes = build_detection_passes(config)
        for pass_cfg in detection_passes:
            _ = model(source=yolo_source,
                    save=False,
                    save_txt=True,
                    conf=pass_cfg['conf'],
                    imgsz=pass_cfg['imgsz'],
                    batch=pass_cfg['batch'],
                    augment=pass_cfg['augment'],
                    device='cpu',
                    project=ARTIFACT_DETECT_DIR,
                    name=pass_cfg["run_name"],
                    exist_ok=True)
    
    # Clean up temp directories if created
    if temp_frame_dir and os.path.exists(temp_frame_dir):
        console.print("Cleaning up temporary frame directory...", style="bold yellow")
        shutil.rmtree(temp_frame_dir)
    if trimmed_videos_dir and os.path.exists(trimmed_videos_dir):
        console.print("Cleaning up temporary trimmed videos directory...", style="bold yellow")
        shutil.rmtree(trimmed_videos_dir)
    
    
# =========================================================================================
# CHANGE 1: Search for multiple video file extensions, not just .mp4
# Support both file path and directory path for videos_path
# =========================================================================================
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'] # Add any other video formats you use
videos = []

# Check if videos_path is a file or directory
if os.path.isfile(config['videos_path']):
    # If it's a file, check if it's a video file
    _, ext = os.path.splitext(config['videos_path'])
    if ext.lower() in video_extensions:
        videos = [config['videos_path']]
    else:
        console.print(f"Warning: {config['videos_path']} is not a recognized video file format", style="bold red")
        videos = []
elif os.path.isdir(config['videos_path']):
    # If it's a directory, search recursively for all video files
    videos = find_video_files(config['videos_path'])
else:
    console.print(f"Warning: {config['videos_path']} does not exist", style="bold red")
    videos = []

videos = natsorted(videos)

if len(videos) == 0:
    console.print(f"Warning: No video files found in {config['videos_path']}", style="bold red")
    console.print("Please check that videos_path points to a valid video file or directory containing video files", style="bold yellow")
else:
    console.print(f"Found {len(videos)} video file(s) to process", style="bold green")
# =========================================================================================

if(config["generate_jsons"]):
    print(f"Generating JSONs for {len(videos)} videos")
    
    for video in track(videos):
        # =========================================================================================
        # CHANGE 2: Use os.path.splitext to robustly get the video name without its extension
        # =========================================================================================
        vid_name, _ = os.path.splitext(os.path.basename(video))
        # =========================================================================================

        # Get video dimensions - try OpenCV first, then imageio, then ffprobe
        vid = cv2.VideoCapture(video)
        can_get_dims = vid.isOpened()
        if can_get_dims:
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            can_get_dims = width > 0 and height > 0
        if can_get_dims:
            vid.release()
        else:
            vid.release()
            # OpenCV can't open, use imageio first
            try:
                reader = imageio.get_reader(video, 'ffmpeg')
                first_frame = reader.get_data(0)
                height, width = first_frame.shape[:2]
                reader.close()
            except Exception as e:
                if debug_log_details:
                    console.print(
                        f"[debug] {vid_name}: imageio dimension lookup failed: {e}",
                        style="cyan"
                    )
                ffprobe_dims = get_video_dimensions_ffprobe(video)
                if ffprobe_dims is None:
                    console.print(f"Warning: Could not get video dimensions for {video}: {e}", style="bold yellow")
                    console.print("Skipping JSON generation for this video", style="bold yellow")
                    continue
                width, height = ffprobe_dims
                console.print(
                    f"[debug] {vid_name}: recovered dimensions via ffprobe -> {width}x{height}",
                    style="cyan"
                )
        
        data_dict = {}
        # The yolo prediction folder name matches the video name without extension
        yolo_vid_name = vid_name
        # Try to find labels directory across any incremented YOLO run name such as
        # yolo_videos_pred, yolo_videos_pred2, yolo_videos_pred5, etc.
        possible_paths = [
            os.path.join(ARTIFACT_DETECT_DIR, f'yolo_videos_pred*/labels/{yolo_vid_name}_*.txt'),
        ]
        annot_dir = []
        matched_pattern = None
        for path_pattern in possible_paths:
            found_files = glob.glob(path_pattern)
            if found_files:
                annot_dir = natsorted(found_files)
                matched_pattern = path_pattern
                break

        if debug_log_details:
            console.print(
                f"[debug] {vid_name}: matched {len(annot_dir)} label file(s) using pattern search",
                style="cyan"
            )
            if matched_pattern:
                console.print(
                    f"[debug] {vid_name}: labels loaded from pattern {matched_pattern}",
                    style="cyan"
                )
        
        if not annot_dir:
            console.print(f"Warning: No annotation files found for {vid_name}", style="bold yellow")
            if debug_log_details:
                console.print(
                    f"[debug] {vid_name}: checked patterns -> {possible_paths}",
                    style="cyan"
                )
        
        for file in annot_dir:
            if (os.path.basename(file).endswith('.txt')):
                # Extract frame number from filename like "540140600_5min_000001.txt"
                # Use the last part after splitting by "_"
                filename_base = os.path.basename(file).replace(".txt", "")
                parts = filename_base.split("_")
                # Frame number is the last part (e.g., "000001")
                frame_num_str = parts[-1]
                try:
                    frame_num = int(frame_num_str)
                except ValueError:
                    console.print(f"Warning: Could not parse frame number from {os.path.basename(file)}", style="bold yellow")
                    continue

                with open(file, 'r') as fin:
                    for line in fin.readlines():
                        if line.strip():  # Skip empty lines
                            try:
                                line_data = [float(item) for item in line.split()[1:]]
                                bbox = pbx.convert_bbox(line_data, from_type="yolo", to_type="voc", image_size=(width,height))
                                if(frame_num not in data_dict.keys()):
                                    data_dict[frame_num] = [] # Initialize as empty list
                                data_dict[frame_num].append(bbox)
                            except Exception as e:
                                console.print(f"Warning: Skipping invalid bbox in {os.path.basename(file)}: {e}", style="bold yellow")
                                continue

        frame_dedupe_iou = float(config.get("frame_dedupe_iou", 0.45))
        frame_dedupe_distance_ratio = float(config.get("frame_dedupe_distance_ratio", 0.7))
        if data_dict:
            for frame_num in list(data_dict.keys()):
                data_dict[frame_num] = dedupe_regions(
                    data_dict[frame_num],
                    min_iou=frame_dedupe_iou,
                    max_center_distance_ratio=frame_dedupe_distance_ratio,
                )
        # Gap-filling: fill short occlusion gaps so brief detector misses do not cause flicker.
        gap_fill_window = config.get("gap_fill_window", 5)
        gap_fill_mode = config.get("gap_fill_mode", "interpolate")
        if gap_fill_window > 0 and data_dict:
            filled = fill_detection_gaps(data_dict, gap_fill_window, gap_fill_mode=gap_fill_mode, config=config)
            if filled > 0:
                console.print(
                    f"Gap-filling: inserted {filled} short-gap region(s) (window={gap_fill_window}, mode={gap_fill_mode})",
                    style="bold cyan"
                )

        carry_forward_window = config.get("carry_forward_window", 2)
        if carry_forward_window > 0 and data_dict:
            carried = carry_forward_recent_detections(data_dict, carry_forward_window, config=config)
            if carried > 0 and debug_log_details:
                console.print(
                    f"[debug] {vid_name}: carried {carried} region(s) across brief misses",
                    style="cyan"
                )

        stabilized = stabilize_detection_tracks(data_dict, width, height, config)
        if stabilized > 0 and debug_log_details:
            console.print(
                f"[debug] {vid_name}: temporal smoothing applied to {stabilized} frame(s)",
                style="cyan"
            )

        try:
            os.makedirs(ARTIFACT_JSON_DIR, exist_ok=True)
            with open(os.path.join(ARTIFACT_JSON_DIR, str(vid_name)+".json"), 'w') as f:
                json.dump(data_dict, f)
            console.print(f"✓ Generated JSON with {len(data_dict)} frames containing detections", style="bold green")
            if debug_log_details and len(data_dict) == 0:
                console.print(
                    f"[debug] {vid_name}: JSON was created but contains 0 detected frames",
                    style="cyan"
                )
        except Exception as e:
            console.print(f'Error saving JSON for {video}: {e}', style="bold red")
            import traceback
            console.print(traceback.format_exc(), style="bold red")


def draw_bboxes(image, regions):
    """
    Draws red bounding boxes on the image for detected license plates.
    """
    image_with_boxes = image.copy()
    for region in regions:
        x1, y1, x2, y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        # Ensure coordinates are within image bounds
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        if x1 < x2 and y1 < y2:
            # Draw red rectangle (BGR format: red = (0, 0, 255))
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image_with_boxes

def blur_regions(image, regions):
    """
    Blurs the image, given the x1,y1,x2,y2 cordinates using Gaussian Blur.
    """
    expand_left_px = int(config.get("blur_expand_left_px", 0))
    expand_right_px = int(config.get("blur_expand_right_px", 0))
    expand_top_px = int(config.get("blur_expand_top_px", 0))
    expand_bottom_px = int(config.get("blur_expand_bottom_px", 0))
    expand_x_ratio = float(config.get("blur_expand_x_ratio", 0.0))
    expand_y_ratio = float(config.get("blur_expand_y_ratio", 0.0))

    for region in regions:
        x1,y1,x2,y2 = region
        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)

        x_pad = int(round(box_w * expand_x_ratio))
        y_pad = int(round(box_h * expand_y_ratio))
        x1 -= expand_left_px + x_pad
        x2 += expand_right_px + x_pad
        y1 -= expand_top_px + y_pad
        y2 += expand_bottom_px + y_pad

        # Ensure coordinates are within image bounds
        y1, y2 = max(0, y1), min(image.shape[0], y2)
        x1, x2 = max(0, x1), min(image.shape[1], x2)
        if x1 < x2 and y1 < y2:
            roi = image[y1:y2, x1:x2]
            # Kernel size must be odd
            blur_k = config["blur_radius"] if config["blur_radius"] % 2 != 0 else config["blur_radius"] + 1
            blurred_roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
            image[y1:y2, x1:x2] = blurred_roi
    return image


if not(os.path.exists(config["output_folder"])):
    console.print(f"Creating Directory {config['output_folder']} to store the anonymized videos", style="bold green")
    os.mkdir(config["output_folder"])

anonymized_videos_path = config["output_folder"]

# Get configuration parameters
save_frames = config.get("save_frames", True)  # Default to True if not specified
time_limit = config.get("time_limit", None)  # None means process entire video

# Create directories for saving frame images only if save_frames is True
frames_with_boxes_dir = None
frames_blurred_dir = None
if save_frames:
    frames_with_boxes_dir = os.path.join(anonymized_videos_path, "frames_with_boxes")
    frames_blurred_dir = os.path.join(anonymized_videos_path, "frames_blurred")
    if not os.path.exists(frames_with_boxes_dir):
        os.makedirs(frames_with_boxes_dir)
    if not os.path.exists(frames_blurred_dir):
        os.makedirs(frames_blurred_dir)
    console.print(f"Frame images will be saved to:", style="bold green")
    console.print(f"  - With boxes: {frames_with_boxes_dir}", style="green")
    console.print(f"  - Blurred: {frames_blurred_dir}", style="green")
else:
    console.print("Frame images saving is disabled (save_frames=False)", style="bold yellow")

if time_limit and time_limit > 0:
    console.print(f"Time limit: Processing only first {time_limit} seconds of each video", style="bold yellow")

def process_one_video(video):
    vid_name, _ = os.path.splitext(os.path.basename(video))
    json_path = os.path.join(ARTIFACT_JSON_DIR, f'{vid_name}.json')
    if debug_log_details:
        console.print(f"[debug] {vid_name}: checking JSON at {json_path}", style="cyan")
    if os.path.exists(json_path):
        with open(json_path) as F:
            data = json.load(F)
            detected_frames = len(data)
            console.print(
                f"[debug] {vid_name}: JSON found with {detected_frames} detected frame(s)",
                style="cyan"
            )

            # Check if OpenCV can open the video
            test_cap = cv2.VideoCapture(video)
            can_open = test_cap.isOpened()
            if can_open:
                ret, _ = test_cap.read()
                can_open = ret
            test_cap.release()
            
            # =========================================================================================
            # CHANGE 4: The output video will always be an MP4 for compatibility with the 'avc1' codec.
            # =========================================================================================
            out_vid_path = osj(anonymized_videos_path, vid_name + '.mp4')
            # =========================================================================================
            
            if can_open:
                # Use OpenCV to read video
                video_capture = cv2.VideoCapture(video)
                frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_size = (frame_width, frame_height)
                
                # Get exact FPS using ffprobe (most accurate)
                fps = get_video_fps_ffprobe(video)
                if fps is None:
                    # Fallback to OpenCV FPS
                    fps = video_capture.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 25.0  # Final fallback
                
                # Calculate max frames based on time limit
                max_frames = None
                if time_limit and time_limit > 0:
                    max_frames = int(time_limit * fps)
                    console.print(f"Processing first {time_limit} seconds ({max_frames} frames at {fps:.2f} fps)", style="yellow")
                
                # Use ffmpeg to write video (more reliable than OpenCV VideoWriter)
                temp_frames_dir = tempfile.mkdtemp(prefix='video_frames_')
                frames_written = 0
                
                count = 1
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    # Check time limit
                    if max_frames and count > max_frames:
                        break
                    
                    if str(count) in data:
                        regions = data[str(count)]
                        # Draw red bounding boxes on original frame
                        frame_with_boxes = draw_bboxes(frame, regions)
                        # Save frame with boxes if enabled
                        if save_frames:
                            frame_with_boxes_path = os.path.join(frames_with_boxes_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success = cv2.imwrite(frame_with_boxes_path, frame_with_boxes)
                            if not success:
                                console.print(f"Warning: Failed to save frame {count} with boxes", style="bold yellow")
                        
                        # Blur the regions
                        frame_blurred = blur_regions(frame.copy(), regions)
                        # Save blurred frame if enabled
                        if save_frames:
                            frame_blurred_path = os.path.join(frames_blurred_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success = cv2.imwrite(frame_blurred_path, frame_blurred)
                            if not success:
                                console.print(f"Warning: Failed to save blurred frame {count}", style="bold yellow")
                        
                        # Use blurred frame for output video
                        frame = frame_blurred
                    else:
                        # No detections in this frame, but still save frames if enabled
                        if save_frames:
                            # Save original frame (no boxes, no blur)
                            frame_with_boxes_path = os.path.join(frames_with_boxes_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success1 = cv2.imwrite(frame_with_boxes_path, frame)
                            # Save same frame to blurred directory (no blur applied)
                            frame_blurred_path = os.path.join(frames_blurred_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success2 = cv2.imwrite(frame_blurred_path, frame)
                            if not (success1 and success2):
                                console.print(f"Warning: Failed to save frame {count}", style="bold yellow")
                    
                    # Save frame to temp directory for ffmpeg encoding
                    temp_frame_path = os.path.join(temp_frames_dir, f"frame_{frames_written:06d}.jpg")
                    cv2.imwrite(temp_frame_path, frame)
                    frames_written += 1
                    count += 1
                
                video_capture.release()
                console.print(
                    f"[debug] {vid_name}: wrote {frames_written} frame(s) through OpenCV path",
                    style="cyan"
                )
                
                # Use ffmpeg to combine frames into video
                console.print(f"Encoding video with ffmpeg ({frames_written} frames at {fps:.2f} fps)...", style="bold yellow")
                frame_pattern = os.path.join(temp_frames_dir, "frame_%06d.jpg")
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', frame_pattern,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    '-preset', 'medium',
                    out_vid_path
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    console.print(f"✓ Video saved successfully: {os.path.basename(out_vid_path)}", style="bold green")
                except subprocess.CalledProcessError as e:
                    console.print(f"Error encoding video: {e.stderr}", style="bold red")
                finally:
                    # Clean up temp frames
                    shutil.rmtree(temp_frames_dir)
            else:
                # OpenCV can't open video, so extract frames with ffmpeg and process them with OpenCV.
                console.print(f"OpenCV cannot open {os.path.basename(video)}, extracting frames with ffmpeg for processing...", style="bold yellow")

                extracted_input_dir = tempfile.mkdtemp(prefix='process_frames_')
                extracted_count, fps = extract_frames_ffmpeg(video, vid_name, extracted_input_dir, time_limit)
                if fps is None or fps <= 0:
                    fps = 25.0
                if extracted_count == 0:
                    console.print(f"Error: No frames extracted for {os.path.basename(video)}; copying original video.", style="bold red")
                    shutil.copy(video, out_vid_path)
                    shutil.rmtree(extracted_input_dir)
                    print(f"Processed Video {vid_name}")
                    return

                temp_frames_dir = tempfile.mkdtemp(prefix='video_frames_')
                input_frame_paths = natsorted(glob.glob(os.path.join(extracted_input_dir, f"{vid_name}_*.jpg")))
                frames_written = 0

                count = 1
                for input_frame_path in input_frame_paths:
                    frame_bgr = cv2.imread(input_frame_path)
                    if frame_bgr is None:
                        console.print(f"Warning: Failed to read extracted frame {input_frame_path}", style="bold yellow")
                        count += 1
                        continue

                    if str(count) in data:
                        regions = data[str(count)]
                        frame_with_boxes = draw_bboxes(frame_bgr, regions)
                        if save_frames:
                            frame_with_boxes_path = os.path.join(frames_with_boxes_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success = cv2.imwrite(frame_with_boxes_path, frame_with_boxes)
                            if not success:
                                console.print(f"Warning: Failed to save frame {count} with boxes", style="bold yellow")

                        frame_blurred = blur_regions(frame_bgr.copy(), regions)
                        if save_frames:
                            frame_blurred_path = os.path.join(frames_blurred_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success = cv2.imwrite(frame_blurred_path, frame_blurred)
                            if not success:
                                console.print(f"Warning: Failed to save blurred frame {count}", style="bold yellow")

                        frame_bgr = frame_blurred
                    else:
                        if save_frames:
                            frame_with_boxes_path = os.path.join(frames_with_boxes_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success1 = cv2.imwrite(frame_with_boxes_path, frame_bgr)
                            frame_blurred_path = os.path.join(frames_blurred_dir, f"{vid_name}_frame_{count:06d}.jpg")
                            success2 = cv2.imwrite(frame_blurred_path, frame_bgr)
                            if not (success1 and success2):
                                console.print(f"Warning: Failed to save frame {count}", style="bold yellow")

                    temp_frame_path = os.path.join(temp_frames_dir, f"frame_{frames_written:06d}.jpg")
                    cv2.imwrite(temp_frame_path, frame_bgr)
                    frames_written += 1
                    count += 1

                console.print(
                    f"[debug] {vid_name}: wrote {frames_written} frame(s) through ffmpeg-extracted-frame path",
                    style="cyan"
                )
                
                # Use ffmpeg to combine frames into video
                console.print(f"Encoding video with ffmpeg ({frames_written} frames at {fps:.2f} fps)...", style="bold yellow")
                frame_pattern = os.path.join(temp_frames_dir, "frame_%06d.jpg")
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', frame_pattern,
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    '-preset', 'medium',
                    out_vid_path
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    console.print(f"✓ Video saved successfully: {os.path.basename(out_vid_path)}", style="bold green")
                except subprocess.CalledProcessError as e:
                    console.print(f"Error encoding video: {e.stderr}", style="bold red")
                finally:
                    shutil.rmtree(extracted_input_dir)
                    shutil.rmtree(temp_frames_dir)
        print(f"Processed Video {vid_name}")
    else:
        console.print(f"No objects detected in file {video}, copying file as is.", style="bold yellow")
        if debug_log_details:
            console.print(
                f"[debug] {vid_name}: JSON missing, so output will be an unchanged copy of the input video",
                style="bold cyan"
            )
        shutil.copy(video, anonymized_videos_path)
        console.print(f"Copied Video {vid_name}", style="bold green")


with ThreadPoolExecutor(max_workers=args.write_workers) as pool:
    futures = {pool.submit(process_one_video, v): v for v in videos}
    for fut in track(as_completed(futures), total=len(futures), description="Writing videos"):
        v = futures[fut]
        try:
            fut.result()
        except Exception as e:
            console.print(f"Error processing {os.path.basename(v)}: {e}", style="bold red")

# remove runs folder
if debug_keep_artifacts:
    console.print(f"Debug mode active: keeping {ARTIFACT_RUNS_DIR} and {ARTIFACT_JSON_DIR} for inspection", style="bold cyan")
else:
    if os.path.exists(ARTIFACT_RUNS_DIR):
        console.print(f"Removing Temporary Files...")
        shutil.rmtree(ARTIFACT_RUNS_DIR)
    if os.path.exists(ARTIFACT_JSON_DIR):
        shutil.rmtree(ARTIFACT_JSON_DIR)

console.print(f"Blurred Videos are stored in {anonymized_videos_path}", style="bold yellow")
