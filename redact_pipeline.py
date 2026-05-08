#!/usr/bin/env python3
"""
redact_pipeline.py — 四步一键打码 pipeline
  Step 1: 乘用车车牌打码    (blur_videos.py + license-plate-finetune-v1x.pt)
  Step 2: 卡车/公交车牌打码 (blur_videos.py + 0422-v1l.pt)
  Step 3: 时间/位置水印打码 (mask_time_location_by_camera.py, NCC 匹配)
  Step 4: 人脸打码          (blur_faces.py + yolov8-face.pt)

输出目录（均在 blurred_videos/ 下）：
  step1_plate/              <- step1 结果
  step2_truck/              <- step2 结果（输入来自 step1）
  step3_osd/                <- step3 结果（输入来自 step2）
  step4_face/               <- step4 最终结果（输入来自 step3）

用法：
  # 冒烟测试（随机抽 5 条视频）
  python redact_pipeline.py --input videos/videos0430 --smoke

  # 全量处理
  python redact_pipeline.py --input videos/videos0430

  # 全量 + 并行（GPU 推理 2 卡，视频写出 8 线程）
  python redact_pipeline.py --input videos/videos0430 --workers 8

  # 跳过已完成步骤（断点续跑）
  python redact_pipeline.py --input videos/videos0430 --start-step 3
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable

# ── 配置文件路径 ───────────────────────────────────────────────
STEP1_CFG_TEMPLATE = REPO / "configs" / "step1_plate_blur.yaml"
STEP2_CFG_TEMPLATE = REPO / "configs" / "step2_truck_blur.yaml"
FACE_CFG           = REPO / "configs" / "face_blur.yaml"
CAMERA_DIR         = REPO / "camera"
CAMERA_MAPPING     = CAMERA_DIR / "video_camera_map.json"

# ── 输出目录 ──────────────────────────────────────────────────
BLURRED_ROOT = REPO / "blurred_videos"
PIPELINE_ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "dashcam_anonymizer_artifacts"
SPECIAL_INPUT_SUFFIXES = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'='*60}")
    print(f"[pipeline] {desc}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print('='*60, flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def list_videos(directory: Path) -> list[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def make_temp_config(template: Path, videos_path: str, output_folder: str, artifact_root: str | None = None) -> Path:
    """Write a temp yaml with videos_path and output_folder filled in."""
    import yaml
    with open(template) as f:
        cfg = yaml.safe_load(f)
    cfg["videos_path"] = videos_path
    cfg["output_folder"] = output_folder
    if artifact_root is not None:
        cfg["artifact_root"] = artifact_root
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=str(REPO / "configs"), prefix="tmp_pipeline_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def update_detect_batch_size(cfg_path: Path, batch_size: int | None) -> None:
    if batch_size is None:
        return
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["detect_batch_size"] = int(batch_size)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def update_time_limit(cfg_path: Path, time_limit: float | None) -> None:
    if time_limit is None:
        return
    import yaml
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["time_limit"] = float(time_limit)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def smoke_sample(src_dir: Path, n: int = 5) -> Path:
    """Copy n random videos into a temp dir and return it."""
    videos = list_videos(src_dir)
    chosen = random.sample(videos, min(n, len(videos)))
    tmp_dir = REPO / "blurred_videos" / "_smoke_input"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for v in chosen:
        shutil.copy2(v, tmp_dir / v.name)
    print(f"[smoke] sampled {len(chosen)} videos -> {tmp_dir}")
    for v in chosen:
        print(f"  {v.name}")
    return tmp_dir


def derive_input_suffix(input_path: Path) -> str:
    raw_name = input_path.name if input_path.is_dir() else input_path.parent.name
    if raw_name in SPECIAL_INPUT_SUFFIXES:
        return SPECIAL_INPUT_SUFFIXES[raw_name]

    candidate = raw_name
    lowered = candidate.lower()
    if lowered.startswith("videos"):
        candidate = candidate[6:].lstrip("_- ")
    if not candidate:
        candidate = raw_name

    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", candidate).strip("_").lower()
    return sanitized


def resolve_output_tag(input_path: Path, explicit_tag: str, smoke: bool) -> str:
    if explicit_tag:
        return f"_{explicit_tag}"

    input_suffix = derive_input_suffix(input_path)
    if smoke:
        return f"_{input_suffix}_smoke" if input_suffix else "_smoke"
    return f"_{input_suffix}" if input_suffix else ""


def _delete_dirs_async(paths: list[Path]) -> None:
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)
        print(f"[smoke] deleted old output: {path}")


def clear_smoke_outputs(out_root: Path, tag: str, start_step: int = 1) -> None:
    """Make prior smoke output dirs disappear immediately, then delete them in background."""
    smoke_dirs_by_step = {
        1: out_root / f"step1_plate{tag}",
        2: out_root / f"step2_truck{tag}",
        3: out_root / f"step3_osd{tag}",
        4: out_root / f"step4_face{tag}",
    }
    smoke_dirs = [path for step_no, path in smoke_dirs_by_step.items() if step_no >= start_step]
    pending_delete: list[Path] = []
    for path in smoke_dirs:
        if path.exists():
            renamed = path.with_name(f"{path.name}.__cleanup__.{int(time.time() * 1000)}")
            path.rename(renamed)
            pending_delete.append(renamed)
            print(f"[smoke] queued old output for cleanup: {path}")
    if pending_delete:
        cleanup_thread = threading.Thread(target=_delete_dirs_async, args=(pending_delete,), daemon=True)
        cleanup_thread.start()


def clear_outputs(out_root: Path, tag: str, start_step: int = 1) -> None:
    """同步清空指定步骤及之后的输出目录。适合非 smoke 场景显式重跑。"""
    dirs_by_step = {
        1: out_root / f"step1_plate{tag}",
        2: out_root / f"step2_truck{tag}",
        3: out_root / f"step3_osd{tag}",
        4: out_root / f"step4_face{tag}",
    }
    for step_no, path in dirs_by_step.items():
        if step_no >= start_step and path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"[pipeline] cleared old output: {path}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _clear_artifact_root(artifact_root: Path) -> None:
    """只清当前步骤专属的中间产物，避免并发任务互相影响。"""
    for artifact_dir in (artifact_root / "runs", artifact_root / "annot_jsons"):
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
            print(f"[pipeline] cleared artifact dir: {artifact_dir}")


def artifact_root_for(output_dir: Path) -> Path:
    """把中间产物放到 /tmp，避免污染最终输出目录。"""
    return PIPELINE_ARTIFACT_ROOT / output_dir.name


def step1_plate(input_dir: Path, output_dir: Path, workers: int, gpu_workers: int, gpu_ids: str, detect_batch_size: int | None, time_limit: float | None) -> None:
    artifact_root = artifact_root_for(output_dir)
    _clear_artifact_root(artifact_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_temp_config(STEP1_CFG_TEMPLATE, str(input_dir), str(output_dir), str(artifact_root))
    update_detect_batch_size(cfg, detect_batch_size)
    update_time_limit(cfg, time_limit)
    try:
        run(
            [PYTHON, "blur_videos.py", "--config", str(cfg), "--write-workers", str(workers), "--gpu-workers", str(gpu_workers), "--gpu-ids", gpu_ids],
            f"Step 1: 乘用车车牌打码  {input_dir.name} -> {output_dir.name}",
        )
    finally:
        cfg.unlink(missing_ok=True)


def step2_truck(input_dir: Path, output_dir: Path, workers: int, gpu_workers: int, gpu_ids: str, detect_batch_size: int | None, time_limit: float | None) -> None:
    artifact_root = artifact_root_for(output_dir)
    _clear_artifact_root(artifact_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_temp_config(STEP2_CFG_TEMPLATE, str(input_dir), str(output_dir), str(artifact_root))
    update_detect_batch_size(cfg, detect_batch_size)
    update_time_limit(cfg, time_limit)
    try:
        run(
            [PYTHON, "blur_videos.py", "--config", str(cfg), "--write-workers", str(workers), "--gpu-workers", str(gpu_workers), "--gpu-ids", gpu_ids],
            f"Step 2: 卡车/公交车牌打码  {input_dir.name} -> {output_dir.name}",
        )
    finally:
        cfg.unlink(missing_ok=True)


def step3_osd(input_dir: Path, output_dir: Path, workers: int, gpu_workers: int, gpu_ids: str, detect_batch_size: int | None, time_limit: float | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "mask_time_location_by_camera.py",
        "--input-dir",  str(input_dir),
        "--output-dir", str(output_dir),
        "--camera-dir", str(CAMERA_DIR),
        "--workers",    str(workers),
        "--overwrite",
        "--force-best-match",
        "--min-match-score",      "0.15",
        "--min-score-margin",     "0.005",
        "--min-fast-match-score", "0.30",
        "--min-fast-score-margin","0.005",
        "--blur-radius", "30",
        "--blur-power",  "6",
        "--pad", "6",
    ]
    if CAMERA_MAPPING.exists():
        cmd.extend(["--mapping", str(CAMERA_MAPPING)])
    run(
        cmd,
        f"Step 3: OSD 水印打码  {input_dir.name} -> {output_dir.name}",
    )


def step4_face(input_dir: Path, output_dir: Path, workers: int, gpu_workers: int, gpu_ids: str, detect_batch_size: int | None, time_limit: float | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root_for(output_dir)
    _clear_artifact_root(artifact_root)
    cmd = [
        PYTHON, "blur_faces.py",
        "--videos",     str(input_dir),
        "--output-dir", str(output_dir),
        "--artifact-root", str(artifact_root),
        "--workers",    str(workers),
        "--gpu-workers", str(gpu_workers),
        "--gpu-ids",    gpu_ids,
        "--overwrite",
    ]
    if detect_batch_size is not None:
        cmd.extend(["--detect-batch-size", str(detect_batch_size)])
    run(
        cmd,
        f"Step 4: 人脸打码  {input_dir.name} -> {output_dir.name}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="四步一键视频打码 pipeline")
    p.add_argument(
        "--input", required=True,
        help="原始视频目录，例如 videos/videos0430",
    )
    p.add_argument(
        "--output-root", default=str(BLURRED_ROOT),
        help="输出根目录，默认 blurred_videos/",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="冒烟测试：随机抽 3 条视频跑完全程",
    )
    p.add_argument(
        "--smoke-n", type=int, default=3,
        help="冒烟测试抽取视频数量（默认 3）",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="视频写出并行线程数（不影响 GPU 推理）",
    )
    p.add_argument(
        "--gpu-workers", type=int, default=4,
        help="YOLO 检测阶段使用的 GPU 并行 worker 数，默认 4。",
    )
    p.add_argument(
        "--gpu-ids", default="0,1,2,3",
        help="用于 YOLO 检测阶段的物理 GPU 编号列表，逗号分隔，默认 0,1,2,3。",
    )
    p.add_argument(
        "--detect-batch-size", type=int, default=None,
        help="覆盖 YOLO 检测 batch size。默认使用配置里的值；1280 建议 16，5120 建议 2。",
    )
    p.add_argument(
        "--time-limit", type=float, default=None,
        help="仅处理每条视频前 N 秒。适合赶时间时快速跑 smoke 或验证链路。",
    )
    p.add_argument(
        "--start-step", type=int, default=1, choices=[1, 2, 3, 4],
        help="从指定步骤开始（断点续跑），1=车牌 2=卡车 3=OSD 4=人脸",
    )
    p.add_argument(
        "--tag", default="",
        help="输出子目录后缀，用于区分多次实验，如 --tag smoke",
    )
    p.add_argument(
        "--clean", action="store_true",
        help="运行前先清空当前 tag 对应的输出目录（从 start-step 开始）。",
    )
    p.add_argument(
        "--no-clean", action="store_true",
        help="不要在运行前清空当前 tag 对应的输出目录。",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    out_root  = Path(args.output_root)
    tag       = resolve_output_tag(input_dir, args.tag, args.smoke)

    if not input_dir.is_dir():
        print(f"ERROR: 输入目录不存在: {input_dir}", file=sys.stderr)
        return 1

    if tag:
        print(f"[pipeline] output suffix resolved to: {tag}")
    else:
        print("[pipeline] output suffix resolved to: <default>")

    # 冒烟测试时替换 input_dir 为随机抽样子集
    if args.smoke:
        smoke_input_dir = REPO / "blurred_videos" / "_smoke_input"
        if args.start_step == 1:
            input_dir = smoke_sample(input_dir, args.smoke_n)
            clear_smoke_outputs(out_root, tag, start_step=1)
        else:
            if not smoke_input_dir.is_dir():
                print(
                    f"ERROR: 无法从第 {args.start_step} 步续跑 smoke，因为采样目录不存在: {smoke_input_dir}",
                    file=sys.stderr,
                )
                return 1
            input_dir = smoke_input_dir
            print(f"[smoke] reusing existing sample -> {smoke_input_dir}")
            clear_smoke_outputs(out_root, tag, start_step=args.start_step)
    else:
        should_clean = args.clean or not args.no_clean
        if should_clean:
            clear_outputs(out_root, tag, start_step=args.start_step)

    # 各步骤输入/输出目录
    dirs = {
        1: (input_dir,                              out_root / f"step1_plate{tag}"),
        2: (out_root / f"step1_plate{tag}",         out_root / f"step2_truck{tag}"),
        3: (out_root / f"step2_truck{tag}",         out_root / f"step3_osd{tag}"),
        4: (out_root / f"step3_osd{tag}",           out_root / f"step4_face{tag}"),
    }

    steps = {
        1: step1_plate,
        2: step2_truck,
        3: step3_osd,
        4: step4_face,
    }

    for step_no in range(args.start_step, 5):
        in_dir, out_dir = dirs[step_no]
        steps[step_no](in_dir, out_dir, args.workers, args.gpu_workers, args.gpu_ids, args.detect_batch_size, args.time_limit)

    final_out = dirs[4][1]
    print(f"\n[pipeline] 全部完成。最终视频在: {final_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
