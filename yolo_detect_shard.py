#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one YOLO detection shard on one GPU.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--physical-gpu-id", default="?")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    import torch
    from ultralytics import YOLO

    print(
        f"[yolo_detect_shard] physical_gpu={args.physical_gpu_id} device_arg={args.device} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"torch_cuda={torch.cuda.is_available()} count={torch.cuda.device_count()} batch={args.batch} source={args.source}",
        flush=True,
    )
    if torch.cuda.is_available() and str(args.device).isdigit():
        local_device = int(args.device)
        if 0 <= local_device < torch.cuda.device_count():
            torch.cuda.set_device(local_device)
        else:
            raise RuntimeError(
                f"Requested local CUDA device {local_device}, but this process can only see "
                f"{torch.cuda.device_count()} device(s). CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )

    model = YOLO(args.model)
    results = model(
        source=args.source,
        stream=True,
        save=False,
        save_txt=True,
        conf=args.conf,
        imgsz=args.imgsz,
        batch=args.batch,
        augment=args.augment,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    for _ in results:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
