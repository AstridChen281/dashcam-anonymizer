# dashcam_anonymizer

行车记录仪视频四步打码 pipeline。依次完成：乘用车车牌 → 卡车/公交车牌 → OSD 时间/位置水印 → 人脸，输出完全脱敏的 mp4 视频。

## 目录结构

```
dashcam_anonymizer_release/
├── redact_pipeline.py          # 主入口，四步串联
├── blur_videos.py              # Step 1 & 2：车牌检测 + 打码（多卡并行）
├── yolo_detect_shard.py        # 子进程 GPU worker，由 blur_videos.py 自动调用
├── mask_time_location_by_camera.py  # Step 3：OSD 水印遮盖（NCC 模板匹配）
├── blur_faces.py               # Step 4：人脸检测 + 打码（多卡并行）
├── setup.sh                    # 依赖安装脚本
├── configs/
│   ├── step1_plate_blur.yaml   # Step 1 配置（乘用车车牌，imgsz=1920）
│   ├── step2_truck_blur.yaml   # Step 2 配置（卡车/公交车牌，imgsz=5120）
│   └── face_blur.yaml          # Step 4 配置（人脸，imgsz=1280）
├── camera/                     # OSD 模板（76 台摄像机的 JSON+JPG，Step 3 必需）
└── weight/
    ├── license-plate-finetune-v1x.pt  # Step 1 模型（乘用车车牌，YOLOv8x）
    ├── 0422-v1l.pt                    # Step 2 模型（卡车/公交车牌，YOLOv8l）
    └── yolov8-face.pt                 # Step 4 模型（人脸检测）
```

## 环境依赖

```bash
bash setup.sh
```

或手动安装：

```bash
pip install ultralytics pybboxes opencv-python "numpy==1.26.4" natsort rich pyyaml
```

运行时还需要系统级 `ffmpeg`（用于视频编码）：

```bash
apt install ffmpeg   # Ubuntu/Debian
```

## 快速开始

### 标准 8 卡并行运行（推荐）

```bash
python redact_pipeline.py \
    --input /path/to/your/videos \
    --gpu-workers 8 \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --workers 8
```

输出目录自动创建在脚本目录下的 `blurred_videos/`：

```
blurred_videos/
├── step1_plate_<tag>/   # 乘用车车牌已打码
├── step2_truck_<tag>/   # 卡车/公交车牌已打码
├── step3_osd_<tag>/     # OSD 时间/位置水印已遮盖
└── step4_face_<tag>/    # 人脸已打码（最终结果）
```

最终结果在 `step4_face_<tag>/`。

### 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必填 | 原始视频目录 |
| `--gpu-workers` | `4` | YOLO 检测阶段并行 GPU 数量 |
| `--gpu-ids` | `0,1,2,3` | 使用哪些物理 GPU，逗号分隔 |
| `--workers` | `4` | 视频写出阶段的并行线程数 |
| `--start-step` | `1` | 断点续跑：从第 N 步开始（1–4） |
| `--smoke` | 关闭 | 冒烟模式：随机抽 3 条视频跑全流程 |
| `--smoke-n` | `3` | 冒烟模式抽取视频数量 |
| `--time-limit` | 无限制 | 每条视频只处理前 N 秒（调试用） |

### 示例

```bash
# 冒烟测试（抽 3 条）
python redact_pipeline.py --input videos/batch01 --smoke

# 从 Step 3 断点续跑
python redact_pipeline.py --input videos/batch01 --start-step 3 \
    --gpu-workers 8 --gpu-ids 0,1,2,3,4,5,6,7 --workers 8

# 只处理每条视频前 60 秒（快速验证链路）
python redact_pipeline.py --input videos/batch01 --time-limit 60 \
    --gpu-workers 8 --gpu-ids 0,1,2,3,4,5,6,7
```

## 四步流程详解

### Step 1 — 乘用车车牌打码

- 模型：`weight/license-plate-finetune-v1x.pt`（YOLOv8x，专门针对行车记录仪场景微调）
- 配置：`configs/step1_plate_blur.yaml`，`imgsz=1920`，`conf=0.18`
- 输入：原始视频；输出：`blurred_videos/step1_plate_<tag>/`

### Step 2 — 卡车/公交车牌打码

- 模型：`weight/0422-v1l.pt`（YOLOv8l，针对大型车辆车牌）
- 配置：`configs/step2_truck_blur.yaml`，`imgsz=5120`，`conf=0.10`
- 输入：Step 1 输出；输出：`blurred_videos/step2_truck_<tag>/`

### Step 3 — OSD 时间/位置水印遮盖

- 方法：NCC（归一化互相关）模板匹配，匹配 `camera/` 下 76 台摄像机模板
- 无需 GPU，纯图像处理
- 输入：Step 2 输出；输出：`blurred_videos/step3_osd_<tag>/`

### Step 4 — 人脸打码

- 模型：`weight/yolov8-face.pt`（YOLOv8 人脸检测）
- 配置：`configs/face_blur.yaml`，`imgsz=1280`，`conf=0.40`
- 输入：Step 3 输出；输出：`blurred_videos/step4_face_<tag>/`（**最终结果**）

## 多卡并行机制

Steps 1、2、4 的 YOLO 检测阶段均支持多卡并行：

- 视频列表按帧数×分辨率均衡分片，每片分配一个 GPU
- 通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 隔离每个子进程，子进程内 `device=0`
- 所有 shard 并发跑完后合并 label 文件，再统一进入打码阶段
- Step 3 不使用 GPU，但支持 `--workers` 多线程并行写视频

## 注意事项

- `yolo_detect_shard.py` 不需要手动调用，由 `blur_videos.py` 和 `blur_faces.py` 自动以子进程启动
- 中间产物默认写到 `/tmp/dashcam_anonymizer_artifacts/`，不污染输出目录
- 如果某台摄像机无法匹配 OSD 模板，Step 3 会自动选取得分最高的模板（`--force-best-match`），极少数情况下水印可能未完全遮盖
- `configs/` 中的 `model_path` 使用相对路径，需从项目根目录运行脚本
