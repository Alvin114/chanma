# AiC 2026 RGB–Infrared–Depth 多模态目标检测

本项目包含 AiC 2026 三模态目标检测的源码、配置、测试、数据准备工具和第三方依赖源码，支持 RGB、红外（T）、深度（D）及其组合，并提供 YOLOv5/ICAFusion/M²D-LIF baseline 与 D-FINE E10–E13 实验链路。

> 源码包不包含比赛数据、预训练权重、训练 checkpoint、推理结果或训练日志。解压后需要单独放置数据和下载预训练权重。

## 1. 数据放置位置

项目直接读取赛事原始 ZIP，不需要解压数据集。两个文件必须放在以下准确路径：

```text
项目根目录/
├── data/
│   ├── train/
│   │   └── AIC2026_Train_2000.zip
│   └── test/
│       └── AIC2026_PHASE_1_1000.zip
├── weights/                 # 预训练权重放这里
├── configs/
├── aic_baseline/
└── README.md
```

创建目录并检查文件：

```bash
mkdir -p data/train data/test weights runs
ls -lh data/train/AIC2026_Train_2000.zip \
       data/test/AIC2026_PHASE_1_1000.zip
```

## 2. 安装环境

推荐 Python 3.10 或 3.11。先安装与 CUDA 匹配的 PyTorch，再安装项目依赖：

```bash
conda create -n aic-rgbtd python=3.11 -y
conda activate aic-rgbtd

# CUDA 12.1 示例；其他版本请使用 PyTorch 官方对应命令。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

部分 `run_*.sh` 中的 `PYTHON` 当前写为 `/root/miniconda3/envs/affect/bin/python`。如果环境路径不同，请先修改脚本中的 `PYTHON=...`。

## 3. 生成数据索引

放好两个 ZIP 后，在项目根目录运行：

```bash
python prepare_data.py \
  --data-root data \
  --output data/prepared/manifests \
  --val-fraction 0.2 \
  --seed 3407
python audit_data.py --manifest-dir data/prepared/manifests
```

程序将生成：

```text
data/prepared/manifests/train.jsonl
data/prepared/manifests/val.jsonl
data/prepared/manifests/test.jsonl
data/prepared/manifests/all_train.jsonl
data/prepared/manifests/audit.json
```

训练时直接随机读取 ZIP。建议将数据放在本地 NVMe，而不是网络文件系统。

## 4. 预训练权重

### YOLOv5 baseline

YOLO 配置默认读取 `weights/yolov5s_state.pt`。若不存在，可下载并转换：

```bash
mkdir -p weights
curl -fL \
  https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt \
  -o weights/yolov5s.pt
pip install -r requirements-convert.txt
python convert_yolov5_weights.py \
  --input weights/yolov5s.pt \
  --output weights/yolov5s_state.pt
```

### D-FINE baseline

E10 默认读取 `weights/dfine_l_obj2coco_e25.pth`。`run_e10_dfine_rgbt.sh` 在文件不存在时会自动下载，也可手动下载：

```bash
mkdir -p weights
curl -fL \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth \
  -o weights/dfine_l_obj2coco_e25.pth
```

## 5. 自动测试

生成 manifest 后运行：

```bash
python -m unittest discover -s tests -v
```

## 6. YOLO/ICAFusion/M²D-LIF baseline

推荐基础实验顺序：

```bash
# E01：RGB-only
python train.py --config configs/rgb.yaml

# E02：RGB-T ICAFusion，依赖 E01
python train.py --config configs/rgbt_icafusion.yaml

# E04：RGB-T-D，依赖 E02
python train.py --config configs/rgbtd.yaml
```

也可运行 `./run_e01_e02_e04.sh`。其他实验配置位于 `configs/`，E07/E08 可运行 `./run_e07_e08_m2d_lif.sh`。

基础模型推理示例：

```bash
python predict.py \
  --checkpoint runs/e04_rgbtd/best.pt \
  --device cuda:0 \
  --output-dir runs/e04_rgbtd/submission_txt \
  --zip runs/e04_rgbtd/submission.zip
```

## 7. D-FINE E10–E13

D-FINE 源码位于 `third_party/D-FINE/`，其中包含本项目的多模态输入、wavelet pyramid、P2 特征层和深度融合修改。推荐按以下依赖顺序运行：

```text
E10 RGB-T D-FINE
 └── E12 RGB-T Wavelet + P2
      └── E13 RGB-T-D + 类别均衡
```

训练并自动生成提交 ZIP：

```bash
./run_e10_train_and_submit.sh
./run_e12_train_and_submit.sh
./run_e13_train_and_submit.sh
```

只训练时使用：

```bash
./run_e10_dfine_rgbt.sh
./run_e12_dfine_rgbt_wavelet_p2.sh
./run_e13_dfine_rgbtd_balanced.sh
```

主要配置位于：

```text
configs/dfine/dfine_l_rgbt_aic.yml
configs/dfine/dfine_l_rgbtd_aic.yml
configs/dfine/dfine_l_rgbt_wavelet_p2_aic.yml
configs/dfine/dfine_l_rgbtd_e13_aic.yml
```

D-FINE 单独推理示例：

```bash
python predict_dfine.py \
  --config configs/dfine/dfine_l_rgbtd_e13_aic.yml \
  --checkpoint runs/e13_dfine_l_rgbtd_balanced/best_stg2.pth \
  --device cuda:0 \
  --confidence 0.01 \
  --output-dir runs/e13_dfine_l_rgbtd_balanced/submission_txt \
  --zip runs/e13_dfine_l_rgbtd_balanced/submission.zip
```

## 8. 项目结构与输出

```text
aic_baseline/             数据、模型、训练、指标和推理核心代码
configs/                  YOLO/ICAFusion/M²D-LIF 配置
configs/dfine/            D-FINE E10–E13 配置
docs/                     模型流程和实验文档
tests/                    自动测试
third_party/              ICAFusion、YOLOv5、M²D-LIF、D-FINE 等源码
data/                     比赛数据与生成的索引（不随包提供）
weights/                  预训练权重（不随包提供）
runs/                     checkpoint、日志与提交文件（不随包提供）
prepare_data.py           审计 ZIP 并生成基础 manifest
prepare_dfine_data.py     生成 D-FINE COCO 元数据
prepare_e13_data.py       生成 E13 分层划分和类别均衡元数据
train.py / validate.py    基础模型训练与验证
predict.py                基础模型推理
predict_dfine.py          D-FINE 推理
```

脚本检测到目标实验目录已存在时通常会停止，以避免覆盖训练结果。重跑前请确认并移动旧目录。

详细模型流程见 [`docs/e10-e12-model-flow.md`](docs/e10-e12-model-flow.md)，文档索引见 [`docs/README.md`](docs/README.md)，第三方许可证见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 9. 源码包排除项

为了便于传输，源码压缩包排除了：

- 本仓库及第三方源码内部的 `.git/`；
- `data/` 中的比赛数据和生成索引；
- `weights/` 中的预训练权重；
- `runs/`、`outputs/`、`wandb/`、`lightning_logs/`；
- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.safetensors` 等模型文件；
- 已有 ZIP/TAR/7z 压缩包；
- Python 缓存、训练日志、TensorBoard 事件和临时备份文件。

解压后按照第 1 节放置数据、按照第 4 节准备预训练权重，即可开始运行。
# chanma
