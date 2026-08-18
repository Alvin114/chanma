# AiC 2026 RGB–Infrared–Depth 多模态目标检测

## 当前进展

项目已经从早期 YOLOv5 多模态 baseline 转向 D-FINE-L。当前实验主线为：

```text
E10：RGB-T 输入残差融合（已完成并取得当前已确认最好成绩）
E12：RGB-T 小波门控四尺度融合 + P2（代码与训练链路已完成）
E13：RGB-T-D 深度质量门控 + 长尾类别增强（当前最新版本，独立训练）
```

| 实验 | 模态 | 核心改进 | 当前状态 | 已记录结果 |
|---|---|---|---|---|
| E10 | RGB + IR | D-FINE-L；IR 低/高频输入残差注入 RGB | 已完成训练与提交 | 严格验证 mAP@50–95 `41.117`；排行榜 `54.825` |
| E12 | RGB + IR | 16:9 高分辨率、P2 小目标层、四尺度小波门控融合 | 实现、测试和一键训练链路已完成 | 仓库暂未记录训练或榜单成绩 |
| E13 | RGB + IR + Depth | 深度有效性门控、分层验证集、长尾 Copy-Paste 与安全精调 | 当前最新实现；从官方公共预训练权重独立训练 | 仓库暂未记录训练或榜单成绩 |

目前唯一经过排行榜验证的最好结果仍是 E10 的 `54.825`。E12 和 E13 是针对 E10 暴露出的“小目标能力弱、高 IoU 定位不足、辅助模态融合过早”问题所做的后续改进；在获得正式训练结果前，不把它们描述为已经带来分数提升。

## E10：已验证的 D-FINE RGB-T 基线

E10 使用 D-FINE-L 和 HGNetv2-B4，以 RGB 为主模态。IR 被拆成低频结构与高频细节，经零初始化适配器生成残差后加入 RGB，再送入主干网络。模型使用 P3/P4/P5 三个检测尺度，输入统一为 `960 × 960`。

已记录结果：

- 严格分组验证集：mAP@50–95 `41.117`、AP50 `68.980`、AP75 `41.953`；
- 排行榜：`54.825`；
- 相比早期 E02 排行榜成绩 `44.298`，提升 `10.527`；
- 主要短板是 Small AP 仅 `16.623`，同时 AP50 与 AP75 差距较大。

相关入口：

- 配置：`configs/dfine/dfine_l_rgbt_aic.yml`
- 训练：`run_e10_dfine_rgbt.sh`
- 训练并生成提交：`run_e10_train_and_submit.sh`

## E12：RGB-T 小波门控四尺度融合

E12 从 E10 最优 checkpoint 初始化，仍使用 D-FINE-L RGB-T，但把融合从单一输入端扩展到主干的 P2/P3/P4/P5 四个尺度：

- 输入由 `960 × 960` 改为保持 16:9 的 `1536 × 864`；
- 新增 stride-4 的 P2 层，重点改善小目标检测；
- IR 使用轻量特征金字塔，在四个尺度执行 Haar 小波低频融合、高频细节选择和空间质量门控；
- 新增融合残差采用零初始化，初始状态保持 E10 的 RGB 主路径；
- 训练时使用 10% IR modality dropout；
- batch size 为 3，梯度累积 5 次，有效 batch size 为 15；
- 强裁剪和 ZoomOut 在第 66 个 epoch 停止，为后半程固定分辨率精调预留空间；
- E10 的 P3/P4/P5 encoder 权重会迁移到 E12 对应层，新 P2 单独初始化。

相关入口：

- 配置：`configs/dfine/dfine_l_rgbt_wavelet_p2_aic.yml`
- 训练：`run_e12_dfine_rgbt_wavelet_p2.sh`
- 训练并生成提交：`run_e12_train_and_submit.sh`
- 结构说明：[`docs/e10-e12-model-flow.md`](docs/e10-e12-model-flow.md)

## E13：当前最新的 RGB-T-D 版本

E13 沿用 E12 的 RGB-T 小波四尺度结构设计并加入 Depth，实际实现仍是 D-FINE-L。RGB 保持主路径，IR 使用小波融合，Depth 作为可退化的辅助残差在 P2/P3/P4/P5 四个尺度注入。为了保证新分组验证集没有被旧实验权重见过，E13 不加载 E10/E12 checkpoint，而是从官方 Objects365 + COCO 预训练权重独立训练。

本阶段已完成的改动包括：

- 将深度编码为深度值、有效性标记和格式标记三个通道；
- 使用有效区域和深度格式信息控制空间质量门，避免无效深度直接污染 RGB-T 特征；
- Depth 融合输出残差采用零初始化，并使用 15% depth modality dropout；
- 新增按序列分组、分辨率、场景代理和多标签分布约束的验证集划分，避免相邻帧泄漏；
- 按类别的图像出现频率进行整组多模态样本重复采样，最多重复 4 次；
- 对尾部类别 `[1, 7, 10, 11]` 使用 RGB、IR、Depth 同步 Copy-Paste；
- batch size 为 2，梯度累积 8 次，有效 batch size 为 16；
- 第 66 个 epoch 停止 Copy-Paste、强裁剪和 ZoomOut，再进入干净精调阶段；
- 官方三尺度权重的 P3/P4/P5 参数会迁移到四尺度模型的对应层，新 P2、IR 和 Depth 模块单独初始化；
- 增加安全的阶段切换与按实际验证 AP 选择 checkpoint，避免第二阶段回载优化器状态造成训练回退。

E13 与 E12 可以独立运行。两者使用不同验证集协议，因此 E13 不能加载使用旧划分训练得到的 E10/E12 权重，否则新验证集可能已经被旧 checkpoint 见过。当前源码包中没有 E12/E13 的 checkpoint、日志或提交成绩。

### E13 数据预处理的实际使用方式

完整三模态 E13 已经接入以下四项数据处理，但它们的作用阶段不同：

| 数据处理 | 当前 E13 是否使用 | 生效范围 | 实际行为 |
|---|---|---|---|
| 双格式深度加载 + 有效性 mask | 是 | 训练、验证、测试 | 16-bit PNG 作为度量深度，只有 300～20000 范围有效；8-bit JPG/其他格式作为相对深度，非零像素有效。Depth 被编码为“深度值 + valid mask + 格式标记”三通道，无效像素的 valid mask 为 0 |
| 尾部类 Copy-Paste | 是 | 仅训练集，第 0～65 epoch | 对类别 `[1, 7, 10, 11]` 做 RGB、IR、Depth 同步矩形 Copy-Paste；概率 0.45，每张最多粘贴 2 个目标。验证集和测试集不做该增强 |
| 含稀有类图片过采样 | 是 | 仅训练集 | 按类别的图片出现频率计算重复倍数，包含稀有类的完整对齐三模态样本最多重复 4 次；当前训练读取 `data/prepared/dfine_e13/train_balanced.json` |
| 分辨率组 × 场景分层验证集 | 是 | 只用于开发阶段的 train/val 划分 | 从 2000 张官方有标签训练图中划出 1600 张训练、400 张验证；相邻帧序列不能跨集合，并约束分辨率、场景代理和类别分布。当前验证集固定包含约 30 张 `640×360` 样本，避免 360p/UAV 专场被普通随机切分扭曲 |

“场景”是根据标注类别得到的粗粒度代理，不是额外人工场景标签；其中 class 10 被归入 `uav` 场景。分辨率×场景分层是验证协议，不是图像增强。

相关入口：

- 数据准备：`prepare_e13_data.py`
- 配置：`configs/dfine/dfine_l_rgbtd_e13_aic.yml`
- 训练：`run_e13_dfine_rgbtd_balanced.sh`
- 训练并生成提交：`run_e13_train_and_submit.sh`
- 最优权重选择：`select_best_dfine_checkpoint.py`

## 预训练权重

当前 D-FINE 主线使用官方发布的 `dfine_l_obj2coco_e25.pth`：D-FINE-L 先在 Objects365 上训练，再在 COCO 上微调至第 25 个 epoch。该权重只包含公开数据预训练结果，没有见过本赛事训练集或 E13 验证集。

| 用途 | 权重 | 保存位置 | 获取方式 |
|---|---|---|---|
| E10、E13 的检测器初始化 | `dfine_l_obj2coco_e25.pth` | `weights/dfine_l_obj2coco_e25.pth` | E10/E13 脚本自动下载，或按下方命令手动下载 |
| HGNetv2-B4 主干初始化缓存 | `PPHGNetV2_B4_stage1.pth` | `weight/hgnetv2/PPHGNetV2_B4_stage1.pth` | D-FINE 首次构建模型时自动下载；离线环境需要提前放置 |
| E12 | E10 的 `best_stg2.pth` 或 `best_stg1.pth` | `runs/e10_dfine_l_rgbt/` | E12 是旧划分上的递进实验，仅用于 E12 |

手动获取：

```bash
mkdir -p weights weight/hgnetv2

curl -fL --retry 5 \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth \
  -o weights/dfine_l_obj2coco_e25.pth

curl -fL --retry 5 \
  https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B4_stage1.pth \
  -o weight/hgnetv2/PPHGNetV2_B4_stage1.pth
```

E13 只能使用上述官方公共预训练权重作为初始化，不能改为 E10/E12 checkpoint，否则新分组验证集可能已经被旧权重见过。

## 早期实验简述

早期方案主要用于验证模态组合、融合方式和检测底座，现已不再作为主线：

| 实验 | 方案 | 结果或结论 |
|---|---|---|
| E01 | YOLOv5s，RGB-only | 验证 mAP@50–95 `27.127`，作为单模态基线 |
| E02 | YOLOv5s + ICAFusion，RGB-T | 验证 `27.595`，排行榜 `44.298`，是早期最好方案 |
| E04 | YOLOv5s + ICA + DepthGate，RGB-T-D | 验证 `27.296`，比 E02 低 `0.299`，不足以证明 Depth 无效 |
| E07 | YOLOv5s，IR-only | 验证 `5.783`，IR 单模态不足以作为强教师 |
| E08 | YOLOv5s + M²D-LIF，RGB-T | 验证 `26.057`，弱教师蒸馏未带来收益 |
| E11 | D-FINE-L 输入残差融合，RGB-T-D | 保留了配置和脚本，但没有形成有效对照结果，后续由 E13 取代 |

这些实验的共同结论是：继续在 YOLOv5s 上更换融合模块收益有限；强检测底座、P2 小目标特征、保持宽高比的高分辨率输入，以及能够在辅助模态质量差时退化回 RGB 的融合方式更值得投入。

## 新 Codex 接手指南

新的 Codex 进入项目后，应把 E13 视为当前工作入口。`docs/archive/` 只保存历史方案，不能用其中关于 E13/E14 的旧规划覆盖当前代码和本 README。

### 1. 先确认工作区和运行环境

- 先运行 `git status --short`，保留已有修改和训练产物；
- 确认 `third_party/D-FINE/train.py` 存在；
- 推荐 Python 3.10 或 3.11，并安装与 CUDA 匹配的 PyTorch 和 `requirements.txt`；
- 当前运行脚本将 Python 固定为 `/root/miniconda3/envs/affect/bin/python`。如果服务器环境不同，应同时修改 E13 的训练脚本和训练提交脚本中的 `PYTHON=...`；
- `runs/e13_dfine_l_rgbtd_balanced` 已存在时脚本会拒绝覆盖。不要删除未知的已有实验，先检查其中的 `log.txt` 和 checkpoint。

### 2. 放置原始数据并生成基础 manifest

项目直接读取 ZIP，不需要解压。目录应为：

```text
data/
├── train/
│   └── AIC2026_Train_2000.zip
└── test/
    └── AIC2026_PHASE_1_1000.zip
```

如果 `data/prepared/manifests/all_train.jsonl` 尚不存在，先执行：

```bash
python prepare_data.py \
  --data-root data \
  --output data/prepared/manifests \
  --val-fraction 0.2 \
  --seed 3407
```

不要手工随机切分 E13 数据。`run_e13_dfine_rgbtd_balanced.sh` 会调用 `prepare_e13_data.py`，用固定 seed `3407` 重新生成序列隔离、分辨率×场景分层的 E13 train/val，以及 `train_balanced.json`。

### 3. 检查权重和回归测试

E13 脚本会自动获取完整 D-FINE-L 权重；无网络环境则按“预训练权重”一节提前下载。开始长时间训练前运行：

```bash
python -m pytest tests/test_e13_data.py tests/test_e13_dfine.py -q
```

当前预期结果为 `7 passed`。测试通过后仍应检查生成的 `data/prepared/dfine_e13/audit.json`：

- `group_overlap` 必须为空；
- 验证集应为 400 张；
- `360p_640x360` 验证样本应为 30 张；
- 当前四个尾部类别应为 `[11, 1, 7, 10]`，与训练配置中的集合一致。

### 4. 继续当前最新实验

E13 是当前完整的 RGB + IR + Depth 三模态主实验，不是只验证数据预处理的过渡实验，也不是等待后续再加入 Depth 的双模态实验。它不需要先训练 E10 或 E12；确认没有同名输出目录后，直接运行：

```bash
./run_e13_train_and_submit.sh 2>&1 | tee train_e13_and_submit.log
```

该入口会依次完成 E13 数据准备、110 epoch 训练、按验证 AP 选择 checkpoint、1000 张测试图推理和提交 ZIP 生成。主要输出为：

```text
runs/e13_dfine_l_rgbtd_balanced/
├── log.txt
├── best_stg1.pth
├── best_stg2.pth
├── last.pth
└── submission.zip
```

不是每次训练都会同时产生三个 checkpoint。最终选择以 `select_best_dfine_checkpoint.py` 读取 `log.txt` 后得到的实际最高验证 AP 为准，不要只按文件名猜测。

### 5. E13 完成后的工作

1. 把最佳 epoch、mAP@50–95、AP50、AP75、Small/Medium/Large AP 和逐类 AP 写回 README；
2. 上传 `submission.zip` 后记录实际排行榜成绩，在结果返回前不要声称 E13 已优于 E10；
3. 如果 E13 的验证和排行榜结果达到预期，锁定当前三模态结构、数据处理、输入尺寸和训练超参数，不再更换主干或融合模块；
4. 下一主线是实现 E13 的全量最终训练。这里的“合并全部 2000 张”是把开发阶段的 1600 张训练图和 400 张验证图重新合并为原始的 2000 张官方有标签训练图；合并基础数据时每张图片只出现一次，之后训练采样器仍可执行稀有类过采样。该数据绝不包含 1000 张无标签测试图，也不使用测试集伪标签；
5. 全量训练继续使用三模态输入、双格式深度与 valid mask、尾部类 Copy-Paste，并根据全部 2000 张重新计算稀有类图片过采样。由于原来的 400 张验证图也已加入训练，全量阶段不再保留内部验证集，也不再执行“分辨率×场景划分”；
6. 全量训练从开发阶段 E13 的最优 checkpoint 出发，训练轮数、学习率和停止位置必须在合并数据前根据 E13 的最佳 epoch 固定。训练过程中不能再用原 400 张图片挑 checkpoint，最终只保存固定日程结束时的单模型/EMA 权重；
7. 1000 张测试图始终只用于最后推理和生成提交 ZIP，不能参与训练、验证、过采样、Copy-Paste、伪标签或超参数选择；
8. 如果 E13 没有达到预期，先根据 AP75、Small AP、尾部类 AP 和 Depth 有效区域表现定位三模态问题，再决定调整深度门控、采样或增强；RGB-T 同协议实验只作为必要时的诊断消融，不是默认下一主线；
9. 不要用旧划分的 E12 指标与 E13 直接做 Depth 消融，也不要让 E13 加载 E10/E12 的赛事 checkpoint。

当前仓库只实现了 1600/400 开发阶段的 E13 训练与提交；“全部 2000 张最终训练”的数据配置和启动脚本尚未实现，应在 E13 开发阶段结果确认后再新增，避免提前固定错误的训练轮数。

训练数据、预训练权重、checkpoint、日志和提交文件不包含在源码包中。
