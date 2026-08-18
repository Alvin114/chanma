# AiC 三模态目标检测 Baseline 搭建方案

> 方案日期：2026-08-09  
> 目标：先得到可信、可复现、可消融、可生成比赛提交文件的单模型 baseline，再逐步加入三模态融合创新。

## 1. 结论

建议把 **ICAFusion 原作者仓库作为主工程底座**，复现其 RGB-T 双流检测，然后增加一个轻量 Depth 几何分支，形成比赛的 RGB-Infrared-Depth 单模型 baseline。

主线不是直接追逐最新版 YOLO，也不是一开始实现完整三分支 Transformer，而是：

1. 在同一套代码、同一划分和同一训练参数下跑通 RGB-only；
2. 复现 ICAFusion 的 RGB-T 双流结构；
3. 增加 RGB-D 消融；
4. 将 ICAFusion 的 RGB-T 融合结果与 Depth 几何特征做门控融合；
5. 最后再比较 CSSA 等轻量模块。

推荐的主模型结构：

```text
RGB ───────────────┐
                   ├─ ICAFusion DMFF ── RGB-T fused ─┐
Infrared ──────────┘                                 ├─ DepthGate ─ YOLOv5 Neck/Head
Depth + valid mask + metric flag ─ Depth stem ──────┘
```

这样做的关键原因：Depth 数据质量和语义并不统一。PNG Depth 是 16-bit 毫米深度，而 JPG Depth 是 8-bit 三通道图，不能把 Depth 当作与 RGB、Infrared 完全对称的第三个视觉分支。

## 2. 使用哪些仓库

### 2.1 主仓库：ICAFusion

- 官方仓库：https://github.com/chanchanchan97/ICAFusion
- 论文：https://arxiv.org/abs/2308.07504
- 论文正式版本：Pattern Recognition 2023, 109913
- 作用：提供 YOLOv5 双流检测框架、双模态数据加载、DMFF/迭代交叉注意力模块、训练与推理入口。
- 注意：仓库是 AGPL-3.0；比赛需要提交源码，使用前仍应确认队伍能够接受其许可证要求。

建议的获取方式：

```bash
mkdir -p third_party
git clone https://github.com/chanchanchan97/ICAFusion.git third_party/ICAFusion
git -C third_party/ICAFusion rev-parse HEAD
```

首次验证后应把实际 commit SHA 记录到 `requirements-lock.txt` 或实验记录中，不能长期跟随 `main` 漂移。

### 2.2 对照仓库：Ultralytics YOLOv5

- 官方仓库：https://github.com/ultralytics/yolov5
- 推荐参考版本：`v7.0`
- 作用：核对标准 YOLOv5 的单流训练、COCO 预训练权重、验证与导出逻辑。
- 是否必须 clone：不是。ICAFusion 已包含 YOLOv5 系代码；只有在核对单流行为或定位上游差异时才需要。

如需本地对照：

```bash
git clone --branch v7.0 --depth 1 https://github.com/ultralytics/yolov5.git third_party/yolov5-v7.0
```

不建议第一阶段切换到最新 `ultralytics/ultralytics`。截至方案编写日，该仓库已经持续演进到更新的 YOLO 系列，虽然训练接口成熟，但会让三模态改造与 ICAFusion 论文代码分裂成两套框架，难以进行公平消融。

### 2.3 CSSA：只复现论文模块

- 论文：Multimodal Object Detection by Channel Switching and Spatial Attention
- CVF 论文页：https://openaccess.thecvf.com/content/CVPR2023W/PBVS/html/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.html
- 用途：作为 ICAFusion 之后的轻量融合对照。
- 当前结论：未核验到明确的原作者官方代码仓库，不应随意选择同名第三方项目声称“论文复现”。

CSSA 应根据论文自行实现，并通过模块输入输出、参数量、FLOPs 和双模态消融验证实现是否合理。

### 2.4 暂不作为 baseline 的工作

- RDTTrack/RGBDT500：输入模态匹配，但任务是单目标跟踪，只借鉴 Depth-Thermal prompt 和冗余约束，不直接复现检测头。
- WaveMamba：实现成本和训练风险较高，等稳定 baseline 后再移植融合块。
- MMYOLO：配置化程度高，但其公开稳定版与依赖栈较旧；本项目样本少、改造集中，第一阶段没有必要再引入一套 OpenMMLab 训练体系。
- 多模型投票或权重平均：赛题明确禁止简单模型集成，不进入方案。

## 3. “复现论文”具体指什么

第一篇且唯一需要完整复现的论文是 **ICAFusion**，但只要求复现与本项目有关的闭环：

1. 使用官方代码成功完成一次双流 smoke test；
2. 在本赛题固定验证集上训练 RGB-T 模型；
3. 验证 RGB-T 的 mAP@50-95 是否稳定优于同设置 RGB-only；
4. 确认 DMFF 在 P3/P4/P5 中的融合位置、输入维度和迭代次数；
5. 保存配置、随机种子、commit SHA、环境版本、权重和逐类 AP。

不要求先复现论文在 KAIST、FLIR 上的原始分数，因为赛事禁止使用外部数据训练，且那不会直接验证当前 12 类数据上的收益。

三模态版本不是论文原结构的机械扩展，而是比赛适配：

```text
F_rt = DMFF(F_rgb, F_ir)
F_d  = DepthStem(depth_norm, valid_mask, metric_flag)
gate = sigmoid(Conv([F_rt, F_d]))
F_out = F_rt + gate * Proj(F_d)
```

Depth 使用残差门控注入。初始化时应让 `gate` 偏小，使模型从可靠的 RGB-T 主干开始学习，避免低质量或无效 Depth 在训练初期破坏特征。

## 4. 计划生成的工程结构

后续代码落地建议使用以下结构，不直接修改原始数据压缩包：

```text
.
├── Baseline搭建方案.md
├── data/
│   ├── train/AIC2026_Train_2000.zip
│   ├── test/AIC2026_PHASE_1_1000.zip
│   └── prepared/                 # 解压或整理后的数据，不提交 Git
├── configs/
│   ├── data/aic2026.yaml
│   ├── models/rgb_yolov5s.yaml
│   ├── models/rgbt_icafusion.yaml
│   └── models/rgbtd_depth_gate.yaml
├── src/
│   ├── data/aic_dataset.py
│   ├── data/depth.py
│   ├── data/split.py
│   ├── models/depth_stem.py
│   ├── models/depth_gate.py
│   ├── train.py
│   ├── validate.py
│   └── predict_submit.py
├── scripts/
│   ├── prepare_data.py
│   ├── audit_data.py
│   ├── smoke_test.py
│   └── make_submission.py
├── tests/
│   ├── test_pairing.py
│   ├── test_depth_decode.py
│   ├── test_sync_augment.py
│   └── test_submission.py
├── third_party/
│   ├── ICAFusion/
│   └── yolov5-v7.0/             # 可选，只做对照
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## 5. 数据层方案

### 5.1 配对键

以文件 stem 为唯一 sample id：

```text
visible/<id>.(png|jpg)
infrared/<id>.(png|jpg)
depth/<id>.(png|jpg)
labels/<id>.txt
```

Dataset 初始化时必须一次性验证三模态和标签集合完全一致，不允许运行中静默跳过缺失模态。

### 5.2 Depth 解码

PNG 与 JPG 分支必须分开：

```text
16-bit PNG:
    valid = depth > 0
    depth = clip(depth, 300, 20000)
    depth_norm = log(depth / 300) / log(20000 / 300)
    metric_flag = 1

8-bit JPG:
    转单通道
    使用样本内或数据集统计进行 0~1 归一化
    valid 根据黑色/恒定区域估计
    metric_flag = 0
```

DepthStem 固定接收三个通道：`depth_norm`、`valid_mask`、`metric_flag`。这里的 `metric_flag` 是与图像同尺寸的常数平面。

### 5.3 Infrared 解码

红外原始文件虽然存成三通道，但三个通道没有普通 RGB 彩色语义。baseline 可先转成单通道，再用独立的 1-channel stem；黑色填充区域应生成有效区域 mask，避免模型把边框当场景特征。

### 5.4 标签处理

- 保留原始标签不变；在 Dataset 输出阶段把越界框裁剪到 `[0, 1]`。
- 每次训练输出裁剪框数量和样本 id，禁止静默修复。
- 保留 12 类全部类别，即使某个验证 fold 中极少出现。

### 5.5 数据划分

禁止普通随机切图。文件名显示大量图片来自相同采集序列，且训练集和测试集存在共享序列前缀。

第一版固定划分：

- 80% train、20% val；
- 按复合文件名的采集序列前缀分组；
- 在 group 约束下尽量做多标签分层，使 tricycle、ball、boat、uav 等稀有类进入验证集；
- 对无法解析序列的简单数字文件名，增加相邻编号分组或感知哈希聚类，避免连续帧跨集合。

最终只保留一个官方开发划分用于选模型；可额外运行 5-fold 验证稳定性，但不能把不同 fold 模型集成提交。

## 6. 数据增强

必须同步执行三模态几何增强：resize、letterbox、flip、crop、mosaic 使用同一组随机参数。

模态独立的光度增强：

- RGB：轻量 HSV、亮度、对比度；
- Infrared：对比度、噪声、局部饱和与整模态 dropout；
- Depth：随机无效块、量化噪声、深度 dropout；
- 不对 Depth 使用 RGB 的 HSV 增强；
- 不分别调用三次随机几何变换。

第一轮 baseline 先关闭 mosaic 和复杂增强，确认数据与指标闭环；第二轮再开启同步 mosaic。

## 7. 实验矩阵

所有实验使用相同划分、seed、训练轮数、输入分辨率和检测头，避免把训练策略变化误认为模态收益。

| ID | 输入/模型 | 目的 | 是否论文复现 |
|---|---|---|---|
| E00 | RGB-only YOLOv5s，640 | smoke test 和最低成本基线 | 否 |
| E01 | RGB-only YOLOv5s，960 | 正式单模态对照 | 否 |
| E02 | RGB-T ICAFusion，960 | 验证红外与 DMFF 收益 | 是，核心复现 |
| E03 | RGB-D DepthGate，960 | 单独测量 Depth 收益 | 否，比赛适配 |
| E04 | RGB-T-D ICAFusion + DepthGate，960 | 主 baseline | 否，基于 ICAFusion 扩展 |
| E05 | E04 + modality dropout | 模态质量下降鲁棒性 | 否 |
| E06 | RGB-T-D CSSA-lite，960 | 轻量融合对照 | 论文模块自实现 |
| E07 | E04，1280 | 评估小目标与高分辨率收益 | 否 |

每个实验至少记录：

- 总体 mAP@50、mAP@75、mAP@50-95；
- 12 类逐类 AP；
- 小/中/大目标 AP；
- 参数量、FLOPs、显存峰值、单图推理时间；
- RGB、Infrared、Depth 缺失或降质情况下的分数；
- 最佳 epoch、训练时长、配置文件、Git SHA 和权重 SHA256。

## 8. 初始训练配置

建议从以下配置开始，显存不足时只调整 batch size 和梯度累积：

```yaml
model: yolov5s
pretrained: COCO
imgsz: 960
epochs: 150
optimizer: SGD
lr0: 0.01
weight_decay: 0.0005
warmup_epochs: 3
patience: 30
seed: 3407
amp: true
max_det: 100
```

训练顺序：

1. 用 16 个样本 overfit，确认 loss 能明显下降；
2. E00 跑 5~10 epochs，确认训练、验证、推理、提交闭环；
3. E01 完整训练并冻结为 RGB 对照；
4. E02 复现 RGB-T；
5. E03/E04 加 Depth；
6. 只有 E04 确认有效后才做复杂模块和高分辨率实验。

类别不均衡优先使用 image-level class-aware sampler 或稀有类图像重复采样，不建议一开始大幅修改分类损失。任何采样策略都要同时报告 macro mAP 和逐类 AP，防止 person/animal 的数量优势掩盖稀有类退化。

## 9. 验证与提交闭环

验证器必须实现或核对赛事定义的 101 点插值 mAP@50-95，并且输出逐类指标。不能只看 ICAFusion README 中常用的 mAP@50。

提交程序必须检查：

1. 恰好生成 1,000 个 TXT；
2. 文件 stem 与测试图完全一致；
3. 每行恰好 6 列；
4. class id 在 `[0, 11]`；
5. 坐标和 confidence 在 `[0, 1]`；
6. 每图最多 100 框，按 confidence 截断；
7. 无预测的图也生成空 TXT；
8. ZIP 根目录直接包含 TXT，不额外嵌套目录。

## 10. 完成标准

Baseline 只有满足以下条件才算搭建完成：

- 数据审计、配对、Depth 解码和同步增强都有自动化测试；
- E00~E04 可由配置文件切换，不通过复制多份代码实现；
- 固定 seed 重跑时指标波动可解释；
- E02 给出 ICAFusion 是否有效的明确结论；
- E04 是单模型、单次前向推理，符合禁止简单集成的规则；
- 能从原始测试 ZIP 一条命令生成合法提交 ZIP；
- README 能让另一台机器从环境安装到生成提交完整复现。

## 11. 实施优先级

```text
P0  数据整理、分组划分、审计脚本、提交格式测试
P1  RGB-only YOLOv5s 闭环
P2  ICAFusion RGB-T 论文复现
P3  Depth 解码、RGB-D 消融
P4  RGB-T-D + DepthGate 主 baseline
P5  modality dropout、CSSA-lite、高分辨率实验
```

第一阶段实际需要 Git 的只有 ICAFusion 原作者仓库；YOLOv5 官方仓库是可选对照。不要同时 clone WaveMamba、RDTTrack、MMYOLO 等大量项目，它们不会帮助最早的 baseline 闭环。
