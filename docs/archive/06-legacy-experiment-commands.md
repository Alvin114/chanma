# D-FINE 多模态打榜命令

当前没有训练进程。以下命令直接使用 `affect` 环境的 Python，不依赖 `conda` 命令。

## 一条命令：训练 E10 并生成提交 ZIP

```bash
./run_e10_train_and_submit.sh 2>&1 | tee train_e10_and_submit.log
```

脚本会完整训练 E10，自动依次选择 `best_stg2.pth`、`best_stg1.pth` 或 `last.pth`，然后对 1000 张测试图推理并生成：

```text
runs/e10_dfine_l_rgbt/submission.zip
```

该脚本不会覆盖已有的 `runs/e10_dfine_l_rgbt`。如果该目录存在，它会直接停止，防止误覆盖权重或中断实验。

当前 RTX 5090 / PyTorch 2.11 环境已经强制使用 FP32；不要给训练命令重新添加 `--use-amp`，原 FP16 路径会使 GradScaler 下溢并产生 NaN 权重。

## 方案

- 主检测器：ICLR 2025 D-FINE-L，加载 Objects365 + COCO 官方权重。
- 主模态：RGB，完整保留预训练检测器。
- 辅助模态：IR；经过低频结构/高频细节分解后，以零初始化残差注入 RGB。
- 三模态版本：在 RGB-T 最优权重上增加同样零初始化的 Depth 辅助分支。
- 输入分辨率 960，类别均衡重复采样，单模型、单 checkpoint。
- 70 分是打榜目标，不是未训练即可保证的结果；主要增益来自更强检测底座和高 IoU 框回归。

参考仓库：

- D-FINE：https://github.com/Peterande/D-FINE
- Fusion-Mamba：https://github.com/EhanDong/Fusion-Mamba
- WaveMamba（ICCV 2025）：https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_WaveMamba_Wavelet-Driven_Mamba_Fusion_for_RGB-Infrared_Object_Detection_ICCV_2025_paper.html

## 1. 训练完整 RGB-T E10

```bash
./run_e10_dfine_rgbt.sh 2>&1 | tee train_e10_dfine_rgbt.log
```

输出目录：

```text
runs/e10_dfine_l_rgbt
```

查看最后几轮指标：

```bash
tail -n 3 runs/e10_dfine_l_rgbt/log.txt
```

优先使用 `best_stg2.pth`；如果不存在，则使用 `best_stg1.pth`。

## 2. 用全部 2000 张训练数据做最终打榜微调

E10 完成后执行：

```bash
./run_e10f_dfine_rgbt_all.sh 2>&1 | tee train_e10f_dfine_rgbt_all.log
```

该阶段从 E10 最优权重继续训练并加入原验证集，只用于最终提交。

## 3. 生成 RGB-T 最终提交 ZIP

```bash
CKPT=runs/e10f_dfine_l_rgbt_all/best_stg2.pth
[[ -s "$CKPT" ]] || CKPT=runs/e10f_dfine_l_rgbt_all/best_stg1.pth

/root/miniconda3/envs/affect/bin/python predict_dfine.py \
  --config configs/dfine/dfine_l_rgbt_all_aic.yml \
  --checkpoint "$CKPT" \
  --device cuda:0 \
  --confidence 0.01 \
  --output-dir runs/e10f_dfine_l_rgbt_all/submission_txt \
  --zip runs/e10f_dfine_l_rgbt_all/submission.zip
```

最终上传：

```text
runs/e10f_dfine_l_rgbt_all/submission.zip
```

## 4. 可选：加入 Depth，训练 RGB-T-D E11

只有 E10 的严格验证 mAP 明显高于旧 E02 后再执行：

```bash
./run_e11_dfine_rgbtd.sh 2>&1 | tee train_e11_dfine_rgbtd.log
```

生成 E11 提交：

```bash
CKPT=runs/e11_dfine_l_rgbtd/best_stg2.pth
[[ -s "$CKPT" ]] || CKPT=runs/e11_dfine_l_rgbtd/best_stg1.pth

/root/miniconda3/envs/affect/bin/python predict_dfine.py \
  --config configs/dfine/dfine_l_rgbtd_aic.yml \
  --checkpoint "$CKPT" \
  --device cuda:0 \
  --confidence 0.01 \
  --output-dir runs/e11_dfine_l_rgbtd/submission_txt \
  --zip runs/e11_dfine_l_rgbtd/submission.zip
```

E11 仍以 RGB 为主模态；IR 和 Depth 都是可退化为零的辅助残差。

## 5. 当前主实验：E12 RGB-T 小波四尺度融合

在项目根目录执行这一条命令：

```bash
./run_e12_train_and_submit.sh 2>&1 | tee train_e12_wavelet_p2.log
```

脚本会从 E10 最优权重初始化，完成 E12 训练，并自动选择
`best_stg2.pth`、`best_stg1.pth` 或 `last.pth` 中优先级最高的有效权重生成提交包。

最终上传文件：

```text
runs/e12_dfine_l_rgbt_wavelet_p2/submission.zip
```

不要同时启动其他训练任务；脚本使用绝对路径调用 `affect` 环境，不依赖当前 shell 的 conda 状态。

