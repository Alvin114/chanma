# 面向城市场景的 RGB-Infrared-Depth 三模态目标检测

> 赛题文件：[[赛题：面向城市场景的视觉多模态目标检测-1(1).pdf]]  
> 论文与链接核验日期：2026-07-31

## 一、赛题筛选条件

- 任务：空间对齐的 RGB、热红外和 Depth 三模态二维目标检测。
- 数据：训练集仅 2,000 组三模态图像；测试阶段各 1,000 组。
- 类别：12 类，包括行人、车辆、船、动物、座椅、标识、球、灯、垃圾箱和无人机等。
- 指标：mAP@50-95，对高 IoU 下的定位质量也很敏感。
- 鲁棒性：低照度、逆光、遮挡、低纹理和模态质量下降。
- 约束：只允许官方训练数据；允许 ImageNet、COCO、Objects365 等公开预训练权重；禁止在线 API 和简单多模型集成。

因此优先筛选：

1. 三模态或可扩展到三模态的特征融合；
2. RGB-T 目标检测中的鲁棒融合；
3. RGB-D 的几何引导；
4. 小目标与高质量边界框回归；
5. 适合小数据集、能够利用公开预训练权重的方案。

## 二、S 级：应该优先读和复现

### 1. RGBDT500 / RDTTrack

[Collaborating Vision, Depth, and Thermal Signals for Multi-Modal Tracking: Dataset and Algorithm](https://arxiv.org/abs/2509.24741)

- **匹配度最高**：输入正是空间对齐的 RGB、Depth、Thermal 三模态。
- 可迁移部分：先融合 Depth 与 Thermal，再作为 prompt 注入 RGB 预训练模型；正交投影约束可用于减少辅助模态冗余。
- 局限：原任务是单目标跟踪，不是多类别检测，检测头和损失不能照搬。

### 2. CSSA

[Multimodal Object Detection by Channel Switching and Spatial Attention](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/html/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.html)

- 赛题 PDF 指定参考文献。
- 使用通道切换和无参数空间注意力，计算量低，适合在 YOLO 特征金字塔上先做 RGB-T 双模态基线，再扩展 Depth 分支。
- 局限：论文只验证 RGB-T，需要自行设计三模态切换或门控规则。

### 3. ICAFusion

[ICAFusion: Iterative Cross-Attention Guided Feature Fusion for Multispectral Object Detection](https://arxiv.org/abs/2308.07504) · [代码](https://github.com/chanchanchan97/ICAFusion)

- 双向 cross-attention 直接面向多光谱目标检测，并通过迭代共享参数降低复杂度。
- 有公开代码，适合快速建立强 RGB-T 融合基线，再增加 Depth-to-RGB 或 Depth-to-Fused 交互。
- 局限：仍是双模态；完整三分支 cross-attention 可能在 2,000 组数据上过拟合。

### 4. RGB-T Tiny Object Detection

[Learning Bi-directional Fusion and Deformation-sensitive Loss for RGB-T Tiny Object Detection](https://doi.org/10.1016/j.inffus.2025.103985)

- 赛题 PDF 指定参考文献。
- 双向、多层级融合适合同时处理行人、车辆、标识、球和无人机等尺度差异大的目标。
- DSSM-LOSS 针对小框的位移和形状误差，和 mAP@50-95 的高 IoU 定位要求直接相关。
- 局限：论文为 RGB-T，需要把 Depth 作为几何引导，而不是简单增加第三个对称分支。

### 5. WaveMamba

[WaveMamba: Wavelet-Driven Mamba Fusion for RGB-Infrared Object Detection](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_WaveMamba_Wavelet-Driven_Mamba_Fusion_for_RGB-Infrared_Object_Detection_ICCV_2025_paper.html)

- 分别处理低频结构与高频细节，对低照度 RGB、热红外轮廓和小目标边缘有价值。
- 可以只移植 WMFB 融合块，不必整体更换检测器。
- 局限：实现复杂度高于 CSSA，建议在基线稳定后再尝试。

## 三、A 级：适合借模块或做消融

### 6. Tri-Modal Fusion Transformers

[Tri-Modal Fusion Transformers for UAV-based Object Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Iaboni_Tri-Modal_Fusion_Transformers_for_UAV-based_Object_Detection_CVPR_2026_paper.html)

- 是真正的三模态目标检测，并系统比较融合深度、门控和 token 交换。
- 可借鉴 modality-aware gated exchange 与分层融合位置。
- 局限：第三模态是 Event 而非 Depth，不能直接照搬输入编码器。

### 7. RGB-D Depth-sensitive Attention

[Deep RGB-D Saliency Detection With Depth-Sensitive Attention and Automatic Multi-Modal Fusion](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Deep_RGB-D_Saliency_Detection_With_Depth-Sensitive_Attention_and_Automatic_Multi-Modal_CVPR_2021_paper.html)

- 可借鉴“Depth 引导 RGB 空间注意力”的思路，尤其适合遮挡、低纹理和前后景分离。
- 局限：任务是显著性检测，不是边界框检测；只建议移植注意力模块。

### 8. YOLO

[You Only Look Once: Unified, Real-Time Object Detection](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)

- 赛题 PDF 指定参考文献，适合作为工程基线。
- 建议先跑通 RGB-only、RGB-T、RGB-D 和 RGB-T-D 四个消融，再引入复杂融合模块。

## 四、B 级：了解即可

### 9. EvaNet

[EvaNet: Towards More Efficient and Consistent Infrared and Visible Image Fusion Assessment](https://arxiv.org/abs/2604.02896)

- 赛题 PDF 指定参考文献，但它研究融合图像的质量评估，不直接优化检测 mAP。
- 可用于分析 RGB-T 融合是否保留信息，不建议作为主模型路线。

### 10. AW-MoE

[AW-MoE: All-Weather Mixture of Experts for Robust Multi-Modal 3D Object Detection](https://arxiv.org/abs/2603.16261)

- 天气感知路由和专家选择有启发。
- 但任务是 LiDAR + 4D Radar 的三维检测，与本赛题传感器、标签和检测头差异很大，优先级低。

## 五、现有检索结果中剔除的条目

| 原条目 | 核验结果 |
|---|---|
| Multi-Modal Diffusion for Urban Object Detection and Uncertainty Estimation | 原链接 `arXiv:2412.01234` 实际是自动驾驶端到端规划论文，与该标题不符 |
| ProS: Prompting-to-simulate Generalized Knowledge for Cross-Domain Multimodal Detection | ProS 实际是 Universal Cross-Domain Retrieval，不是多模态检测 |
| Decomposition-Integration Enhancing Multimodal Insights for Object Detection | DIEM 实际是图像问答/多模态推理方法，不是目标检测 |
| RGB-D Co-Attention Network for Pedestrian and Vehicle Detection | 能核验到的是 2020 年 RGB-D Semantic Segmentation 论文，不是所列检测论文 |
| PromptT、Illumination-Aware Multi-Spectral KD、Unified Feature Prompting、Spatially-Disentangled Cross-Modal Fusion、Depth-Guided Spatial Attention、CMFlow | 未找到与原题名、年份和任务同时吻合的可靠论文记录，不进入比赛阅读清单 |

## 六、对这个赛题最实际的实验顺序

1. **数据审计**：检查三模态文件一一对应、框是否一致；统计类别和目标尺寸；单独统计 Depth 的 0 值/无效区域。
2. **四个输入基线**：RGB-only、RGB-T、RGB-D、RGB-T-D，确认每种模态的真实增益。
3. **第一版融合**：YOLO + 三分支浅层编码 + CSSA/轻量门控；公开预训练权重初始化 RGB 主干。
4. **训练鲁棒性**：三模态同步几何增强；对辅助模态做 modality dropout、亮度/噪声和 Depth 无效区模拟。
5. **小目标与定位**：尝试双向多尺度融合及 DSSM 类定位损失，重点看 uav、ball、sign 等类别的 AP。
6. **第二版融合**：基线稳定后，再比较 ICAFusion、RDTTrack 式正交约束或 WaveMamba 模块。

### 必须避免

- 三个模态分别做随机裁剪、翻转或旋转，破坏空间对齐；
- 把 16-bit Depth 当普通 8-bit RGB 图读入；
- 将热红外的三个重复通道误当成彩色语义；
- 为追求分数做多个模型的简单投票或平均，赛题明确禁止；
- 在只有 2,000 组训练样本时一开始就训练大型三分支 Transformer。
