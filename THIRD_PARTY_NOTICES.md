# Third-party notices

## ICAFusion

- Repository: https://github.com/chanchanchan97/ICAFusion
- Pinned commit: `6c06b831c70f484b7a434d51cd71e2a2586d1998`
- License: AGPL-3.0, retained at `third_party/ICAFusion/LICENSE`
- Use in this project: architectural reference and an adapted implementation of the DMFF dual cross-attention/iterative fusion block in `aic_baseline/models/fusion.py`.

The adapted fusion implementation is a derivative of the upstream design. Review AGPL-3.0 obligations before redistribution or competition code submission.

## Ultralytics YOLOv5

- Repository: https://github.com/ultralytics/yolov5
- Release/commit: v7.0 / `915bbf294bb74c859f0b41f1c23bc395014ea679`
- License: GPL-3.0, retained in `third_party/yolov5-v7.0/LICENSE`
- Use in this project: architecture reference and loading/conversion of the official COCO-pretrained YOLOv5s checkpoint.

The project implements its own compact CSP/PAN/Detect training path but follows the YOLOv5 architecture and loss formulation. Review GPL-3.0 obligations before redistribution.


## M²D-LIF

- Repository: https://github.com/Zhao-Tian-yi/M2D-LIF
- Pinned commit: `210c7ca22b7670e7d4629cca995427bf7e74653f`
- Paper: ICCV 2025, “Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning”
- Use in this project: architectural reference for local illumination-aware fusion (LIF), dual mono-modal teacher CWD, and cross-modal CAD losses.

The pinned repository does not contain a standalone top-level license file. Its modified Ultralytics source files carry AGPL-3.0 notices. Confirm redistribution and competition-submission terms with the upstream authors before distributing derived code.
