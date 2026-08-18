from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aic_baseline.constants import DEFAULT_ANCHORS, NUM_CLASSES
from aic_baseline.models.blocks import CSPBackbone, PANNeck
from aic_baseline.models.fusion import (
    DepthGate,
    ICAFusionBlock,
    LocalIlluminationEstimator,
    LocalIlluminationFusion,
)


class Detect(nn.Module):
    def __init__(self, num_classes: int, channels: tuple[int, int, int], anchors=DEFAULT_ANCHORS):
        super().__init__()
        self.nc = num_classes
        self.no = num_classes + 5
        self.nl = 3
        self.na = len(anchors[0])
        self.register_buffer("anchors_pixel", torch.tensor(anchors, dtype=torch.float32))
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))
        self.m = nn.ModuleList(nn.Conv2d(channel, self.na * self.no, 1) for channel in channels)
        self.grid: list[torch.Tensor | None] = [None] * self.nl
        self._initialize_biases()

    @property
    def anchors(self) -> torch.Tensor:
        return self.anchors_pixel / self.stride.view(-1, 1, 1)

    def _initialize_biases(self) -> None:
        for conv, stride in zip(self.m, self.stride):
            bias = conv.bias.detach().view(self.na, -1)
            bias[:, 4] += math.log(8 / (640 / float(stride)) ** 2)
            bias[:, 5:] += math.log(0.6 / (self.nc - 0.99))
            conv.bias = nn.Parameter(bias.view(-1), requires_grad=True)

    @staticmethod
    def _make_grid(nx: int, ny: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y, x = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
        return torch.stack((x, y), dim=2).view(1, 1, ny, nx, 2).to(dtype)

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        raw = []
        decoded = []
        for index, (feature, conv) in enumerate(zip(features, self.m)):
            output = conv(feature)
            batch, _, ny, nx = output.shape
            output = output.view(batch, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            raw.append(output)
            if not self.training:
                grid = self._make_grid(nx, ny, output.device, output.dtype)
                prediction = output.sigmoid()
                xy = (prediction[..., 0:2] * 2 - 0.5 + grid) * self.stride[index]
                wh = (prediction[..., 2:4] * 2) ** 2 * self.anchors_pixel[index].view(1, self.na, 1, 1, 2)
                decoded.append(torch.cat((xy, wh, prediction[..., 4:]), dim=-1).view(batch, -1, self.no))
        return raw if self.training else (torch.cat(decoded, dim=1), raw)


class MultiModalYOLO(nn.Module):
    def __init__(
        self,
        input_mode: str = "rgbtd",
        num_classes: int = NUM_CLASSES,
        width: float = 0.5,
        depth: float = 0.33,
        fusion_heads: int = 8,
        fusion_loops: int = 1,
        fusion_dropout: float = 0.1,
        fusion_type: str = "ica",
    ) -> None:
        super().__init__()
        self.input_mode = input_mode.lower()
        if self.input_mode not in {"rgb", "ir", "rgbt", "rgbd", "rgbtd"}:
            raise ValueError(f"Unsupported input mode: {input_mode}")
        self.fusion_type = fusion_type.lower()
        if self.fusion_type not in {"ica", "lif"}:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        if self.fusion_type == "lif" and "t" not in self.input_mode:
            raise ValueError("LIF fusion requires an RGB+IR input mode")

        self.nc = num_classes
        self.rgb_backbone = CSPBackbone(2 if self.input_mode == "ir" else 3, width, depth)
        channels = self.rgb_backbone.out_channels
        self.ir_backbone = CSPBackbone(2, width, depth) if "t" in self.input_mode else None
        self.depth_backbone = CSPBackbone(3, width, depth) if "d" in self.input_mode else None
        anchor_sizes = (20, 16, 10)
        self.rgbt_fusion = nn.ModuleList(
            ICAFusionBlock(channel, anchors, fusion_heads, fusion_loops, fusion_dropout)
            for channel, anchors in zip(channels, anchor_sizes)
        ) if "t" in self.input_mode and self.fusion_type == "ica" else None
        self.illumination_estimator = LocalIlluminationEstimator() if self.fusion_type == "lif" else None
        self.lif_fusion = nn.ModuleList(LocalIlluminationFusion() for _ in channels) if self.fusion_type == "lif" else None
        self.depth_gates = nn.ModuleList(DepthGate(channel) for channel in channels) if "d" in self.input_mode else None
        self.neck = PANNeck(channels, depth)
        self.detect = Detect(num_classes, channels)
        self.gr = 1.0
        self.hyp: dict[str, Any] = {}

    @property
    def stride(self) -> torch.Tensor:
        return self.detect.stride

    def extract_modal_features(self, images: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, ...]]:
        """Return pre-fusion backbone features for mono- and cross-modal distillation."""
        modal_features = {}
        if self.input_mode == "ir":
            modal_features["ir"] = self.rgb_backbone(images["infrared"])
            return modal_features
        modal_features["rgb"] = self.rgb_backbone(images["rgb"])
        if self.ir_backbone is not None:
            modal_features["ir"] = self.ir_backbone(images["infrared"])
        if self.depth_backbone is not None:
            modal_features["depth"] = self.depth_backbone(images["depth"])
        return modal_features

    def forward(self, images: dict[str, torch.Tensor], return_aux: bool = False):
        modal_features = self.extract_modal_features(images)
        illumination = None
        if self.input_mode == "ir":
            features = modal_features["ir"]
        else:
            features = modal_features["rgb"]
        if self.ir_backbone is not None:
            if self.fusion_type == "lif":
                illumination = self.illumination_estimator(images["rgb"])
                features = tuple(
                    block(rgb, ir, illumination)
                    for block, rgb, ir in zip(self.lif_fusion, features, modal_features["ir"])
                )
            else:
                features = tuple(
                    block(rgb, ir) for block, rgb, ir in zip(self.rgbt_fusion, features, modal_features["ir"])
                )
        if self.depth_backbone is not None:
            features = tuple(
                gate(main, depth) for gate, main, depth in zip(self.depth_gates, features, modal_features["depth"])
            )
        predictions = self.detect(self.neck(features))
        if return_aux:
            return predictions, {"modal_features": modal_features, "illumination": illumination}
        return predictions

    def initialize_auxiliary_backbones(self, branches: tuple[str, ...] = ("ir", "depth")) -> None:
        source = self.rgb_backbone.state_dict()
        candidates = (("ir", self.ir_backbone), ("depth", self.depth_backbone))
        for name, backbone in candidates:
            if name not in branches:
                continue
            if backbone is None:
                continue
            target = backbone.state_dict()
            copied = {}
            for key, value in target.items():
                source_value = source.get(key)
                if source_value is None:
                    continue
                if source_value.shape == value.shape:
                    copied[key] = source_value
                elif key == "layers.0.conv.weight" and source_value.ndim == 4:
                    mean = source_value.mean(dim=1, keepdim=True)
                    copied[key] = mean.repeat(1, value.shape[1], 1, 1) * (source_value.shape[1] / value.shape[1])
            backbone.load_state_dict(copied, strict=False)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]
        if isinstance(model, dict):
            return model
        if hasattr(model, "float"):
            return model.float().state_dict()
    if isinstance(checkpoint, dict) and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def _map_yolov5_state(source: dict[str, torch.Tensor], model: MultiModalYOLO) -> dict[str, torch.Tensor]:
    mapping = {index: f"rgb_backbone.layers.{index}." for index in range(10)}
    mapping.update(
        {
            10: "neck.reduce_p5.",
            13: "neck.c3_p4.",
            14: "neck.reduce_p4.",
            17: "neck.c3_p3.",
            18: "neck.down_p3.",
            20: "neck.c3_n4.",
            21: "neck.down_p4.",
            23: "neck.c3_n5.",
        }
    )
    target_state = model.state_dict()
    converted = {}
    for key, value in source.items():
        if key.startswith("module."):
            key = key[7:]
        pieces = key.split(".")
        if len(pieces) < 3 or pieces[0] != "model" or not pieces[1].isdigit():
            continue
        layer = int(pieces[1])
        suffix = ".".join(pieces[2:])
        target_key = None
        if layer in mapping:
            target_key = mapping[layer] + suffix
        elif layer == 24 and suffix.startswith("m."):
            target_key = "detect." + suffix
        if target_key not in target_state:
            continue
        if target_state[target_key].shape == value.shape:
            converted[target_key] = value
        elif target_key == "rgb_backbone.layers.0.conv.weight" and value.ndim == 4:
            target_channels = target_state[target_key].shape[1]
            mean = value.mean(dim=1, keepdim=True)
            converted[target_key] = (
                mean.repeat(1, target_channels, 1, 1) * (value.shape[1] / target_channels)
            )
    return converted


def load_model_weights(model: MultiModalYOLO, path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    repository_root = Path(__file__).resolve().parents[2]
    candidates = (
        repository_root / "third_party" / "yolov5-v7.0",
        repository_root / "third_party" / "ICAFusion",
    )
    for third_party in reversed(candidates):
        if third_party.exists() and str(third_party) not in sys.path:
            sys.path.insert(0, str(third_party))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source = _extract_state_dict(checkpoint)
    target_state = model.state_dict()
    direct = {}
    for key, value in source.items():
        target_key = key.removeprefix("module.")
        if target_key not in target_state:
            continue
        if value.shape == target_state[target_key].shape:
            direct[target_key] = value
        elif target_key == "rgb_backbone.layers.0.conv.weight" and value.ndim == 4:
            target_channels = target_state[target_key].shape[1]
            mean = value.mean(dim=1, keepdim=True)
            direct[target_key] = mean.repeat(1, target_channels, 1, 1) * (value.shape[1] / target_channels)
    state = direct if direct else _map_yolov5_state(source, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    branches = []
    if model.ir_backbone is not None and not any(key.startswith("ir_backbone.") for key in state):
        branches.append("ir")
    if model.depth_backbone is not None and not any(key.startswith("depth_backbone.") for key in state):
        branches.append("depth")
    model.initialize_auxiliary_backbones(tuple(branches))
    return {
        "loaded": len(state),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "checkpoint": checkpoint if isinstance(checkpoint, dict) else {},
    }
