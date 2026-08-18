from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aic_baseline.models.blocks import Conv


class LearnableCoefficient(nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class CrossAttention(nn.Module):
    """Dual cross-attention adapted from the official ICAFusion implementation."""

    def __init__(self, channels: int, heads: int = 8, attention_dropout: float = 0.1, residual_dropout: float = 0.1):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_ir = nn.LayerNorm(channels)
        self.q_rgb = nn.Linear(channels, channels)
        self.k_rgb = nn.Linear(channels, channels)
        self.v_rgb = nn.Linear(channels, channels)
        self.q_ir = nn.Linear(channels, channels)
        self.k_ir = nn.Linear(channels, channels)
        self.v_ir = nn.Linear(channels, channels)
        self.out_rgb = nn.Linear(channels, channels)
        self.out_ir = nn.Linear(channels, channels)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _split(self, x: torch.Tensor, transpose_key: bool = False) -> torch.Tensor:
        batch, tokens, _ = x.shape
        result = x.view(batch, tokens, self.heads, self.head_dim)
        return result.permute(0, 2, 3, 1) if transpose_key else result.permute(0, 2, 1, 3)

    def forward(self, rgb: torch.Tensor, infrared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = self.norm_rgb(rgb)
        infrared = self.norm_ir(infrared)
        q_rgb, k_rgb, v_rgb = self._split(self.q_rgb(rgb)), self._split(self.k_rgb(rgb), True), self._split(self.v_rgb(rgb))
        q_ir, k_ir, v_ir = self._split(self.q_ir(infrared)), self._split(self.k_ir(infrared), True), self._split(self.v_ir(infrared))
        scale = math.sqrt(self.head_dim)
        attention_rgb = self.attention_dropout(torch.softmax(torch.matmul(q_ir, k_rgb) / scale, dim=-1))
        attention_ir = self.attention_dropout(torch.softmax(torch.matmul(q_rgb, k_ir) / scale, dim=-1))
        batch, _, tokens, _ = attention_rgb.shape
        output_rgb = torch.matmul(attention_rgb, v_rgb).permute(0, 2, 1, 3).reshape(batch, tokens, self.channels)
        output_ir = torch.matmul(attention_ir, v_ir).permute(0, 2, 1, 3).reshape(batch, tokens, self.channels)
        return self.residual_dropout(self.out_rgb(output_rgb)), self.residual_dropout(self.out_ir(output_ir))


class IterativeCrossTransformer(nn.Module):
    def __init__(self, channels: int, heads: int, expansion: int, loops: int, dropout: float):
        super().__init__()
        self.loops = loops
        self.attention = CrossAttention(channels, heads, dropout, dropout)
        self.norm = nn.LayerNorm(channels)
        self.mlp_rgb = nn.Sequential(nn.Linear(channels, expansion * channels), nn.GELU(), nn.Linear(expansion * channels, channels), nn.Dropout(dropout))
        self.mlp_ir = nn.Sequential(nn.Linear(channels, expansion * channels), nn.GELU(), nn.Linear(expansion * channels, channels), nn.Dropout(dropout))
        self.coefficients = nn.ModuleList(LearnableCoefficient() for _ in range(8))

    def forward(self, rgb: torch.Tensor, infrared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.coefficients
        for _ in range(self.loops):
            rgb_cross, ir_cross = self.attention(rgb, infrared)
            rgb_attention = c[0](rgb) + c[1](rgb_cross)
            ir_attention = c[2](infrared) + c[3](ir_cross)
            rgb = c[4](rgb_attention) + c[5](self.mlp_rgb(self.norm(rgb_attention)))
            infrared = c[6](ir_attention) + c[7](self.mlp_ir(self.norm(ir_attention)))
        return rgb, infrared


class ICAFusionBlock(nn.Module):
    """ICAFusion DMFF block with fixed-size pooled tokens and iterative parameter sharing.

    The implementation is adapted from ``chanchanchan97/ICAFusion`` at commit
    ``6c06b831c70f484b7a434d51cd71e2a2586d1998``.
    """

    def __init__(self, channels: int, anchors: int, heads: int = 8, loops: int = 1, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.anchors = anchors
        self.position_rgb = nn.Parameter(torch.zeros(1, anchors * anchors, channels))
        self.position_ir = nn.Parameter(torch.zeros(1, anchors * anchors, channels))
        self.pool_rgb = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.pool_ir = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.transformer = IterativeCrossTransformer(channels, heads, 4, loops, dropout)
        self.output = Conv(channels * 2, channels, 1, 1)

    def _pool(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, (self.anchors, self.anchors))
        maximum = F.adaptive_max_pool2d(x, (self.anchors, self.anchors))
        return avg * weights[0] + maximum * weights[1]

    def forward(self, rgb: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = rgb.shape
        rgb_tokens = self._pool(rgb, self.pool_rgb).flatten(2).transpose(1, 2) + self.position_rgb
        ir_tokens = self._pool(infrared, self.pool_ir).flatten(2).transpose(1, 2) + self.position_ir
        rgb_tokens, ir_tokens = self.transformer(rgb_tokens, ir_tokens)
        rgb_cross = rgb_tokens.transpose(1, 2).reshape(batch, channels, self.anchors, self.anchors)
        ir_cross = ir_tokens.transpose(1, 2).reshape(batch, channels, self.anchors, self.anchors)
        rgb_cross = F.interpolate(rgb_cross, size=(height, width), mode="bilinear", align_corners=False) + rgb
        ir_cross = F.interpolate(ir_cross, size=(height, width), mode="bilinear", align_corners=False) + infrared
        return self.output(torch.cat((rgb_cross, ir_cross), dim=1))


class DepthGate(nn.Module):
    def __init__(self, channels: int, initial_bias: float = -2.0):
        super().__init__()
        self.projection = Conv(channels, channels, 1, 1)
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, initial_bias)

    def forward(self, main: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        depth = self.projection(depth)
        gate = torch.sigmoid(self.gate(torch.cat((main, depth), dim=1)))
        return main + gate * depth



class LocalIlluminationEstimator(nn.Module):
    """Predict the P3-scale RGB illumination map used by M2D-LIF."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Conv(3, 32, 3, 1),
            nn.AvgPool2d(2, 2),
            Conv(32, 64, 3, 1),
            nn.AvgPool2d(2, 2),
            Conv(64, 64, 3, 1),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.layers(rgb)


class LocalIlluminationFusion(nn.Module):
    """Spatially weight RGB and IR features following the official LIF rule."""

    def __init__(self, beta: float = 0.4):
        super().__init__()
        self.beta = beta

    def forward(self, rgb: torch.Tensor, infrared: torch.Tensor, illumination: torch.Tensor) -> torch.Tensor:
        weight = F.interpolate(illumination, size=rgb.shape[-2:], mode="area")
        weight = self.beta * torch.clamp((weight - 0.31) / 0.63, max=0.5) + 0.5
        return weight * rgb + (1.0 - weight) * infrared
