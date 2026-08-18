from __future__ import annotations

import torch
import torch.nn as nn


FeaturePyramid = tuple[torch.Tensor, ...]


def _normalize_feature(feature: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Parameter-free equivalent of the non-affine BN used by M2D-LIF."""
    feature = feature.float()
    dimensions = (0, 2, 3)
    mean = feature.mean(dim=dimensions, keepdim=True)
    variance = feature.var(dim=dimensions, unbiased=False, keepdim=True)
    return (feature - mean) * torch.rsqrt(variance + eps)


class ChannelWiseDistillationLoss(nn.Module):
    """Spatial channel-wise KL divergence (CWD)."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, student: FeaturePyramid, teacher: FeaturePyramid) -> torch.Tensor:
        if len(student) != len(teacher):
            raise ValueError("Student and teacher feature pyramids must have equal length")
        losses = []
        for student_feature, teacher_feature in zip(student, teacher):
            if student_feature.shape != teacher_feature.shape:
                raise ValueError(f"Feature shape mismatch: {student_feature.shape} != {teacher_feature.shape}")
            student_feature = _normalize_feature(student_feature)
            teacher_feature = _normalize_feature(teacher_feature.detach())
            spatial_size = student_feature.shape[-2] * student_feature.shape[-1]
            student_logits = student_feature.flatten(2).reshape(-1, spatial_size) / self.temperature
            teacher_logits = teacher_feature.flatten(2).reshape(-1, spatial_size) / self.temperature
            teacher_probability = teacher_logits.softmax(dim=1)
            loss = torch.sum(
                teacher_probability * (teacher_logits.log_softmax(dim=1) - student_logits.log_softmax(dim=1))
            )
            losses.append(loss * self.temperature**2 / (student_feature.shape[0] * student_feature.shape[1]))
        return torch.stack(losses).sum()


class CrossAttentionDistillationLoss(nn.Module):
    """Teacher-attention-weighted Pearson loss (CAD) from M2D-LIF."""

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)

    def forward(self, student: FeaturePyramid, teacher: FeaturePyramid) -> torch.Tensor:
        if len(student) != len(teacher):
            raise ValueError("Student and teacher feature pyramids must have equal length")
        losses = []
        for student_feature, teacher_feature in zip(student, teacher):
            if student_feature.shape != teacher_feature.shape:
                raise ValueError(f"Feature shape mismatch: {student_feature.shape} != {teacher_feature.shape}")
            student_feature = _normalize_feature(student_feature)
            teacher_feature = _normalize_feature(teacher_feature.detach())
            centered = teacher_feature - teacher_feature.mean(dim=(2, 3), keepdim=True)
            sample_count = max(teacher_feature.shape[-2] * teacher_feature.shape[-1] - 1, 1)
            attention = torch.sigmoid(
                centered.square()
                / (4 * (centered.square().sum(dim=(2, 3), keepdim=True) / sample_count + self.eps))
                + 0.5
            )
            student_flat = (student_feature * attention).flatten(2)
            teacher_flat = (teacher_feature * attention).flatten(2)
            student_flat = student_flat - student_flat.mean(dim=-1, keepdim=True)
            teacher_flat = teacher_flat - teacher_flat.mean(dim=-1, keepdim=True)
            numerator = (student_flat * teacher_flat).sum(dim=-1)
            denominator = torch.sqrt(
                student_flat.square().sum(dim=-1) * teacher_flat.square().sum(dim=-1) + self.eps
            )
            losses.append(1.0 - (numerator / denominator).mean())
        return torch.stack(losses).sum()


class M2DDistillationLoss(nn.Module):
    """Dual-teacher intra-modal CWD plus cross-modal CAD."""

    def __init__(self, temperature: float = 1.0, normal: bool = True, cross: bool = True):
        super().__init__()
        self.normal = bool(normal)
        self.cross = bool(cross)
        self.cwd = ChannelWiseDistillationLoss(temperature)
        self.cad = CrossAttentionDistillationLoss()

    def forward(
        self,
        student_rgb: FeaturePyramid,
        student_ir: FeaturePyramid,
        teacher_rgb: FeaturePyramid,
        teacher_ir: FeaturePyramid,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = student_rgb[0].new_zeros((), dtype=torch.float32)
        intra = zero
        cross = zero
        if self.normal:
            intra = self.cwd(student_rgb, teacher_rgb) + self.cwd(student_ir, teacher_ir)
        if self.cross:
            cross = self.cad(student_rgb, teacher_ir) + self.cad(student_ir, teacher_rgb)
        return intra, cross
