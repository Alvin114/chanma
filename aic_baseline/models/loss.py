from __future__ import annotations

import math

import torch
import torch.nn as nn


def smooth_bce(epsilon: float = 0.0) -> tuple[float, float]:
    return 1.0 - 0.5 * epsilon, 0.5 * epsilon


class FocalLoss(nn.Module):
    def __init__(self, base: nn.Module, gamma: float = 1.5, alpha: float = 0.25):
        super().__init__()
        self.base = base
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = base.reduction
        self.base.reduction = "none"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base(prediction, target)
        probability = prediction.sigmoid()
        p_t = target * probability + (1 - target) * (1 - probability)
        alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        loss *= alpha_factor * (1 - p_t) ** self.gamma
        return loss.mean() if self.reduction == "mean" else loss.sum()


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, ciou: bool = False, epsilon: float = 1e-7) -> torch.Tensor:
    # Inputs are transposed xywh: [4, n].
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    intersection = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(0) * (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(0)
    width1, height1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + epsilon
    width2, height2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + epsilon
    union = width1 * height1 + width2 * height2 - intersection + epsilon
    iou = intersection / union
    if not ciou:
        return iou
    convex_width = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    convex_height = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    center_distance = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    diagonal = convex_width**2 + convex_height**2 + epsilon
    aspect = (4 / math.pi**2) * torch.pow(torch.atan(width2 / height2) - torch.atan(width1 / height1), 2)
    with torch.no_grad():
        alpha = aspect / (aspect - iou + 1 + epsilon)
    return iou - center_distance / diagonal - aspect * alpha


class YoloLoss:
    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.device = next(model.parameters()).device
        self.hyp = config
        cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.get("cls_pw", 1.0)], device=self.device))
        obj_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.get("obj_pw", 1.0)], device=self.device))
        gamma = float(config.get("focal_gamma", 0.0))
        if gamma > 0:
            cls_loss, obj_loss = FocalLoss(cls_loss, gamma), FocalLoss(obj_loss, gamma)
        self.bce_cls = cls_loss
        self.bce_obj = obj_loss
        self.positive, self.negative = smooth_bce(float(config.get("label_smoothing", 0.0)))
        self.balance = [4.0, 1.0, 0.4]
        self.detect = model.detect
        self.num_classes = self.detect.nc

    def __call__(self, predictions: list[torch.Tensor], targets: torch.Tensor):
        box_loss = torch.zeros(1, device=self.device)
        object_loss = torch.zeros(1, device=self.device)
        class_loss = torch.zeros(1, device=self.device)
        classes, boxes, indices, anchors = self.build_targets(predictions, targets)
        for layer_index, prediction in enumerate(predictions):
            batch, anchor_index, grid_y, grid_x = indices[layer_index]
            object_target = torch.zeros_like(prediction[..., 0], device=self.device)
            count = batch.shape[0]
            if count:
                selected = prediction[batch, anchor_index, grid_y, grid_x]
                predicted_xy = selected[:, :2].sigmoid() * 2 - 0.5
                predicted_wh = (selected[:, 2:4].sigmoid() * 2) ** 2 * anchors[layer_index]
                predicted_box = torch.cat((predicted_xy, predicted_wh), dim=1)
                iou = bbox_iou(predicted_box.T, boxes[layer_index], ciou=True)
                box_loss += (1 - iou).mean()
                score = iou.detach().clamp(0).to(object_target.dtype)
                order = torch.argsort(score)
                batch, anchor_index, grid_y, grid_x, score = batch[order], anchor_index[order], grid_y[order], grid_x[order], score[order]
                object_target[batch, anchor_index, grid_y, grid_x] = score
                if self.num_classes > 1:
                    target_classes = torch.full_like(selected[:, 5:], self.negative)
                    target_classes[range(count), classes[layer_index]] = self.positive
                    class_loss += self.bce_cls(selected[:, 5:], target_classes)
            object_loss += self.bce_obj(prediction[..., 4], object_target) * self.balance[layer_index]

        box_loss *= float(self.hyp.get("box", 0.05))
        object_loss *= float(self.hyp.get("obj", 1.0))
        class_loss *= float(self.hyp.get("cls", 0.5)) * self.num_classes / 80.0
        batch_size = predictions[0].shape[0]
        total = (box_loss + object_loss + class_loss) * batch_size
        return total, torch.cat((box_loss, object_loss, class_loss)).detach()

    def build_targets(self, predictions: list[torch.Tensor], targets: torch.Tensor):
        num_anchors = self.detect.na
        num_targets = targets.shape[0]
        target_classes, target_boxes, indices, matched_anchors = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        anchor_indices = torch.arange(num_anchors, device=targets.device).float().view(num_anchors, 1).repeat(1, num_targets)
        expanded = torch.cat((targets.repeat(num_anchors, 1, 1), anchor_indices[:, :, None]), dim=2)
        offset_vectors = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * 0.5

        for layer_index, prediction in enumerate(predictions):
            anchors = self.detect.anchors[layer_index]
            gain[2:6] = torch.tensor(prediction.shape, device=targets.device)[[3, 2, 3, 2]]
            current = expanded * gain
            if num_targets:
                ratio = current[:, :, 4:6] / anchors[:, None]
                keep = torch.maximum(ratio, 1 / ratio).amax(dim=2) < float(self.hyp.get("anchor_t", 4.0))
                current = current[keep]
                grid_xy = current[:, 2:4]
                inverse = gain[[2, 3]] - grid_xy
                left, top = ((grid_xy % 1 < 0.5) & (grid_xy > 1)).T
                right, bottom = ((inverse % 1 < 0.5) & (inverse > 1)).T
                mask = torch.stack((torch.ones_like(left), left, top, right, bottom))
                current = current.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(grid_xy)[None] + offset_vectors[:, None])[mask]
            else:
                current = expanded[0]
                offsets = 0
            batch, classes = current[:, :2].long().T
            grid_xy, grid_wh = current[:, 2:4], current[:, 4:6]
            grid_ij = (grid_xy - offsets).long()
            grid_x, grid_y = grid_ij.T
            anchor_index = current[:, 6].long()
            indices.append((batch, anchor_index, grid_y.clamp_(0, prediction.shape[2] - 1), grid_x.clamp_(0, prediction.shape[3] - 1)))
            target_boxes.append(torch.cat((grid_xy - grid_ij, grid_wh), dim=1))
            matched_anchors.append(anchors[anchor_index])
            target_classes.append(classes)
        return target_classes, target_boxes, indices, matched_anchors

