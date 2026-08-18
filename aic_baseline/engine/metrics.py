from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import torch

try:
    from torchvision.ops import nms as torchvision_nms
except Exception:  # pragma: no cover - fallback for minimal server images
    torchvision_nms = None

from aic_baseline.constants import CLASS_NAMES, NUM_CLASSES


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    result = boxes.clone()
    result[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    result[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    result[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    result[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return result


def box_iou(box1: torch.Tensor, box2: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)
    intersection = (
        (torch.minimum(box1[:, None, 2:], box2[:, 2:]) - torch.maximum(box1[:, None, :2], box2[:, :2]))
        .clamp(0)
        .prod(2)
    )
    return intersection / (area1[:, None] + area2 - intersection + epsilon)


def _fallback_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current].view(1, 4), boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep) if keep else torch.zeros(0, dtype=torch.long, device=boxes.device)


def non_max_suppression(
    prediction: torch.Tensor,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    max_detections: int = 100,
    max_candidates: int = 30000,
) -> list[torch.Tensor]:
    outputs = []
    for image_prediction in prediction:
        image_prediction = image_prediction[image_prediction[:, 4] > confidence_threshold]
        if not len(image_prediction):
            outputs.append(torch.zeros((0, 6), device=prediction.device))
            continue
        boxes = xywh_to_xyxy(image_prediction[:, :4])
        class_scores = image_prediction[:, 5:] * image_prediction[:, 4:5]
        scores, classes = class_scores.max(dim=1)
        keep = scores > confidence_threshold
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep].float()
        if len(scores) > max_candidates:
            top = scores.argsort(descending=True)[:max_candidates]
            boxes, scores, classes = boxes[top], scores[top], classes[top]
        offsets = classes.view(-1, 1) * 4096
        nms_function = torchvision_nms or _fallback_nms
        selected = nms_function(boxes + offsets, scores, iou_threshold)[:max_detections]
        outputs.append(torch.cat((boxes[selected], scores[selected, None], classes[selected, None]), dim=1))
    return outputs


def scale_boxes_to_original(boxes: torch.Tensor, meta: dict) -> torch.Tensor:
    boxes = boxes.clone()
    pad_x, pad_y = meta["pad"]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / meta["ratio"]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / meta["ratio"]
    height, width = meta["original_shape"]
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)
    return boxes


def targets_for_image(targets: torch.Tensor, batch_index: int, image_size: int, meta: dict) -> torch.Tensor:
    selected = targets[targets[:, 0] == batch_index, 1:]
    if not len(selected):
        return torch.zeros((0, 5), device=targets.device)
    boxes = xywh_to_xyxy(selected[:, 1:5] * image_size)
    boxes = scale_boxes_to_original(boxes, meta)
    return torch.cat((selected[:, 0:1], boxes), dim=1)


def match_predictions(predictions: torch.Tensor, labels: torch.Tensor, iou_thresholds: torch.Tensor) -> torch.Tensor:
    correct = torch.zeros((len(predictions), len(iou_thresholds)), dtype=torch.bool, device=predictions.device)
    if not len(predictions) or not len(labels):
        return correct
    iou = box_iou(labels[:, 1:], predictions[:, :4])
    class_match = labels[:, 0:1] == predictions[:, 5]
    label_index, prediction_index = torch.where((iou >= iou_thresholds[0]) & class_match)
    if not len(label_index):
        return correct
    matches = torch.stack((label_index, prediction_index, iou[label_index, prediction_index]), dim=1).cpu().numpy()
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    matches = matches[matches[:, 2].argsort()[::-1]]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    prediction_indices = torch.tensor(matches[:, 1], dtype=torch.long, device=predictions.device)
    match_ious = torch.tensor(matches[:, 2], device=predictions.device)
    correct[prediction_indices] = match_ious[:, None] >= iou_thresholds[None]
    return correct


def exact_101_point_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    values = []
    for threshold in np.linspace(0, 1, 101):
        eligible = envelope[recall >= threshold]
        values.append(float(eligible.max()) if eligible.size else 0.0)
    return float(np.mean(values))


def compute_metrics(stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    thresholds = np.linspace(0.5, 0.95, 10)
    if stats:
        correct = np.concatenate([item[0] for item in stats], axis=0)
        confidence = np.concatenate([item[1] for item in stats], axis=0)
        predicted_class = np.concatenate([item[2] for item in stats], axis=0)
        target_class = np.concatenate([item[3] for item in stats], axis=0)
    else:
        correct = np.zeros((0, 10), dtype=bool)
        confidence = predicted_class = target_class = np.zeros((0,))
    order = np.argsort(-confidence)
    correct, confidence, predicted_class = correct[order], confidence[order], predicted_class[order]
    ap = np.zeros((NUM_CLASSES, 10), dtype=np.float64)
    precision_at_50 = np.zeros(NUM_CLASSES)
    recall_at_50 = np.zeros(NUM_CLASSES)
    target_counts = np.bincount(target_class.astype(np.int64), minlength=NUM_CLASSES)
    prediction_counts = np.bincount(predicted_class.astype(np.int64), minlength=NUM_CLASSES)
    for class_id in range(NUM_CLASSES):
        mask = predicted_class == class_id
        num_targets = target_counts[class_id]
        if num_targets == 0 or not mask.any():
            continue
        false_positive = (~correct[mask]).cumsum(axis=0)
        true_positive = correct[mask].cumsum(axis=0)
        recalls = true_positive / max(num_targets, 1)
        precisions = true_positive / np.maximum(true_positive + false_positive, 1e-16)
        for index in range(10):
            ap[class_id, index] = exact_101_point_ap(recalls[:, index], precisions[:, index])
        precision_at_50[class_id] = precisions[-1, 0]
        recall_at_50[class_id] = recalls[-1, 0]
    per_class = {
        CLASS_NAMES[index]: {
            "targets": int(target_counts[index]),
            "predictions": int(prediction_counts[index]),
            "precision_50": float(precision_at_50[index]),
            "recall_50": float(recall_at_50[index]),
            "ap50": float(ap[index, 0]),
            "ap75": float(ap[index, 5]),
            "ap50_95": float(ap[index].mean()),
        }
        for index in range(NUM_CLASSES)
    }
    return {
        "map50": float(ap[:, 0].mean()),
        "map75": float(ap[:, 5].mean()),
        "map50_95": float(ap.mean()),
        "ap_by_iou": {f"{threshold:.2f}": float(ap[:, index].mean()) for index, threshold in enumerate(thresholds)},
        "per_class": per_class,
    }


def write_submission(predictions: dict[str, torch.Tensor], output_dir: Path, zip_path: Path | None = None) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, detections in predictions.items():
        lines = []
        for x1, y1, x2, y2, confidence, class_id in detections.tolist():
            width = max(x2 - x1, 0.0)
            height = max(y2 - y1, 0.0)
            # Detections supplied here must already be normalized to original image size.
            lines.append(f"{int(class_id)} {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {width:.8f} {height:.8f} {confidence:.8f}")
        (output_dir / f"{sample_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if zip_path is None:
        return None
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for text_file in sorted(output_dir.glob("*.txt")):
            archive.write(text_file, arcname=text_file.name)
    return zip_path


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

