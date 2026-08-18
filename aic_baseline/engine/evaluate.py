from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from aic_baseline.engine.metrics import (
    compute_metrics,
    match_predictions,
    non_max_suppression,
    save_metrics,
    scale_boxes_to_original,
    targets_for_image,
)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.65,
    max_detections: int = 100,
    output_path: Path | None = None,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    iou_thresholds = torch.linspace(0.5, 0.95, 10, device=device)
    stats = []
    elapsed = 0.0
    images_seen = 0
    for batch_index, (images, targets, metas) in enumerate(tqdm(dataloader, desc="validate", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
        targets = targets.to(device)
        start = time.perf_counter()
        decoded, _ = model(images)
        detections = non_max_suppression(decoded, confidence_threshold, iou_threshold, max_detections)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        for index, (prediction, meta) in enumerate(zip(detections, metas)):
            if len(prediction):
                prediction = prediction.clone()
                prediction[:, :4] = scale_boxes_to_original(prediction[:, :4], meta)
            labels = targets_for_image(targets, index, meta["input_shape"][0], meta)
            correct = match_predictions(prediction, labels, iou_thresholds)
            stats.append(
                (
                    correct.cpu().numpy(),
                    prediction[:, 4].detach().cpu().numpy(),
                    prediction[:, 5].detach().cpu().numpy(),
                    labels[:, 0].detach().cpu().numpy(),
                )
            )
            images_seen += 1
    metrics = compute_metrics(stats)
    metrics["images"] = images_seen
    metrics["inference_ms_per_image"] = 1000 * elapsed / max(images_seen, 1)
    if output_path is not None:
        save_metrics(metrics, output_path)
    return metrics
