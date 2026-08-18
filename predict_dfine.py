#!/usr/bin/env python3
"""Generate an AiC submission ZIP from an E10/E11 D-FINE checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from aic_baseline.engine.metrics import write_submission


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    if args.zip.exists():
        raise RuntimeError(f"Submission ZIP already exists: {args.zip}")

    sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "D-FINE"))
    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = "data/prepared/dfine/test.json"
    cfg.yaml_cfg["val_dataloader"]["dataset"]["input_mode"] = cfg.yaml_cfg["HGNetv2"]["input_mode"]

    device = torch.device(args.device)
    model = cfg.model.to(device)
    postprocessor = cfg.postprocessor.to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "ema" in state and "module" in state["ema"]:
        model_state = state["ema"]["module"]
    elif "model" in state:
        model_state = state["model"]
    else:
        raise KeyError("Checkpoint contains neither ema.module nor model")
    model.load_state_dict(model_state, strict=True)
    model.eval()
    postprocessor.eval()

    loader = cfg.val_dataloader
    predictions = {}
    try:
        for samples, targets in tqdm(loader, desc="predict-dfine"):
            samples = samples.to(device, non_blocking=True)
            outputs = model(samples)
            sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = postprocessor(outputs, sizes)
            for target, result in zip(targets, results):
                width, height = target["orig_size"].tolist()
                keep = result["scores"] >= args.confidence
                boxes = result["boxes"][keep].clone()
                scores = result["scores"][keep]
                labels = result["labels"][keep]
                # Advanced indexing returns a copy in PyTorch, so .div_() on
                # boxes[:, [0, 2]] does not update boxes. Assign the scaled
                # coordinates back explicitly before clipping.
                boxes[:, [0, 2]] = boxes[:, [0, 2]] / float(width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]] / float(height)
                boxes.clamp_(0, 1)
                valid = (
                    torch.isfinite(boxes).all(dim=1)
                    & torch.isfinite(scores)
                    & (boxes[:, 2] > boxes[:, 0])
                    & (boxes[:, 3] > boxes[:, 1])
                )
                boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
                detections = torch.cat(
                    (boxes, scores[:, None], labels[:, None].to(boxes.dtype)), dim=1
                )
                predictions[target["sample_id"]] = detections.cpu()
    finally:
        if hasattr(loader.dataset, "close"):
            loader.dataset.close()

    if len(predictions) != 1000:
        raise RuntimeError(f"Expected 1000 predictions, got {len(predictions)}")

    total_boxes = sum(len(detections) for detections in predictions.values())
    degenerate_boxes = sum(
        int(
            (
                (detections[:, 2] <= detections[:, 0])
                | (detections[:, 3] <= detections[:, 1])
            ).sum()
        )
        for detections in predictions.values()
        if len(detections)
    )
    if degenerate_boxes:
        raise RuntimeError(
            "Refusing to create submission: "
            f"{degenerate_boxes}/{total_boxes} boxes are degenerate"
        )

    zip_path = write_submission(predictions, args.output_dir, args.zip)
    print(f"created {len(predictions)} TXT files and {zip_path}")


if __name__ == "__main__":
    main()
