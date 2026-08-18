from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aic_baseline.config import deep_update, parse_overrides
from aic_baseline.engine.checkpoint import load_checkpoint
from aic_baseline.engine.evaluate import evaluate
from aic_baseline.factory import build_dataloader, build_model
from aic_baseline.utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an AiC baseline checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/validation.json"))
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only; omit for complete validation")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = deep_update(checkpoint["config"], parse_overrides(args.set))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    loader, dataset = build_dataloader(config, "val", augment=False)
    try:
        metrics = evaluate(
            model,
            loader,
            device,
            confidence_threshold=float(config.get("inference", {}).get("val_confidence", 0.001)),
            iou_threshold=float(config.get("inference", {}).get("nms_iou", 0.65)),
            max_detections=int(config.get("inference", {}).get("max_detections", 100)),
            output_path=args.output,
            max_batches=args.max_batches,
        )
    finally:
        dataset.close()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
