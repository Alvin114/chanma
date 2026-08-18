from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from aic_baseline.config import deep_update, parse_overrides
from aic_baseline.engine.checkpoint import load_checkpoint
from aic_baseline.engine.metrics import non_max_suppression, scale_boxes_to_original, write_submission
from aic_baseline.factory import build_dataloader, build_model
from aic_baseline.utils import select_device


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Infer the AiC test set and create a submission ZIP.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/submission/txt"))
    parser.add_argument("--zip", type=Path, default=Path("runs/submission/submission.zip"))
    parser.add_argument("--max-batches", type=int, default=None, help="Debug only; partial output is not a valid submission")
    parser.add_argument("--allow-partial", action="store_true", help="Permit partial output when using --max-batches")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = deep_update(checkpoint["config"], parse_overrides(args.set))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    loader, dataset = build_dataloader(config, "test", augment=False)
    all_predictions = {}
    inference = config.get("inference", {})
    try:
        for batch_index, (images, _, metas) in enumerate(tqdm(loader, desc="predict")):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = {key: value.to(device, non_blocking=True) for key, value in images.items()}
            decoded, _ = model(images)
            detections = non_max_suppression(
                decoded,
                confidence_threshold=float(inference.get("submit_confidence", 0.01)),
                iou_threshold=float(inference.get("nms_iou", 0.65)),
                max_detections=int(inference.get("max_detections", 100)),
            )
            for detection, meta in zip(detections, metas):
                detection = detection.clone()
                if len(detection):
                    detection[:, :4] = scale_boxes_to_original(detection[:, :4], meta)
                    height, width = meta["original_shape"]
                    detection[:, [0, 2]] /= width
                    detection[:, [1, 3]] /= height
                    detection[:, :4].clamp_(0, 1)
                all_predictions[meta["id"]] = detection.cpu()
    finally:
        dataset.close()
    if len(all_predictions) != len(dataset) and not args.allow_partial:
        raise RuntimeError(f"Expected {len(dataset)} predictions, got {len(all_predictions)}")
    zip_path = write_submission(all_predictions, args.output_dir, args.zip)
    print(f"created {len(all_predictions)} TXT files and {zip_path}")


if __name__ == "__main__":
    main()
