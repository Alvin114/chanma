from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a trusted YOLOv5 pickle checkpoint to a portable tensor state dict.")
    parser.add_argument("--input", type=Path, default=Path("weights/yolov5s.pt"))
    parser.add_argument("--output", type=Path, default=Path("weights/yolov5s_state.pt"))
    args = parser.parse_args()
    source_tree = Path("third_party/yolov5-v7.0").resolve()
    if not source_tree.exists():
        raise FileNotFoundError("third_party/yolov5-v7.0 is required to load the original checkpoint")
    sys.path.insert(0, str(source_tree))
    # Loading with weights_only=False is intentional and must only be used for the
    # official trusted Ultralytics checkpoint documented in README.md.
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None or not hasattr(model, "state_dict"):
        raise ValueError("Input does not contain a YOLOv5 model")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": state,
            "source": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
            "source_commit": "915bbf294bb74c859f0b41f1c23bc395014ea679",
        },
        args.output,
    )
    print(f"saved {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
