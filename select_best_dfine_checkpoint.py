#!/usr/bin/env python3
"""Select the checkpoint with the highest logged COCO AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in (args.output_dir / "log.txt").read_text().splitlines()
        if line.strip()
    ]
    ap_by_epoch = {
        int(row["epoch"]): float(row["test_coco_eval_bbox"][0]) for row in rows
    }
    candidates = []
    for name in ("best_stg1.pth", "best_stg2.pth", "last.pth"):
        path = args.output_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(state.get("last_epoch", -1))
        candidates.append((ap_by_epoch.get(epoch, float("-inf")), -epoch, path))
    if not candidates:
        raise RuntimeError(f"No usable checkpoint found in {args.output_dir}")
    print(max(candidates, key=lambda item: (item[0], item[1]))[2])


if __name__ == "__main__":
    main()
