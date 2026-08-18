#!/usr/bin/env python3
"""Build E13 resolution/scene-stratified manifests and image-balanced COCO JSON."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

from aic_baseline.data.manifest import (
    read_jsonl,
    resolution_scene_grouped_split,
    write_jsonl,
)
from prepare_dfine_data import convert, write_json


def image_class_counts(records: list[dict]) -> collections.Counter:
    counts = collections.Counter()
    for record in records:
        counts.update({int(label[0]) for label in record.get("labels", [])})
    return counts


def repeat_records_by_image_frequency(
    records: list[dict], power: float, maximum: int
) -> tuple[list[dict], collections.Counter]:
    """Repeat complete aligned samples using per-class image frequency."""
    if power < 0:
        raise ValueError("repeat power must be non-negative")
    if maximum < 1:
        raise ValueError("maximum repeat must be at least one")
    counts = image_class_counts(records)
    largest = max(counts.values(), default=1)
    output: list[dict] = []
    repeat_histogram = collections.Counter()
    for record in records:
        classes = {int(label[0]) for label in record.get("labels", [])}
        factor = max(
            ((largest / max(counts[class_id], 1)) ** power for class_id in classes),
            default=1.0,
        )
        copies = max(1, min(maximum, int(math.ceil(factor))))
        output.extend([record] * copies)
        repeat_histogram[copies] += 1
    return output, repeat_histogram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, default=Path("data/prepared/manifests")
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("data/prepared/manifests_e13")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/prepared/dfine_e13")
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--repeat-power", type=float, default=0.35)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--tail-class-count", type=int, default=4)
    args = parser.parse_args()

    all_train = read_jsonl(args.source_dir / "all_train.jsonl")
    test = read_jsonl(args.source_dir / "test.jsonl")
    train, val, split_audit = resolution_scene_grouped_split(
        all_train, args.val_fraction, args.seed
    )
    balanced, repeat_histogram = repeat_records_by_image_frequency(
        train, args.repeat_power, args.max_repeat
    )

    counts = image_class_counts(train)
    tail_classes = [
        class_id
        for class_id, _ in sorted(counts.items(), key=lambda item: (item[1], item[0]))[
            : args.tail_class_count
        ]
    ]

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(all_train, args.manifest_dir / "all_train.jsonl")
    write_jsonl(train, args.manifest_dir / "train.jsonl")
    write_jsonl(val, args.manifest_dir / "val.jsonl")
    write_jsonl(test, args.manifest_dir / "test.jsonl")

    write_json(convert(train), args.output_dir / "train.json")
    write_json(convert(balanced), args.output_dir / "train_balanced.json")
    write_json(convert(val), args.output_dir / "val.json")
    write_json(convert(test), args.output_dir / "test.json")
    write_json(convert(all_train), args.output_dir / "all_train.json")

    audit = {
        **split_audit,
        "oversampling": {
            "strategy": "whole_aligned_sample_repeat_by_class_image_frequency",
            "repeat_power": args.repeat_power,
            "max_repeat": args.max_repeat,
            "original_train_samples": len(train),
            "balanced_train_samples": len(balanced),
            "class_image_counts": {str(key): value for key, value in sorted(counts.items())},
            "repeat_histogram": {
                str(key): value for key, value in sorted(repeat_histogram.items())
            },
            "tail_classes": tail_classes,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"tail classes: {tail_classes}")
    print(f"audit: {args.output_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
