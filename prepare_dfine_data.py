#!/usr/bin/env python3
"""Convert AiC JSONL manifests to COCO metadata without extracting the ZIPs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from aic_baseline.constants import CLASS_NAMES


def read_manifest(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def repeat_records(records, power, maximum):
    if power <= 0:
        return records
    counts = Counter(int(label[0]) for row in records for label in row.get("labels", []))
    largest = max(counts.values(), default=1)
    output = []
    for row in records:
        classes = {int(label[0]) for label in row.get("labels", [])}
        factor = max((largest / max(counts[c], 1)) ** power for c in classes) if classes else 1
        copies = max(1, min(int(maximum), int(math.ceil(factor))))
        output.extend([row] * copies)
    return output


def convert(records):
    images, annotations = [], []
    annotation_id = 1
    for image_id, row in enumerate(records, 1):
        visible_meta = row["metadata"]["visible"]
        width, height = int(visible_meta["width"]), int(visible_meta["height"])
        images.append({
            "id": image_id,
            "sample_id": str(row["id"]),
            "file_name": row["members"]["visible"],
            "width": width,
            "height": height,
            "archive": row["archive"],
            "members": row["members"],
            "metadata": row.get("metadata", {}),
        })
        for label in row.get("labels", []):
            category, cx, cy, bw, bh = map(float, label)
            x = max(0.0, (cx - bw / 2.0) * width)
            y = max(0.0, (cy - bh / 2.0) * height)
            box_w = min(width - x, bw * width)
            box_h = min(height - y, bh * height)
            if box_w <= 0 or box_h <= 0:
                continue
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category),
                "bbox": [x, y, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
            })
            annotation_id += 1
    return {
        "info": {"description": "AiC 2026 multimodal detection"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index, "name": name, "supercategory": "object"}
                       for index, name in enumerate(CLASS_NAMES)],
    }


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"{path}: {len(data['images'])} images, {len(data['annotations'])} boxes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path,
                        default=Path("data/prepared/manifests"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/prepared/dfine"))
    parser.add_argument("--repeat-power", type=float, default=0.25)
    parser.add_argument("--max-repeat", type=int, default=3)
    args = parser.parse_args()

    for split in ("train", "val", "test", "all_train"):
        source = args.manifest_dir / f"{split}.jsonl"
        if source.exists():
            records = read_manifest(source)
            write_json(convert(records), args.output_dir / f"{split}.json")
            if split in {"train", "all_train"}:
                balanced = repeat_records(records, args.repeat_power, args.max_repeat)
                write_json(convert(balanced),
                           args.output_dir / f"{split}_balanced.json")


if __name__ == "__main__":
    main()
