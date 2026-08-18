from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_baseline.data.manifest import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prepared AiC manifests.")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/prepared/manifests"))
    args = parser.parse_args()
    output = {}
    for split in ("train", "val", "test"):
        records = read_jsonl(args.manifest_dir / f"{split}.jsonl")
        counts = [0] * 12
        for record in records:
            for label in record.get("labels", []):
                counts[int(label[0])] += 1
        output[split] = {
            "samples": len(records),
            "groups": len({record["group"] for record in records}),
            "boxes_per_class": counts,
        }
    output["train_val_group_overlap"] = sorted(
        {record["group"] for record in read_jsonl(args.manifest_dir / "train.jsonl")}
        & {record["group"] for record in read_jsonl(args.manifest_dir / "val.jsonl")}
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

