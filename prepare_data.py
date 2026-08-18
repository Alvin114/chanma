from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_baseline.data.manifest import prepare_manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AiC archives and build leakage-aware manifests.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/manifests"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    result = prepare_manifests(args.data_root.resolve(), args.output, args.val_fraction, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

