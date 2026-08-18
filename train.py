from __future__ import annotations

import argparse

from aic_baseline.config import deep_update, load_config, parse_overrides
from aic_baseline.engine.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AiC multimodal YOLO baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = deep_update(load_config(args.config), parse_overrides(args.set))
    output = train(config)
    print(f"training artifacts: {output.resolve()}")


if __name__ == "__main__":
    main()

