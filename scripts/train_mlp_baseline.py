"""Train the Transformer VAE + MLP decoder baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vkid.training.mlp_baseline import train_mlp_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_mlp_mvp.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    result = train_mlp_baseline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
