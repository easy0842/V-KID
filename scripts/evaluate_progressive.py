"""Evaluate progressive dynamics identification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from vkid.analysis.progressive import evaluate_progressive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval_progressive_mvp.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    outputs = evaluate_progressive(config)
    print("Progressive evaluation outputs:")
    for name, path in outputs.items():
        print(f"{name}: {path}")

    summary = pd.read_csv(outputs["summary"])
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
