"""Summarize a generated VKID simulation dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from vkid.analysis.dataset_summary import summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/vkid_mvp.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures/simulation_mvp/summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = summarize_dataset(args.dataset, args.output_dir)
    print(f"Summary written for dataset: {args.dataset}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
