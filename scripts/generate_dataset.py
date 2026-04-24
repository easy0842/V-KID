"""Generate VKID simulation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vkid.simulation.dataset import generate_dataset, save_dataset
from vkid.simulation.plots import plot_sanity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sim_mvp.yaml"))
    parser.add_argument("--output", type=Path, default=None, help="Override dataset output path from config.")
    parser.add_argument("--plot-dir", type=Path, default=Path("outputs/figures/simulation_mvp"))
    parser.add_argument("--no-plots", action="store_true", help="Skip sanity-check plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output_path = args.output or Path(config["dataset"]["output_path"])
    print(f"Generating dataset from {args.config}...")
    dataset = generate_dataset(config)
    save_dataset(dataset, output_path)
    print(f"Saved dataset: {output_path}")
    print(
        "Shapes: "
        f"states={dataset['states'].shape}, "
        f"actions={dataset['actions'].shape}, "
        f"target_delta={dataset['target_delta'].shape}"
    )

    if not args.no_plots:
        plot_path = plot_sanity(dataset, args.plot_dir)
        print(f"Saved sanity plot: {plot_path}")


if __name__ == "__main__":
    main()
