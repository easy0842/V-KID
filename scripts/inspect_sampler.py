"""Inspect VKID cross-sequence sampler batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vkid.data.sampler import CrossSequenceBatchSampler, VkidSimulationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_mlp_mvp.yaml"))
    parser.add_argument("--split", choices=["train", "val", "all"], default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset = VkidSimulationDataset(config["data"]["dataset_path"])
    sampler = CrossSequenceBatchSampler(
        dataset,
        context_min=config["data"]["train_context_min"],
        context_max=config["data"]["train_context_max"],
        queries_per_context=config["data"]["queries_per_context"],
        seed=config["seed"],
        normalize=config["data"].get("normalize", True),
    )
    batch = sampler.sample_batch(config["train"]["batch_size"], split=args.split)

    print(f"dataset: {dataset.path}")
    print(f"conditions={dataset.n_conditions}, sequences={dataset.n_sequences}, steps={dataset.n_steps}")
    print(f"normalize={sampler.normalize}")
    for key, value in batch.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
    print("first condition ids:", batch["condition_id"][:8].tolist())
    print("first negative ids:", batch["negative_condition_id"][:8].tolist())
    print("first context lengths:", batch["context_length"][:8].tolist())


if __name__ == "__main__":
    main()
