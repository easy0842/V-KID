"""Cross-sequence dataset sampler for VKID training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Split = Literal["train", "val", "all"]


@dataclass(frozen=True)
class ContextSample:
    condition_id: int
    sequence_id: int
    start_index: int
    length: int
    values: np.ndarray


@dataclass(frozen=True)
class QuerySample:
    condition_id: int
    sequence_id: int
    indices: np.ndarray
    inputs: np.ndarray
    targets: np.ndarray


class VkidSimulationDataset:
    """Thin loader around a generated VKID simulation ``.npz`` file."""

    def __init__(self, dataset_path: str | Path):
        self.path = Path(dataset_path)
        with np.load(self.path, allow_pickle=False) as data:
            self.time = data["time"].astype(np.float32)
            self.states = data["states"].astype(np.float32)
            self.actions = data["actions"].astype(np.float32)
            self.inputs = data["inputs"].astype(np.float32)
            self.target_delta = data["target_delta"].astype(np.float32)
            self.target_next = data["target_next"].astype(np.float32)
            self.condition_params = data["condition_params"].astype(np.float32)
            self.condition_param_names = data["condition_param_names"]
            self.condition_ids = data["condition_ids"].astype(np.int64)
            self.train_condition_ids = data["train_condition_ids"].astype(np.int64)
            self.val_condition_ids = data["val_condition_ids"].astype(np.int64)
            self.sample_rate_hz = float(data["sample_rate_hz"])
            self.sequence_length_s = float(data["sequence_length_s"])

    @property
    def n_conditions(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def n_sequences(self) -> int:
        return int(self.inputs.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.inputs.shape[2])

    @property
    def input_dim(self) -> int:
        return int(self.inputs.shape[-1])

    @property
    def target_dim(self) -> int:
        return int(self.target_delta.shape[-1])

    def condition_ids_for_split(self, split: Split) -> np.ndarray:
        if split == "train":
            return self.train_condition_ids
        if split == "val":
            return self.val_condition_ids
        if split == "all":
            return self.condition_ids
        raise ValueError(f"Unknown split: {split}")


class CrossSequenceBatchSampler:
    """Sample anchor/positive/negative contexts and cross-sequence queries."""

    def __init__(
        self,
        dataset: VkidSimulationDataset,
        context_min: int,
        context_max: int,
        queries_per_context: int,
        seed: int = 0,
    ):
        if context_min < 1:
            raise ValueError("context_min must be positive")
        if context_max < context_min:
            raise ValueError("context_max must be >= context_min")
        if context_max > dataset.n_steps - 1:
            raise ValueError("context_max must leave at least one transition step")
        if queries_per_context < 1:
            raise ValueError("queries_per_context must be positive")
        if dataset.n_sequences < 2:
            raise ValueError("At least two sequences per condition are required")

        self.dataset = dataset
        self.context_min = int(context_min)
        self.context_max = int(context_max)
        self.queries_per_context = int(queries_per_context)
        self.rng = np.random.default_rng(seed)

    def sample_context(
        self,
        condition_id: int,
        sequence_id: int | None = None,
        length: int | None = None,
    ) -> ContextSample:
        if sequence_id is None:
            sequence_id = int(self.rng.integers(0, self.dataset.n_sequences))
        if length is None:
            length = int(self.rng.integers(self.context_min, self.context_max + 1))
        max_start = self.dataset.n_steps - length
        start_index = int(self.rng.integers(0, max_start + 1))
        values = self.dataset.inputs[condition_id, sequence_id, start_index : start_index + length]
        return ContextSample(condition_id, sequence_id, start_index, length, values)

    def sample_queries(
        self,
        condition_id: int,
        exclude_sequence_id: int | None = None,
    ) -> QuerySample:
        candidate_sequences = np.arange(self.dataset.n_sequences)
        if exclude_sequence_id is not None and self.dataset.n_sequences > 1:
            candidate_sequences = candidate_sequences[candidate_sequences != exclude_sequence_id]
        sequence_id = int(self.rng.choice(candidate_sequences))

        # target_delta has one fewer time step than inputs. Query input at t predicts delta to t+1.
        n_transition_steps = self.dataset.target_delta.shape[2]
        replace = self.queries_per_context > n_transition_steps
        indices = self.rng.choice(n_transition_steps, size=self.queries_per_context, replace=replace)
        indices = np.sort(indices).astype(np.int64)
        return QuerySample(
            condition_id=condition_id,
            sequence_id=sequence_id,
            indices=indices,
            inputs=self.dataset.inputs[condition_id, sequence_id, indices],
            targets=self.dataset.target_delta[condition_id, sequence_id, indices],
        )

    def sample_batch(self, batch_size: int, split: Split = "train") -> dict[str, np.ndarray]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        available_condition_ids = self.dataset.condition_ids_for_split(split)
        if available_condition_ids.size < 2:
            raise ValueError(f"Split '{split}' needs at least two conditions for negative sampling")

        anchor_contexts: list[ContextSample] = []
        positive_contexts: list[ContextSample] = []
        negative_contexts: list[ContextSample] = []
        query_samples: list[QuerySample] = []

        for _ in range(batch_size):
            condition_id = int(self.rng.choice(available_condition_ids))
            negative_candidates = available_condition_ids[available_condition_ids != condition_id]
            negative_condition_id = int(self.rng.choice(negative_candidates))

            anchor = self.sample_context(condition_id)
            positive_sequence_candidates = np.arange(self.dataset.n_sequences)
            positive_sequence_candidates = positive_sequence_candidates[positive_sequence_candidates != anchor.sequence_id]
            positive_sequence_id = int(self.rng.choice(positive_sequence_candidates))
            positive = self.sample_context(condition_id, sequence_id=positive_sequence_id, length=anchor.length)
            negative = self.sample_context(negative_condition_id, length=anchor.length)
            query = self.sample_queries(condition_id, exclude_sequence_id=anchor.sequence_id)

            anchor_contexts.append(anchor)
            positive_contexts.append(positive)
            negative_contexts.append(negative)
            query_samples.append(query)

        max_length = max(context.length for context in anchor_contexts)
        return {
            "anchor_context": _pad_contexts(anchor_contexts, max_length, self.dataset.input_dim),
            "anchor_mask": _make_masks(anchor_contexts, max_length),
            "positive_context": _pad_contexts(positive_contexts, max_length, self.dataset.input_dim),
            "positive_mask": _make_masks(positive_contexts, max_length),
            "negative_context": _pad_contexts(negative_contexts, max_length, self.dataset.input_dim),
            "negative_mask": _make_masks(negative_contexts, max_length),
            "query_input": np.stack([sample.inputs for sample in query_samples], axis=0).astype(np.float32),
            "query_target": np.stack([sample.targets for sample in query_samples], axis=0).astype(np.float32),
            "query_indices": np.stack([sample.indices for sample in query_samples], axis=0),
            "condition_id": np.array([sample.condition_id for sample in anchor_contexts], dtype=np.int64),
            "positive_condition_id": np.array([sample.condition_id for sample in positive_contexts], dtype=np.int64),
            "negative_condition_id": np.array([sample.condition_id for sample in negative_contexts], dtype=np.int64),
            "anchor_sequence_id": np.array([sample.sequence_id for sample in anchor_contexts], dtype=np.int64),
            "query_sequence_id": np.array([sample.sequence_id for sample in query_samples], dtype=np.int64),
            "context_length": np.array([sample.length for sample in anchor_contexts], dtype=np.int64),
        }


def _pad_contexts(contexts: list[ContextSample], max_length: int, input_dim: int) -> np.ndarray:
    padded = np.zeros((len(contexts), max_length, input_dim), dtype=np.float32)
    for row, context in enumerate(contexts):
        padded[row, : context.length] = context.values
    return padded


def _make_masks(contexts: list[ContextSample], max_length: int) -> np.ndarray:
    mask = np.zeros((len(contexts), max_length), dtype=bool)
    for row, context in enumerate(contexts):
        mask[row, : context.length] = True
    return mask
