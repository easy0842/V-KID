from pathlib import Path

import numpy as np

from vkid.analysis.dataset_summary import (
    make_condition_table,
    make_range_summary,
    make_sequence_table,
    summarize_dataset,
)


def _toy_dataset() -> dict[str, np.ndarray]:
    states = np.zeros((2, 2, 4, 3), dtype=np.float64)
    actions = np.zeros((2, 2, 4, 2), dtype=np.float64)
    poses = np.zeros((2, 2, 4, 3), dtype=np.float64)
    states[..., 0] = 10.0
    actions[..., 1] = 100.0
    poses[..., 0] = np.arange(4)
    return {
        "states": states,
        "actions": actions,
        "inputs": np.concatenate([states, actions], axis=-1),
        "poses": poses,
        "target_delta": states[:, :, 1:, :] - states[:, :, :-1, :],
        "target_next": states[:, :, 1:, :],
        "condition_params": np.array(
            [
                [10.0, 1.2, 0.9, -0.5, 1500.0, 2400.0, 0.48],
                [11.0, 1.3, 1.0, -0.4, 1600.0, 2500.0, 0.52],
            ]
        ),
        "condition_param_names": np.array(["B", "C", "D", "E", "m", "Iz", "lf_ratio"]),
        "condition_ids": np.array([0, 1]),
        "train_condition_ids": np.array([0]),
        "val_condition_ids": np.array([1]),
        "sample_rate_hz": np.array(20.0),
        "sequence_length_s": np.array(0.2),
    }


def test_summary_tables_have_expected_rows() -> None:
    dataset = _toy_dataset()
    assert make_condition_table(dataset).shape[0] == 2
    assert make_sequence_table(dataset).shape[0] == 4
    summary = make_range_summary(dataset)
    assert summary["total_sequences"] == 4
    assert summary["total_transitions"] == 12


def test_summarize_dataset_writes_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "toy.npz"
    np.savez_compressed(dataset_path, **_toy_dataset())
    outputs = summarize_dataset(dataset_path, tmp_path / "summary")
    for path in outputs.values():
        assert path.exists()
