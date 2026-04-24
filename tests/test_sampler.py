import numpy as np

from vkid.data.sampler import CrossSequenceBatchSampler, VkidSimulationDataset


def _write_toy_dataset(path) -> None:
    n_conditions = 3
    n_sequences = 3
    n_steps = 12
    states = np.zeros((n_conditions, n_sequences, n_steps, 3), dtype=np.float32)
    actions = np.zeros((n_conditions, n_sequences, n_steps, 2), dtype=np.float32)
    for condition in range(n_conditions):
        for sequence in range(n_sequences):
            states[condition, sequence, :, 0] = condition
            states[condition, sequence, :, 1] = sequence
            states[condition, sequence, :, 2] = np.arange(n_steps)
            actions[condition, sequence, :, 0] = 10 + condition
            actions[condition, sequence, :, 1] = 20 + sequence
    inputs = np.concatenate([states, actions], axis=-1)
    np.savez_compressed(
        path,
        time=np.arange(n_steps, dtype=np.float32),
        states=states,
        actions=actions,
        inputs=inputs,
        poses=np.zeros((n_conditions, n_sequences, n_steps, 3), dtype=np.float32),
        target_delta=states[:, :, 1:, :] - states[:, :, :-1, :],
        target_next=states[:, :, 1:, :],
        condition_params=np.zeros((n_conditions, 7), dtype=np.float32),
        condition_param_names=np.array(["B", "C", "D", "E", "m", "Iz", "lf_ratio"]),
        condition_ids=np.arange(n_conditions),
        train_condition_ids=np.array([0, 1]),
        val_condition_ids=np.array([2]),
        sample_rate_hz=np.array(20.0),
        sequence_length_s=np.array(0.6),
    )


def test_dataset_loader_and_sampler_batch_shapes(tmp_path) -> None:
    path = tmp_path / "toy.npz"
    _write_toy_dataset(path)
    dataset = VkidSimulationDataset(path)
    sampler = CrossSequenceBatchSampler(dataset, context_min=2, context_max=5, queries_per_context=4, seed=123)

    batch = sampler.sample_batch(batch_size=6, split="train")

    assert batch["anchor_context"].shape[0] == 6
    assert batch["anchor_context"].shape[2] == 5
    assert batch["anchor_mask"].shape == batch["anchor_context"].shape[:2]
    assert batch["query_input"].shape == (6, 4, 5)
    assert batch["query_target"].shape == (6, 4, 3)
    assert np.all(batch["condition_id"] == batch["positive_condition_id"])
    assert np.all(batch["condition_id"] != batch["negative_condition_id"])
    assert np.all(batch["anchor_sequence_id"] != batch["query_sequence_id"])
    assert np.all(batch["anchor_mask"].sum(axis=1) == batch["context_length"])


def test_sampler_can_return_raw_values(tmp_path) -> None:
    path = tmp_path / "toy.npz"
    _write_toy_dataset(path)
    dataset = VkidSimulationDataset(path)
    sampler = CrossSequenceBatchSampler(
        dataset,
        context_min=2,
        context_max=2,
        queries_per_context=2,
        seed=123,
        normalize=False,
    )

    batch = sampler.sample_batch(batch_size=2, split="train")

    assert np.isin(batch["anchor_context"][..., 0], [0.0, 1.0]).all()


def test_sampler_rejects_val_split_with_one_condition(tmp_path) -> None:
    path = tmp_path / "toy.npz"
    _write_toy_dataset(path)
    dataset = VkidSimulationDataset(path)
    sampler = CrossSequenceBatchSampler(dataset, context_min=2, context_max=5, queries_per_context=4, seed=123)

    try:
        sampler.sample_batch(batch_size=1, split="val")
    except ValueError as exc:
        assert "at least two conditions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for val split with one condition")
