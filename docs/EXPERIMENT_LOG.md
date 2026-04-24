# Experiment Log

Use this file to keep short, reproducible notes for each experiment.

## Template

```text
Date:
Experiment ID:
Config:
Commit or code state:
Dataset:
Goal:
Result:
Key plots:
Notes:
Next action:
```

## Runs

## 2026-04-24: Simulator MVP Smoke and Dataset Generation

Date: 2026-04-24
Experiment ID: sim_mvp_initial
Config: `configs/sim_mvp.yaml`
Dataset: `data/raw/vkid_mvp.npz`
Goal: Verify 3-DOF bicycle simulator, action generator, parameter sampling, dataset serialization, and sanity plot generation.
Result: Success.
Key plots: `outputs/figures/simulation_mvp/sanity_condition_00_sequence_00.png`
Summary artifacts: `outputs/figures/simulation_mvp/summary/`
Notes:

```text
states=(12, 4, 1000, 3)
actions=(12, 4, 1000, 2)
target_delta=(12, 4, 999, 3)
NaN states=0
NaN actions=0
Vx range ~= 25.2-118.1 km/h
Vy range ~= -37.1-34.6 km/h
yaw-rate range ~= -57.2-51.4 deg/s
```

Next action: implement dataset loader and cross-sequence sampler for the MLP VAE baseline.

## 2026-04-24: Cross-Sequence Sampler Inspection

Date: 2026-04-24
Experiment ID: sampler_mvp_initial
Config: `configs/train_mlp_mvp.yaml`
Dataset: `data/raw/vkid_mvp.npz`
Goal: Verify anchor/positive/negative context sampling and cross-sequence query sampling.
Result: Success.
Notes:

```text
anchor_context=(32, 197, 5)
positive_context=(32, 197, 5)
negative_context=(32, 197, 5)
query_input=(32, 64, 5)
query_target=(32, 64, 3)
```

Next action: implement Transformer VAE encoder and MLP decoder baseline.

## 2026-04-24: MLP Baseline Smoke Training

Date: 2026-04-24
Experiment ID: mlp_smoke_initial
Config: `configs/train_mlp_smoke.yaml`
Dataset: `data/raw/vkid_mvp.npz`
Goal: Verify Transformer VAE encoder, MLP decoder, losses, optimizer step, validation loop, and checkpoint writing.
Result: Success.
Notes:

```text
epoch 1 train_rmse ~= 0.739, val_rmse ~= 0.641
epoch 2 train_rmse ~= 0.986, val_rmse ~= 0.732
checkpoint_dir = outputs/checkpoints/mlp_smoke
```

The RMSE is in normalized target-delta units for this smoke test.

Next action: run a longer MLP MVP training job and inspect progressive sigma behavior.
