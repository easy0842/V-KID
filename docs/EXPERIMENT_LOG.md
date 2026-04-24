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
