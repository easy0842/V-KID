# VKID: Variational Koopman Identification of Dynamics

VKID is a project for progressive vehicle dynamics fingerprinting and matching from driving time-series data.

The core idea is to infer a latent dynamics fingerprint `z_dyn` from short context windows, track uncertainty as the context grows, and use the latent space to retrieve dynamically similar vehicle+tire systems. A Deep Koopman decoder will be compared against an MLP decoder baseline.

## Current Scope

The initial milestone is an MVP:

1. Generate simulated 3-DOF bicycle-model data across multiple tire and vehicle parameter conditions.
2. Train a Transformer VAE encoder with an MLP dynamics decoder.
3. Evaluate progressive uncertainty convergence and latent matching accuracy.
4. Add a z-conditioned Deep Koopman decoder after the baseline is stable.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full working plan.

## Repository Layout

```text
configs/                 Experiment and dataset configs
data/                    Local generated datasets
docs/                    Planning and experiment notes
notebooks/               Exploratory analysis
outputs/                 Figures, checkpoints, logs
scripts/                 CLI entry points
src/vkid/                Python package source
tests/                   Unit and smoke tests
```

## First Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Generate the MVP simulation dataset:

```bash
python3 scripts/generate_dataset.py --config configs/sim_mvp.yaml
```

This writes `data/raw/vkid_mvp.npz` and a sanity-check plot under `outputs/figures/simulation_mvp/`.

Planned next scripts:

```bash
python3 scripts/train_mlp_baseline.py --config configs/train_mlp_mvp.yaml
python3 scripts/evaluate_progressive.py --config configs/eval_progressive_mvp.yaml
```
