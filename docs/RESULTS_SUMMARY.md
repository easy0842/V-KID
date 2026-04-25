# VKID Results Summary

Date: 2026-04-25

## 1. Project Snapshot

VKID aims to identify a vehicle+tire system's dynamics fingerprint from driving time-series data.

The intended pipeline is:

```text
driving context -> Transformer VAE encoder -> z_dyn = N(mu, sigma^2)
z_dyn + query state/action -> MLP dynamics decoder -> delta state prediction
context length increases -> expected sigma decreases
latent distance -> dynamics condition matching
```

Current implementation status:

- Phase 1 simulation dataset generation is complete.
- Cross-sequence sampler is implemented.
- Transformer VAE + MLP decoder baseline is implemented.
- GPU long training has been run.
- Progressive sigma/matching evaluation has been run.

## 2. Simulation Dataset

Dataset:

```text
data/raw/vkid_mvp.npz
```

Dataset scale:

```text
conditions: 12
sequences per condition: 4
steps per sequence: 1000
total sequences: 48
total timesteps: 48,000
total transitions: 47,952
sample rate: 50 Hz
sequence length: 20 s
```

Train/validation split:

```text
train conditions: 9
validation conditions: 3
```

Condition parameters:

```text
B, C, D, E, m, Iz, lf_ratio
```

Range sanity check:

```text
Vx range: approximately 25.2-118.1 km/h
Vy range: approximately -37.1-34.6 km/h
yaw-rate range: approximately -57.2-51.4 deg/s
NaN states: 0
NaN actions: 0
```

### Key Figures

Overall trajectories:

![All trajectories](../outputs/figures/simulation_mvp/all_trajectories_xy.png)

Condition-wise trajectory grid:

![Trajectory grid](../outputs/figures/simulation_mvp/summary/trajectory_grid.png)

Feature histograms:

![Feature histograms](../outputs/figures/simulation_mvp/summary/feature_histograms.png)

Single-sequence sanity check:

![Simulator sanity check](../outputs/figures/simulation_mvp/sanity_condition_00_sequence_00.png)

## 3. Baseline Training

Model:

```text
Transformer VAE encoder + MLP dynamics decoder
```

Training command:

```bash
python3 scripts/train_mlp_baseline.py --config configs/train_mlp_mvp_long.yaml
```

Checkpoint:

```text
outputs/checkpoints/mlp_mvp_long/best.pt
```

Training log:

```text
outputs/logs/mlp_mvp_long/history.csv
```

Long training result:

```text
epoch 1   train_rmse ~= 1.052, val_rmse ~= 0.668
epoch 8   train_rmse ~= 0.783, val_rmse ~= 0.478
epoch 17  train_rmse ~= 0.591, val_rmse ~= 0.407
epoch 24  train_rmse ~= 0.486, val_rmse ~= 0.372
epoch 25  train_rmse ~= 0.462, val_rmse ~= 0.384
```

Best validation:

```text
best_val_mse ~= 0.140
best_val_rmse ~= 0.372
```

Note: RMSE is measured in normalized target-delta units.

## 4. Progressive Identification Evaluation

Evaluation command:

```bash
python3 scripts/evaluate_progressive.py --config configs/eval_progressive_mvp.yaml
```

Evaluation setup:

```text
checkpoint: outputs/checkpoints/mlp_mvp_long/best.pt
split: validation
context lengths: 5, 10, 20, 50, 100, 200
context times: 0.1, 0.2, 0.4, 1.0, 2.0, 4.0 s
```

Expected behavior:

```text
as context length increases, sigma should decrease
```

Observed behavior:

```text
sigma did not decrease with context length
```

Validation summary:

| Context Steps | Time [s] | Mean Sigma | Top-1 | Top-3 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 0.1 | 0.3729 | 0.167 | 0.417 |
| 10 | 0.2 | 0.3797 | 0.250 | 0.417 |
| 20 | 0.4 | 0.3795 | 0.000 | 0.500 |
| 50 | 1.0 | 0.3820 | 0.250 | 0.417 |
| 100 | 2.0 | 0.3809 | 0.250 | 0.500 |
| 200 | 4.0 | 0.3842 | 0.250 | 0.500 |

### Key Figures

Sigma vs context:

![Sigma vs context](../outputs/figures/progressive_mvp/sigma_vs_context.png)

Matching accuracy:

![Matching vs context](../outputs/figures/progressive_mvp/matching_vs_context.png)

Mu distance to reference:

![Mu distance vs context](../outputs/figures/progressive_mvp/mu_distance_vs_context.png)

Sigma by latent dimension:

![Sigma by dimension](../outputs/figures/progressive_mvp/sigma_by_dimension.png)

Sigma by condition:

![Sigma by condition](../outputs/figures/progressive_mvp/sigma_by_condition.png)

Sigma vs driving excitation:

![Sigma vs excitation](../outputs/figures/progressive_mvp/sigma_vs_excitation.png)

## 5. Main Diagnostic Finding

The naive VAE baseline does not automatically produce progressive identification uncertainty.

The sigma values are nearly flat or slightly increasing as context length grows. Dimension-wise analysis shows no consistent monotonic decrease.

This suggests that current sigma behaves more like:

```text
prediction difficulty / maneuver complexity / latent noise scale
```

rather than:

```text
system identification uncertainty
```

## 6. Excitation Correlation

Sigma is positively correlated with driving excitation.

Spearman correlations with mean sigma:

| Feature | Correlation |
| --- | ---: |
| steering std | 0.50 |
| yaw-rate std | 0.52 |
| Fx std | 0.46 |
| Vy std | 0.45 |
| max steering | 0.51 |
| max yaw-rate | 0.56 |

Interpretation:

```text
More excited/aggressive driving segments tend to produce larger sigma.
```

This supports the hypothesis that current sigma is responding to local prediction difficulty or dynamic complexity, not pure identification uncertainty.

## 7. Interpretation

The current result does not invalidate the project.

Instead, it shows a useful baseline result:

```text
Prediction-trained VAE sigma is not automatically calibrated as progressive dynamics-identification uncertainty.
```

Therefore, VKID needs an explicit progressive uncertainty objective.

## 8. Recommended Next Step

Add a short-context vs long-context progressive loss.

For the same condition and sequence:

```text
T_short < T_long
context_short -> mu_short, sigma_short
context_long  -> mu_long,  sigma_long
```

Desired constraints:

```text
sigma_long < sigma_short
mu_long is closer to condition reference
mu_short approaches stopgrad(mu_long)
```

Candidate loss:

```text
L_progress =
  ||mu_short - stopgrad(mu_long)||^2
  + relu(mean_sigma_long - mean_sigma_short + margin)
```

This would directly align the VAE posterior with the intended interpretation:

```text
more observed driving data -> lower system-identification uncertainty
```

## 9. Current Git Commits

Recent commits:

```text
d665e51 Expand progressive sigma diagnostics
5a8b399 Add progressive identification evaluation
10be1d1 Add MVP long training config
a2fd52f Add VAE MLP baseline training
```
