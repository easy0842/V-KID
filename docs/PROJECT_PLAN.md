# VKID Project Plan

## 1. Project Goal

Build a framework that progressively identifies system-level vehicle dynamics behavior from driving time-series data.

The target output is:

```text
short driving context -> z_dyn distribution -> dynamics fingerprint + uncertainty
longer context -> lower uncertainty -> stable matching
latent distance -> dynamically similar system retrieval
Koopman operator K(z_dyn) -> interpretable dynamics modes
```

This project does not primarily estimate individual physical parameters such as tire `B`, vehicle mass `m`, or yaw inertia `Iz`. Instead, it learns a latent representation of the combined vehicle+tire behavior produced by those parameters.

## 2. Research Questions

1. Can a Transformer VAE infer a stable dynamics fingerprint from short driving context?
2. Does the VAE uncertainty decrease as context length increases?
3. Does latent distance correspond to behavior-level dynamics similarity?
4. Does a z-conditioned Deep Koopman decoder provide useful interpretation through `K(z_dyn)` eigenvalues?
5. How much driving excitation is needed for reliable identification?

## 3. Milestone Strategy

The project should be built in layers.

| Milestone | Purpose | Success Criterion |
| --- | --- | --- |
| M0: Repository setup | Create project structure, configs, and plan | Scripts/config paths are defined |
| M1: Simulator MVP | Generate varied vehicle+tire dynamics data | Trajectories are physically plausible |
| M2: MLP VAE baseline | Prove progressive fingerprinting works | Sigma decreases and matching improves with context |
| M3: Matching analysis | Validate behavior-level latent similarity | Latent distance correlates with behavior distance |
| M4: Deep Koopman decoder | Add interpretable structured decoder | Competitive prediction and meaningful eigenvalue trends |
| M5: Final experiments | Package figures and comparisons | Report-ready plots and tables |

## 4. Phase 1: Simulation Data Generation

### 4.1 Vehicle Model

Use a 3-DOF bicycle model:

```text
x = [Vx, Vy, wz]
u = [delta, Fx]
```

Equations:

```text
m(Vxdot - Vy*wz) = Fxf*cos(delta) - Fyf*sin(delta) + Fxr
m(Vydot + Vx*wz) = Fyf*cos(delta) + Fxf*sin(delta) + Fyr
Iz*wzdot = lf*(Fyf*cos(delta) + Fxf*sin(delta)) - lr*Fyr
```

Initial simplification:

- RWD or fixed longitudinal force distribution.
- Same tire parameters for front and rear.
- Clamp low `Vx` to avoid unstable slip-angle division.

### 4.2 Tire Model

Use Magic Formula lateral force:

```text
Fy = D * sin(C * atan(B*alpha - E*(B*alpha - atan(B*alpha))))
```

Slip angles:

```text
alpha_f = delta - atan((Vy + lf*wz) / Vx)
alpha_r = -atan((Vy - lr*wz) / Vx)
```

### 4.3 Parameter Sampling

Use Latin Hypercube Sampling over 7 dimensions.

| Parameter | Range |
| --- | --- |
| tire B | 8-14 |
| tire C | 1.0-1.6 |
| tire D | 0.5-1.1 |
| tire E | -1.5-0.5 |
| vehicle m | 1200-2000 kg |
| vehicle Iz | 1500-3500 kg m^2 |
| lf ratio | 0.45-0.55 |

MVP dataset:

```text
conditions: 12
sequences per condition: 4
sequence length: 20 s
sampling rate: 50 Hz
```

Full dataset:

```text
conditions: 35
sequences per condition: 8
sequence length: 30-60 s
sampling rate: 100 Hz
```

### 4.4 Action Generation

Generate varied driving excitation:

- Steering: filtered random signal, plus occasional step steer events.
- Longitudinal force: filtered random signal, plus occasional hard acceleration or braking.
- Different random seed per sequence.

### 4.5 Data Format

Store each sequence with:

```text
time: [T]
state: [T, 3] = [Vx, Vy, wz]
action: [T, 2] = [delta, Fx]
input: [T, 5] = [Vx, Vy, wz, delta, Fx]
target_delta: [T-1, 3] = x[t+1] - x[t]
target_next: [T-1, 3] = x[t+1]
condition_id
metadata: B, C, D, E, m, Iz, lf, lr
```

## 5. Phase 2: MLP VAE Baseline

### 5.1 Architecture

```text
context [T_context, 5]
-> input projection
-> positional encoding
-> Transformer encoder
-> mean pooling
-> mu, logvar
-> reparameterized z_dyn
-> MLP decoder([z_dyn, x_t, u_t])
-> delta x prediction
```

### 5.2 Training Batch

For each iteration:

1. Sample condition `k`.
2. Sample context from sequence A of condition `k`.
3. Encode context into `mu`, `logvar`, and `z_dyn`.
4. Sample query points from sequence B of the same condition.
5. Predict `delta x` for query points.
6. Sample positive context from sequence C of condition `k`.
7. Sample negative context from sequence D of condition `j != k`.
8. Compute dynamics loss, KL loss, and triplet consistency loss.

### 5.3 Losses

Start simple:

```text
L_total = MSE(delta_x_hat, delta_x) + beta * KL
```

Then add consistency:

```text
L_total = L_dyn + lambda_consist * L_triplet + beta * KL
```

Optionally add heteroscedastic NLL later:

```text
L_dyn = 0.5*logvar_delta + (delta_x - mu_delta)^2 / (2*exp(logvar_delta))
```

## 6. Phase 3: Progressive Identification

Evaluate context lengths:

```text
T = [5, 10, 20, 50, 100, 200]
```

For each test sequence:

```text
context[0:T] -> mu_T, sigma_T
nearest = argmin distance(mu_T, z_database)
```

Primary plots:

- Context length vs mean sigma.
- Context length vs top-1/top-3 matching accuracy.
- Latent trajectory as context grows.
- Matching rank stability over time.

## 7. Phase 4: Behavior Similarity Validation

Define a behavior distance independent of parameter distance.

For each pair of conditions, run the same input sequence:

```text
behavior_distance(i, j) = RMSE(trajectory_i, trajectory_j)
```

Then compare:

```text
latent_distance(i, j) vs behavior_distance(i, j)
```

Report:

- Spearman correlation.
- Scatter plot.
- Nearest-neighbor examples where parameters differ but behavior is similar.

## 8. Phase 5: Deep Koopman Decoder

Recommended control-affine structure:

```text
phi(x_t) -> g_t
K(z_dyn) -> [d_lift, d_lift]
B(z_dyn) -> [d_lift, d_u]
g_next = K(z_dyn) g_t + B(z_dyn) u_t
psi(g_next) -> x_next_hat
```

Loss:

```text
L_recon = ||psi(K(z)phi(x_t) + B(z)u_t) - x_{t+1}||^2
L_linear = ||phi(x_{t+1}) - K(z)phi(x_t) - B(z)u_t||^2
L_total = L_recon + lambda_linear*L_linear + lambda_consist*L_triplet + beta*KL
```

Suggested training order:

1. Pretrain `phi` and `psi` as a state autoencoder.
2. Train one-step Koopman prediction.
3. Add VAE, triplet, and KL objectives.

## 9. Phase 6: Driving Content Analysis

Split contexts by excitation:

| Content Type | Expected Result |
| --- | --- |
| Straight-only | Lateral/yaw uncertainty remains high |
| Moderate cornering | Lateral dynamics uncertainty decreases |
| Aggressive maneuvers | Fast uncertainty convergence |

Use this to estimate how much informative driving content is needed for identification.

## 10. Baselines

Recommended baselines, in priority order:

1. MLP decoder baseline.
2. DMD or EDMD matrix matching.
3. Supervised parameter regression.

The MLP baseline is mandatory. DMD/EDMD is the most course-aligned comparison.

## 11. Deliverables

| Deliverable | Description |
| --- | --- |
| Simulator sanity plots | Vehicle trajectories, tire forces, slip angles |
| Prediction metrics | RMSE for MLP and Koopman decoders |
| Sigma convergence curves | Context length vs uncertainty |
| Matching accuracy curves | Context length vs top-1/top-3 |
| Latent visualization | UMAP or t-SNE by dynamics condition |
| Behavior similarity validation | Latent distance vs rollout behavior distance |
| Koopman eigenvalue analysis | Eigenvalue shifts across learned fingerprints |
| Final report figures | Clean figures for presentation/report |

## 12. Six-Week Schedule

| Week | Goal |
| --- | --- |
| 1 | Simulator, parameter sampling, dataset generation, sanity plots |
| 2 | Dataset loader, Transformer VAE, MLP baseline training |
| 3 | Progressive identification, latent DB, matching accuracy |
| 4 | Behavior similarity metric, driving content analysis, DMD/EDMD baseline |
| 5 | Deep Koopman decoder, eigenvalue analysis |
| 6 | Final experiments, plots, report and presentation |

## 13. Immediate Next Tasks

- [x] Implement simulation parameter dataclasses.
- [x] Implement Magic Formula tire force.
- [x] Implement bicycle-model derivative function.
- [x] Implement filtered steering and longitudinal-force generator.
- [x] Implement dataset generation script.
- [x] Generate MVP dataset.
- [x] Plot simulator sanity checks.
- [x] Implement Phase 1 dataset summary script.
- [x] Implement dataset loader and cross-sequence sampler.
- [ ] Implement Transformer VAE encoder.
- [ ] Implement MLP decoder baseline.

## 14. Current Simulator Status

Implemented files:

- `src/vkid/simulation/parameters.py`
- `src/vkid/simulation/actions.py`
- `src/vkid/simulation/dynamics.py`
- `src/vkid/simulation/dataset.py`
- `src/vkid/simulation/plots.py`
- `scripts/generate_dataset.py`

Current MVP dataset command:

```bash
python3 scripts/generate_dataset.py --config configs/sim_mvp.yaml --plot-dir outputs/figures/simulation_mvp
python3 scripts/summarize_dataset.py --dataset data/raw/vkid_mvp.npz --output-dir outputs/figures/simulation_mvp/summary
```

Verified MVP output:

```text
states: (12, 4, 1000, 3)
actions: (12, 4, 1000, 2)
target_delta: (12, 4, 999, 3)
train condition ids: 0-8
validation condition ids: 9-11
```

The simulator includes a sanity filter that regenerates sequences with too-low forward speed, excessive lateral velocity, or excessive yaw rate.

Summary artifacts:

- `condition_summary.csv`
- `sequence_summary.csv`
- `range_summary.json`
- `trajectory_grid.png`
- `feature_histograms.png`

## 15. Current Phase 2 Data Sampler Status

Implemented files:

- `src/vkid/data/sampler.py`
- `scripts/inspect_sampler.py`
- `tests/test_sampler.py`

Sampler inspection command:

```bash
python3 scripts/inspect_sampler.py --config configs/train_mlp_mvp.yaml --split train
```

Verified batch shapes:

```text
anchor_context: (32, T_max, 5)
anchor_mask: (32, T_max)
positive_context: (32, T_max, 5)
negative_context: (32, T_max, 5)
query_input: (32, 64, 5)
query_target: (32, 64, 3)
context_length: (32,)
```

`T_max` varies by batch according to the longest sampled context in that batch.
