# VKID Project Rethink Notes

Date: 2026-04-25

## Short Conclusion

The original VKID idea is promising, but the uncertainty framing needs to be revised.

The initial assumption was:

```text
as driving sequence length increases, VAE sigma should decrease
```

The experiments showed that this is too simplistic.

In the current baseline, sigma does not behave as progressive system-identification uncertainty. Instead, sigma appears to respond more to driving excitation, maneuver difficulty, or prediction complexity.

Therefore, the project should be reframed around:

```text
coverage-conditioned dynamics uncertainty
```

rather than:

```text
time-length-conditioned uncertainty
```

## What Has Been Implemented

- 3-DOF bicycle simulator.
- Magic Formula tire model.
- Latin Hypercube vehicle+tire condition sampling.
- Random steering and longitudinal-force action generation.
- MVP dataset:

```text
12 conditions
4 sequences per condition
1000 steps per sequence
50 Hz
48,000 total timesteps
47,952 transitions
```

- Dataset summary and trajectory plots.
- Cross-sequence sampler.
- Transformer VAE encoder.
- MLP dynamics decoder baseline.
- GPU training.
- Progressive sigma and matching evaluation.

## Main Experimental Result

The MLP VAE baseline can reduce prediction loss, but its VAE sigma does not decrease as context length increases.

Validation progressive evaluation:

| Context Steps | Time [s] | Mean Sigma | Top-1 | Top-3 |
| --- | ---: | ---: | ---: | ---: |
| 5 | 0.1 | 0.3729 | 0.167 | 0.417 |
| 10 | 0.2 | 0.3797 | 0.250 | 0.417 |
| 20 | 0.4 | 0.3795 | 0.000 | 0.500 |
| 50 | 1.0 | 0.3820 | 0.250 | 0.417 |
| 100 | 2.0 | 0.3809 | 0.250 | 0.500 |
| 200 | 4.0 | 0.3842 | 0.250 | 0.500 |

This means:

```text
longer observed context does not automatically imply lower sigma
```

## Important Diagnostic Finding

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
more excited/aggressive driving segments tend to produce larger sigma
```

Thus, the current sigma is closer to:

```text
prediction difficulty / maneuver complexity / latent noise scale
```

than:

```text
system identification uncertainty
```

## Why the Original Framing Is Ambiguous

The phrase:

```text
vehicle dynamics uncertainty decreases as more driving data accumulates
```

is too broad.

The core problem:

```text
How can the model know which future maneuver domain it should be confident about?
```

For example:

- Straight driving may identify longitudinal behavior but not lateral handling.
- Cornering may identify lateral/yaw behavior but not braking/traction behavior.
- Aggressive maneuvers may reveal nonlinear tire behavior but may also increase prediction difficulty.

Therefore, a single scalar or averaged sigma cannot represent all future dynamics uncertainty.

## Better Conceptual Framing

Uncertainty should be conditional on both:

```text
1. observed context coverage C
2. future query or maneuver domain Q
```

Better formulation:

```text
VKID learns p(z_dyn | C)

Uncertainty should be evaluated as:

U(C, Q) = Var_{z ~ p(z|C)} [ decoder(z, q) ], q in Q
```

In words:

```text
Given observed driving context C,
how uncertain is the model when predicting dynamics over a specific query maneuver domain Q?
```

This avoids claiming global uncertainty without specifying the future driving domain.

## Recommended Reframing

Replace:

```text
progressive identification = longer sequence -> lower sigma
```

with:

```text
progressive identification = increasing coverage over relevant maneuver domains -> lower uncertainty for those domains
```

More precise statement:

```text
Uncertainty decreases only for dynamics modes sufficiently excited by the observed driving context.
```

## Suggested New Experiment Structure

Define context types:

```text
straight-only
acceleration/braking
cornering
cornering + acceleration/braking
mixed/aggressive
```

Define query domains:

```text
longitudinal queries
lateral/yaw queries
aggressive/nonlinear queries
```

Then evaluate a matrix:

```text
rows = observed context type
columns = query domain
value = predictive uncertainty or prediction RMSE
```

Expected pattern:

```text
straight context -> low uncertainty for longitudinal queries, high for lateral queries
cornering context -> lower uncertainty for lateral queries, higher for longitudinal/traction queries
mixed context -> lower uncertainty across multiple query domains
```

This is more meaningful than simply plotting:

```text
context length vs sigma
```

## Model/Loss Direction If Continuing

If continuing the current VAE approach, the model should separate:

```text
identification uncertainty: p(z_dyn | context)
predictive uncertainty: Var decoder(z, query)
```

Recommended next direction:

1. Keep the Transformer VAE encoder.
2. Keep the MLP decoder baseline for now.
3. Define maneuver coverage metrics.
4. Define query-domain-specific evaluation sets.
5. Use Monte Carlo samples from `p(z|C)` to compute predictive variance over each query domain.
6. Treat sigma as meaningful only when tied to a query domain or coverage domain.

Possible loss improvement:

```text
coverage-aware uncertainty calibration
```

But this should come after redefining the evaluation protocol.

## Key Takeaway For Redesign

The project should not claim:

```text
more time -> less uncertainty
```

It should claim something closer to:

```text
observed driving coverage determines which regions of vehicle dynamics are identifiable;
uncertainty must be evaluated conditionally on future maneuver/query domains.
```

This reframing is more physically correct and more defensible.

## Useful Existing Artifacts

Important files:

```text
docs/RESULTS_SUMMARY.md
outputs/figures/progressive_mvp/sigma_vs_context.png
outputs/figures/progressive_mvp/sigma_by_dimension.png
outputs/figures/progressive_mvp/sigma_vs_excitation.png
outputs/figures/simulation_mvp/all_trajectories_xy.png
outputs/figures/simulation_mvp/summary/trajectory_grid.png
```

Recent relevant commits:

```text
d665e51 Expand progressive sigma diagnostics
5a8b399 Add progressive identification evaluation
10be1d1 Add MVP long training config
a2fd52f Add VAE MLP baseline training
```
