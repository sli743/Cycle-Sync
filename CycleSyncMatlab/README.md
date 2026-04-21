# CycleSync MATLAB

A GitHub-ready MATLAB repository for Cycle-Sync camera location synchronization on synthetic and precomputed real-data inputs.

Included methods:

- Cycle-Sync with T-AAB initialization and cycle-message reweighting.
- LUD.
- ShapeFit.
- BATA.
- Full FusedTA baseline.
- Uniform and cycle-consistent/adversarial synthetic corruption models.
- Robust signed-scale/translation alignment for evaluation.

The default Cycle-Sync parameters are set in `src/core/default_cycle_params.m`:

```text
tmax = 20
beta = 20
lambda_t = t/(t+10)
delta = 1e-8
rho(x) = 1 - exp(-4 |x|)
w_ij = exp(-4 h_ij)/(h_ij + delta)
initial weight = exp(-20 * T-AAB score)
```

## Setup

```matlab
startup
```

## Demo: one synthetic experiment

```matlab
run('demo/run_one_synthetic_experiment.m')
```

## Demo: one real-data experiment

```matlab
run('demo/run_one_real_dataset.m')
```

## Compare methods on all included real-data files

```matlab
run('experiments/compare_methods_real.m')
```

## Smoke test

```matlab
run('tests/smoke_test.m')
```

The real-data evaluation uses robust trimmed-Cauchy signed-scale/translation alignment by default. Full comparison tables are written to `results/`.

## License

See `LICENSE`.
