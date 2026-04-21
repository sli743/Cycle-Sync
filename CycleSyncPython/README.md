# CycleSync Python

A GitHub-ready Python repository for testing Cycle-Sync camera location synchronization on synthetic and precomputed real-data inputs.

Included methods:

- Cycle-Sync with T-AAB initialization and cycle-message reweighting.
- LUD.
- ShapeFit.
- BATA.
- Full FusedTA baseline port.
- Uniform and cycle-consistent/adversarial synthetic corruption models.
- Robust signed-scale/translation alignment for evaluation.

The repository contains small precomputed ETH3D-style `.mat` location inputs under `data/real_precomputed/`. These files include the location-estimation graph, pairwise directions, ground-truth locations, and global rotation alignment; they do not include raw images.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo: one synthetic experiment

```bash
python demo/run_one_synthetic_experiment.py --n 100 --p 0.5 --model uniform --q 0.8 --sigma 0
```

For a quick smoke run:

```bash
python demo/run_one_synthetic_experiment.py --n 30 --p 0.6 --model uniform --q 0.4 --sigma 0 --fast
```

## Demo: one real-data experiment

```bash
python demo/run_one_real_dataset.py data/real_precomputed/pipes_location\(2\).mat --alignment robust --fast
```

`--alignment robust` uses trimmed-Cauchy signed-scale/translation alignment before computing all-camera errors. This prevents a few extreme outlier cameras from distorting the fitted global scale and translation.

## Compare methods on real data

```bash
python experiments/compare_methods_real.py \
  --data-dir data/real_precomputed \
  --out-dir results/real_method_comparison \
  --alignment robust \
  --fast
```

## Reproduce a small synthetic sweep

```bash
python experiments/reproduce_synthetic_fig3_small.py --n 100 --p 0.5 --model uniform --sigma 0 --qlist 0.2 0.4 0.6 0.8 --trials 10
```

## Default Cycle-Sync parameters

```text
tmax = 20
beta = 20
lambda_t = t/(t+10)
delta = 1e-8
rho(x) = 1 - exp(-4 |x|)
w_ij = exp(-4 h_ij)/(h_ij + delta)
initial weight = exp(-20 * T-AAB score)
```

These defaults are set directly in `cyclesync/cyclesync.py` through `CycleSyncParams`.

## Tests

```bash
pytest -q
```

The Python WLS solver uses an active-set formulation of the same constrained WLS subproblem used by the MATLAB implementation. For a fixed active set of edges with `alpha_ij=1`, free `alpha_ij` variables are eliminated by perpendicular projection; the remaining centered linear system is solved with the gauge `sum_i t_i=0`.

## License

See `LICENSE`.
