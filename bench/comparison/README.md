# Cross-package comparison benchmark

Benchmarks libKriging (`pylibkriging`, `rlibkriging`) against classic kriging
packages on standard response surfaces, with **identical, randomized designs**
shared across all packages and both languages.

| | Packages |
|---|---|
| Python | pylibkriging, scikit-learn, GPy, SMT, OpenTURNS |
| R | rlibkriging, DiceKriging, RobustGaSP |

## Protocol

- Test functions ([Surjanovic & Bingham](https://www.sfu.ca/~ssurjano/)):
  Branin (d=2), Hartmann-3, Hartmann-6, Borehole (d=8);
  training sizes from 50 up to 1000 points.
- Designs: Latin Hypercube (scipy `qmc`, seeded per `(n, rep)`), **10
  repetitions** by default; a common 2000-point LHS test set per function.
  Designs are generated once (`make_datasets.py`) and written to CSV, then
  consumed as-is by the Python and R runners — every package sees exactly the
  same conditioning points and test points.
- Model: Matern 5/2 anisotropic kernel, constant trend, interpolation
  (no nugget), hyperparameters by MLE with each package's default optimizer.
- Metrics: fit time, prediction time, RMSE, Q², NLPD on the test set;
  per-fit wall-clock budget (default 300 s), timeouts/errors recorded.
- Report: median [q25; q75] over repetitions, per (function, n), written to
  the GitHub Actions step summary and uploaded as an artifact.

## Running locally

```sh
cd bench/comparison
pip install "numpy<2" scipy pandas pylibkriging scikit-learn GPy smt openturns
python make_datasets.py --repeats 10          # or --quick
python run_python.py                          # results/python.csv
Rscript run_r.R data results/r.csv 300        # needs rlibkriging, DiceKriging, RobustGaSP
python aggregate.py                           # results/all.csv + summary.md
```

## CI

`.github/workflows/bench-comparison.yml` — manual `workflow_dispatch`
(inputs: `repeats`, `quick`, `budget`) plus a monthly schedule. It never runs
on push/PR (too heavy). Python is pinned to 3.11 and `numpy<2` for GPy
compatibility.

## Fairness caveats

Packages differ in optimizer, restarts, bounds and internal rescaling; this
compares *default MLE fits* under a common kernel/trend, not tuned setups.
Contributions refining per-package settings are welcome — please keep any
change symmetric across packages.
