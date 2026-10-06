# GPU comparison benchmark — GPyTorch vs libKriging (n > 1000)

Repetitions per case: 1. Identical LHS designs. **Fixed shared reference theta** = `theta_frac × per-dimension input range`, `sigma2 = var(y)`. Interpolation only (jitter 1e-10). `theta_frac` small = short correlation / well-conditioned R; large = long correlation / ill-conditioned R.

Times are seconds, median [q25; q75] over repetitions. `—` = no successful run.

- **fit** = one likelihood(+grad) evaluation at theta_ref (one optimizer-step-equivalent).
- **pred mean** = posterior mean on the 2000-point test set.
- **pred stdev** = stdev on a subsample (one CG solve per point on the libKriging side).
- **update** = re-condition on n + ⌈0.25n⌉ fresh points at the same theta, then predict the test mean.

## sine_sum (d=4, n=2000, matern5_2, theta_frac=0.3)

| backend | device | fit | pred mean | pred stdev | update | RMSE | Q² | NLPD | cov90 | CG conv | ok/total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpytorch-cpu | CPU | 3.83 [3.83; 3.83] | 2.18 [2.18; 2.18] | 0.00237 [0.00237; 0.00237] | 4.71 [4.71; 4.71] | 0.02827 [0.02827; 0.02827] | 0.9996 [0.9996; 0.9996] | 0.387 [0.387; 0.387] | 1 [1; 1] | 0/1 | 1/1 |
| libkriging-cpu | — | — | — | — | — | — | — | — | — | — | 0/1 |
| libkriging-gpu | CUDA | 77.5 [77.5; 77.5] | 41.1 [41.1; 41.1] | 229 [229; 229] | 264 [264; 264] | 0.01654 [0.01654; 0.01654] | 0.9999 [0.9999; 0.9999] | -2.746 [-2.746; -2.746] | 0.95 [0.95; 0.95] | 1/1 | 1/1 |

### Reading the sweep

- **theta_frac** is the knob: small → short correlation → well-conditioned R → both sides' CG is fast; large (and small `d`) → long correlation → ill-conditioned R → libKriging's CG runs many iterations (slow but converges), GpyTorch's CG hits its iteration cap (fast but its `Q²` / `CG conv` degrade — see `CG conv 0/n`).
- Whether a given theta_frac is a *good hyperparameter* for a function is separate from the timing: at theta_frac far from the surface's natural length-scale, both packages predict poorly (RMSE/Q²), yet the fit/predict/update **timings** still compare like-for-like.
- `libkriging-cpu` is the same binary with the CUDA CG toggled off; same RMSE/Q² as `libkriging-gpu`, ~10–50× slower (the GPU-batched matvec + SLQ is what closes that gap). `gpytorch-cpu` uses `torch.device('cpu')`.
- NLPD can blow up (1e15+) when the surface has tiny amplitude and is near-perfectly interpolated: `sigma2 = var(y)` then badly under-estimates the process variance, so predicted stdev → 0. A metric artifact of the fixed-theta protocol, not a solver issue.

### Caveats

- Fixed shared theta isolates each package's **matrix-free CG linear algebra**; it is not a comparison of optimizers or of end-to-end MLE fits (see `run_gpu.py` and `docs/comparisons/libKriging_vs_GPyTorch.ipynb`).
- GPyTorch's `max_cg_iterations` / `cg_tolerance` and libKriging's `predictIterative` tol are a speed vs convergence trade-off on both sides; `CG conv` flags whether GPyTorch's CG hit its iteration cap before tolerance (libKriging's fit-side CG budget is not tunable from Python — reported as converged).
- libKriging `LLIterative` is `NoiseModel::None`-only; both sides run pure interpolation.
- The machine's GPUs are shared; absolute times carry run-to-run contention noise — compare within a run, not across.
