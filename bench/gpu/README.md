# `bench/gpu/` — manual libKriging vs GPyTorch benchmark

`bench_gpu.py` is a **standalone, run-by-hand** benchmark (not wired into
CI). It runs one fixed sweep — `sine_sum`, d=4, matern5_2, shared
`theta=0.15`, n ∈ {250, 500, 1000, 2000} by default (`--sizes` to change,
e.g. to add 4000) — and compares seven backends, each named
`<lib>-<method>-<linalg lib>` (the linalg name is auto-detected from the
shared-library linkage / `torch.__config__`):

| backend | what it exercises |
|---|---|
| `libKriging-Cholesky-<BLAS>` | exact dense path (`objective="LL"`) — the **reference** for the log-likelihood value and posterior mean |
| `libKriging-Iterative-CUDA` | `set_cuda_iterative_enabled(True)` — device-batched CG + SLQ log-det + Hutchinson gradient; materializes R (or `dR/dtheta_k`) once per evaluation for a separable kernel within `LK_ITERATIVE_CUDA_DENSE_MAX_MB` and matvecs via `cublasDgemm`, else a hand-written CUDA kernel |
| `libKriging-Iterative-OpenMP` | `set_cuda_iterative_enabled(False)` — the same, materializing R within `LK_ITERATIVE_DENSE_MAX_MB` and using BLAS-3 `R*V`, else a hand-written OpenMP matvec loop |
| `GPyTorch-BBMM-CUDA` / `-<BLAS>` | GPyTorch `ExactGP`, raised CG/Lanczos/preconditioner settings and `max_cholesky_size=0` (GPyTorch's default, 800, silently uses exact Cholesky at/below it — forced off so every `n` here genuinely runs BBMM), on `cuda` / `cpu` |
| `GPyTorch-Cholesky-CUDA` / `-<BLAS>` | the SAME model, `max_cholesky_size` forced far above every `n` instead, so GPyTorch always solves exactly — a second reference (independent of libKriging's) for whether BBMM has converged |

`GPyTorch-BBMM-*` and `GPyTorch-Cholesky-*` only ever differ in that one
setting, so comparing them directly checks BBMM convergence without
involving the covariance-argument-convention offset from libKriging's own
Cholesky reference (see `dMean/rms` and the Verdict section below).

`theta=0.15` is the largest length-scale at which the SLQ log-determinant's
Lanczos quadrature, `predictIterative`'s CG and GPyTorch's BBMM CG all
converge with sane iteration budgets — so this is an "everything actually
converges" comparison.

Each run writes `results/<GPU-slug>__<CPU-slug>.md` (+ `.csv`) — the file
name encodes the machine, so results from several machines are committed
side by side (`bench/gpu/results/` is de-ignored in `.gitignore` for this).
Re-running on the same machine overwrites its file.

## Running it

Needs `pylibkriging` (built with `-DENABLE_CUDA_ITERATIVE=ON` for the CUDA
row), plus `torch` and `gpytorch` importable from the same environment.

```sh
# whole sweep, all seven backends, auto-named output
python bench/gpu/bench_gpu.py

# quick smoke run, iterative backends only
python bench/gpu/bench_gpu.py --sizes 250,500 --backends chol,iter-cuda,iter-omp

# machine with no CUDA: the -cuda backends are dropped automatically
python bench/gpu/bench_gpu.py
```

Flags: `--sizes`, `--theta`, `--backends` (keys: `chol`, `iter-cuda`,
`iter-omp`, `gpt-cuda`, `gpt-cpu`, `gpt-chol-cuda`, `gpt-chol-cpu`),
`--tag`, `--outdir`.

## Reading the output

Three timings per backend, plus accuracy vs the Cholesky reference:

* **`fit`** — the `Kriging(...)` constructor: dense Cholesky for `LL`; one
  CG+SLQ commit for the light `LLIterative` fit (no dense R factor);
  GPyTorch model/likelihood build. **Not comparable across libraries as a
  standalone column**: `gpytorch.models.ExactGP.__init__` does no linear
  algebra — it's lazy, so it's always ~1ms flat regardless of `n`
  (confirmed empirically) — while libKriging's constructor eagerly
  factorizes/CG-fits. All the kernel/solve/backward cost GPyTorch defers
  out of `fit` shows up in its `logLik` (first forward call) instead;
  compare `fit + logLik` per backend for a fair "time to a log-likelihood
  value" total.
* **`logLik`** — one log-likelihood **+ gradient** evaluation at `theta`
  (`logLikelihoodFun` / `logLikelihoodIterativeFun` / one `-mll().backward()`).
* **`predict`** — a *cold* posterior mean on the 300-point test set
  (`predict` / `predictIterative` with a raised `max_iter` + Nyström
  preconditioner / GPyTorch `.eval()` posterior; GPyTorch's per-fit cache is
  dropped each rep).
* All timings are the **min of up to 5 reps** (1 rep once a call > 3 s).
* **`RMSE`/`Q²`** on the test set; **`dLogLik/n`** = `|ll − ll_chol|/n`
  (blank for GPyTorch — its `-mll` is differently normalised); **`dMean/rms`**
  = `max|mean − mean_chol| / rms(y_test)`.

The libKriging iterative objective is `LLIterative(30,0,40)` — the third
argument (40 SLQ Lanczos steps per probe) keeps the iterative log-likelihood
*value* close to exact at `theta=0.15`; see
[`docs/math/Iterative.md`](../../docs/math/Iterative.md) and
[`docs/comparisons/libKriging_vs_GPyTorch.ipynb`](../../docs/comparisons/libKriging_vs_GPyTorch.ipynb).
