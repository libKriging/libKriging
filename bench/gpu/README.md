# `bench/gpu/` — manual libKriging vs GPyTorch benchmark

`bench_gpu.py` is a **standalone, run-by-hand** benchmark (not wired into
CI). It runs one fixed sweep — `sine_sum`, d=4, matern5_2, shared
`theta=0.15`, n ∈ {250, 500, 1000, 2000} — and compares five backends,
each named `<lib>-<method>-<linalg lib>` (the linalg name is auto-detected
from the shared-library linkage / `torch.__config__`):

| backend | what it exercises |
|---|---|
| `libKriging-Cholesky-<BLAS>` | exact dense path (`objective="LL"`) — the **reference** for the log-likelihood value and posterior mean |
| `libKriging-Iterative-CUDA` | `set_cuda_iterative_enabled(True)` — device-batched CG + SLQ log-det + Hutchinson gradient (hand-written CUDA kernels, not cuBLAS) |
| `libKriging-Iterative-OpenMP` | `set_cuda_iterative_enabled(False)` — the same, on hand-written OpenMP matvec loops |
| `GPyTorch-BBMM-CUDA` | GPyTorch `ExactGP` + BBMM on `cuda`, raised CG/Lanczos/preconditioner settings |
| `GPyTorch-BBMM-<BLAS>` | same on `cpu` (torch's own BLAS named) |

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
row), plus `torch` and `gpytorch` importable from the same environment. See
[`bench/comparison-gpu/README.md`](../comparison-gpu/README.md) and the
`gpu-bench-env` project note for the build/env recipe used on the reference
machine.

```sh
# whole sweep, all five backends, auto-named output
python bench/gpu/bench_gpu.py

# quick smoke run, iterative backends only
python bench/gpu/bench_gpu.py --sizes 250,500 --backends chol,iter-cuda,iter-omp

# machine with no CUDA: the -cuda backends are dropped automatically
python bench/gpu/bench_gpu.py
```

Flags: `--sizes`, `--theta`, `--backends` (keys: `chol`, `iter-cuda`,
`iter-omp`, `gpt-cuda`, `gpt-cpu`), `--tag`, `--outdir`.

## Reading the output

Three timings per backend, plus accuracy vs the Cholesky reference:

* **`fit`** — the `Kriging(...)` constructor: dense Cholesky for `LL`; one
  CG+SLQ commit for the light `LLIterative` fit (no dense R factor);
  GPyTorch model/likelihood build (near zero).
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
