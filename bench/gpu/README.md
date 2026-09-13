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

### Budget matching

Both libraries run at the **same** relative-residual CG tolerance, set by
`--cg-tol` (default `1e-4`): it drives libKriging's `predictIterative(tol=)`
*and* GPyTorch's `cg_tolerance` / `eval_cg_tolerance`. This has to be one
shared number. Earlier revisions of this harness hardcoded `tol=1e-8` on the
libKriging side while leaving GPyTorch at `1e-4`, i.e. compared a solver
converged four orders of magnitude tighter against a deliberately truncated
one, and reported the difference as a speed gap. At n=4000 that setting alone
accounted for a 3× difference in libKriging's predict time (0.253 s at 1e-8 vs
0.085 s at 1e-4) — for a posterior mean that was already far closer to the
exact Cholesky answer than GPyTorch's at either tolerance.

Use `--cg-tol 1e-8` to reproduce the old, non-budget-matched rows, and
`--lk-precond-rank` (default 128, `0` = off) to vary or disable the Nyström
preconditioner used by `predictIterative`.

Each run writes `results/<GPU-slug>__<CPU-slug>.html` (+ `.csv`) — the file
name encodes the machine, so results from several machines are committed
side by side (`bench/gpu/results/` is de-ignored in `.gitignore` for this).
Re-running on the same machine overwrites its file. The HTML report embeds
an interactive Plotly chart (time, log scale, vs `n`) with a dropdown to
switch between fit/logLik/predict and a legend to isolate/hide backends;
open it directly in a browser (loads Plotly from a CDN, so needs network
the first time, or vendor the script locally for offline viewing).

On a multi-GPU host, select the device with `CUDA_VISIBLE_DEVICES` and run
the sweep once per GPU: the slug comes from the *visible* device, so each
run lands in its own file rather than overwriting the previous one.

```sh
CUDA_VISIBLE_DEVICES=0 python bench/gpu/bench_gpu.py   # -> NVIDIA-L40S__...
CUDA_VISIBLE_DEVICES=1 python bench/gpu/bench_gpu.py   # -> NVIDIA-H100-NVL__...
```

Beware when comparing two GPUs: everything here is FP64, and consumer/
workstation-class parts throttle FP64 hard (L40S is ~1:64 vs FP32, i.e.
~1.4 TFLOPS, against ~34 TFLOPS on an H100). A slug-to-slug slowdown of
5-15x on *every* row, GPyTorch included, is that ratio and not a
regression.

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

Flags: `--sizes`, `--theta`, `--cg-tol`, `--lk-precond-rank`, `--backends`
(keys: `chol`, `iter-cuda`, `iter-omp`, `gpt-cuda`, `gpt-cpu`,
`gpt-chol-cuda`, `gpt-chol-cpu`), `--tag`, `--outdir`.

## Reading the output

Three timings per backend, plus accuracy vs the Cholesky reference:

* **`fit`** — solving `R(theta)` once at the fixed, given theta (no
  hyperparameter optimization anywhere in this sweep): the `Kriging(...)`
  constructor (dense Cholesky for `LL`; one CG+SLQ commit for the light
  `LLIterative` fit, no dense R factor); for GPyTorch, model/likelihood
  build **plus one forced no-grad `mll(model(x), y)` forward** (a
  Cholesky, or one BBMM CG+SLQ solve). The forced forward is needed
  because `gpytorch.models.ExactGP.__init__` itself does no linear
  algebra — it's lazy, always ~1ms flat regardless of `n` (confirmed
  empirically) — so without it, all of GPyTorch's kernel/solve cost would
  silently land in `logLik` (its first forward call) instead, making `fit`
  measure object construction rather than an actual fit.
* **`logLik`** — a further, independent log-likelihood **+ gradient**
  evaluation at `theta` (`logLikelihoodFun` / `logLikelihoodIterativeFun` /
  one `-mll().backward()`). GPyTorch's train-mode forward has no cache, so
  this genuinely re-solves `R(theta)` from scratch, same as libKriging's
  `logLikelihoodFun`/`logLikelihoodIterativeFun` re-solving independently
  of what `fit` already did.
* **`predict`** — a *cold* posterior mean on the 300-point test set
  (`predict` / `predictIterative` with a raised `max_iter` + Nyström
  preconditioner / GPyTorch `.eval()` posterior; GPyTorch's per-fit cache is
  dropped each rep).
* All timings are the **min of up to 5 reps** (1 rep once a call > 3 s).
* **`RMSE`/`Q²`** on the test set; **`dLogLik/n`** = `|ll − ll_chol|/n`
  (blank for GPyTorch — its `-mll` is differently normalised); **`dMean/rms`**
  = `max|mean − mean_chol| / rms(y_test)`, against the **libKriging** Cholesky
  mean.

### Both libraries must fit the same model, or `dMean/rms` means nothing

`dMean/rms` is only a *convergence* figure if the two libraries are solving the
same problem. Two alignments are needed, and neither is the obvious default:

* **The kernel.** `gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d)` is
  *radial after scaling* — it collapses the per-dimension offsets into
  `r = ||dx/l||₂` and evaluates one Matérn function of it — while libKriging's
  `matern5_2` (like every `covType` it offers) is **separable**,
  `∏ₖ f(|dxₖ|/θₖ)`. At `d=4, θ=0.15, dx=(0.05,0.10,0.02,0.07)`: 0.589472 vs
  0.556929. They agree only at `d=1`. The harness therefore builds a
  `ProductKernel` of one-dimensional Matérn-5/2 factors.
* **The trend.** libKriging does *universal* kriging and profiles out the
  **GLS** constant `β = (F'R⁻¹F)⁻¹F'R⁻¹y`, not `mean(y)` (its OLS counterpart).
  GPyTorch's fixed `ConstantMean` is set to libKriging's `β`.

Before these fixes, *every* GPyTorch row — including the **exact**
`GPyTorch-Cholesky` ones, which is the tell — reported `dMean/rms ≈ 4e-02`, a
model offset that drowned the ~1e-3 iterative-convergence signal, and RMSE/Q²
differed between libraries. After them, `GPyTorch-Cholesky` sits at **4.1e-08**
of the libKriging Cholesky reference and RMSE/Q² agree to six digits across all
seven backends — so the remaining `dMean/rms` on the `-BBMM` and `-Iterative`
rows is solver convergence and nothing else. Cost: about **+13% on GPyTorch's
`predict`** (`d` lazily-evaluated kernels instead of one fused one), which is
the price of a valid comparison.

The libKriging iterative objective is `LLIterative(30,0,40)` — the third
argument (40 SLQ Lanczos steps per probe) keeps the iterative log-likelihood
*value* close to exact at `theta=0.15`; see
[`docs/math/Iterative.md`](../../docs/math/Iterative.md) and
[`docs/comparisons/libKriging_vs_GPyTorch.ipynb`](../../docs/comparisons/libKriging_vs_GPyTorch.ipynb).
