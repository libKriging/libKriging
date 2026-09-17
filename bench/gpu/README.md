# `bench/gpu/` — manual libKriging vs GPyTorch benchmark

`bench_gpu.py` is a **standalone, run-by-hand** benchmark (not wired into
CI). It runs one fixed sweep — `sine_sum`, d=4, matern5_2, shared
`theta=0.15`, n ∈ {250, 500, 1000, 2000, 4000, 8000} by default (`--sizes` to
change) — and compares seven backends, each named
`<lib>-<method>-<linalg lib>` (the linalg name is auto-detected from the
shared-library linkage / `torch.__config__`):

The exact-Cholesky backends (`libKriging-Cholesky-*`, `GPyTorch-Cholesky-*`
— `CHOL_FAMILY_KEYS`) are capped at n=8000 regardless of `--sizes`: an O(n³)
dense factorization is not meant to scale past that. Two GPU backends go
further (`SWEEP_EXTRA_SIZES_BY_KEY`), independently capped:
`libKriging-Iterative-CUDA` up to **n=32000**, `GPyTorch-BBMM-CUDA` up to
**n=16000** only. GPyTorch stops at 16000 because n=32000 hits a hard `CUDA
out of memory` building its BBMM pipeline (~78 GiB, confirmed independent
of the CG iteration budget — giving it 38x more iterations changed
nothing) on a card libKriging's matrix-free kernels use ~1.5 GiB on at the
same n; not something this script's settings can work around.
`libKriging-Iterative-CUDA` reaches n=32000 only with
`LK_ITERATIVE_CUDA_DENSE_MAX_MB` raised well past its 4096 MiB default (the
`d` `dR/dtheta_k` blocks alone need ~31 GiB at n=32000, d=4 — see
`docs/math/Iterative.md`), e.g.:

```sh
LK_ITERATIVE_CUDA_DENSE_MAX_MB=40960 CUDA_VISIBLE_DEVICES=1 python bench/gpu/bench_gpu.py
```

Both GPU iterative backends' `max_cg_iterations` (`LK_ITER_CG_MAX_ITER_MULT
* n`, same multiplier as libKriging's own `cg_max_iter_mult`) scale with
`n` too — GPyTorch's old fixed 5000-iteration cap left CG at an average
residual norm of 0.857 against a 1e-4 target at n=32000 (essentially
unsolved, not just under-tolerance), which is why the shared n=16000 row is
what makes libKriging vs. GPyTorch there an apples-to-apples comparison of
the same iteration budget. The CPU (OpenMP/MKL) iterative backends stay
capped at the base `--sizes`: their unbatched O(n²)-per-iteration matvec
makes n=16000/32000 impractically slow for a by-hand sweep. No `chol` row
exists at n>8000, so there is no reference log-likelihood/posterior-mean to
diff against: `dMean/rms`/`dLogLik/n` are simply absent there, and GPyTorch
falls back to a `mean(y)` trend instead of
libKriging's GLS `beta` (only computed by the `chol` row) — pure scalability
numbers past n=8000, not an accuracy comparison.

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

`bench/bench_gpu.sh` wraps `bench_gpu.py` for this: it picks (or configures) a
build dir, rebuilds `_pylibkriging` if the C++ sources are newer than what's
built, wires up `LD_LIBRARY_PATH`/`PYTHONPATH`, checks `numpy`/`torch`/
`gpytorch` are importable (erroring out with the `pip install` to run if not —
it won't guess a CUDA wheel for you), and forwards every argument to
`bench_gpu.py`:

```sh
# whole sweep, all seven backends, auto-named output
./bench/bench_gpu.sh

# quick smoke run, iterative backends only
./bench/bench_gpu.sh --sizes 250,500 --backends chol,iter-cuda,iter-omp

# machine with no CUDA: the -cuda backends are dropped automatically
./bench/bench_gpu.sh

# pick a build dir / interpreter explicitly, or skip the build check
BUILD_DIR=./build_cuda_iterative PYTHON_BIN=/usr/bin/python3.12 ./bench/bench_gpu.sh
SKIP_BUILD=1 ./bench/bench_gpu.sh
```

Calling `bench_gpu.py` directly (e.g. from a notebook or a differently-set-up
environment) still works the same way it always did — the wrapper only
automates the env plumbing:

```sh
python bench/gpu/bench_gpu.py --sizes 250,500 --backends chol,iter-cuda,iter-omp
```

Flags: `--sizes`, `--theta`, `--cg-tol`, `--lk-precond-rank`, `--backends`
(keys: `chol`, `iter-cuda`, `iter-omp`, `gpt-cuda`, `gpt-cpu`,
`gpt-chol-cuda`, `gpt-chol-cpu`), `--tag`, `--outdir`.

By default every backend is capped to exactly `--sizes` (the `*-Cholesky-*`
backends are further capped to n<=8000 regardless, see `CHOL_FAMILY_KEYS`).
Pass `--extra-sizes` to additionally run `iter-cuda` at n=16000/32000 and
`gpt-cuda` at n=16000 (`SWEEP_EXTRA_SIZES_BY_KEY`) — these rows can take well
over an hour and need tens of GiB of GPU memory, so they're opt-in.

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
* **`RMSE`/`Q²`** on the test set; **`dLogLik/n`** = `|ll − ll_chol|/n`;
  **`dMean/rms`** = `max|mean − mean_chol| / rms(y_test)`, against the
  **libKriging** Cholesky mean.

### The two log-likelihood columns

`logLik (native)` is what each library returns as-is, and the two sides look
nothing alike — e.g. `2183.8` against `−0.29` at n=4000. That gap is **pure
convention**, not disagreement:

* GPyTorch's `ExactMarginalLogLikelihood` returns `log p(y|X) / n`, a
  **per-datapoint** figure.
* libKriging returns the **concentrated** likelihood: `σ²` is profiled out
  analytically, so the quadratic form `q = r′K⁻¹r` is replaced by
  `n·log(q/n) + n`.

`logLik (libK conv.)` undoes both, via
`ll_libK = n·mll + q/2 − (n·log(q/n) + n)/2`, with `q` taken from GPyTorch's
*own* solve so a BBMM row keeps its own CG error rather than borrowing
libKriging's. Verified: the exact `GPyTorch-Cholesky` rows land within
**5e-09 … 2e-08** per point of the libKriging Cholesky reference. This is the
column `dLogLik/n` is computed from.

Making it comparable exposed something the previously-blank column hid. At an
**identical** stochastic budget — 30 Hutchinson probes and 40 SLQ Lanczos
steps per probe on both sides (GPyTorch's `num_trace_samples` /
`max_lanczos_quadrature_iterations` are set from libKriging's `LLIterative`
fields, not left at their own defaults) and the same CG tolerance —
libKriging's log-determinant estimate is roughly **two orders of magnitude**
more accurate than GPyTorch BBMM's:

| n | libKriging-Iterative `dLogLik/n` | GPyTorch-BBMM `dLogLik/n` |
|--:|--:|--:|
| 250 | 4.1e-03 | 2.2e-02 |
| 500 | 9.0e-03 | 2.1e-03 |
| 1000 | 2.0e-03 | 3.6e-01 |
| 2000 | 2.1e-03 | 2.7e-01 |
| 4000 | 1.3e-02 | 1.6e-01 |

Both estimators are stochastic, so individual cells move between runs (note
GPyTorch's n=500, which happened to land well) — but the *envelope* is stable
and reproduces on both the H100 and the L40S: libKriging stays in the low
`1e-03`…`1e-02` band at every n, GPyTorch BBMM reaches `1e-01` at n ≥ 1000.

The two libraries are better at different things: BBMM's posterior **mean** is
slightly closer to exact (`dMean/rms` 2.6e-04 vs 6.5e-04 at n=4000),
libKriging's **log-likelihood** markedly so — which matters, because the
log-likelihood is what hyperparameter optimization actually descends, and this
sweep deliberately holds `theta` fixed and so never exercises that.

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

#### The actual GPyTorch model code

Both alignments live in `_gpt_modules()` / `run_gpytorch()` in `bench_gpu.py`.
The kernel — a `ProductKernel` of 1-D Matérn-5/2 factors, one per input
dimension, instead of `MaternKernel(ard_num_dims=d)`:

```python
d = train_x.shape[1]
base = gpytorch.kernels.ProductKernel(
    *[gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[k]) for k in range(d)]
)
self.covar_module = gpytorch.kernels.ScaleKernel(base)
```

The trend — `ConstantMean` pinned to libKriging's GLS `beta` (from the
Cholesky reference fit), not `mean(y)`:

```python
model.mean_module.constant.fill_(
    float(np.mean(y)) if trend_const is None else float(trend_const))
```

`trend_const` is `m.beta()` read off the `libKriging-Cholesky-*` run for the
same `n` (see `one_point`/`beta_ref` in `bench_gpu.py`), so every GPyTorch
backend is fit against the exact same trend constant, not its own estimate.

`fit` is forced to actually solve `R(theta)` once — `ExactGP.__init__` alone
does no linear algebra, so without this the cost silently landed in the first
`logLik` call instead:

```python
model, lik = build()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
model.train(); lik.train()
with torch.no_grad():
    mll(model(tx), ty)  # forces the R(theta) solve/logdet now, not on first logLik call
```

BBMM vs. exact Cholesky is a single setting (everything else — CG/Lanczos
budgets, preconditioner size, `cg_tolerance`/`eval_cg_tolerance` tied to
`--cg-tol` — held fixed between the two, see `_gpt_converged_ctx()`):

```python
gpytorch.settings.max_cholesky_size(0 if bbmm else 1_000_000)
```

`0` means "never small enough to fall back to exact Cholesky" (GPyTorch's own
default, 800, would silently do that at every `n` in this sweep); a huge value
means "always small enough", forcing the exact solve GPyTorch-Cholesky is
supposed to be.

The libKriging iterative objective is
`LLIterative(30,0,40,6,{cg_tol},{probes_cg_tol})` — the third argument (40 SLQ
Lanczos steps per probe) keeps the iterative log-likelihood *value* close to
exact at `theta=0.15`; the fourth raises the CG iteration budget to `6n`
(needed for the probe solve to converge at all up to `n=8000`); the fifth is
`--cg-tol`, shared with GPyTorch and `predictIterative` for a fair comparison;
the sixth is `--lk-probes-cg-tol` (default `1e-2`, libKriging-only — no
GPyTorch equivalent) which loosens ONLY the gradient's Hutchinson-probe CG
solve, the term that otherwise dominates `logLik`'s cost at large `n` (its CG
iteration count grows far faster with `n` than the `[F|y]` solve's — see
`docs/math/Iterative.md`) with no measurable effect on `predictIterative`'s
own accuracy, unlike loosening the shared `--cg-tol`. See
[`docs/math/Iterative.md`](../../docs/math/Iterative.md) and
[`docs/comparisons/libKriging_vs_GPyTorch.ipynb`](../../docs/comparisons/libKriging_vs_GPyTorch.ipynb).
