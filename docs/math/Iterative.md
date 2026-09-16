# Matrix-free CG + stochastic log-det fit (`objective="LLIterative(m)"`)

## Idea

Unlike `LLVecchia`/`LLNystrom` (which each replace R by a cheaper
*structured* approximation — local conditioning / global low rank),
`LLIterative` keeps R itself exact: every term of the concentrated
log-likelihood except `log|R|` is computed via conjugate gradient
(`LinearAlgebra::conjugateGradientBatched` — all right-hand sides share
one matvec per iteration) instead of a dense O(n³) Cholesky
factorization, mathematically the same quantity a full factorization
would give (up to CG's own convergence tolerance) but computed via
O(n²) matvecs — matrix-free by default, or against a `R` materialized
once per evaluation when it fits (see *Dense fast path* below). `log|R|` — the one term CG cannot produce
directly — is replaced by a Stochastic Lanczos Quadrature (SLQ) estimate,
and the gradient's `trace(R⁻¹ ∂R/∂θₖ)` term by a Hutchinson estimate
sharing the same probe vectors. This is the same overall strategy as
GPyTorch's BBMM/Lanczos-based inference (see
[libKriging_vs_GPyTorch.ipynb](../comparisons/libKriging_vs_GPyTorch.ipynb)).

```r
k <- Kriging(y, X, "matern5_2", objective = "LLIterative(30)")   # or "LLIterative"
```

Where this sits relative to the other scaling methods:

- **`LLVecchia`/`LLNystrom`** trade exactness for a cheaper R — the
  objective itself is only an approximation of the true concentrated
  log-likelihood, but each evaluation is genuinely O(n·m³)/O(n·k²).
- **`LLIterative`** keeps the true R throughout (β/σ²/the quadratic
  form in the gradient are exact, up to CG's tolerance); the only
  approximation is the stochastic log-determinant. Each evaluation is
  still O(n²) per CG iteration though (R is exact, not structured), so
  it doesn't reduce the matvec cost the way Vecchia/Nystrom do — its
  payoff is avoiding the O(n³) dense factorization and O(n²) memory,
  same rationale as [`predictIterative`](PredictIterative.md) but applied to *fit*
  rather than just predict.

## Mathematical description

- **CG solves for β/σ²**: one batched CG call solves `R⁻¹·[F | y]`
  together (β̂'s design matrix F has p ≤ a few columns, so this is p+1
  independent Krylov solves sharing the same matvec — not block-CG
  subspace sharing, but far cheaper than p+1 separate O(n³)
  factorizations either way). β̂ and σ̂² then follow the usual GLS
  formulas, same as the exact objective.
- **SLQ log-determinant**: `LinearAlgebra::stochasticLogDetBatched`
  estimates `log|R| ≈ (n/nprobe) Σᵢ zᵢᵀ log(R) zᵢ` via a
  reorthogonalized Lanczos quadrature on each Rademacher probe `zᵢ`
  (Ubaru, Chen & Saad 2017) — R is only ever accessed through matvecs.
  Every probe's Lanczos recurrence is advanced **in lockstep** so the
  matvec `R·[z₁|…|z_nprobe]` is evaluated once per step on the whole
  block, sharing a single covariance sweep across all probes — the same
  batched matvec engine (`LinearAlgebra::conjugateGradientBatched` /
  the GPU backend) used by the CG solves. (An earlier attempt to read
  the quadrature straight off the CG scalars — true "matrix-free BBMM"
  mBCG, no extra matvecs — was dropped: without full reorthogonalization
  the reconstructed tridiagonal loses accuracy as `cond(R)` grows, and
  restoring orthogonality *is* running Lanczos again.)
- **Hutchinson gradient trace term**: the envelope-theorem gradient
  (same principle as `_logLikelihoodVecchia`/`_logLikelihoodNystrom` —
  β̂/σ̂²'s own θ-dependence doesn't contribute at their profiled values)
  needs `trace(R⁻¹ ∂R/∂θₖ)` per parameter. This is a Hutchinson estimate
  `trace(R⁻¹ ∂R/∂θₖ) ≈ mean_p(wₚ · (∂R/∂θₖ · zₚ))` where `wₚ = R⁻¹zₚ`
  comes from **one more batched CG call** (all probes solved together),
  reusing the exact same probe vectors as the SLQ log-determinant.
  **These two are independent stochastic estimators of related but
  distinct quantities**, not exact derivatives of one another (unlike
  `LLNystrom`'s Woodbury identities, which are exact given the fixed
  landmarks): SLQ is a *truncated* Lanczos quadrature of `zᵀ log(R) z`,
  the gradient's trace term is a plain Hutchinson estimate using the
  CG-exact `R⁻¹`. Don't expect a tight finite-difference match between
  the analytic gradient and a numerical differentiation of the SLQ
  objective — check order-of-magnitude/sign agreement instead (see
  `KrigingIterativeTest.cpp`).
- **Fixed probes, smooth objective**: like `LLNystrom`'s landmarks,
  the `nprobe` Rademacher probes are drawn ONCE per fit (fixed seed,
  `make_iterative_probes`) and held fixed across every θ evaluation
  during optimization — re-drawing fresh probes at every evaluation
  would make the objective (and its gradient) noisy/non-smooth between
  optimizer iterations.
- **Optional CG preconditioner**: `objective="LLIterative(m,precond_rank)"`
  opts into the same Nystrom/Woodbury preconditioner as
  [`predictIterative`](PredictIterative.md#preconditioning): a rank-`precond_rank`
  Nystrom factor built from a FIXED landmark set (chosen once, same
  greedy pivoted-Cholesky selection as `LLNystrom`'s landmarks, at a
  θ-neutral reference kernel) is passed as `Pinv` to both CG calls,
  fewer Krylov iterations needed to reach `tol` on the typically
  ill-conditioned R. Unlike `predictIterative` (which only ever
  preconditions-solves at one fixed, already-fitted θ*), the
  preconditioner here is **rebuilt from the fixed landmarks at the
  current θ on every objective/gradient evaluation** — the landmark
  *set* is what's held fixed (for the same θ-smoothness reason as
  `LLNystrom`'s landmarks), not the factorization itself. Off by
  default (`precond_rank` omitted or 0).
  The preconditioner is also applied to the **SLQ log-determinant**, not
  just the CG solves: the Lanczos quadrature then runs on the whitened
  operator `R̃ = L⁻¹ R L⁻ᵀ` where `L L' = P = D + U U'` (a square,
  non-symmetric factor whose inverse applies in O(n·precond_rank) via
  `WoodburyFactorization::whitenL`/`whitenLt` — a thin-QR + k×k
  eigendecomposition of the low-rank structure, never an n×n factor),
  and `log|R| = log|P| + log|P⁻¹R|` with `log|P|` added back exactly
  (`LinearAlgebra::woodbury_logdet`). `R̃` is far better conditioned than
  `R`, so its quadrature needs fewer Lanczos steps for the same
  accuracy — preconditioning tightens the log-determinant bias too, not
  only the solve iteration count. (The Hutchinson trace still needs
  plain isotropic probes, so it keeps its own separate preconditioned
  solve on the unwhitened probes.)
- **SLQ Lanczos steps**:
  `objective="LLIterative(m,precond_rank,lanczos_steps)"` sets the number
  of Lanczos steps per probe in the stochastic log-determinant estimate
  (default 20). The Lanczos quadrature converges to the exact `log|R|` as
  `lanczos_steps → n`; 20 steps under-resolve the spectrum of a strongly
  ill-conditioned R, biasing `log|R|` (and hence the concentrated
  log-likelihood *value* — the gradient's Hutchinson trace is a separate,
  less affected estimator). Raise it (e.g. `LLIterative(30,0,40)`) when
  the iterative log-likelihood drifts from the exact `"LL"` objective at
  large `n` / long θ; the cost per probe grows ~linearly in
  `lanczos_steps` (plus an `O(lanczos_steps²)` tridiagonal `eig_sym`,
  negligible). A non-zero `precond_rank` (2nd arg) *also* reduces this
  bias — the quadrature then runs on the well-conditioned whitened `R̃`
  (see the preconditioner bullet) — so `LLIterative(m,precond_rank)` and
  `LLIterative(m,0,lanczos_steps)` are two independent levers on the same
  drift; how much the preconditioner helps depends on how well the
  fixed-rank Nystrom `P` approximates `R` (little at small `n`, where the
  θ-neutral reference kernel yields few above-tolerance pivots; more as
  `n` grows — which is the regime the iterative path is for).
- **CG iteration budget / non-convergence**: every CG solve here (the
  `[F|y]` solve and, when a gradient is requested, the Hutchinson probe
  solve) defaults to `max_iter = 2n`, `tol = 1e-4` (see the next bullet). That budget is not a
  law of nature: at a long enough θ relative to `n` (`R` ill-conditioned
  enough), the true number of CG iterations needed for `tol` can exceed
  `2n` — observed directly at `n=8000` in `bench/gpu`'s sweep, where the
  probe solve still hadn't reached `tol` at the full `2n=16000` budget.
  `objective="LLIterative(m,precond_rank,lanczos_steps,cg_max_iter_mult)"`
  raises the budget to `cg_max_iter_mult * n` (default 2, so
  `LLIterative(30,0,40,6)` gives a 6n budget) for exactly this case.
  Hitting the budget without converging is no longer silent: every CG
  solve (CPU and every GPU backend) prints a `[WARNING]` via
  `LinearAlgebra::cgNonConvergenceWarning` (`LinearAlgebra::warn_cg`,
  default on; `LinearAlgebra::set_cg_warning(false)` to opt out) and
  `Kriging::iterative_cg_converged()` /
  `iterative_cg_n_unconverged()` let a caller check this
  programmatically after `logLikelihoodIterativeFun`/`logLikelihoodFun`
  instead of parsing stdout. Only the `[F|y]`/probe solves are tracked
  this way; `predictIterative`'s own CG solve still gets the printed
  warning (same underlying function) but has no dedicated accessor yet.
- **CG tolerance**:
  `objective="LLIterative(m,precond_rank,lanczos_steps,cg_max_iter_mult,cg_tol)"`
  sets the
  relative-residual tolerance of the solves (default `1e-4`). This default
  is deliberately loose, and it is loose for the same reason GPyTorch's
  `cg_tolerance` is: the log-determinant it sits next to is a *stochastic*
  SLQ estimate carrying a percent-level bias, so the linear solves are not
  the accuracy-limiting step. Measured at `n = 4000`, `d = 4`,
  `matern5_2`, θ = 0.15 (H100, CUDA backend):

  | `cg_tol` | CG iterations (30 probes) | evaluation | `|ll−ll_exact|/|ll_exact|` |
  |---|--:|--:|--:|
  | `1e-8` | 6550 | 0.94 s | 6.6e-03 |
  | `1e-6` | — | 0.66 s | 6.6e-03 |
  | `1e-4` (default) | ~100 | 0.40 s | 6.6e-03 |
  | `1e-2` | — | 0.22 s | 4.2e-03 |

  i.e. eight orders of magnitude of extra residual accuracy buy nothing at
  all in the returned log-likelihood, and cost 2.3x in time. Tighten it
  only if you have first made the log-determinant itself exact enough that
  the solves start to matter (more `lanczos_steps`, or a preconditioner).
- **A separate tolerance for the probe solve**:
  `objective="LLIterative(m,precond_rank,lanczos_steps,cg_max_iter_mult,cg_tol,probes_cg_tol)"`.
  `cg_tol` (5th field) is shared by default between the `[F|y]` solve and
  the gradient's Hutchinson-probe solve (`W = R⁻¹·probes`), but these two
  do NOT need the same iteration count to reach the SAME relative-residual
  tolerance: CG iterations-to-tolerance depend on the right-hand side's
  spectral content, not just `cond(R)`. `F`/`y` are smooth and project
  mostly onto `R`'s dominant eigenmodes; the isotropic Rademacher probes
  don't, and this gap widens as `R` gets more ill-conditioned (fixed θ,
  growing `n` in a fixed-volume domain). Measured on an H100, `d=4`,
  `matern5_2`, θ=0.15, `cg_tol=1e-4`:

  | `n` | `[F\|y]` CG iterations | probe-solve CG iterations |
  |--:|--:|--:|
  | 2000 | 180 | 900 |
  | 4000 | 400 (×2.2) | 2270 (×2.5) |
  | 8000 | 1100 (×2.75) | 13470 (×5.9) |

  so at `n=8000` the probe solve alone dominates the evaluation's wall
  time. `probes_cg_tol` (6th field, defaults to `cg_tol` when omitted) lets
  it be loosened on its own — it feeds the SAME stochastic Hutchinson trace
  whose own sampling error already swamps a tight tolerance (same argument
  as `cg_tol` above), so there is no accuracy reason to keep it as tight as
  `[F|y]`'s. Unlike `cg_tol`, do NOT loosen it via `predictIterative`'s own
  `tol` parameter -- that solve has no equivalent stochastic floor and its
  accuracy is exactly its CG tolerance.
- **Dense fast path (CPU)**: strictly matrix-free (R never stored) is the
  fallback, not the only mode. For a *separable* kernel — `gauss`, `exp`,
  `matern3_2`, `matern5_2`, i.e. `Cov(dx,θ) = exp(-Σₖ sₖ(|dxₖ|/θₖ))` — and
  an `n` whose dense `n×n` `R` (plus the `d` `∂R/∂θₖ` blocks when a
  gradient is wanted) fits a memory budget (`LK_ITERATIVE_DENSE_MAX_MB`,
  default 6144; `0` forces matrix-free), `_logLikelihoodIterative`
  materializes `R` **once** with an inlined, OpenMP-parallel symmetric
  build and then runs every CG iteration / Lanczos step as a BLAS-3
  `R·V`. The dozens-to-hundreds of transcendental covariance sweeps a
  single objective+gradient evaluation would otherwise do collapse to
  one; results are unchanged to the SLQ/Hutchinson noise floor (only the
  matvec summation order differs). This is the regime the iterative path
  targets anyway — `n` large enough that a *resident* Cholesky factor is
  the problem, not `O(n²)` scratch that a dense fit would also allocate.
  Non-separable kernels and `n` past the budget stay on the matrix-free
  loops.
- **Dense fast path (CUDA)**: the same idea, independently, on the device
  side. `LinearAlgebraCuda::conjugateGradient`/`rmulBatched`/`dRmulBatched`
  materialize `R` (or the `dR/dtheta_k` blocks) ONCE per call with a
  dedicated `build_cov_kernel` — one CUDA thread per `(i,j)` pair, no
  symmetry trick needed since GPU parallelism is cheap — then every matvec
  is a single `cublasDgemm` against it, instead of the hand-written
  `rmul_batched_kernel`/`drmul_batched_kernel` recomputing every
  transcendental on every CG iteration / Lanczos step. Governed by its own
  memory budget, `LK_ITERATIVE_CUDA_DENSE_MAX_MB` (default 4096 MiB,
  independent of the CPU path's host-RAM budget). This is what actually
  makes the GPU faster than the CPU dense path at moderate-to-large `n` --
  before it, `libKriging-Iterative-CUDA` was *slower* than
  `libKriging-Iterative-OpenMP` (dense) at every `n` in `bench/gpu`'s
  sweep, because the hand-written CUDA kernel paid the same
  recompute-every-iteration cost the CPU path used to. Measured ~25x
  faster at `n=2000`, ~114x at `n=4000` (results unchanged to the SLQ
  noise floor); `dRmulBatched`'s per-`k` `cublasDgemm` writes straight
  into its interleaved output slot via a `ldc = dimX*n` leading dimension,
  no extra scatter kernel. The `cublasDgemm` call is fp64 throughout, same
  as everything else here — the dominant matvec is now the only place left
  where a **TF32/fp32 matvec + fp64 residual correction** (mixed-precision
  CG, using the existing 50-iteration exact-residual restart as the
  correction step) could plausibly help further on an H100, at the cost of
  actually changing the numerics rather than just how they're computed
  (unlike every fast path above, which is bit-identical-to-noise-floor by
  construction). Deliberately not done here — flagged as a candidate
  follow-up, not started.
- **CUDA batched CG scalar/vector kernels stay hand-written, not cuBLAS**:
  `lk_cuda_batched_dot/axpy/update_p_launch` are custom kernels, not
  `cublasDdot`/`cublasDaxpy` — tried once before (see the git history this
  project's own `CMakeLists.txt` comment references) and reverted, because
  looping ncols separate per-column cuBLAS calls paid ncols launches for
  BLAS-1-sized work. With the dense fast path's `dgemm` now the dominant
  O(n²·ncols) cost per iteration, these three are O(n·ncols) — well under
  1% of an iteration at the `n` this sweep covers — so there is no case
  for revisiting that decision.
- **Cost model**: a gradient evaluation is a CG solve over `nprobe`
  right-hand sides (each up to `2n` Krylov iterations, each an O(n²)
  matvec on the dense-fast-path `R` or a matrix-free covariance sweep,
  or cheaper per-iteration with the preconditioner enabled but at extra
  O(n·precond_rank²) setup cost per evaluation). A free BFGS
  fit multiplies that by however many iterations BFGS needs against a
  somewhat noisy stochastic objective surface — this can get expensive
  fast for anything beyond small/moderate `n`/`nprobe`. Prefer
  `optim="none"` with a fixed θ whenever a test isn't actually about
  optimizer convergence.
- **`optim="none"` never sets the light-fit flag**: exactly like
  `LLNystrom`/`LLVecchia`, `m_iterative_light` (and everything gated
  behind it — `predict` routing, blocked `simulate`/`update_simulate`/`save`)
  is only set on the actual multistart-BFGS commit path. A fixed-theta
  `optim="none"` fit always does the plain exact O(n³) factorization
  regardless of objective — existing, consistent behavior across
  Nystrom/Vecchia/Iterative.
- **Incremental `update`**: `updateIterative` mirrors `update_nystrom`'s
  strategy — extend `m_X`/`m_y`/`m_F` with the new rows (the fixed
  `precond_rank`-landmark set, if enabled, stays valid since rows are only
  ever appended), redraw `m_iterative_probes` at the new n (they're sized
  per-point, unlike the landmarks, so can't just be left as-is), then either
  re-profile β/σ² at the current θ (`refit=false`) or first do a
  warm-restart single BFGS from the current θ over the same (fixed) probes
  and landmarks (`refit=true`) before re-profiling. No O(n³)/O(n²) matrix is
  ever built by this path.

## Usage

```r
library(rlibkriging)

set.seed(1)
n <- 1000
X <- matrix(runif(2 * n), ncol = 2)
y <- sin(3 * X[, 1]) * cos(3 * X[, 2]) + rnorm(n, sd = 0.05)

# Exact objective would need an O(n^3) factorization at every theta
# evaluated during BFGS; LLIterative solves R exactly via CG instead of
# factorizing it, and only approximates the log-determinant.
k <- Kriging(y, X, "matern5_2", objective = "LLIterative(30)")

# Opt into a rank-50 Nystrom-preconditioned CG for both fit solves:
k2 <- Kriging(y, X, "matern5_2", objective = "LLIterative(30,50)")

Xnew <- matrix(runif(2 * 10), ncol = 2)
pred <- predict(k, Xnew, stdev = TRUE)   # routes to predictIterative (light fit)
```

## Current limitations (v1)

- `NoiseModel::None` only (no nugget/noise channel).
- Permanent light fit like `LLNystrom`/`LLVecchia`: `predict()` routes to
  `predictIterative`. `update()` has its own incremental path (`updateIterative`,
  see above); `simulate`/`update_simulate`/`save` are still intentionally
  **blocked** — simulating from a matrix-free model would need a genuine
  stochastic sampling technique (e.g. Lanczos-based sampling) rather than
  the explicit covariance square root the exact/Nystrom `simulate` paths
  use, and isn't implemented yet.
- No preconditioning inside `predictIterative` is inherited automatically from
  an `LLIterative(m,precond_rank)` fit — `predictIterative`'s own
  `use_nystrom_precond`/`precond_rank` arguments are independent and
  must be passed explicitly at call time (see [PredictIterative.md](PredictIterative.md)).
- The SLQ log-determinant and the Hutchinson gradient trace term are
  independent estimators (see above) — don't expect the analytic
  gradient to match a finite-difference of the objective as tightly as
  `LLNystrom`'s exact-Woodbury gradient does.

## See also

[lliterative_vs_cholesky.ipynb](lliterative_vs_cholesky.ipynb) for a
worked notebook covering this objective's math in detail, including an
honest, empirically-found caveat about free-BFGS convergence on its
stochastic objective.
[Scalability.md](Scalability.md) for how this compares to `LLVecchia`,
`LLNystrom` and `predictIterative`, and how to pick between them.
[PredictIterative.md](PredictIterative.md) for the Nystrom-preconditioned CG idea this
reuses, and [Nystrom.md](Nystrom.md) for the fixed-landmark rationale
both this and the preconditioner share.

## References

- Ubaru, S., Chen, J., & Saad, Y. (2017). *Fast estimation of tr(f(A))
  via stochastic Lanczos quadrature*. SIAM Journal on Matrix Analysis
  and Applications, 38(4), 1075-1099.
- Hutchinson, M. F. (1990). *A stochastic estimator of the trace of the
  influence matrix for Laplacian smoothing splines*. Communications in
  Statistics - Simulation and Computation, 19(2), 433-450.
- Gardner, J., Pleiss, G., Weinberger, K. Q., Bindel, D., & Wilson, A. G.
  (2018). *GPyTorch: Blackbox matrix-matrix Gaussian process inference
  with GPU acceleration*. Advances in Neural Information Processing
  Systems, 31 — the BBMM strategy this objective mirrors (CG-based
  linear solves + SLQ log-determinant), minus GPyTorch's GPU batching
  and pivoted-Cholesky preconditioner refinements.
