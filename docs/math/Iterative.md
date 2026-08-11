# Matrix-free CG + stochastic log-det fit (`objective="LLIterative(m)"`)

## Idea

Unlike `LLVecchia`/`LLNystrom` (which each replace R by a cheaper
*structured* approximation — local conditioning / global low rank),
`LLIterative` keeps R itself exact: every term of the concentrated
log-likelihood except `log|R|` is computed via matrix-free conjugate
gradient (`LinearAlgebra::conjugateGradient`) instead of a dense O(n³)
Cholesky factorization, mathematically the same quantity a full
factorization would give (up to CG's own convergence tolerance) but
computed via O(n²) matvecs. `log|R|` — the one term CG cannot produce
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
- **SLQ log-determinant**: `LinearAlgebra::stochasticLogDet` estimates
  `log|R| ≈ (n/nprobe) Σᵢ zᵢᵀ log(R) zᵢ` via Lanczos quadrature on each
  Rademacher probe `zᵢ` (Ubaru, Chen & Saad 2017) — R is only ever
  accessed through matvecs.
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
- **Cost model**: a gradient evaluation is a CG solve over `nprobe`
  right-hand sides (each up to `2n` Krylov iterations, each an O(n²)
  matvec, or cheaper per-iteration with the preconditioner enabled but
  at extra O(n·precond_rank²) setup cost per evaluation). A free BFGS
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
