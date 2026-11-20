# Matrix-free conjugate-gradient prediction (`predict(..., method="iterative")`)

## Idea

`predict`'s mean/stdev both need `R⁻¹ v` for some vector(s) v, where R is
the n×n correlation matrix at the fitted θ*. The default path gets this
from a stored dense O(n²) Cholesky factor -- fast to reuse across many
`predict` calls, but O(n²) memory, and it has to have been computed and
kept resident in the first place.

`predict(..., method="iterative")` is a predict-*only* alternative: it solves `R x = v` with
conjugate gradient (`LinearAlgebra::conjugateGradientBatched`) instead of a
stored factor -- matrix-free by default (evaluating `R * p` on the fly,
O(n²) time per matvec, O(n) memory, R itself never materialized), or
against an `R` materialized once per call for a separable kernel within a
memory budget (see *Dense fast path* below). Useful when a model's dense
factor either was never computed (e.g. after a light Vecchia/Nystrom fit
-- though `predictVecchia`/`predictNystrom` are cheaper still there,
since they only need the fit's own O(n·m³)/O(n·k²) factors, not a fresh
CG solve) or simply isn't worth keeping resident just for prediction on
an otherwise ordinary exact fit.

```cpp
Kriging model(y, X, "matern5_2");  // ordinary exact fit
auto [mean, stdev, cov, dm, ds] = model.predict(Xnew, true, false, false, "iterative");  // matrix-free predict
```

Also available on `WarpKriging`/`MLPKriging` (`predict(..., method)`, same semantics)
via the shared `KrigingImpl::predictIterative_impl` -- the algorithm only needs
`m_X`/`m_theta`/`_Cov`/`m_F`/`m_beta`/`m_sigma2`, all common `KrigingImpl`
state, plus the same optional feature map `predict_impl` takes for the
warped/MLP kernel space. Not available on `NestedKriging` (a composition of
independent `Kriging` submodels, not a single `KrigingImpl`) or on any
model with a per-point noise channel (`m_noise` non-empty).

## Mathematical description

- **Mean**: one CG solve of `R w = (y - Fβ)` (shared across every
  prediction point), then `mean = F_n β + R_on.t() * w`, same
  universal-kriging formula as the exact predictor, just with `w` from CG
  instead of a triangular solve. O(n²·iters) total, iters defaulting to
  `2n` (see below).
- **Stdev (opt-in)**: each prediction point's variance needs its own
  `R⁻¹ r_on` (`r_on` = that point's own correlation column), so
  `return_stdev=true` runs **one CG solve per prediction point** --
  O(n²·iters·q) total for q prediction points, since `R_on`'s q columns
  don't share a Krylov subspace. This is the dominant cost by far and is
  why it's opt-in (default `false`): fine for a handful of points on a
  model without a resident factor, but scales worse than the exact
  predictor's O(n²) *total* cost (not per point) as q grows -- see
  [PredictIterativeTest.cpp](../../tests/KrigingPredictIterativeTest.cpp) and
  `bench/bench-predictiterative.cpp` for the concrete crossover. Whenever the fit
  has an estimated trend (any `regmodel != "none"`, including the default
  `"constant"`), the variance also needs the GLS correction term
  `u.t() * (F.t() R⁻¹ F)⁻¹ * u` (`u = F_n - R_on.t() * R⁻¹F`) that accounts
  for β's own estimation uncertainty -- same term the exact predictor
  computes via its Cholesky factor, here from one additional CG solve of
  `R⁻¹F` (shared across all prediction points, same as the mean's solve).
- **Iteration budget**: `max_iter=0` defaults to `2n`. CG's classical
  exact-arithmetic bound is n iterations, but GP covariance matrices are
  commonly ill-conditioned enough (smooth kernels, clustered designs)
  that round-off keeps the true error shrinking well past that point in
  practice; `2n` is a more realistic default than the textbook bound.
- **Residual restart**: the recursively-updated CG residual drifts from
  the true residual under round-off well before `max_iter` is reached on
  these matrices, and can make the solution measurably *worse* past that
  drift point rather than better. `LinearAlgebra::conjugateGradient`
  recomputes the exact residual from scratch every 50 iterations and does
  a **full restart** (`p = r`, not a blended Fletcher-Reeves update) at
  that point -- blending a stale ratio into a freshly-corrected residual
  was tried first and made convergence worse, not better.
- **Tolerance**: `tol` (default `1e-8`) is a relative-residual early-stop
  criterion (`‖Ax - b‖ / ‖b‖ < tol`), checked at every iteration and at
  each restart.
- **Preconditioning (opt-in)**: `use_nystrom_precond=true` builds a
  rank-`precond_rank` Nystrom factor of R (`LinearAlgebra::nystromFactor`,
  same machinery as [`nystrom(k)`](Nystrom.md)) at the model's own
  already-fitted θ*, and uses `LinearAlgebra::woodbury_solve` bound to
  that factor as the CG preconditioner (`Pinv`) — same idea as
  GPyTorch's pivoted-Cholesky preconditioner: fewer CG iterations needed
  to reach `tol` on the typically ill-conditioned R, at a one-time
  O(n·precond_rank²) setup cost. Off by default (`precond_rank=50` when
  enabled). Since the iterative predictor is only ever called at one fixed θ*, the
  factor is built once per `predict(..., method="iterative")` call and doesn't need the
  fixed-landmark-set machinery `nystrom(k)`/`iterative` use to stay
  smooth across varying θ. Measured ~38x accuracy improvement at a fixed
  tight iteration budget on an ill-conditioned fit — see
  `KrigingPredictIterativeTest.cpp`.
- **Scope**: `NoiseModel::None` only (no nugget/noise channel); throws if
  the model wasn't fitted, or on `X_n` dimension mismatch.
- **Dense fast path (CPU)**: shared with `Kriging::_logLikelihoodIterative`
  (`KrigingImpl::build_separable_cov`, see [Iterative.md](Iterative.md#mathematical-description)) --
  for a separable kernel (`gauss`/`exp`/`matern3_2`/`matern5_2`) and an `n`
  whose dense `n×n` `R` fits a memory budget (`LinearAlgebra::set_iterative_dense_max_mb`,
  default 6144 MiB; `0` forces matrix-free), `R` is materialized ONCE per
  `predict(..., method="iterative")` call and every CG iteration becomes a BLAS-3 `R·V`
  instead of a fresh covariance sweep -- all of the mean solve, the
  per-point stdev solves, and the GLS-correction solve share it. Measured
  ~40x faster at `n=4000` (50.7s → 1.3s) with results unchanged to ~1e-7;
  skipped when a GPU backend is bound (it has its own dense fast path, next).
- **Dense fast path (CUDA)**: `LinearAlgebraCuda::conjugateGradient`
  materializes `R` once per `predict(..., method="iterative")` call (its own
  `build_cov_kernel` + `cublasDgemm`, governed by
  `Gpu::set_dense_max_mb`, default 4096 MiB) instead of
  recomputing every covariance entry's transcendentals on every CG
  iteration. Speedup grows with `n` (more CG iterations to amortize the
  build over): ~1.4x at `n≤1000` up to ~3x at `n=4000` in this project's
  own measurements, on top of the CUDA backend's existing per-iteration
  launch/host-sync savings. See [Iterative.md](Iterative.md#mathematical-description)
  for the same fast path applied to `iterative`'s objective/gradient,
  where the effect is much larger (~25x at `n=2000`) because that path
  runs many more matvecs per evaluation (SLQ Lanczos + multiple CG solves)
  to amortize the build over.

## Usage

```cpp
#include "libKriging/Kriging.hpp"

Kriging model(y, X, "matern5_2");

// Mean only (cheap: one shared CG solve, same asymptotic cost class as
// the exact predictor's O(n^2), just with a larger constant).
auto [mean, s0, c0, d0, e0] = model.predict(Xnew, false, false, false, "iterative");

// Mean + stdev (one extra CG solve PER point in Xnew).
auto [mean2, stdev, c1, d1, e1] = model.predict(Xnew, true, false, false, "iterative");

// CG budget 10 n (4th field) and tolerance 1e-6 (5th field); defaults 2 n, 1e-8.
auto [mean3, stdev3, c2, d2, e2] = model.predict(Xnew, true, false, false, "iterative(30,0,20,10,1e-6)");

// Nystrom-preconditioned CG of rank 50 (2nd field): fewer iterations on an
// ill-conditioned R, at a one-time O(n*50^2) setup cost.
auto [mean4, stdev4, c3, d3, e3] = model.predict(Xnew, true, false, false, "iterative(30,50)");
```

The first and third fields (probes, Lanczos steps) only matter for the fit;
on an iterative fit, `"iterative"` uses the fitted spec.

## Current limitations

- Noise model None only (no nugget / noise channel).
- `return_stdev=true`'s O(n²·iters·q) cost makes it a poor fit for large
  q; prefer the exact `predict` when a factorization is resident.
- Without a preconditioner rank in the spec, an automatic Nystrom
  preconditioner of rank 50 is built and kept only when it captures enough
  of R (`LinearAlgebra::set_cg_auto_precond`).

## See also

[Scalability.md](Scalability.md) for how this compares to `vecchia(m)`,
`nystrom(k)` and `NestedKriging`, and how to pick between them.
[Iterative.md](Iterative.md) reuses this same Nystrom-preconditioned-CG
idea for the fit's own CG solves, not just prediction.

## References

- Hestenes, M. R., & Stiefel, E. (1952). *Methods of conjugate gradients
  for solving linear systems*. Journal of Research of the National
  Bureau of Standards, 49(6), 409-436.
- Gardner, J., Pleiss, G., Weinberger, K. Q., Bindel, D., & Wilson, A. G.
  (2018). *GPyTorch: Blackbox matrix-matrix Gaussian process inference
  with GPU acceleration*. Advances in Neural Information Processing
  Systems, 31 -- the same matrix-free CG idea taken much further (batched
  matrix-matrix solves, pivoted-Cholesky preconditioning, GPU), see
  [libKriging_vs_GPyTorch.ipynb](../comparisons/libKriging_vs_GPyTorch.ipynb)
  §4 for a direct comparison against `nystrom(k)`'s low-rank alternative.
