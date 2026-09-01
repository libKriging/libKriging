# Matrix-free conjugate-gradient prediction (`predictIterative`)

## Idea

`predict`'s mean/stdev both need `R⁻¹ v` for some vector(s) v, where R is
the n×n correlation matrix at the fitted θ*. The default path gets this
from a stored dense O(n²) Cholesky factor -- fast to reuse across many
`predict` calls, but O(n²) memory, and it has to have been computed and
kept resident in the first place.

`predictIterative` is a predict-*only* alternative: it solves `R x = v` with
matrix-free conjugate gradient (`LinearAlgebra::conjugateGradient`)
instead, evaluating `R * p` on the fly (O(n²) time per matvec, O(n)
memory -- R itself is never materialized). Useful when a model's dense
factor either was never computed (e.g. after a light Vecchia/Nystrom fit
-- though `predictVecchia`/`predictNystrom` are cheaper still there,
since they only need the fit's own O(n·m³)/O(n·k²) factors, not a fresh
CG solve) or simply isn't worth keeping resident just for prediction on
an otherwise ordinary exact fit.

```cpp
Kriging model(y, X, "matern5_2");                 // ordinary exact fit
auto [mean, stdev] = model.predictIterative(Xnew, true);  // matrix-free predict
```

Also available on `WarpKriging`/`MLPKriging` (same signature and semantics)
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
  same machinery as [`LLNystrom`](Nystrom.md)) at the model's own
  already-fitted θ*, and uses `LinearAlgebra::woodbury_solve` bound to
  that factor as the CG preconditioner (`Pinv`) — same idea as
  GPyTorch's pivoted-Cholesky preconditioner: fewer CG iterations needed
  to reach `tol` on the typically ill-conditioned R, at a one-time
  O(n·precond_rank²) setup cost. Off by default (`precond_rank=50` when
  enabled). Since predictIterative is only ever called at one fixed θ*, the
  factor is built once per `predictIterative` call and doesn't need the
  fixed-landmark-set machinery `LLNystrom`/`LLIterative` use to stay
  smooth across varying θ. Measured ~38x accuracy improvement at a fixed
  tight iteration budget on an ill-conditioned fit — see
  `KrigingPredictIterativeTest.cpp`.
- **Scope**: `NoiseModel::None` only (no nugget/noise channel); throws if
  the model wasn't fitted, or on `X_n` dimension mismatch.

## Usage

```cpp
#include "libKriging/Kriging.hpp"

Kriging model(y, X, "matern5_2");

// Mean only (cheap: one shared CG solve, same asymptotic cost class as
// the exact predictor's O(n^2), just with a larger constant).
auto [mean, stdev_empty] = model.predictIterative(Xnew);

// Mean + stdev (one extra CG solve PER point in Xnew -- opt-in on purpose).
auto [mean2, stdev] = model.predictIterative(Xnew, /*return_stdev=*/true);

// Explicit iteration budget / tolerance (defaults: max_iter=2n, tol=1e-8).
auto [mean3, stdev3] = model.predictIterative(Xnew, true, /*max_iter=*/500, /*tol=*/1e-6);

// Nystrom-preconditioned CG (opt-in): fewer iterations to reach `tol` on
// an ill-conditioned R, at a one-time O(n*precond_rank^2) setup cost.
auto [mean4, stdev4] = model.predictIterative(Xnew, true, /*max_iter=*/50, /*tol=*/1e-6,
                                       /*use_nystrom_precond=*/true, /*precond_rank=*/50);
```

## Current limitations (v1)

- `NoiseModel::None` only (no nugget/noise channel).
- `return_stdev=true`'s O(n²·iters·q) cost makes it a poor fit for large
  q; prefer the exact `predict` (if a factor is already resident) or
  batch/limit q when only `predictIterative` is available.
- Preconditioning is opt-in and per-call: `use_nystrom_precond` isn't
  inherited from an `LLIterative(m,precond_rank)` fit's own
  preconditioner (see [Iterative.md](Iterative.md)) — the two are
  independent and each must be enabled explicitly where used.

## See also

[predictiterative_vs_cholesky.ipynb](predictiterative_vs_cholesky.ipynb)
for a worked, executed notebook: the linear-algebra idea illustrated on a
1D running example, plus accuracy/timing benchmarks.
[Scalability.md](Scalability.md) for how this compares to `LLVecchia`,
`LLNystrom` and `NestedKriging`, and how to pick between them.
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
  §4 for a direct comparison against `LLNystrom`'s low-rank alternative.
