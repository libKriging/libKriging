# Nystrom approximation (`objective="LLNystrom(k)"`)

## Idea

Exact Gaussian process likelihood evaluation costs O(n³) (Cholesky
factorization of the n×n covariance matrix R), which becomes impractical
somewhere in the n = 10³–10⁴ range. The Nystrom approximation replaces R
with a global rank-k factorization built from a fixed subset of k
"landmark" points:

  R ≈ R_ns · R_ss⁻¹ · R_ns.t()

where S is the set of k landmarks and R_ns / R_ss are the n×k / k×k
covariance blocks between all points / among the landmarks. This turns
the O(n³) exact factorization into O(n·k²) work per likelihood
evaluation via the Woodbury identity, without ever materializing the
n×n matrix R.

libKriging exposes it as an alternative fitting objective,
`objective="LLNystrom"` (default k = 50 landmarks) or
`objective="LLNystrom(k)"` for an explicit rank k, usable from any
binding since `objective` is just a string forwarded to the C++ core.

```r
k <- Kriging(y, X, "matern5_2", objective = "LLNystrom(50)")   # or "LLNystrom"
```

Complementary to `LLVecchia`: Vecchia is a *local* approximation (each
point conditions on a handful of nearest neighbors) whereas Nystrom is a
*global* low-rank approximation (every point shares the same k landmarks).
Nystrom tends to degrade more gracefully with input dimension than
Vecchia (no reliance on spatial nearest-neighbor structure), at the cost
of needing k large enough to capture the process' effective rank —
which grows with how short the correlation range is relative to the
domain.

## Mathematical description

- **Landmarks**: a fixed set S of k points, chosen ONCE (before
  optimization starts, from a theta-neutral reference kernel scaled to
  the data's own extent) via `LinearAlgebra::nystromFactor`'s greedy
  pivoted-Cholesky selection, and held fixed across every θ evaluation
  during the fit. Re-selecting landmarks at each θ would make the
  objective (and its gradient) discontinuous in θ, since the pivot
  choice itself depends on the current covariance values.
- **Low-rank factor**: once S is fixed, R_ss (k×k) and R_ns (n×k) are
  ordinary, smooth functions of θ. With L_ss = chol(R_ss),
  U := R_ns · L_ss⁻ᵀ satisfies U·Uᵀ = R_ns · R_ss⁻¹ · R_ns.t() exactly.
- **Woodbury identity**: R⁻¹ and log|R| are obtained from U (and a
  small jittered residual diagonal keeping the factorization SPD)
  through `LinearAlgebra::woodbury_solve` / `woodbury_logdet`, both
  O(n·k²) — the dense n×n matrix R is never formed.
- **Profiling**: matches the exact `"LL"` objective — σ² has a closed
  form and β is profiled by generalized least squares, using the same
  GLS formulas as the exact objective, just solved via Woodbury instead
  of a dense Cholesky.
- **Gradient**: analytic (envelope theorem — β̂/σ̂² don't contribute
  their own θ-dependence since ∂ll/∂β = ∂ll/∂σ² = 0 at the profiled
  values), same principle as the Vecchia gradient. An earlier version
  used a finite-difference gradient instead; it worked but was more
  sensitive to the likelihood surface's local curvature near
  small-θ / near-singular-R regions than the analytic form is.
- **Exactness at k = n**: with k equal to the sample size, the
  factorization is exact and `LLNystrom(n)` reproduces the exact
  concentrated log-likelihood at any θ (used as a correctness check in
  `KrigingNystromTest.cpp`).
- **Prediction**: after fitting, `predict` uses one exact O(n³)
  factorization at the fitted θ* by default (small/medium n).
  `predictNystrom` instead reuses the committed rank-k factors (U, D)
  from the fit via the Woodbury identity — no n×n factorization, usable
  after any `"LLNystrom(k)"` fit. `simulateNystrom` similarly draws
  joint sample trajectories through the same low-rank machinery.
- **Update/save**: a Nystrom fit never holds an exact n×n
  factorization (unlike Vecchia's optional "light" mode, this is the
  *only* mode Nystrom has). `update` routes through a dedicated
  incremental path (`update_nystrom`, extends the landmarks-fixed
  factorization without a full O(n³) refit) and `save`/`load` serialize
  the committed rank-k factors (U, D) directly; `update_simulate`
  raises a clear error (no such path exists yet, see Current
  limitations).

## Usage

```r
library(rlibkriging)

set.seed(1)
n <- 2000
X <- matrix(runif(2 * n), ncol = 2)
y <- sin(3 * X[, 1]) * cos(3 * X[, 2]) + rnorm(n, sd = 0.05)

# Exact objective would cost O(n^3) ~ 8e9 ops; Nystrom costs O(n*k^2).
k <- Kriging(y, X, "matern5_2", objective = "LLNystrom(50)")

Xnew <- matrix(runif(2 * 10), ncol = 2)
pred <- predict(k, Xnew, stdev = TRUE)
```

## Current limitations (v1)

- `NoiseModel::None` only (no nugget/noise channel).
- Landmarks are not serialized (rebuilt on refit).
- Rank k is a fixed hyperparameter chosen by the caller; there is no
  automatic rank selection based on a target approximation accuracy.

## See also

[Scalability.md](Scalability.md) for how this compares to `LLVecchia`,
`NestedKriging` and `predictCG`, and how to pick between them.

## References

- Williams, C. K. I., & Seeger, M. (2001). *Using the Nystrom method to
  speed up kernel machines*. Advances in Neural Information Processing
  Systems, 13, 682–688.
- Drineas, P., & Mahoney, M. W. (2005). *On the Nystrom method for
  approximating a Gram matrix for improved kernel-based learning*.
  Journal of Machine Learning Research, 6, 2153–2175.
