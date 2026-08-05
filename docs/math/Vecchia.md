# Vecchia approximation (`objective="VLL(m)"`)

## Idea

Exact Gaussian process likelihood evaluation costs O(n³) (Cholesky
factorization of the n×n covariance matrix), which becomes impractical
somewhere in the n = 10³–10⁴ range. Vecchia's approximation (1988)
replaces the exact joint density with a product of low-dimensional
conditionals, each conditioning on only a handful of "neighbor" points
instead of on all the others. This turns an O(n³) factorization into
n independent O(m³) factorizations — cheap, parallelizable, and exact
in the limit m → n−1.

libKriging exposes it as an alternative fitting objective,
`objective="VLL"` (default m = 30 neighbors) or `objective="VLL(m)"`
for an explicit m, usable from any binding since `objective` is just a
string forwarded to the C++ core.

```r
k <- Kriging(y, X, "matern5_2", objective = "VLL(30)")   # or "VLL"
```

## Mathematical description

For a Gaussian vector y = (y₁, …, yₙ), the joint density factors exactly
as a product of conditionals in any fixed order:

  p(y) = ∏ᵢ p(yᵢ | y₁, …, yᵢ₋₁)

Vecchia's approximation truncates each conditioning set to a small
subset N(i) ⊂ {1, …, i−1} of size at most m:

  log p(y) ≈ Σᵢ log p(yᵢ | y_N(i))

- **Ordering**: points are ordered by a greedy *maxmin* sequence
  (Guinness 2018) — each new point maximizes its minimum distance to
  already-ordered points — which conditions well and concentrates
  approximation error near the start of the sequence.
- **Neighbors**: N(i) is the m nearest previously-ordered points
  (Euclidean, in normalized input space), fixed before optimization
  since they don't depend on the correlation parameters θ.
- **Cost**: O(n·m³) per likelihood evaluation instead of O(n³); the
  approximation is exact for m = n − 1.
- **Profiling**: as with the exact objective, σ² has a closed form and
  β is profiled by generalized least squares per-conditional (constant,
  linear and quadratic trends are supported); the gradient in θ is
  analytic (envelope theorem handles β̂).
- **Prediction**: after fitting, `predict` uses one exact O(n³)
  factorization at the fitted θ* by default (small/medium n). For
  large n, a local Vecchia predictor (Katzfuss & Guinness 2021,
  response-only) conditions each prediction point on its own m nearest
  neighbors — O(q·m³), embarrassingly parallel, usable after any fit.
  It gives the universal-kriging mean with the fitted β but a
  simple-kriging variance (no cross-covariance between prediction
  points — use the exact `predict` for the joint distribution).
- **Screening**: the approximation degrades in higher input dimension
  because nearest neighbors become less informative; it is recommended
  for d ≲ 5, and complements `NestedKriging` (which is dimension-robust)
  for scaling beyond that.

## Simple example

```r
library(rlibkriging)

set.seed(1)
n <- 2000
X <- matrix(runif(2 * n), ncol = 2)
y <- sin(3 * X[, 1]) * cos(3 * X[, 2]) + rnorm(n, sd = 0.05)

# Exact objective would cost O(n^3) ~ 8e9 ops; Vecchia costs O(n*m^3).
k <- Kriging(y, X, "matern5_2", objective = "VLL(30)")

Xnew <- matrix(runif(2 * 10), ncol = 2)
pred <- predict(k, Xnew, stdev = TRUE)
```

For n large enough that even the final exact commit (O(n³)) is too
costly, `set_vecchia_exact_commit(FALSE)` before fitting skips it
entirely: θ* comes from the optimizer, β/σ² from the VLL profile, and
`predict` automatically routes through the local Vecchia predictor
(mean/stdev only — `return_cov`/`return_deriv`, `simulate`, `update`
and `save` raise a clear error on such a "light" model).

## Current limitations (v1)

- `NoiseModel::None` only (no nugget/noise channel).
- The default exact commit after optimization is still O(n³)
  time/memory — practical up to n ~ 2·10⁴; use the "light" mode above
  beyond that.
- Vecchia neighbor sets are not serialized (rebuilt on refit).

## References

- Vecchia, A. V. (1988). *Estimation and model identification for
  continuous spatial processes*. Journal of the Royal Statistical
  Society, Series B, 50(2), 297–312.
- Guinness, J. (2018). *Permutation and grouping methods for sharpening
  Gaussian process approximations*. Technometrics, 60(4), 415–429.
- Katzfuss, M., & Guinness, J. (2021). *A general framework for Vecchia
  approximations of Gaussian processes*. Statistical Science, 36(1),
  124–141.
