# Vecchia approximation (`objective="LLVecchia(m)"`)

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
`objective="LLVecchia"` (default m = 30 neighbors) or `objective="LLVecchia(m)"`
for an explicit m, usable from any binding since `objective` is just a
string forwarded to the C++ core.

```r
k <- Kriging(y, X, "matern5_2", objective = "LLVecchia(30)")   # or "LLVecchia"
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
k <- Kriging(y, X, "matern5_2", objective = "LLVecchia(30)")

Xnew <- matrix(runif(2 * 10), ncol = 2)
pred <- predict(k, Xnew, return_stdev = TRUE)
```

For n large enough that even the final exact commit (O(n³)) is too
costly, calling `set_vecchia_exact_commit(false)` before fitting skips it
entirely (**C++ API only**: no binding exposes this method, nor `predictVecchia`,
so the light mode cannot be used from Python, R, Julia or Octave/Matlab): θ* comes from the optimizer, β/σ² from the LLVecchia profile, and
`predict` automatically routes through the local Vecchia predictor
(mean/stdev only — `return_cov`/`return_deriv`, `update_simulate`,
`update` and `save` raise a clear error on such a "light" model), and
`simulate` through `simulateVecchia` (below; `will_update=true` raises).

## Vecchia simulation (`simulateVecchia`, C++ only)

`simulateVecchia(nsim, seed, X_n, m)` draws joint trajectories at the
q rows of `X_n` by sequential conditioning, in the row order of `X_n`
("response-first" ordering, Katzfuss et al. 2020): the t-th point is
drawn from its simple-kriging conditional law (β treated as known, as in
`predictVecchia`) given its m nearest neighbors among the n observations
**and** the t−1 points already simulated:

$$
Z_t \mid Z_{N(t)} \sim \mathcal{N}\big(f_t^\top\beta + r_t^\top R_{N}^{-1}(z_{N} - F_{N}\beta),\;
\sigma^2 (1 - r_t^\top R_{N}^{-1} r_t)\big).
$$

- Cost O(q (n + q) d) for the neighbor search plus O(q m³) for the local
  solves (shared by the nsim trajectories), instead of O(q³) for the
  exact joint factorization: usable for large n **and** large q.
- Exact (β known) when every point conditions on all its predecessors,
  i.e. m ≥ n + q − 1 (chain rule). For smaller m, correlations beyond the
  m nearest neighbors are truncated, and the trajectories depend on the
  row order of `X_n`.
- Usable after any `NoiseModel::None` fit (m defaults to
  `vecchia_neighbors()` after an LLVecchia fit, 30 otherwise); no
  `will_update` / `update_simulate`.

## Current limitations

- `NoiseModel::None` only (no nugget/noise channel).
- The default exact commit after optimization is still O(n³)
  time/memory — practical up to n ~ 2·10⁴; use the "light" mode above
  beyond that.
- Vecchia neighbor sets are not serialized (rebuilt on refit).

## See also

[llvecchia_vs_cholesky.ipynb](llvecchia_vs_cholesky.ipynb) for a worked,
executed notebook: the math above illustrated on a running example, with
convergence/accuracy/timing plots and references.
[Scalability.md](Scalability.md) for how this compares to `LLNystrom`
and `NestedKriging`, and how to pick between them.

## References

- Vecchia, A. V. (1988). *Estimation and model identification for
  continuous spatial processes*. Journal of the Royal Statistical
  Society, Series B, 50(2), 297–312.
- Guinness, J. (2018). *Permutation and grouping methods for sharpening
  Gaussian process approximations*. Technometrics, 60(4), 415–429.
- Katzfuss, M., & Guinness, J. (2021). *A general framework for Vecchia
  approximations of Gaussian processes*. Statistical Science, 36(1),
  124–141.
