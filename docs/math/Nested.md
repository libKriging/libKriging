# NestedKriging — divide-and-conquer Kriging (`NestedKriging`)

## Idea

Exact Kriging costs O(n³) to fit and O(n²) per prediction (Cholesky
factorization of the n×n covariance matrix), which becomes impractical
somewhere around n ~ 10⁴. `NestedKriging` (Rullière, Durrande, Bachoc,
Chevalier, 2018) scales this to n ~ 10⁵–10⁶ by splitting the design
`(X, y)` into p groups, fitting one independent submodel per group
(cheap: O(n/p)³ each, so O(n³/p²) total), unifying their
hyperparameters, and then *aggregating* the p submodel predictions
into a single prediction. Aggregation can be parallelized across
groups and, for the `NK` variant, across pairs of groups.

```r
nk <- NestedKriging(y, X, kernel = "matern5_2", nb_groups = 20,
                     aggregation = "NK", partition = "kmeans")
pred <- predict(nk, Xnew, stdev = TRUE)
```

## Mathematical description

### Partitioning and submodels

The n design points are split into p groups (`partition = "kmeans"` or
`"random"`). Each group g fits an ordinary/simple-kriging submodel on
its own points, but all p submodels share one common set of
hyperparameters (θ, σ², β) — either refit jointly by profiling a
common likelihood, or (when combined with the Vecchia objective, see
below) taken directly from one global fit. Fixing the hyperparameters
before per-group refits means each submodel's Cholesky factor Lg only
needs to be computed once, closed-form, at those shared parameters.

### Aggregation

Given the p submodel predictors (mean Mᵢ(x), variance Kᵢᵢ(x)) at a
prediction point x, the aggregation methods differ in how they combine
them:

- **PoE / gPoE / BCM / rBCM** (Deisenroth & Ng, 2015): precision-weighted
  products of experts. Cheap — O(q·n²/p) for q prediction points — but
  the resulting predictive variance is not a consistent Gaussian-process
  posterior (can be over- or under-confident, particularly `PoE`/`BCM`
  near group boundaries). Works with any trend.

- **NK** (default): treats the p submodel predictions themselves as
  noisy observations of the *true* underlying GP value at x, and Krige
  them. Concretely, for prediction point x:

  1. Each submodel gives Mᵢ(x) = β₀ + rᵢ(x)ᵀ Rᵢ⁻¹ (yᵢ − β₀) and residual
     variance Kᵢᵢ(x) = σ²(1 − rᵢ(x)ᵀ Rᵢ⁻¹ rᵢ(x)).
  2. The cross-covariance between two submodels' predictors at x,
     Kᵢⱼ(x), is obtained from the shared kernel evaluated between the
     two groups' points, projected through both submodels' kriging
     weights.
  3. The p×p matrix K(x) = [Kᵢⱼ(x)] (diagonal Kᵢᵢ(x), off-diagonal
     Kᵢⱼ(x)) is treated as the covariance of a size-p "meta-observation"
     M(x) = (M₁(x), …, Mₚ(x)) of the same unknown value f(x), and a
     simple-kriging step aggregates them:

     w(x) = K(x)⁻¹ k(x),  where k(x) = diag(K(x))

     mean(x)  = β₀ + w(x)ᵀ (M(x) − β₀)
     var(x)   = σ² · max(0, 1 − w(x)ᵀ k(x))

  This *is* a proper kriging predictor in the meta-model sense: it
  interpolates the data exactly (with p=1 it reduces to ordinary
  kriging) and gives internally consistent variances. It requires a
  **`"constant"` trend** (β₀ fixed before aggregation, simple-kriging
  theory) — the PoE family has no such restriction. As group size
  grows, NK's aggregate converges to the full-GP predictor, which is
  why it is also the natural fallback target for plain `Kriging` on
  small/medium problems.

  Cost: O(q·n²) worst case (all-pairs cross-covariances), but
  parallelized over the p(p−1)/2 group pairs, so effectively O(q·n²/p)
  wall-clock with p-way parallelism.

### Combination with the Vecchia objective

`NestedKriging(..., objective="LLVecchia(m)")` estimates the shared prior
(θ, σ², β) with a single global Vecchia-light fit (O(n·m³), see
[Vecchia.md](Vecchia.md)) instead of averaging p local MLEs — this
uses inter-group information the purely-local estimate discards, and
is statistically preferable at any group size. It is not supported
together with `warping`.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
n = 5000
X = rng.uniform(size=(n, 2))
y = np.sin(3 * X[:, 0]) * np.cos(3 * X[:, 1]) + rng.normal(scale=0.05, size=n)

# Full O(n^3) fit would be ~1e11 ops; 20 groups of ~250 points instead.
model = lk.NestedKriging(y, X, "matern5_2", nb_groups=20,
                          aggregation="NK", partition="kmeans", seed=123,
                          regmodel="constant")

Xnew = rng.uniform(size=(10, 2))
mean, stdev = model.predict(Xnew, return_stdev=True)
```

Order of magnitude: n = 10⁵ split into p = 100 groups gives submodels
of ~10³ points each — the fit is ~10⁴× faster than full kriging on the
same n (which would not fit in memory anyway).

## Current limitations

- `NoiseModel::None` only — no nugget/noise channel.
- `normalize` is not supported.
- `save`/`load` are not yet implemented.
- Combined with `warping`: NK aggregation requires evaluating the
  warped kernel between arbitrary points (supported since `WarpKriging`
  exposes a public `covMat`); the common (θ, warp) prior is estimated
  from a single reference fit on a subsample of size
  `min(n, warp_subsample)` (default 1000, tunable via
  `set_warp_subsample`), then submodels are fit closed-form
  (`optim="none"`) — one warp training total instead of p.

## See also

[Scalability.md](Scalability.md) for how this compares to `LLVecchia`
and `LLNystrom`, and how to pick between them.

## References

- Rullière, D., Durrande, N., Bachoc, F., & Chevalier, C. (2018).
  *Nested Kriging predictions for datasets with a large number of
  observations*. Statistics and Computing, 28(4), 849–867.
- Deisenroth, M. P., & Ng, J. W. (2015). *Distributed Gaussian
  Processes*. Proceedings of ICML 2015 (PoE/BCM/rBCM aggregation).
