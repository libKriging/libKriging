# Subset-of-data pre-fit reduction (`Kriging::subsetOfData`)

## Idea

The cheapest of libKriging's large-n options: pick `n_max` rows out of a
design `X` with `n > n_max`, and fit an ordinary exact Kriging model
(O(n_max³)) on that subset — at the cost of discarding `n - n_max` points
outright. Unlike `LLVecchia`/`LLNystrom`/`NestedKriging`
(which all still use every point, in a cheaper structured or partitioned
way), `subsetOfData` doesn't change the objective or the model at all; it
changes what data the model ever sees.

```r
idx <- subsetOfData(X, 200)   # row-indices to keep, 1-based in R
k <- Kriging(y[idx], X[idx, ], "matern5_2")   # ordinary exact fit on the subset
```

It's a standalone utility, not a fit objective or predict method — call
it once before fitting, then fit however you like on the reduced
`(X[idx], y[idx])`.

## Mathematical description

- **Selection**: `method="kmeans"` (default) runs `n_max` k-means
  centroids on `X`, then snaps each centroid to its nearest *actual*
  data point — so the subset always consists of real observations, not
  synthetic centroid coordinates. Falls back to `method="random"`
  (uniform subsample without replacement) if k-means degenerates (e.g.
  `n_max` close to `n` producing near-empty clusters).
- **Coverage metric**: validated via *fill-distance* (the maximum
  distance from any point in the full design to its nearest selected
  point) — not intra-subset spacing. Fill-distance is the right metric
  here: it measures how well the subset *covers* the full domain, which
  is what matters for prediction accuracy elsewhere in the domain;
  intra-subset spacing only measures how spread out the kept points are
  from each other, which is backwards for this purpose (see
  `KrigingSubsetOfDataTest.cpp`'s test comments).
- **No-op case**: if `n_max >= X.n_rows`, returns all indices
  unchanged (sorted `0, 1, ..., n-1`) — safe to call unconditionally
  without a size check first.
- **Cost**: O(n_max) k-means passes over `X`, negligible next to the
  O(n_max³) fit that follows.

## Usage

```r
library(rlibkriging)

set.seed(1)
n <- 20000
X <- matrix(runif(2 * n), ncol = 2)
y <- sin(3 * X[, 1]) * cos(3 * X[, 2]) + rnorm(n, sd = 0.05)

# An exact fit on all 20000 points would cost O(n^3); subsetOfData
# reduces to a manageable size FIRST, then fits exactly on the subset.
idx <- subsetOfData(X, 500)
k <- Kriging(y[idx], X[idx, ], "matern5_2")

Xnew <- matrix(runif(2 * 10), ncol = 2)
pred <- predict(k, Xnew, stdev = TRUE)
```

## Current limitations (v1)

- Discards `n - n_max` points outright — no way to recover their
  information the way Vecchia/Nystrom/NestedKriging do (they all use
  every point, just more cheaply). Prefer one of those if losing data
  isn't acceptable and `n` is large mainly because a genuinely global
  fit is needed.
- `method="kmeans"`'s centroid-snapping is a heuristic coverage
  criterion, not an optimal design (e.g. not a maximin or minimax
  design) — good enough as a cheap pre-fit reduction, not a substitute
  for deliberate experimental design when the points themselves are
  still to be chosen (unlike here, where `X` is already fixed).

## See also

[subsetofdata_vs_cholesky.ipynb](subsetofdata_vs_cholesky.ipynb) for a
worked, executed notebook: coverage (fill-distance) vs. naive random
subsampling, and accuracy/cost vs. `n_max`, with references.
[Scalability.md](Scalability.md) for how this compares to
`LLVecchia`/`LLNystrom`/`NestedKriging`, and how to pick between them.

## References

- Lloyd, S. (1982). *Least squares quantization in PCM*. IEEE
  Transactions on Information Theory, 28(2), 129-137 — the k-means
  algorithm used for centroid selection.
