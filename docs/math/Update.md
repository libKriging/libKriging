# Incremental update — `update`, `simulate`, `update_simulate`

## Idea

Three related operations let a fitted Kriging model react to new data
without paying the full O(n³) refit cost each time:

- **`update(y_u, X_u, refit=...)`**: extend a fitted model with new
  observations `(X_u, y_u)`, reusing the existing Cholesky factor
  instead of recomputing it from scratch — O(n·n_u²) instead of
  O((n+n_u)³).
- **`simulate(nsim, seed, X_n)`**: draw `nsim` sample paths of the GP
  posterior at new points `X_n`, conditional on the fitted data —
  exact conditional (simple/universal) kriging simulation.
- **`update_simulate(y_u, X_u)`**: given paths already simulated at
  `X_n` (via `simulate(..., will_update=TRUE)`), and new *real*
  observations `(X_u, y_u)` that arrive afterwards, correct those
  already-drawn paths so they become consistent with the enlarged
  data set — without resimulating from scratch and without needing
  `X_u` to have been part of `X_n`.

The last operation matters for sequential design / Bayesian
optimization loops: you can simulate the future once, then cheaply
"rewind" the simulated scenarios each time a real evaluation comes in,
instead of re-running a fresh simulation after every new point.

All three are, underneath, the *same* piece of linear algebra applied
three times — the Schur complement — so it is worth stating once
before going through each operation.

## The Schur complement

For a symmetric positive-definite block matrix

  M = [ A   B ]      with Cholesky factor      L_M = [ L_A     0  ]
      [ Bᵀ  C ]                                      [ L_ABᵀ  L_S ]

where `L_A` is the (already known) Cholesky factor of `A`
(`A = L_A L_Aᵀ`), the bottom-right block is obtained from

  L_AB = L_A⁻¹ B                       (forward-substitution, not a matrix inverse)
  S    = C − L_ABᵀ L_AB   =   C − Bᵀ A⁻¹ B      ← the Schur complement of A in M
  L_S  = chol(S)

This is exactly the classical block-Cholesky identity, and `S` is *the
Schur complement of A in M* by definition. The key practical point:
**building `L_M` this way costs O(size(C)³ + size(A)·size(C)²)
instead of O(size(M)³)** — you never refactorize the `A` block, only
solve a triangular system against it and factorize the (usually much
smaller) Schur-complement block `S`. This single identity is the
engine behind all three operations below:

- **`update`**: `A = R_o` (already-factorized training correlation),
  `B = R_ou`, `C = R_uu` — the new Cholesky factor is built by a Schur
  update instead of a full refactorization of `R_ou`, the augmented
  correlation matrix.
- **`simulate`/`predict`**: the conditional covariance of a Gaussian
  vector is *itself* a Schur complement — for jointly Gaussian
  `(y_o, Y(X_n))` with joint covariance `[[σ²R_o, σ²R_on],[σ²R_noᵀ,
  σ²R_nn]]`, the posterior covariance `Cov[Y(X_n) | y_o]` is exactly
  the Schur complement `σ²(R_nn − R_noᵀR_o⁻¹R_no)` of `σ²R_o` in that
  joint covariance — the textbook "conditioning a Gaussian" formula
  (Rasmussen & Williams, 2006, §A.2) is a Schur complement in disguise.
- **`update_simulate`**: the conditional-kriging weights that rewind
  already-simulated paths onto new real data are built from the same
  block-inversion identity, applied to the joint covariance of
  `(y_o, Y(X_n), Y(X_u))` rather than recomputed from scratch.

## Mathematical description

All three operations reuse the same building block: the Cholesky
factor `L` of the correlation matrix `R` of the design points, and the
QR decomposition of `L⁻¹F` (F = trend basis) used to profile β and σ²
by generalized least squares. Everything below is written for the
plain Gaussian kernel / linear trend case; nugget and warping variants
plug their own correlation function into the same formulas ρ(dx, θ)
→ e.g. the nugget model rescales it by α = σ²/(σ²+nugget) off the
diagonal (see [Noise.md](Noise.md)).

### `update`: Schur-complement Cholesky update

Given the existing factor `L_o` (so `R_o = L_o L_oᵀ`) and new points
`X_u` with cross-correlation `R_ou` and self-correlation `R_uu`, apply
the Schur-complement identity above with `A=R_o`, `B=R_ou`, `C=R_uu`:

  L_oCu = L_o⁻¹ R_ou
  L_uCu = chol(R_uu − L_oCuᵀ L_oCu)          ← Schur complement of R_o

  L_ou = [ L_o        0   ]
         [ L_oCuᵀ   L_uCu ]

β, the profiled residual, and σ² are then recomputed from the QR
decomposition of `L_ou⁻¹ [F_ou | y_ou]`, exactly as in a from-scratch
fit but working with an (n+n_u)-size triangular system built from two
already-triangular blocks rather than a fresh (n+n_u)×(n+n_u) Cholesky.
`refit=FALSE` keeps θ/σ² fixed (β is still recomputed by GLS on the
enlarged data); `refit=TRUE` additionally re-optimizes the
hyperparameters.

### `simulate`: conditional sample paths

At new points `X_n`, the joint law of (Y(X_n) | y_o) is Gaussian with
mean and covariance given by the usual universal-kriging formulas —
the Schur complement of `R_o` in the joint covariance, plus the trend
part's contribution:

  mean(X_n)  = F_n β̂ + R_on^⋆ᵀ · Ê⋆       (Ê⋆ = L_o⁻¹-transformed residual)
  Σ(X_n)     = σ² · ( R_nn − R_on^⋆ᵀ R_on^⋆ + Ê_n Ê_nᵀ )

where `R_on^⋆ᵀ R_on^⋆` is the Schur-complement reduction term (same
role as `L_oCuᵀL_oCu` above) and the extra `Ê_n Ê_nᵀ` term adds back
the variance coming from not knowing β exactly (trend/universal-
kriging uncertainty) on top of the simple-kriging reduction. A sample
path is then

  Y(X_n) = mean(X_n) + σ · chol(Σ(X_n)/σ²) · Z,   Z ~ N(0, I)

using libKriging's seeded RNG so paths are reproducible. When called
with `will_update=TRUE`, `simulate` additionally keeps the drawn paths,
the seed and the conditional factors at `X_n` (`R_on^⋆`, `Ê_n` and the
Cholesky factor of `Σ(X_n)`), so that `update_simulate` only has to
compute the blocks involving the new points `X_u`.

### `update_simulate`: rewinding simulated paths onto new real data

This is the "condition on a simulation, then condition further on
reality" trick. Say paths `Y_sim(X_n)` were drawn conditional on `y_o`
only. New real data `(X_u, y_u)` arrives. Instead of resimulating
`Y(X_n) | y_o, y_u` directly, the identity

  Y(X_n) | y_o, y_u  =  E[Y(X_n) | y_o, Y(X_u)] evaluated by
  substituting the *simulated* Y(X_u) with the *known* y_u, corrected
  by the kriging weights W̃ between X_n and X_u conditional on (y_o, X_n)

is used:

  1. Extend each drawn path to the new points `X_u`, jointly with the
     already-drawn `Y_sim(X_n)`. With `Σ` the conditional covariance of
     `(Y(X_n), Y(X_u))` given `y_o` (including the trend uncertainty) and
     `L_n` the Cholesky factor of its `X_n` block, this is a
     block-Cholesky extension:

     Y_sim(X_u) = mean(X_u) + σ · ( Aᵀ Z_n + L_u Z_u ),
     A = L_n⁻¹ Σ_nu / σ²,   L_u L_uᵀ = Σ_uu / σ² − Aᵀ A

     where `Z_n` are the standard normals behind `Y_sim(X_n)`
     (regenerated from the seed of `simulate`), `Z_u` are fresh normals,
     and `L_u` is a positive semi-definite square root of the Schur
     complement (zero when `X_u` lies on `X_n`). No (n_o + n_n)-size
     matrix is inverted.
  2. Compute the conditional-kriging weights of `X_n` on `X_u` given
     `y_o`: `W̃_{n|u} = Σ_nu Σ_uu⁻¹`, a small `n_u × n_u` system.
  3. Correct each already-drawn path:

     Y_upd(X_n) = Y_sim(X_n) + W̃_{n|u} · (y_u − Y_sim(X_u))

Because `(Y_sim(X_n), Y_sim(X_u))` is a joint draw from the correct
conditional law given `y_o`, replacing `Y_sim(X_u)` by the true `y_u`
and propagating the discrepancy through the conditional-kriging weights
yields paths distributed exactly as `Y(X_n) | y_o, y_u`, for fixed
hyperparameters. This is the law that `update(y_u, X_u, refit=FALSE)`
followed by a fresh `simulate(X_n)` would give, at a fraction of the
cost. The two agree in distribution, not path by path. All of this is
done on the internal (normalized) output scale, so `normalize=TRUE` is
handled consistently.

## Simple example

```r
library(rlibkriging)

f <- function(x) 1 - 0.5 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
X_o <- matrix(seq(0, 1, length.out = 5))   # X is always a matrix (n x d)
y_o <- f(X_o)

k <- Kriging(y_o, X_o, kernel = "gauss", regmodel = "linear")

Xnew <- seq(0, 1, , 21)
p <- predict(k, Xnew)                       # exact conditional mean/stdev

# simulate 10 paths at Xnew, keep internals for a later update_simulate
sims <- simulate(k, nsim = 1000, seed = 123, x = Xnew, will_update = TRUE)

# a new real observation arrives
X_u <- 0.5; y_u <- f(X_u)

# cheaply correct the already-drawn paths onto the new observation,
# without resimulating from scratch:
sims_updated <- update_simulate(k, y_u, X_u)

# reference with the same law (but more expensive): update then resimulate
k2 <- copy(k)
update(k2, y_u, X_u, refit = FALSE)
sims_ref <- simulate(k2, nsim = 1000, seed = 123, x = Xnew)
# same distribution, not the same paths:
# rowMeans(sims_updated) ≈ predict(k2, Xnew)$mean, and likewise for the stdev
```

## Current notes

- `update`/`simulate`/`update_simulate` are implemented for `Kriging`
  (all `noise_model` variants, see [Noise.md](Noise.md)), `WarpKriging`,
  `MLPKriging` and `MultiOutputKriging` (see
  [MultiOutput.md](MultiOutput.md)).
- `update_simulate` requires a preceding `simulate(..., will_update=TRUE)`
  call on the same model: it reuses that call's paths, seed and cached
  factors.
- The moments of the updated paths are tested against the updated model
  (`tests/KrigingUpdateSimulateTest.cpp`), together with Kolmogorov-Smirnov
  checks for the nugget and noise variants. With the `gauss` kernel,
  `simulate` itself adds a small numerical nugget, which leaves a floor
  on very small posterior variances.
- Numerical correctness is validated by mockup R scripts comparing a
  from-scratch reference implementation (independent Cholesky/QR
  formulas) against the library's `predict`/`simulate`/`update`, for
  both the plain and nugget kernels.

## References

- Chevalier, C., Ginsbourger, D., & Emery, X. (2014). *Corrected
  kriging update formulae for batch-sequential data assimilation*.
  Mathematics of Planet Earth, 119–122 (block Cholesky / Schur-complement
  kriging update formulas).
- Chevalier, C., Emery, X., & Ginsbourger, D. (2015). *Fast update of
  conditional simulation ensembles*. Mathematical Geosciences, 47(7),
  771–789 (the update-simulate / "rewinding" trick).
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. MIT Press, Appendix A.2 (Gaussian conditioning
  as a Schur complement).
- Haynsworth, E. V. (1968). *On the Schur Complement*. Basel
  Mathematical Notes, 20 (the general block-matrix identity by name).
