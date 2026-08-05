# The `objective="LMP"` fitting criterion (log-marginal-posterior)

## Idea

Maximizing the plain concentrated likelihood (`objective="LL"`, see
[Kriging.md](Kriging.md)) can occasionally drive θ into
numerically-degenerate regions — a length-scale θₖ → 0 (the
correlation matrix collapses towards the identity, effectively
treating every point as unrelated noise) or θₖ → ∞ (the correlation
matrix collapses towards an all-ones matrix, effectively fitting a
pure linear trend) — especially with few observations, where the
likelihood surface can be nearly flat in those directions. `"LMP"`
(RobustGaSP's default; Gu, Wang & Berger, 2018) fixes this by
maximizing a *log-marginal-posterior* instead: the ordinary marginal
likelihood plus a "jointly robust" reference prior on θ that penalizes
both degenerate extremes, without requiring the user to supply any
prior hyperparameters — the ones used are set automatically from `n`
and `d`.

```r
k <- Kriging(y, X, kernel = "matern5_2", objective = "LMP")
```

## Mathematical description

### Marginal likelihood (β integrated out, not just plugged in)

Unlike `"LL"`, which *profiles* β (plugs in its GLS estimate β̂(θ) into
the likelihood), `"LMP"` uses the proper Bayesian marginal likelihood
with a flat (improper, non-informative) prior on β integrated out
analytically — the same quantity used in REML. With `R(θ)=LLᵀ` and
`X_Rinv_X = Xᵀ R⁻¹ X = L_X L_Xᵀ`:

  S² = yᵀR⁻¹y − yᵀR⁻¹X (XᵀR⁻¹X)⁻¹ XᵀR⁻¹y
  log p(y | θ) = −Σ log(diag L) − Σ log(diag L_X) − (n−p)/2 · log(S²/(n−p))

The extra `−Σ log(diag L_X)` term (absent from the plain `"LL"`
objective) is the log-determinant of the Fisher information for β —
it is what makes this a genuine marginal likelihood (β's own
uncertainty is accounted for) rather than a profile likelihood.

### Jointly robust reference prior

θ is reparametrized through a per-dimension characteristic length
`CLₖ = (max(Xₖ) − min(Xₖ)) / n^{1/d}` (the typical inter-point spacing
along dimension k, for a space-filling design), and

  t(θ) = Σₖ CLₖ/θₖ  (+ nugget_ratio, if `noise_model="nugget"` — see below)
  log p_prior(θ) = a·log(t) − b·t,   with defaults a = 0.2,  b = (a+d) / n^{1/d}

This is the "jointly robust prior" of Gu, Wang & Berger (2018): it
behaves like `t^a·e^{−bt}` — a Gamma-shaped density in `t = Σ CLₖ/θₖ`
— which vanishes both as `t→0` (θ→∞, over-smooth/degenerate towards a
pure trend) and as `t→∞` (θ→0, degenerate towards treating all points
as uncorrelated), with `a`, `b` chosen automatically from `n` and `d`
so no user tuning is required.

### Combined objective and nugget coupling

  LMP(θ) = log p(y | θ) + log p_prior(θ)

optimized exactly like `"LL"` (L-BFGS-B on θ, analytic gradient — a
finite-difference fallback is used only when σ² is held fixed, an
uncommon configuration). With `noise_model="nugget"` (see
[Noise.md](Noise.md)), the nugget ratio `(1−α)/α` (α =
σ²/(σ²+nugget)) is folded additively into `t`, so the same reference
prior simultaneously discourages a degenerate θ *and* a degenerate
nugget-to-signal ratio — one prior term regularizes both.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
n = 6  # deliberately few points: LL alone can be unstable here
X = rng.uniform(size=(n, 1))
y = np.sin(3 * X[:, 0]) + rng.normal(scale=0.02, size=n)

k_ll = lk.Kriging(y, X, "matern5_2", objective="LL")
k_lmp = lk.Kriging(y, X, "matern5_2", objective="LMP")
print(k_ll.theta(), k_lmp.theta())
# with few points, LL can drift to a very large or very small theta;
# LMP's reference prior keeps it in a well-conditioned range
```

## References

- Gu, M., Wang, X., & Berger, J. O. (2018). *Robust Gaussian stochastic
  process emulation*. The Annals of Statistics, 46(6A), 3038–3066
  (the jointly robust reference prior and `objective="LMP"`, as
  implemented in RobustGaSP).
- Berger, J. O., De Oliveira, V., & Sansó, B. (2001). *Objective
  Bayesian analysis of spatially correlated data*. Journal of the
  American Statistical Association, 96(456), 1361–1374 (reference
  priors for GP correlation parameters, the framework `LMP`'s prior
  builds on).
- Harville, D. A. (1977). *Maximum likelihood approaches to variance
  component estimation and to related problems*. Journal of the
  American Statistical Association, 72(358), 320–338 (restricted
  maximum likelihood / REML — the marginal-likelihood-with-β-
  integrated-out idea `LMP` also relies on).
