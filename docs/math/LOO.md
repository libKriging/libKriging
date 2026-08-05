# The `objective="LOO"` fitting criterion (leave-one-out)

## Idea

Instead of maximizing the (marginal or profiled) likelihood, `"LOO"`
picks θ to directly minimize leave-one-out cross-validation error: for
each training point i, "how well would the model predict yᵢ if it had
never seen it?" This is a more *predictive* criterion than `"LL"` — it
can behave better when the goal is accuracy at (or near) the design
points specifically, rather than the best global fit of the assumed
GP model, and it is less sensitive to the correlation-family
mis-specification the likelihood is comparing against.

Computing this naively would mean refitting the model n times (once
per left-out point) — libKriging instead uses the closed-form
"virtual LOO" trick shared with DiceKriging, which gets the exact same
answer from *one* factorization of the full design.

```r
k <- Kriging(y, X, kernel = "matern5_2", objective = "LOO")
```

## Mathematical description

### The virtual-LOO trick (no refitting)

Let `R = LLᵀ` be the correlation matrix's Cholesky factor and define
the "hat-removing" matrix

  Q = R⁻¹ − R⁻¹F(FᵀR⁻¹F)⁻¹FᵀR⁻¹

(the same `Q` that appears in the GLS residual projection — it is the
matrix that maps `y` onto the vector of *trend-adjusted* residuals).
Then, remarkably, the leave-one-out prediction error and variance at
point i are obtained directly from the diagonal of `Q`, without ever
removing point i and refitting (Dubrule, 1983; Sundararajan & Keerthi,
2001, give the same identity for GP regression):

  σ²_LOO,ᵢ = 1 / Qᵢᵢ
  errorLOO,ᵢ = σ²_LOO,ᵢ · (Qy)ᵢ    =    yᵢ − ŷ₋ᵢ(xᵢ)

i.e. `ŷ₋ᵢ(xᵢ) = yᵢ − errorLOO,ᵢ` is exactly the prediction the model
*would* have made at xᵢ had it been fit on the other n−1 points only —
obtained here in O(n²) total (dominated by the one initial Cholesky
factorization and inversion) instead of O(n) separate O((n−1)³) fits.

### Objective and gradient

  LOO(θ) = (1/n) · Σᵢ errorLOO,ᵢ²

is minimized over θ (equivalently, libKriging's optimizer maximizes
`−LOO(θ)`, so it fits the same `optim="BFGS..."` machinery as every
other objective). The analytic gradient reuses the same `Q`-based
identities — for each range parameter θₖ, with `gradR_k = ∂R/∂θₖ`:

  ∂Qᵢᵢ/∂θₖ  = −[Q · gradR_k · Q]ᵢᵢ
  ∂σ²_LOO,ᵢ/∂θₖ = −σ⁴_LOO,ᵢ · ∂Qᵢᵢ/∂θₖ
  ∂errorLOO/∂θₖ = (∂σ²_LOO/∂θₖ) ⊙ (Qy) − σ²_LOO ⊙ (Q · gradR_k · Qy)
  ∂LOO/∂θₖ = (2/n) · errorLOOᵀ · (∂errorLOO/∂θₖ)

— again all obtained from the same cached `Q`/Cholesky factors, no
finite differences.

### When to prefer it over `"LL"`

`"LOO"` directly targets predictive accuracy at the design points,
which can matter more than the full-likelihood fit when the assumed
kernel family is only an approximation of the true correlation
structure, or when the trend model is a poor global fit — cases where
maximizing `"LL"` can trade a bit of local predictive accuracy for a
better overall Gaussian fit. It has no equivalent for extrapolation
quality away from the data, and (being a cross-validation criterion,
not a proper likelihood) it does not extend naturally to model
comparison across different kernel families the way `"LL"`/`"LMP"` do.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(20, 1))
y = np.sin(3 * X[:, 0]) + rng.normal(scale=0.02, size=20)

k_ll = lk.Kriging(y, X, "matern5_2", objective="LL")
k_loo = lk.Kriging(y, X, "matern5_2", objective="LOO")

yhat, sd = k_loo.leaveOneOutVec(k_loo.theta())  # LOO predictions/stdev at each X_i
```

## References

- Dubrule, O. (1983). *Cross validation of kriging in a unique
  neighborhood*. Mathematical Geology, 15(6), 687–699 (the virtual-LOO
  formula for kriging, no refitting required).
- Sundararajan, S., & Keerthi, S. S. (2001). *Predictive Approaches for
  Choosing Hyperparameters in Gaussian Processes*. Neural Computation,
  13(5), 1103–1118 (LOO as a GP hyperparameter-selection criterion,
  with the same closed-form identity).
- Roustant, O., Ginsbourger, D., & Deville, Y. (2012). *DiceKriging,
  DiceOptim: Two R Packages for the Analysis of Computer Experiments by
  Kriging-Based Metamodeling and Optimization*. Journal of Statistical
  Software, 51(1), 1–55 (`leaveOneOutFun`/`leaveOneOutGrad`, the
  reference implementation this objective mirrors).
