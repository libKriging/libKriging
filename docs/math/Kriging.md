# Kriging — the base Gaussian process model

## Idea

`Kriging` is the core model every other class in libKriging builds on
(`NestedKriging` fits many of them, `WarpKriging`/`MLPKriging` wrap
them with a learned input feature map, `objective="LLVecchia(m)"` swaps the
likelihood for a scalable approximation — see
[Vecchia.md](Vecchia.md), [Nested.md](Nested.md),
[Warping-Affine.md](Warping-Affine.md) *et al.*). It treats the
unknown response as a Gaussian process: a trend (regression) part
capturing the mean behaviour, plus a zero-mean stationary GP capturing
spatial correlation between residuals — the standard Kriging /
Gaussian-process-regression setup (Rasmussen & Williams, 2006).

```r
k <- Kriging(y, X, kernel = "matern5_2", regmodel = "constant", objective = "LL")
p <- predict(k, Xnew, stdev = TRUE)
```

## Mathematical description

### Model

  y(x) = f(x)ᵀβ + Z(x),   Z ~ GP(0, σ²·ρ(x, x′; θ))

- **Trend** `f(x)ᵀβ`: `f(x)` is the regression basis selected by
  `regmodel` — `"none"` (zero mean), `"constant"` (ordinary kriging,
  `f(x)=1`), `"linear"` (`f(x)=[1, x₁, …, x_d]`), `"interactive"`
  (adds pairwise products), `"quadratic"` (adds squared terms). β is
  profiled out by generalized least squares at every likelihood
  evaluation (see below), not optimized directly.
- **Correlation** `ρ(x, x′; θ)`: one of `"gauss"`, `"exp"`,
  `"matern3_2"`, `"matern5_2"` — a stationary, isotropic-per-dimension
  kernel with one length-scale θₖ per input dimension (ARD). See
  [Kernels.md](Kernels.md) for the exact formulas and smoothness
  properties of each.
- **σ²**: the process variance, also profiled out analytically.

### Concentrated (profiled) likelihood

Given θ, let `R(θ)` be the n×n correlation matrix of the design points
and `F` the n×p trend matrix. Using the Cholesky factor `R = LLᵀ`:

  β̂(θ) = (Fᵀ R⁻¹ F)⁻¹ Fᵀ R⁻¹ y     (generalized least squares)
  σ̂²(θ) = (y − Fβ̂)ᵀ R⁻¹ (y − Fβ̂) / n

so the concentrated log-likelihood, a function of θ alone, is

  LL(θ) = −n/2·[1 + log(2π) + log σ̂²(θ)] − ½ log|R(θ)|

Fitting maximizes `LL(θ)` over θ (L-BFGS-B by default, `optim="BFGSk"`
for k random restarts, or `optim="none"` to keep supplied parameters
fixed); β̂ and σ̂² are then simply the closed-form values at the
optimal θ*. The analytic gradient `∂LL/∂θ` is available (no finite
differences needed) via the envelope theorem for β̂.

### Alternative fitting objectives (`objective=`)

- **`"LL"`** (default): as above.
- **`"LOO"`**: leave-one-out cross-validation score, obtained without
  refitting n times — the virtual-LOO trick (Dubrule, 1983; as used by
  DiceKriging's `leaveOneOutFun`): with `Q = R⁻¹ − R⁻¹F(FᵀR⁻¹F)⁻¹FᵀR⁻¹`,
  the LOO residual and variance at point i are `errorsLOO_i = (Qy)_i /
  Q_{ii}` and `σ²LOO_i = 1/Q_{ii}`, computed from one n×n inverse
  instead of n separate n−1-point fits. See [LOO.md](LOO.md) for the
  full derivation and gradient.
- **`"LMP"`** (log-marginal-posterior): a Bayesian-flavoured objective
  from RobustGaSP (Gu, Wang & Berger, 2018) — the marginal likelihood
  (β integrated out, not just plugged in) plus an approximated
  reference-prior term `a·log(t) − b·t` (jointly-robust prior over θ)
  that discourages θ from drifting into numerically-degenerate regions
  (θ → 0 or θ → ∞) that the pure likelihood alone can favour with
  sparse data. See [LMP.md](LMP.md) for the full derivation.
- **`"LLVecchia"` / `"LLVecchia(m)"`**: Vecchia approximation for large n — see
  [Vecchia.md](Vecchia.md).

### Prediction

At new points X_n, the usual universal-kriging formulas give an exact
Gaussian posterior — mean, standard deviation, and (optionally) the
full covariance / mean-and-stdev derivatives w.r.t. x — profiling in
the same trend uncertainty term as the concentrated likelihood. See
[Update.md](Update.md) for the exact formulas shared
by `predict`, `simulate`, `update` and `update_simulate`, all
implemented via the same cached Cholesky/QR factorization rather than
independent from-scratch computations.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(20, 1))
y = np.sin(3 * X[:, 0]) + rng.normal(scale=0.02, size=20)

model = lk.Kriging(y, X, "matern5_2", regmodel="constant", objective="LL")
Xnew = np.linspace(0, 1, 50).reshape(-1, 1)
mean, stdev = model.predict(Xnew, return_stdev=True)
print(model.theta(), model.sigma2())
```

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. MIT Press.
- Roustant, O., Ginsbourger, D., & Deville, Y. (2012). *DiceKriging,
  DiceOptim: Two R Packages for the Analysis of Computer Experiments by
  Kriging-Based Metamodeling and Optimization*. Journal of Statistical
  Software, 51(1), 1–55.
- Dubrule, O. (1983). *Cross validation of kriging in a unique
  neighborhood*. Mathematical Geology, 15(6), 687–699 (leave-one-out
  formula used by `objective="LOO"`).
- Gu, M., Wang, X., & Berger, J. O. (2018). *Robust Gaussian stochastic
  process emulation*. The Annals of Statistics, 46(6A), 3038–3066
  (reference-prior objective used by `objective="LMP"`, as in
  RobustGaSP).
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). *A Limited Memory
  Algorithm for Bound Constrained Optimization*. SIAM Journal on
  Scientific Computing, 16(5), 1190–1208 (L-BFGS-B, the default
  optimizer).
