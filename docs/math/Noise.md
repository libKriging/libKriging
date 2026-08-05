# Noise models — `noise_model` (`none` / `nugget` / `heterogeneous`)

## Idea

Real observations are rarely noiseless. `Kriging` supports three ways
of treating observation noise, selected by `NoiseModel` /
`noise_model` at construction:

- **`none`**: the classical interpolating GP — predictions pass
  exactly through the data.
- **`nugget`**: a single, *estimated*, homogeneous noise variance
  shared by all observations (equivalent to the historical
  `NuggetKriging` class, now folded into `Kriging`) — use when the
  data are noisy but the noise level itself is unknown and roughly
  constant across the design.
- **`heterogeneous`**: a *known*, per-observation noise variance
  supplied by the caller (equivalent to the historical `NoiseKriging`
  class) — use when each observation comes with its own known
  measurement uncertainty (e.g. replicated runs with different sample
  sizes, or a simulator that reports its own numerical error per run).

```r
k <- Kriging(y, X, kernel = "matern5_2", noise_model = "nugget")
k2 <- Kriging(y, X, kernel = "matern5_2", noise_model = "heterogeneous",
              parameters = list(), noise = noise_variances)
```

`NuggetKriging(...)` / `NoiseKriging(...)` constructor calls have been
removed from every binding — use `Kriging(..., noise=...)` /
`noise_model=` instead (`noise=NULL` ⟹ `none`, `noise="nugget"` ⟹
`nugget`, `noise=<vector>` ⟹ `heterogeneous`, matching the per-language
convenience wrappers described in `bindings/README.md`).

## Mathematical description

All three modes share the same concentrated-likelihood machinery as
plain `Kriging` (see [Kriging.md](Kriging.md)) — only the correlation
matrix `R` fed into that machinery changes:

  R_none            = ρ(θ)                                       (n×n correlation, diag = 1)
  R_nugget          = α·ρ(θ) + (1 − α)·I                          α = σ² / (σ² + τ²)  ∈ (0, 1)
  R_heterogeneous   = ρ(θ) + diag(noise) / σ²                     noise_i known, in raw y-units

where τ² is the (estimated) nugget variance and `noise_i` the
(user-supplied) per-observation noise variance.

- **`nugget`**: reparametrized in terms of `α = σ²/(σ²+τ²) ∈ (0,1)`
  rather than τ² directly — a numerically better-behaved unconstrained
  parametrization for the optimizer (α=1 ⟹ no nugget, α→0 ⟹ pure
  noise). `σ̂²` and `τ̂²` are recovered from the profiled variance `var`
  and α at the optimum: `σ̂² = α·var`, `τ̂² = (1−α)·var` (or, if one of
  σ²/nugget is supplied and fixed, the other is solved for from the
  fixed one and the estimated α). α is optimized jointly with θ
  (`gamma = [theta, alpha]` in the profile likelihood), with an
  analytic gradient `∂LL/∂α` derived alongside `∂LL/∂θ`.
- **`heterogeneous`**: `noise_i` enters as a fixed diagonal shift, not
  as a parameter to optimize — only θ (and profiled σ², β) are fit;
  `gamma = [theta, sigma2]` couples the noise scaling into the same
  profile-likelihood machinery as the `nugget` case, but with the
  per-point `noise_i/σ²` diagonal held fixed by the caller rather than
  estimated as a single shared α.

Both modes plug directly into the same `predict`/`simulate`/`update`/
`update_simulate` machinery described in
[Update.md](Update.md) — a fitted noisy model does
not interpolate the training data exactly (its predictive mean at a
training point sits between the observation and the trend, weighted by
how much noise variance was attributed to that point), unlike the
noiseless case.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
n = 30
X = rng.uniform(size=(n, 1))
true_noise = 0.05
y = np.sin(3 * X[:, 0]) + rng.normal(scale=true_noise, size=n)

# unknown, homogeneous noise level -> estimate it jointly with theta
k_nugget = lk.Kriging(y, X, "matern5_2", noise="nugget")
print(k_nugget.nugget())  # ~ true_noise**2

# known, heterogeneous noise (e.g. varying replication counts)
noise_var = rng.uniform(0.01, 0.1, size=n) ** 2
k_hetero = lk.Kriging(y, X, "matern5_2", noise=noise_var)
```

## References

- Gramacy, R. B., & Lee, H. K. H. (2012). *Cases for the nugget in
  modeling computer experiments*. Statistics and Computing, 22(3),
  713–722 (homogeneous-nugget rationale, `noise_model="nugget"`).
- Ranjan, P., Haynes, R., & Karsten, R. (2011). *A Computationally
  Stable Approach to Gaussian Process Interpolation of Deterministic
  Computer Simulation Data*. Technometrics, 53(4), 366–378
  (nugget/α reparametrization for numerical stability).
- Binois, M., Gramacy, R. B., & Ludkovski, M. (2018). *Practical
  Heteroscedastic Gaussian Process Modeling for Large Simulation
  Experiments*. Journal of Computational and Graphical Statistics,
  27(4), 808–821 (`noise_model="heterogeneous"`, known per-point noise).
