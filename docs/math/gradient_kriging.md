# Gradient-Enhanced Kriging (GEK)

Gradient-enhanced kriging conditions a Gaussian-process (GP) surrogate on both
observed function values **and** observed gradients. When derivatives are
available at (some of) the design points — from adjoint solvers, automatic
differentiation, or finite differences already computed by the simulation
code — feeding them into the fit typically gives a markedly more accurate
surrogate for the same number of design points, since each observation now
carries `1 + d` pieces of information instead of one.

This is exposed in libKriging as an optional trailing argument on the
existing `fit`, rather than a separate method or class:

```cpp
k.fit(y, X, regmodel, normalize, optim, objective, parameters, /*grady=*/dy);
```

```python
k = pylibkriging.Kriging(y, X, "gauss", grady=dy)
```

where `dy` is an `n × d` matrix with `dy(a, j) = ∂y/∂x_j` evaluated at
`X.row(a)`. Passing nothing (`grady = std::nullopt` / `grady=None`) recovers
the ordinary value-only fit exactly — the augmented code paths are inert
whenever no gradients are supplied.

## 1. The augmented Gaussian process

An ordinary kriging model treats `y(x)` as one realization of a GP
`Z(x) ~ GP(m(x), σ²k(x, x'; θ))`. If `Z` is mean-square differentiable, its
partial derivatives `∂Z/∂x_i` are *also* jointly Gaussian with `Z` — nothing
new needs to be assumed, this is a property of the same prior. GEK simply
conditions on more of that joint distribution: instead of observing
`y = Z(X) + noise`, it observes both `y` and `∇y = ∇Z(X) + noise`.

Concretely, the model is fit on the augmented observation vector

```
y_aug = [ y ; vec(∇y) ]         length N = n(1 + d)
```

with matching augmented trend and covariance:

```
F_aug = [ F ; ∂F/∂x ]                        (N × p)
K_aug = σ² · R_aug(θ)                        (N × N)
```

where `F` is the usual trend/regression matrix and `∂F/∂x` stacks its
per-point Jacobians (`Trend::regressionModelDerivativeMatrix`). The kriging
equations themselves (GLS for β, concentrated MLE for σ², BLUP predictor and
its variance) are **unchanged** — they are simply applied to `(F_aug, y_aug,
R_aug)` instead of `(F, y, R)`.

## 2. Covariance blocks

Write `k(h; θ)` for the stationary correlation between two points at lag
`h = x_a − x_b`. Differentiating the joint Gaussian process gives four block
types, assembled once per pair of points `(a, b)`:

| block | expression |
|---|---|
| `Cov(Z(a), Z(b))` | `k(h)` |
| `Cov(∂Z(a)/∂x_i, Z(b))` | `∂k/∂x_i(h)` |
| `Cov(Z(a), ∂Z(b)/∂x_j)` | `−∂k/∂x_j(h)` |
| `Cov(∂Z(a)/∂x_i, ∂Z(b)/∂x_j)` | `∂²k/∂x_i∂x'_j(h)` |

The sign flip on the mixed first-derivative block follows directly from the
chain rule (`h` depends on `x_a` with a `+1` coefficient and on `x_b` with
`−1`). The full `N × N` matrix `R_aug` is exactly this `(1+d) × (1+d)` block
structure repeated over every pair of the `n` design points — see
`LinearAlgebra::covMat_sym_X_grad` for the assembly and
`Covariance::DCovDx` / `Covariance::D2CovDxDxp` for the per-kernel formulas.

### Admissible kernels

Only kernels that are twice mean-square differentiable at the origin admit a
well-defined `∂²k/∂x_i∂x'_j(0)` (the variance of `∂Z/∂x_i`), so GEK is
restricted to:

- `gauss`
- `matern3_2`
- `matern5_2`

`exp` (a corner at the origin) and `whitenoise` are rejected outright — the
values at `h = 0` below are also a useful cross-check, since they must equal
the known derivative variance of each process:

| kernel | `∂²k/∂x_i²` at `h = 0` |
|---|---|
| gauss | `1/θᵢ²` |
| matern3_2 | `3/θᵢ²` |
| matern5_2 | `5/(3θᵢ²)` |

## 3. A small example

```python
import numpy as np
import pylibkriging as lk

def f(x):
    return np.sin(3 * x[0]) + np.cos(5 * x[1])

def grad_f(x):
    return np.array([3 * np.cos(3 * x[0]), -5 * np.sin(5 * x[1])])

rng = np.random.default_rng(0)
X = rng.random((15, 2))
y = np.array([f(x) for x in X])
dy = np.array([grad_f(x) for x in X])          # n x d observed gradients

k_plain = lk.Kriging(y, X, "gauss")             # values only
k_grad  = lk.Kriging(y, X, "gauss", grady=dy)   # values + gradients

X_test = rng.random((200, 2))
y_test = np.array([f(x) for x in X_test])

rmse_plain = np.sqrt(np.mean((k_plain.predict(X_test)[0].flatten() - y_test) ** 2))
rmse_grad  = np.sqrt(np.mean((k_grad.predict(X_test)[0].flatten()  - y_test) ** 2))
# rmse_grad < rmse_plain, typically by a wide margin at small n
```

Predicting derivatives (via `predict(..., return_deriv=True)`, already
available on an ordinary fit) works exactly as before on a gradient-enhanced
model — the extra observations only change what the model is *conditioned*
on, not the shape of `predict`'s output.

## 4. Cost

The augmented system has `N = n(1+d)` rows instead of `n`, so every
linear-algebra step (Cholesky, solves) that was `O(n³)` becomes `O(n³d³)`.
Gradient observations are worth it when derivative evaluations are cheap
relative to function evaluations (adjoint/AD codes) and `n·d` stays
moderate; for large `d` or large `n`, consider combining GEK with
[`NestedKriging`](../dev/NestedKriging.md) (see §5) to keep each submodel's
augmented system small.

## 5. Where it's supported

| class | status |
|---|---|
| `Kriging` (`NoiseModel::None`/`Nugget`/`Heterogeneous`) | full support |
| `WarpKriging` | supported with a **frozen** warp: `optim="none"` (θ and warp parameters must be given, not jointly re-optimized), non-joint (per-dimension) warp only, not combined with per-observation noise |
| `NestedKriging` | supported on the plain path (no warping, no `VLL` objective) with `PoE`/`gPoE`/`BCM`/`rBCM` aggregation; **not** `NK` aggregation (its cross-group aggregation bypasses the submodels and rebuilds directly from values, so gradients would silently have no effect) |
| `MLPKriging` | not yet — it doesn't share the common `KrigingImpl` base that the augmented-likelihood machinery lives in |

Simulation (`simulate`, `update_simulate`) and incremental `update` are not
yet available on a model fit with gradient observations; predicting (mean,
stdev, covariance, and derivatives) is fully supported.

## References

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for
  Machine Learning*, §9.4 (derivative observations). MIT Press.
- Solak, E., Murray-Smith, R., Leithead, W. E., Leith, D. J., & Rasmussen,
  C. E. (2003). *Derivative observations in Gaussian process models of
  dynamic systems*. NeurIPS 15.
- Forrester, A., Sobester, A., & Keane, A. (2008). *Engineering Design via
  Surrogate Modelling: A Practical Guide*, §2.4 (gradient-enhanced kriging /
  indirect co-kriging). Wiley.
- Laurent, L., Le Riche, R., Soulier, B., & Boucard, P.-A. (2019). *An
  overview of gradient-enhanced metamodels with applications*. Archives of
  Computational Methods in Engineering, 26(1), 61–106.
- Bouhlel, M. A., & Martins, J. R. R. A. (2019). *Gradient-enhanced kriging
  for high-dimensional problems*. Engineering with Computers, 35(1), 157–173.
  (GEKPLS, as implemented in [SMT](https://smt.readthedocs.io/) — see the
  comparison notebook in `docs/comparison/`.)
