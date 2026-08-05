# Input warping — `knots(K)` / `knots(t1:t2:…:tK)`

## Idea

A piecewise-linear monotone warp (Xiong, Chen, Apley & Ding, 2007),
already familiar from `DiceKriging`'s `knots` argument. It partitions
the (normalized) input range [0, 1] into intervals separated by fixed
breakpoints, and learns one positive slope per interval — giving the
GP a way to stretch/compress its effective correlation length
piecewise across the input range, more flexibly than a single global
θ but with far fewer parameters than a full neural warp (`neural_mono`,
`mlp`). It is the natural choice when non-stationarity is expected to
be *localized* to specific regions of the input.

```r
wk <- WarpKriging(y, X, warping = c("knots(4)"), kernel = "gauss")
# or with explicit knot positions:
wk <- WarpKriging(y, X, warping = c("knots(0.2:0.5:0.8)"), kernel = "gauss")
```

## Mathematical description

Breakpoints  0 = t₀ < t₁ < … < t_K < t_{K+1} = 1  partition [0, 1] into
K+1 intervals (K interior knots, uniform by default, or given
explicitly as `knots(t1:...:tK)`). Each interval k carries a positive
slope `sₖ = exp(rₖ)`, with rₖ the unconstrained learnable parameter
(so K+1 parameters total; initialised at rₖ = 0, i.e. all slopes = 1,
identity-like). The warp is the running integral of the piecewise
slopes:

  w(x) = Σ_{j<k} sⱼ·(t_{j+1} − tⱼ)  +  sₖ·(x − tₖ),   for x ∈ [tₖ, t_{k+1})

i.e. `w` is continuous, piecewise-linear, and strictly increasing (each
slope > 0 by the `exp` reparametrization) — the only source of
non-stationary behaviour vs. a plain linear rescaling is that the
*slope itself changes across intervals*.

- **Input derivative**: `dw/dx = sₖ` for x in interval k — piecewise
  constant, discontinuous at the knots (by construction: a
  piecewise-linear function's derivative is a step function).
- **Parameter gradient**: for interval k's log-slope rₖ,
  `dw/drₖ = sₖ·(x − tₖ)` if x is in interval k, and
  `dw/drₖ = sₖ·(t_{k+1} − tₖ)` for every later interval j > k (the
  cumulative-sum term); zero for earlier intervals. Backpropagated as
  usual: `dL/drₖ = Σᵢ (dL/dφ)ᵢ · (dw/drₖ)ᵢ`.

Choosing more knots (or moving them) increases flexibility at the cost
of more parameters to estimate jointly with (θ, σ², β) — with sparse
data, prefer few knots (the default K = 3) or a smoother warp
(`kumaraswamy`, `neural_mono`) instead.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(40, 1))
# sharper variation for x<0.3, flatter beyond -> knots should learn a larger slope there
y = np.where(X[:, 0] < 0.3, 5 * X[:, 0], 1.5 + 0.2 * X[:, 0]) + rng.normal(scale=0.03, size=40)

model = lk.WarpKriging(y, X, ["knots(3)"], kernel="gauss",
                        regmodel="constant", normalize=True)
```

## References

- Xiong, Y., Chen, W., Apley, D., & Ding, X. (2007). *A non-stationary
  covariance-based Kriging method for metamodelling in engineering
  design*. International Journal for Numerical Methods in Engineering,
  71(6), 733–756.
