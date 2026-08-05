# Input warping — `affine`

## Idea

The simplest per-variable warp: a linear rescaling of the input before
it enters the GP kernel. On its own it changes nothing a kernel's own
length-scale θ couldn't already absorb (an affine map is exactly
compensated by rescaling θ), but as one entry of a `WarpKriging`
warping vector it lets an otherwise nonlinear warp on another variable
share the same joint optimisation loop, and it is the natural
"identity-plus-learnable-scale" baseline other warps are compared
against.

```r
wk <- WarpKriging(y, X, warping = c("affine", "none"), kernel = "gauss")
```

## Mathematical description

  w(x) = a·x + b,   2 learnable parameters (a, b), initialised at
  (a, b) = (1, 0) (identity).

- **Forward**: `w(x) = a·x + b`.
- **Input derivative**: `dw/dx = a`.
- **Parameter gradient**: given the backpropagated loss gradient
  `dL/dφ` (one scalar per observation),

    dL/da = Σᵢ (dL/dφ)ᵢ · xᵢ
    dL/db = Σᵢ (dL/dφ)ᵢ

Because the warp is monotone and linear, it does not change the
qualitative shape of the correlation structure — it only reparametrizes
the length-scale θ that the GP already estimates. It mainly exists as
a lightweight default/building block, not something to reach for on
its own; prefer `boxcox`, `kumaraswamy` or `knots` when the goal is to
actually change the correlation's shape in x-space.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(30, 1))
y = np.sin(3 * X[:, 0]) + rng.normal(scale=0.05, size=30)

model = lk.WarpKriging(y, X, ["affine"], kernel="gauss", regmodel="constant")
print(model.warping())  # e.g. "Affine(a=1.83, b=-0.41)"
```

## References

- Snelson, E., Rasmussen, C. E., & Ghahramani, Z. (2004). *Warped
  Gaussian Processes*. Advances in Neural Information Processing
  Systems 16 (NIPS) — general input/output warping framework of which
  affine rescaling is the linear special case.
