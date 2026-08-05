# Input warping — `boxcox`

## Idea

The Box-Cox transform (Box & Cox, 1964) is the classical power
transform used to stabilize variance / make a skewed variable more
"Gaussian-like". Used as an input warp instead of an output transform,
it lets the GP treat an input whose *effect* on the response is
naturally multiplicative or power-law (concentrations, sizes, rates —
quantities that are positive and often skewed) on a scale where a
stationary kernel's constant length-scale assumption is more
reasonable.

```r
wk <- WarpKriging(y, X, warping = c("boxcox"), kernel = "matern5_2")
```

## Mathematical description

  w(x) = (xᵏ − 1) / λ   for λ ≠ 0,      w(x) = log(x)   for λ → 0

1 learnable parameter λ (stored internally unconstrained on ℝ; the
active regime is selected by whether |λ| exceeds a numerical
tolerance, ~1e−6, below which the log limit is used to avoid the
0/0 singularity). Requires x > 0 (inputs are clamped to a small
positive floor, 1e−10, to avoid the transform blowing up right at the
boundary).

- **Input derivative**: `dw/dx = x^(λ−1)` (for λ ≠ 0), `dw/dx = 1/x`
  in the log limit.
- **Parameter gradient** (λ ≠ 0):

    dw/dλ = [xᵏ(λ·ln(x) − 1) + 1] / λ²

  backpropagated as `dL/dλ = Σᵢ (dL/dφ)ᵢ · (dw/dλ)ᵢ`.

λ = 1 recovers an affine map (shifted identity); λ → 0 gives the log
transform; λ = 0.5 is a square-root-like compression. Because λ is
optimised jointly with the GP hyperparameters by maximum likelihood,
the model picks whichever power best linearizes the input–output
relationship for that variable.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
# a response driven by a power-law effect of a positive input
X = rng.uniform(0.1, 5.0, size=(30, 1))
y = np.log(X[:, 0]) + rng.normal(scale=0.05, size=30)

model = lk.WarpKriging(y, X, ["boxcox"], kernel="matern5_2", regmodel="constant")
print(model.warping())  # lambda close to 0 (log-like warp) is expected here
```

## References

- Box, G. E. P., & Cox, D. R. (1964). *An Analysis of Transformations*.
  Journal of the Royal Statistical Society: Series B, 26(2), 211–252.
- Snelson, E., Rasmussen, C. E., & Ghahramani, Z. (2004). *Warped
  Gaussian Processes*. Advances in Neural Information Processing
  Systems 16 (NIPS).
