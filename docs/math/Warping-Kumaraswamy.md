# Input warping — `kumaraswamy`

## Idea

The Kumaraswamy CDF (Kumaraswamy, 1980) is a flexible, closed-form,
strictly monotone S-shaped map on [0, 1] — a cheap, smooth alternative
to the Beta CDF that avoids the incomplete-Beta function. Snoek et al.
(2014) popularized it as an input warp for Bayesian optimization: it
lets the GP compress or stretch different regions of a bounded input
range (e.g. spend more "correlation length" resolution near 0 or near
1) to correct for non-stationary behaviour without changing the base
kernel family.

```r
wk <- WarpKriging(y, X, warping = c("kumaraswamy"), kernel = "gauss")
```

## Mathematical description

  w(x) = 1 − (1 − x^a)^b,    x ∈ [0, 1], a, b > 0

2 learnable parameters, stored as `log_a`, `log_b` (unconstrained
reals) so that `a = exp(log_a)`, `b = exp(log_b)` stay positive by
construction; initialised at a = b = 1 (identity map). Inputs are
clamped to `[1e−10, 1 − 1e−10]` to keep the transform well-defined at
the boundaries.

- **Input derivative**:

    dw/dx = a·b · x^(a−1) · (1 − x^a)^(b−1)

- Both a and b control the S-curve's steepness/skew: a < 1 stretches
  near 0, a > 1 compresses near 0 (and symmetrically for b near 1);
  a = b = 1 is the identity. Gradients w.r.t. `log_a`/`log_b` are
  obtained by backpropagating `dL/dφ` through the chain rule (product
  and power rules on the expression above).

Requires the raw input to already live in [0, 1] — combine with
`normalize=true` at the `WarpKriging` level, or pre-scale the column,
if the natural range of the variable isn't already [0, 1].

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(30, 1))
# response with sharper variation near x=0 -> kumaraswamy should skew there
y = np.exp(-8 * X[:, 0]) + rng.normal(scale=0.02, size=30)

model = lk.WarpKriging(y, X, ["kumaraswamy"], kernel="gauss",
                        regmodel="constant", normalize=True)
print(model.warping())  # e.g. "Kumaraswamy(a=..., b=...)"
```

## References

- Kumaraswamy, P. (1980). *A generalized probability density function
  for double-bounded random processes*. Journal of Hydrology, 46(1–2),
  79–88.
- Snoek, J., Swersky, K., Zemel, R., & Adams, R. P. (2014). *Input
  Warping for Bayesian Optimization of Non-Stationary Functions*.
  Proceedings of the 31st ICML, PMLR 32(2), 1674–1682.
