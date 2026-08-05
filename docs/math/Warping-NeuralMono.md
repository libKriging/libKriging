# Input warping — `neural_mono(H)`

## Idea

A small one-hidden-layer neural network, constrained by construction
to be a monotone function of its scalar input, used as a flexible
learned generalization of `boxcox`/`kumaraswamy`: instead of picking a
fixed parametric family, `neural_mono` lets the shape of the warp be
learned directly from the likelihood, while retaining the guarantee
that the warp cannot fold the input space back on itself (important
because a non-monotone warp could make two different x-values produce
the same, or wrongly ordered, correlation behaviour).

```r
wk <- WarpKriging(y, X, warping = c("neural_mono(8)"), kernel = "matern5_2")
```

## Mathematical description

Architecture (H hidden units, default H = 8):

  h = softplus(|W₁|·x + b₁),    w(x) = W₂ᵀ h + b₂

with `softplus(z) = log(1 + eᶻ)` — smooth, strictly increasing, with
positive derivative everywhere — and `|W₁| = exp(raw_W₁)`,
`|W₂| = exp(raw_W₂)` enforced by storing the weights in log-space so
that both layers are guaranteed positive. Since softplus is increasing
and a positive-weighted sum/composition of increasing functions is
increasing, `w` is monotone increasing in x for any parameter values —
no constraint needs to be enforced during optimisation.

Parameter count: 3H + 1 (`raw_W₁`, `b₁`, `raw_W₂` each of size H, plus
scalar bias `b₂`).

- **Forward** (per input xᵢ): `zⱼ = W₁ⱼ·xᵢ + b₁ⱼ`,
  `hⱼ = softplus(zⱼ)`, `φ(xᵢ) = Σⱼ W₂ⱼ·hⱼ + b₂`.
- **Input derivative**: chain rule through softplus'(z) = sigmoid(z):

    dw/dx = Σⱼ W₂ⱼ · sigmoid(zⱼ) · W₁ⱼ

  which is a sum of positive terms (all factors positive) — confirming
  monotonicity numerically, not just by construction.
- **Parameter gradients**: standard backprop through the two linear
  layers and the softplus nonlinearity, with the extra chain-rule
  factor `exp(raw_W)` from the positivity reparametrization.

Compared to `mlp` (unconstrained, can have any shape, and any output
dimension), `neural_mono` trades expressiveness for the monotonicity
guarantee — appropriate when the modeller knows the true input effect
should be monotone (e.g. warping a variable known to have a monotone
but unknown-shape effect) and wants the GP kernel evaluated on that
guaranteed-monotone rescaling rather than risk overfitting a
non-monotone wiggle into a single input dimension.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(40, 1))
y = np.log1p(5 * X[:, 0]) + rng.normal(scale=0.03, size=40)  # monotone, non-power-law shape

model = lk.WarpKriging(y, X, ["neural_mono(8)"], kernel="matern5_2", regmodel="constant")
mean, stdev = model.predict(np.linspace(0, 1, 21).reshape(-1, 1), return_stdev=True)
```

## References

- Snelson, E., Rasmussen, C. E., & Ghahramani, Z. (2004). *Warped
  Gaussian Processes*. Advances in Neural Information Processing
  Systems 16 (NIPS) — general learned-warp framework.
- Archer, N. P., & Wang, S. (1993). *Application of the back
  propagation neural network algorithm with monotonicity constraints
  for two-group classification problems*. Decision Sciences, 24(1),
  60–75; Lang, B. (2005). *Monotonic Multi-Layer Perceptron Networks
  as Universal Approximators*. ICANN 2005 — positive-weight
  construction of provably monotone networks, the principle
  `neural_mono` applies at small scale.
