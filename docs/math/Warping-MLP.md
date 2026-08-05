# Input warping — `mlp(h1:h2,d_out,act)`

## Idea

An unconstrained multi-layer perceptron applied to one input variable,
mapping it to a (possibly multi-dimensional) feature vector before the
GP kernel sees it. Unlike `neural_mono`, `mlp` is not constrained to be
monotone and can output more than one feature per input — it is the
most expressive of the per-variable warps and, per the class
documentation, "subsumes all continuous warps (Affine, BoxCox,
Kumaraswamy, NeuralMono) as special cases" a wide-enough network could
in principle approximate. Use it when there is enough data to afford
the extra parameters and no reason to believe the input's effect is
monotone.

```r
wk <- WarpKriging(y, X, warping = c("mlp(16:8,2,selu)"), kernel = "gauss")
```

## Mathematical description

For hidden layer sizes (h₁, h₂, …) and output dimension `d_out`
(default `mlp` ⟹ `mlp(16:8,2,selu)`):

  H₀ = x  (scalar input, batched over observations)
  Hₗ = act(Hₗ₋₁ Wₗ + bₗ)   for hidden layers,   φ(x) = H_L = H_{L−1} W_L + b_L

i.e. a standard feed-forward network with an activation on every
hidden layer and a linear (no activation) output layer producing
`φ(x) ∈ ℝ^{d_out}`. Supported activations: `relu`, `selu` (default),
`tanh`, `sigmoid`, `elu`.

- **Forward/backward**: standard dense-layer forward pass and
  backpropagation (`backward` returns `dL/d(all weights and biases)`
  given `dL/dφ`); the input-side Jacobian `deriv_input` is obtained the
  same way, propagating a one-hot seed back through the same layers.
- **Parameter count**: the usual sum of `(fan_in+1)·fan_out` per dense
  layer.
- **Activation choice matters for optimisation, not just expressivity**:
  the GP is fit by single-start gradient-based optimisation
  (`optim="BFGS"` by default with no restarts) on top of this warp;
  `selu`'s kink at z=0 can make the profiled likelihood surface locally
  jagged there and destabilize a single BFGS run (the analytic
  gradient stays correct — it is an optimization-landscape issue, not
  a bug). Prefer `activation="tanh"` (smooth everywhere) unless
  `selu`'s less-saturating behaviour is specifically wanted, especially
  for deep/wide networks.

For the corresponding *joint* (all-inputs-at-once) network, see
[Warping-MLPJoint.md](Warping-MLPJoint.md) — `mlp` here still warps one
input column independently, like every other per-variable warp in this
family.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(60, 1))
y = np.sin(3 * X[:, 0]) * np.cos(9 * X[:, 0]) + rng.normal(scale=0.03, size=60)

model = lk.WarpKriging(y, X, ["mlp(8:4,2,tanh)"], kernel="matern5_2",
                        regmodel="constant", normalize=True)
```

## References

- Calandra, R., Peters, J., Rasmussen, C. E., & Deisenroth, M. P.
  (2016). *Manifold Gaussian Processes for Regression*. IEEE
  International Joint Conference on Neural Networks (IJCNN), 3338–3345.
- Snelson, E., Rasmussen, C. E., & Ghahramani, Z. (2004). *Warped
  Gaussian Processes*. Advances in Neural Information Processing
  Systems 16 (NIPS).
