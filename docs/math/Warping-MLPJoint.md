# Input warping — `mlp_joint(h1:h2,d_out,act)` (a.k.a. `MLPKriging`)

## Idea

Where every other warp in this family is applied independently, one
input column at a time, `mlp_joint` feeds the *entire* input vector
through a single MLP jointly — letting the network learn interactions
between input dimensions before the GP kernel ever sees them. This is
deep kernel learning / "Manifold GP" (Calandra et al., 2016): the
network learns a feature map Φ: ℝ^d → ℝ^{d_out} such that a plain
stationary kernel on Φ(x) captures cross-variable, non-stationary
structure a per-variable warp cannot. `MLPKriging` is a thin facade
that always uses exactly this warp — `WarpKriging(y, X,
["mlp_joint(...)"], kernel)` and `MLPKriging(y, X, hidden_dims=...,
d_out=..., ...)` are the same model.

```r
mk <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2, activation = "selu", kernel = "gauss")
```

When used, the warping spec must be exactly one `mlp_joint(…)` entry —
it replaces all per-variable warps, since it consumes every input
dimension at once.

## Mathematical description

Model:

  y(x) = f(Φ(x))ᵀβ + ζ(x),    Φ(x) = MLP(x; W) ∈ ℝ^{d_out}
  Cov[ζ(x), ζ(x′)] = σ² · k_base(Φ(x), Φ(x′) ; θ)

For hidden layer sizes (h₁, h₂, …) and output dimension `d_out`
(default `mlp_joint` ⟹ `mlp_joint(h1:h2)` with `d_out=2`,
`activation=selu`):

  H₀ = x  (full input row, d_in-dimensional)
  Hₗ = act(Hₗ₋₁ Wₗ + bₗ)   for hidden layers,   Φ(x) = H_L = H_{L−1} W_L + b_L

identical dense-layer structure to `mlp` (§ [Warping-MLP.md](Warping-MLP.md)),
but operating on the full row vector `x ∈ ℝ^d` instead of a scalar
column, so `W₁` has shape `d_in × h₁` rather than `1 × h₁`. The
analytical input Jacobian `∂Φ/∂x` (d_out × d_in) is available for
downstream use (e.g. predictive-derivative queries) via the same
backprop machinery.

All parameters — network weights W, kernel range θ, profiled (σ², β) —
are optimised jointly by maximising the marginal log-likelihood; θ, σ²
and β are concentrated out analytically at each network-weight
iterate, exactly as in plain `Kriging`.

`(θ, σ², β) as functions of W` recovers a standard GP regression on the
learned features, so the interesting degrees of freedom are entirely in
W; unlike `mlp`'s per-column use as *one factor* among several warps,
here the network has to reconstruct any needed input scaling itself
(there is no separate per-variable normalization once `mlp_joint` is
in effect for that variable's contribution to Φ).

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
n = 200
X = rng.uniform(size=(n, 2))
# response with a genuine cross-variable interaction
y = np.sin(3 * (X[:, 0] + X[:, 1])) + rng.normal(scale=0.03, size=n)

model = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation="selu",
                       kernel="gauss", regmodel="constant")
mean, stdev = model.predict(rng.uniform(size=(10, 2)), return_stdev=True)
```

## References

- Calandra, R., Peters, J., Rasmussen, C. E., & Deisenroth, M. P.
  (2016). *Manifold Gaussian Processes for Regression*. IEEE
  International Joint Conference on Neural Networks (IJCNN), 3338–3345.
- Wilson, A. G., Hu, Z., Salakhutdinov, R., & Xing, E. P. (2016). *Deep
  Kernel Learning*. Proceedings of the 19th International Conference
  on Artificial Intelligence and Statistics (AISTATS), PMLR 51,
  370–378.
