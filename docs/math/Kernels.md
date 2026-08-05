# Covariance kernels — `kernel` / `covType`

## Idea

The kernel (a.k.a. correlation function) `ρ(x, x′; θ)` is what encodes
the modelling assumption "points that are close in input space have
correlated responses" — it's the piece of `Kriging` (and every class
built on it) that determines how smooth the fitted surface is allowed
to be. libKriging exposes four stationary, separable (ARD)
correlation families — `"gauss"`, `"exp"`, `"matern3_2"`,
`"matern5_2"` — plus an internal `"whitenoise"` building block that is
not a modelling choice on its own.

```r
k <- Kriging(y, X, kernel = "matern5_2")
```

## Mathematical description

All four are **product-form, one length-scale per dimension** (ARD —
automatic relevance determination): for `dx = x − x′` and per-dimension
range parameters `θ = (θ₁, …, θ_d)`,

  ρ(dx; θ) = ∏ₖ ρ₁(dxₖ / θₖ)

so the joint kernel factorizes across input dimensions, and each θₖ
independently controls how quickly correlation decays along that axis
— a small θₖ means the response can vary quickly along dimension k, a
large θₖ means it must vary slowly (near-linear) along that axis. The
per-dimension factor `ρ₁(u)` is:

| `kernel` | ρ₁(u), u = dxₖ/θₖ | Smoothness (mean-square differentiability) |
|---|---|---|
| `"gauss"` | exp(−u²/2) | C^∞ — infinitely differentiable, analytic |
| `"matern5_2"` | (1 + √5·\|u\| + 5u²/3)·exp(−√5·\|u\|) | twice mean-square differentiable |
| `"matern3_2"` | (1 + √3·\|u\|)·exp(−√3·\|u\|) | once mean-square differentiable |
| `"exp"` | exp(−\|u\|) | continuous but not differentiable (rough, Ornstein–Uhlenbeck) |
| `"whitenoise"` | 1 if dx=0, else 0 | discontinuous — pure nugget-like building block, not a standalone model choice |

`matern3_2`/`matern5_2` are the general-purpose Matérn family (Matérn,
1960; Stein, 1999) at half-integer smoothness ν=3/2 and ν=5/2 — `gauss`
is the ν→∞ limit of the same family. Practical guidance (already in
`SKILL.md` §2): `"matern5_2"` is the sane default — smoother than
`matern3_2` but, unlike `"gauss"`, its correlation matrix does not
become numerically near-singular as points cluster close together
(`"gauss"` is over-smooth and prone to ill-conditioning unless the
underlying function is known to be very smooth/analytic). `"exp"` is
appropriate only when the response is known to be genuinely rough
(non-differentiable).

- **Analytic gradients**: each kernel ships both `∂ln ρ/∂θ` (used by
  the L-BFGS-B optimizer fitting θ, see [Kriging.md](Kriging.md)) and
  `∂ln ρ/∂x` (used by `predict`'s derivative outputs and by the warp
  kernels' chain rule, see [Warping-MLP.md](Warping-MLP.md) *et al.*) —
  no finite differences needed for either.
- **Combining with warping**: `WarpKriging`/`MLPKriging` evaluate the
  same `ρ(·;θ)` on the warped features Φ(x) instead of x directly —
  `k(x,x′) = σ²·ρ(Φ(x) − Φ(x′); θ)` — the kernel family and its
  gradients are unchanged, only what enters `dx` differs (see
  [Warping-MLP.md](Warping-MLP.md), [Warping-MLPJoint.md](Warping-MLPJoint.md)).

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(20, 1))
y_smooth = np.sin(3 * X[:, 0]) + rng.normal(scale=0.01, size=20)
y_rough = np.sign(np.sin(20 * X[:, 0])) + rng.normal(scale=0.05, size=20)

k_smooth = lk.Kriging(y_smooth, X, "gauss")       # analytic underlying function
k_rough  = lk.Kriging(y_rough, X, "exp")          # rough / discontinuous-like function
k_default = lk.Kriging(y_smooth, X, "matern5_2")  # safe general-purpose default
```

## References

- Matérn, B. (1960). *Spatial Variation*. Meddelanden från Statens
  Skogsforskningsinstitut, 49(5) (the Matérn covariance family
  underlying `matern3_2`/`matern5_2`).
- Stein, M. L. (1999). *Interpolation of Spatial Data: Some Theory for
  Kriging*. Springer (smoothness properties of the Matérn/Gaussian
  families, and why the Gaussian kernel's extreme smoothness causes
  numerical ill-conditioning in practice).
- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes
  for Machine Learning*. MIT Press, §4.2 (covariance function
  properties, ARD).
