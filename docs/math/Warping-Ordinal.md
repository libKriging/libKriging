# Input warping — `ordinal(L)` / `ordinal(["low","med","high"])`

## Idea

Between a fully unordered categorical variable (`categorical`, no
notion of order) and a continuous one (implicit equal spacing) sits
the ordinal variable: levels have a known order (low < medium < high;
grade 1 < grade 2 < grade 3) but the *gaps* between consecutive levels
are not necessarily equal, and shouldn't be assumed equal by treating
the level index as a plain number. `ordinal` learns those gaps
directly, placing the `L` levels at estimated positions on the real
line while guaranteeing the order is preserved — so the GP kernel
still sees a genuine 1-D distance, just not a uniformly-spaced one.

```r
wk <- WarpKriging(y, X, warping = c("ordinal(4)"), kernel = "matern5_2")
# or with named levels:
wk <- WarpKriging(y, X, warping = c('ordinal(["low","med","high"])'), kernel = "matern5_2")
```

## Mathematical description

Input encodes each observation's level as an integer code
`ℓ ∈ {0, …, L−1}` respecting the known order. Positions are built from
L−1 learnable positive gaps:

  z₀ = 0,     zₗ = z_{l−1} + exp(gap_{l−1})   for l = 1, …, L−1
  φ(x) = z_{round(x)}

so `z₀ < z₁ < … < z_{L−1}` is guaranteed for any parameter values (each
increment is `exp(·) > 0`) — order is enforced by construction, exactly
as `knots`' positive-slope reparametrization guarantees monotonicity.
L−1 learnable parameters (`raw_gaps`, unconstrained), randomly
initialised.

- **Forward**: table lookup `φ(xᵢ) = z_{round(xᵢ)}`, `round(xᵢ)` must
  satisfy `0 ≤ level < L` (validated, as for `categorical`).
- **Input derivative**: zero, same reasoning as `categorical` — the
  mapping between discrete codes and positions is not something to
  differentiate with respect to the raw code.
- **Parameter gradient**: `∂zₗ/∂gap_k = exp(gap_k)` for every `k < l`
  (each earlier gap shifts every later position by its own increment),
  zero for `k ≥ l`; backpropagated as
  `dL/d(gap_k) = Σᵢ (dL/dφ)ᵢ · (∂z_{level(i)}/∂gap_k)`.

Compare to feeding the raw level index directly as a continuous
input: that would implicitly fix all gaps to 1, which `ordinal`
relaxes while still respecting the known order — useful when, e.g.,
"medium" is known to behave much closer to "high" than to "low".

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
n = 60
levels = rng.integers(0, 3, size=n)   # ordered: 0=low, 1=medium, 2=high
# medium and high behave similarly (small gap); low is far from both
level_effect = np.array([0.0, 3.0, 3.3])
X = levels.astype(float).reshape(-1, 1)
y = level_effect[levels] + rng.normal(scale=0.05, size=n)

model = lk.WarpKriging(y, X, ["ordinal(3)"], kernel="matern5_2", regmodel="constant")
```

## References

- Qian, P. Z. G., Wu, H., & Wu, C. F. J. (2008). *Gaussian Process
  Models for Computer Experiments with Qualitative and Quantitative
  Factors*. Technometrics, 50(3), 383–396.
- Garrido-Merchán, E. C., & Hernández-Lobato, D. (2020). *Dealing with
  categorical and integer-valued variables in Bayesian Optimization
  with Gaussian processes*. Neurocomputing, 380, 20–35.
