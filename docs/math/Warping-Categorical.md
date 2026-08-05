# Input warping — `categorical(L,q)` / `categorical(["a","b","c"],q)`

## Idea

Standard Gaussian-process kernels assume the input lives in a metric
space where "distance" is meaningful — which breaks down for a
categorical variable with unordered levels (a material type, a factory
site, a product family). `categorical` embeds each of the `L` levels
as a learned point `eₗ` in a small Euclidean space ℝ^q; the GP kernel
then measures distance between *embeddings*, letting the model
discover which levels behave similarly (small embedded distance) or
differently (large embedded distance), purely from the data's
likelihood — the analogue for GPs of a categorical embedding layer in
a neural network (Garrido-Merchán & Hernández-Lobato, 2020).

```r
wk <- WarpKriging(y, X, warping = c("categorical(5,2)", "none"), kernel = "gauss")
# or with named levels:
wk <- WarpKriging(y, X, warping = c('categorical(["A","B","C"],2)'), kernel = "gauss")
```

## Mathematical description

Input column encodes each observation's level as an integer code
`ℓ ∈ {0, …, L−1}` (or a name, mapped to a code via `level_names`). The
warp is a direct table lookup:

  φ(x) = E[ℓ, :] ∈ ℝ^q,     E ∈ ℝ^{L×q} a learned embedding matrix

L·q learnable parameters (one length-q embedding vector per level),
randomly initialised. There is no notion of an "input derivative" —
the mapping is not continuous — so `deriv_input` returns zero
(consistent with treating the level as a discrete label, not something
to differentiate a prediction with respect to).

- **Forward**: `φ(xᵢ) = E[round(xᵢ), :]` (the level code, rounded to
  the nearest integer for robustness, must satisfy
  `0 ≤ level < L` — validated at fit/predict/simulate/update time,
  raising a clear out-of-range error otherwise).
- **Parameter gradient**: `dL/dE[level, :] += (dL/dφ)ᵢ` for each
  observation i whose level equals `level` — a scatter-add, since
  every observation at that level shares the same row of E.

The embedding dimension q trades expressiveness (larger q lets levels
be arranged more freely) against overfitting risk (L·q parameters must
be estimated jointly with θ/σ²/β from n observations — keep q small,
q=2 by default, unless there are many observations per level).

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
n = 60
levels = rng.integers(0, 4, size=n)              # 4 categorical levels, encoded 0..3
x_cont = rng.uniform(size=n)
X = np.column_stack([levels.astype(float), x_cont])
level_effect = np.array([0.0, 1.5, 1.5, -2.0])   # levels 1 and 2 behave similarly
y = level_effect[levels] + np.sin(3 * x_cont) + rng.normal(scale=0.05, size=n)

model = lk.WarpKriging(y, X, ["categorical(4,2)", "none"], kernel="matern5_2",
                        regmodel="constant")
```

## References

- Garrido-Merchán, E. C., & Hernández-Lobato, D. (2020). *Dealing with
  categorical and integer-valued variables in Bayesian Optimization
  with Gaussian processes*. Neurocomputing, 380, 20–35.
- Roustant, O., Padonou, E., Deville, Y., et al. (2020). *Group kernels
  for Gaussian process metamodels with categorical inputs*. SIAM/ASA
  Journal on Uncertainty Quantification, 8(2), 775–806.
- Saves, P., et al. (2023). *SMT 2.0: A surrogate modeling toolbox with
  a focus on hierarchical and mixed variables Gaussian processes*.
  Advances in Engineering Software.
