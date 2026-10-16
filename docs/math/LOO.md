# The `objective="LOO"` fitting criterion (leave-one-out)

## Idea

Instead of maximizing the (marginal or profiled) likelihood, `"LOO"`
picks θ to directly minimize leave-one-out cross-validation error: for
each training point i, "how well would the model predict yᵢ if it had
never seen it?" This is a more *predictive* criterion than `"LL"` — it
can behave better when the goal is accuracy at (or near) the design
points specifically, rather than the best global fit of the assumed
GP model, and it is less sensitive to the correlation-family
mis-specification the likelihood is comparing against.

Computing this naively would mean refitting the model n times (once
per left-out point) — libKriging instead uses the closed-form
"virtual LOO" trick shared with DiceKriging, which gets the exact same
answer from *one* factorization of the full design.

```r
k <- Kriging(y, X, kernel = "matern5_2", objective = "LOO")
```

## Mathematical description

### The virtual-LOO trick (no refitting)

Let `R = LLᵀ` be the correlation matrix's Cholesky factor and define
the "hat-removing" matrix

  Q = R⁻¹ − R⁻¹F(FᵀR⁻¹F)⁻¹FᵀR⁻¹

(the same `Q` that appears in the GLS residual projection — it is the
matrix that maps `y` onto the vector of *trend-adjusted* residuals).
Then, remarkably, the leave-one-out prediction error and variance at
point i are obtained directly from the diagonal of `Q`, without ever
removing point i and refitting (Dubrule, 1983; Sundararajan & Keerthi,
2001, give the same identity for GP regression):

  σ²_LOO,ᵢ = 1 / Qᵢᵢ
  errorLOO,ᵢ = σ²_LOO,ᵢ · (Qy)ᵢ    =    yᵢ − ŷ₋ᵢ(xᵢ)

i.e. `ŷ₋ᵢ(xᵢ) = yᵢ − errorLOO,ᵢ` is exactly the prediction the model
*would* have made at xᵢ had it been fit on the other n−1 points only —
obtained here in O(n²) total (dominated by the one initial Cholesky
factorization and inversion) instead of O(n) separate O((n−1)³) fits.

### Objective and gradient

  LOO(θ) = (1/n) · Σᵢ errorLOO,ᵢ²

is minimized over θ (equivalently, libKriging's optimizer maximizes
`−LOO(θ)`, so it fits the same `optim="BFGS..."` machinery as every
other objective). The analytic gradient reuses the same `Q`-based
identities — for each range parameter θₖ, with `gradR_k = ∂R/∂θₖ`:

  ∂Qᵢᵢ/∂θₖ  = −[Q · gradR_k · Q]ᵢᵢ
  ∂σ²_LOO,ᵢ/∂θₖ = −σ⁴_LOO,ᵢ · ∂Qᵢᵢ/∂θₖ
  ∂errorLOO/∂θₖ = (∂σ²_LOO/∂θₖ) ⊙ (Qy) − σ²_LOO ⊙ (Q · gradR_k · Qy)
  ∂LOO/∂θₖ = (2/n) · errorLOOᵀ · (∂errorLOO/∂θₖ)

— again all obtained from the same cached `Q`/Cholesky factors, no
finite differences.

### Approximate versions (`method="vecchia(m)"` / `"nystrom(k)"`)

With an approximate fit method (noise-free models only: on a nugget /
heterogeneous-noise model, the fit and every approximate evaluation —
`leaveOneOut("vecchia")`, `logMargPostNystromFun`, … — raise),
`objective="LOO"` is the leave-one-out criterion **of the approximate
Gaussian model**: the approximation replaces the correlation precision
R⁻¹ by a precision P that is cheap to apply, and the virtual-LOO
formulas above are evaluated with P in place of R⁻¹:

  Q = P − PF(FᵀPF)⁻¹FᵀP,   Qᵢᵢ = Pᵢᵢ − gᵢᵀ(FᵀPF)⁻¹gᵢ  (gᵢ = row i of PF)
  errorLOO,ᵢ = (Qy)ᵢ / Qᵢᵢ = (P(y − Fβ̂))ᵢ / Qᵢᵢ,   σ²_LOO,ᵢ = σ²/Qᵢᵢ
  LOO(θ) = (1/n) Σᵢ errorLOO,ᵢ²

(β̂ = GLS estimate under P; with a known β, Q = P). This is the exact
leave-one-out error of the *approximate joint law* N(Fβ, σ² P⁻¹),
computed in one pass:

- **Vecchia**: P = BᵀD⁻¹B, with B unit lower triangular in the maxmin
  order (row t: 1 at the t-th ordered point, −aₜ on its m neighbors
  N(t), aₜ = R_NN⁻¹ r_N) and D = diag(vₜ), vₜ = 1 − r_Nᵀaₜ — the same
  m × m solves as the Vecchia likelihood. Pᵢᵢ = Σₜ Bₜᵢ²/vₜ is a scatter
  over the O(n m) nonzeros of B, and P·v costs O(n m), so the whole
  criterion costs O(n m³ + n m p) — no n × n matrix.
- **Nystrom**: P = R̂⁻¹ with R̂ = UUᵀ + diag(D) (fixed landmarks),
  through the block Cholesky factorization of
  [Nystrom.md](Nystrom.md#numerical-factorization) (landmark rows
  first): diag(P) and P·v cost O(n k); total O(n k²).

**This is not the error of the Vecchia (local) predictor.**
`predict(…, method="vecchia")` predicts xᵢ from its own m nearest
*observations*, whereas the virtual LOO of the Vecchia joint law
conditions yᵢ on all the other observations through P (both its
"parents" N(i) and the later points that have i as a neighbor). The two
coincide only for m = n − 1 (P = R⁻¹, exact LOO). The Vecchia LOO also
depends on the maxmin **ordering** (P does, unless m = n − 1). The
Nystrom LOO inherits the low-rank **smoothing bias**: R̂ under-represents
the short-range correlation not captured by the k landmarks (it is put
in the diagonal D), so for small k the LOO errors look like those of a
smoother model plus a local "nugget" D — it is exact for k = n only.

**Gradient** (analytic, FD-verified in `tests/KrigingApproxObjectivesTest.cpp`).
With r = y − Fβ̂, A = FᵀPF, G = PF, H = I − FA⁻¹FᵀP (so Q = HᵀPH and
dQ = Hᵀ dP H), c = e/diag(Q) and w = c ∘ e (e the LOO errors):

  ∂LOO/∂θₖ = (2/n) [ (Hc)ᵀ ∂ₖP r − tr(∂ₖP · H diag(w) Hᵀ) ]
  H diag(w) Hᵀ = diag(w) + F C₁ᵀ + C₁ Fᵀ + F C₂ Fᵀ,
  C₁ = −diag(w) G A⁻¹,  C₂ = A⁻¹ Gᵀ diag(w) G A⁻¹

so only the bilinear forms uᵀ ∂ₖP v (for u, v among r, Hc, F, C₁) and the
weighted diagonal Σᵢ wᵢ (∂ₖP)ᵢᵢ are needed:

- Vecchia: ∂P = ∂BᵀD⁻¹B + BᵀD⁻¹∂B − BᵀD⁻¹∂D D⁻¹B, with ∂aₜ, ∂vₜ from the
  same block solves as the Vecchia likelihood gradient;
  (∂P)ᵢᵢ = Σₜ 2Bₜᵢ∂Bₜᵢ/vₜ − Bₜᵢ²∂vₜ/vₜ². O(n m p) per θₖ.
- Nystrom: ∂P = −P ∂R̂ P with ∂R̂ = ∂M Wᵀ + W ∂Mᵀ − W ∂K Wᵀ − diag(δ)
  (M = R_ns, K = R_ss, W = MK⁻¹; δ removes the diagonal of the low-rank
  part, since R̂ᵢᵢ ≡ 1 while Dᵢ is not clamped). The weighted diagonal is
  −tr(∂R̂ · P diag(w) P), and P diag(w) P = diag(w/D²) − (w/D ∘ Y)Yᵀ −
  Y(w/D ∘ Y)ᵀ + Y(Yᵀdiag(w)Y)Yᵀ is diagonal plus rank-k terms, so each
  trace is a k × k bilinear form; in particular (Q∘Q)-type sums
  Σⱼ δⱼ Pᵢⱼ² reduce to yᵢᵀ(Σⱼ δⱼ yⱼyⱼᵀ)yᵢ. O(n k² ) per θₖ.

The committed σ² and β follow the exact LOO commit: β = GLS under P and
σ² = (y − Fβ̂)ᵀP(y − Fβ̂)/n. `leaveOneOut(method)`,
`leaveOneOutVec(θ, method)` (C++ `leaveOneOutVecchiaFun` /
`leaveOneOutNystromFun` for the value and gradient at a given θ)
evaluate this criterion; `leaveOneOut()` stays the exact (O(n³)) one,
available on any fit. As the exact functions, they work in the original
scale: with `normalize=true`, θ is read in the units of X, the value is the
mean squared LOO error in the units of y (squared) and the gradient is
chain-ruled back, so `leaveOneOut(method)` equals
`leaveOneOutVecchiaFun(theta())` (resp. Nystrom), and a `normalize=true`
model gives the `normalize=false` values whenever the approximation is
itself scale-free (m = n − 1, k = n; otherwise the Vecchia neighbors and
Nystrom landmarks are chosen in the normalized inputs).

### When to prefer it over `"LL"`

`"LOO"` directly targets predictive accuracy at the design points,
which can matter more than the full-likelihood fit when the assumed
kernel family is only an approximation of the true correlation
structure, or when the trend model is a poor global fit — cases where
maximizing `"LL"` can trade a bit of local predictive accuracy for a
better overall Gaussian fit. It has no equivalent for extrapolation
quality away from the data, and (being a cross-validation criterion,
not a proper likelihood) it does not extend naturally to model
comparison across different kernel families the way `"LL"`/`"LMP"` do.

## Simple example

```python
import numpy as np
import pylibkriging as lk

rng = np.random.default_rng(0)
X = rng.uniform(size=(20, 1))
y = np.sin(3 * X[:, 0]) + rng.normal(scale=0.02, size=20)

k_ll = lk.Kriging(y, X, "matern5_2", objective="LL")
k_loo = lk.Kriging(y, X, "matern5_2", objective="LOO")

yhat, sd = k_loo.leaveOneOutVec(k_loo.theta())  # LOO predictions/stdev at each X_i
```

## References

- Dubrule, O. (1983). *Cross validation of kriging in a unique
  neighborhood*. Mathematical Geology, 15(6), 687–699 (the virtual-LOO
  formula for kriging, no refitting required).
- Sundararajan, S., & Keerthi, S. S. (2001). *Predictive Approaches for
  Choosing Hyperparameters in Gaussian Processes*. Neural Computation,
  13(5), 1103–1118 (LOO as a GP hyperparameter-selection criterion,
  with the same closed-form identity).
- Roustant, O., Ginsbourger, D., & Deville, Y. (2012). *DiceKriging,
  DiceOptim: Two R Packages for the Analysis of Computer Experiments by
  Kriging-Based Metamodeling and Optimization*. Journal of Statistical
  Software, 51(1), 1–55 (`leaveOneOutFun`/`leaveOneOutGrad`, the
  reference implementation this objective mirrors).
