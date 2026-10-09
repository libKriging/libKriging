# Multi-output Kriging (`MultiOutputKriging`)

## Idea

A simulator often returns several outputs per run: a few scalar quantities,
or a curve sampled at q time steps. `MultiOutputKriging` fits all of them at
once on an **isotopic** design: the q outputs are observed at the same n
points, with no missing value.

The data is a matrix `Y` (n × q), with rows as observations, in the same layout
as `X` (n × d). The `outputModel` string chooses how the outputs are
linked:

| `outputModel` | Model | Hyperparameters | When |
|---|---|---|---|
| `"pca"`, `"pca(K)"`, `"pca(v)"` | Karhunen-Loève reduction, one `Kriging` per score | one (θ, σ², β) per component | many correlated outputs (curves, fields), q ≫ n possible |
| `"shared"` | parallel partial GP: one θ, outputs independent given θ | θ shared, (β_j, σ_j²) per output | a few to many outputs of similar regularity |
| `"separable"` | intrinsic coregionalization model (ICM), free Σ | θ shared, β_j per output, Σ (q × q) | a few correlated outputs, joint covariance or joint simulations needed |

```python
import pylibkriging as lk
mo = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="pca(0.999)")
mean, stdev, cov, deriv = mo.predict(Xnew)        # mean, stdev: m × q
sims = mo.simulate(nsim=100, seed=1, X=Xnew)      # m × q × nsim
```

All three models use the same kernels, trends (`regmodel`), `normalize`
flag and optimizer as `Kriging` (see [Kriging.md](Kriging.md)). A worked
example (a curve-valued simulator, the three output models, an
independent-`Kriging` baseline and `update_simulate`) exists in each
binding: [Python](../../bindings/Python/multioutputkriging_py.ipynb),
[R](../../bindings/R/multioutputkriging_r.ipynb),
[Julia](../../bindings/Julia/multioutputkriging_julia.ipynb),
[Octave](../../bindings/Octave/multioutputkriging_octave.ipynb).

## Mathematical description

### `"pca"`: Karhunen-Loève reduction

Following Higdon et al. (2008), the outputs are centered by their column
means ȳ and, with `normalize=true`, scaled by their column standard deviations
s, so the PCA is computed on correlations instead of covariances. The centered
matrix is decomposed by an SVD:

  Y_c = (Y − 1 ȳᵀ) diag(s)⁻¹ = U S Vᵀ,   λ_k = S_kk² / (n − 1)

The first K right singular vectors give an orthonormal basis Φ (q × K). The
output model is then

  Y ≈ 1 ȳᵀ + (A Φᵀ) diag(s),   A = Y_c Φ   (n × K scores)

- **Choice of K.** `"pca(K)"` with an integer K ≥ 1 keeps K components.
  `"pca(v)"` with 0 < v < 1 keeps the smallest K whose cumulative explained
  variance reaches v. The default `"pca"` is `"pca(0.99)"`. K is capped by
  min(n − 1, rank(Y_c)).
- **Signs.** They are made deterministic: the loading of largest magnitude of
  each axis is positive.
- **Score models.** Each score column A_k gets its own `Kriging`, fitted with
  the same kernel, trend, `normalize`, `optim` and `objective`. The scores are
  uncorrelated on the design, and they are modelled as independent GPs.
- **Truncation residual.** The residual Ρ = Y − reconstruction (n × q,
  original scale) is not thrown away. It is modelled as white noise across
  prediction points, N(0, Σ_res) with Σ_res = ΡᵀΡ / (n − 1). This keeps the
  correlation between outputs without forming the q × q matrix.

Prediction at x combines the K latent predictions (μ_k(x), v_k(x)):

  mean_j(x) = ȳ_j + s_j Σ_k Φ_jk μ_k(x)
  var_j(x)  = s_j² Σ_k Φ_jk² v_k(x) + (Σ_res)_jj

The joint covariance over vec(Y_n) is

  Cov = Σ_k (w_k w_kᵀ) ⊗ C_k + Σ_res ⊗ I_m,   with w_k = s ∘ Φ_k

where C_k is the predictive covariance of latent model k.

- **`simulate`.** Each latent `Kriging` draws its own paths (seed + k). The
  residual is drawn as Ρᵀw / √(n − 1) with w ~ N(0, I_n), independently per
  point and per simulation (seed + K).
- **`update` and `update_simulate`.** Both keep the basis: the new outputs
  Y_u are projected on Φ, and each latent model is conditioned on its new
  scores. `update(refit=true)` recomputes the basis and refits everything.
- **LOO.** `leaveOneOut` is the closed-form leave-one-out of each latent
  model, mapped back to the outputs. The basis itself is computed on all
  data, so the LOO error is slightly optimistic.

### `"shared"`: parallel partial Gaussian process

Gu & Berger (2016) give each output its own trend and variance, with one
correlation shared by all outputs:

  Y_j ~ GP(F β_j, σ_j² r_θ),   outputs independent given θ

- **One factorization.** With R = R_θ(X, X) = L Lᵀ, all outputs share one
  Cholesky factor per likelihood evaluation. The whitened residuals of all
  outputs are solved at once as an n × q right-hand side.
- **Profiling.** β̂_j (GLS) and σ̂_j² = ‖L⁻¹(y_j − F β̂_j)‖² / n are profiled
  out, and θ maximizes the summed concentrated log-likelihood

  ℓ(θ) = Σ_j −½ ( n log(2π σ̂_j²) + log|R| + n )

  Its analytic gradient is the `Kriging` gradient with q whitened columns,
  which costs about one `Kriging` gradient, not q of them.
- **LOO objective.** `objective="LOO"` minimizes the leave-one-out squared
  errors summed over outputs on the normalized scale (Dubrule 1983, closed
  form, analytic gradient).
- **Normalization.** As in `Kriging`, `normalize=true` maps each output to
  [0, 1] through its min and range.
- **Optimizer.** θ bounds are the union of the per-output `Kriging` bounds.
  Starting points, restarts and the `"BFGS<k>"` multistart are those of
  `Kriging`.
- **Constant outputs.** An output that is constant on the design (e.g. the
  initial value of a curve) is reproduced exactly by any trend with an
  intercept. It gets σ_j² = 0 and does not enter the likelihood.
- **q = 1.** With a single output, `"shared"` gives the same model as
  `Kriging(y, X, …)`.

Prediction is the `Kriging` prediction of each output with the shared θ. The
dense joint covariance is block diagonal: Cov = diag(σ_j² s_j²) ⊗ C_x, where
C_x is the universal-kriging correlation of the prediction points given the
data. `update_simulate` conditions the stored draws exactly, as described
below for `"separable"`.

### `"separable"`: intrinsic coregionalization model

The isotopic ICM (Conti & O'Hagan 2010) lets the outputs covary through a
free q × q matrix Σ:

  Cov(vec Y) = Σ ⊗ R_θ

- **Closed form for Σ.** On an isotopic design Σ has a closed form,
  Σ̂ = E*ᵀ E* / n, where E* = L⁻¹(Y − F B̂) are the whitened residuals.
- **Profiled likelihood.** Then

  −2ℓ(θ) = nq log 2π + n log|Σ̂| + q log|R| + nq

  and its gradient is the shared-θ gradient with the residuals also whitened
  across outputs by Σ̂^{-1/2}.
- **Same mean as `"shared"`.** The GLS trend B̂ is the same as in `"shared"`
  for any Σ (autokrigeability of the isotopic ICM). The predictive mean is
  therefore identical; only θ, which maximizes a different likelihood, and
  the covariances differ.
- **Joint covariance.** The covariance couples the outputs:

  Cov(vec Y_n) = Σ_raw ⊗ C_x,   Σ_raw = diag(s) Σ̂ diag(s)

  `predictCovFactors(X_n)` returns the two factors (C_x, Σ_raw) without
  forming the dense mq × mq matrix.
- **Joint simulations.** `simulate` draws one standard normal field per
  output and mixes them with the Cholesky factor of Σ̂. The simulated outputs
  are thus jointly distributed with covariance Σ̂ ⊗ C_x.
- **Requirements.** Σ̂ needs n − p ≥ q, with p the number of trend
  functions. The fit refuses a numerically singular Σ̂, which happens with
  outputs that are nearly linearly dependent, such as finely sampled smooth
  curves. In that case use `"pca"`.

### `update_simulate` for `"shared"` and `"separable"`

With θ and Σ kept, let S be the joint conditional correlation of
(Y(X_n), Y(X_u)) given the data, including the trend uncertainty
(universal kriging). For each stored draw y_n:

1. Draw y_u ~ N(μ_u + S_un S_nn⁻¹ (y_n − μ_n), S_uu − S_un S_nn⁻¹ S_nu),
   mixed across outputs by Σ as in `simulate`. Then (y_n, y_u) is a joint
   draw.
2. Correct by kriging: y_n ← y_n + S_nu S_uu⁻¹ (Y_u − y_u).

The result is distributed as `update(Y_u, X_u, refit=false)` followed by a
new `simulate`. The two agree in distribution, not path by path.

## Cost

| Model | Fit, per objective evaluation | Prediction at m points |
|---|---|---|
| `"pca"` | K × `Kriging` (O(n³) each), plus one SVD O(n q min(n, q)) | K × `Kriging` + O(m q K) |
| `"shared"` | one O(n³) Cholesky + O(n² q) | O(n² m + n m q) |
| `"separable"` | same as `"shared"` + O(q³) | same + O(q²) |

## Current limitations

- **Isotopic design only.** `Y` must be finite (no missing value), and one
  `regmodel` is used for all outputs.
- **No noise channel.** There is no `nugget` and no `noise` for now.
- **Objectives.** `"shared"` and `"separable"` accept `objective="LL"` or
  `"LOO"`. `"pca"` forwards `objective` to each latent `Kriging`.
- **Not implemented yet.** `"separable(<kernel>)"` (Σ = σ² R_t(φ) built from
  output coordinates set with `set_output_coordinates`) is parsed, but `fit`
  throws.
- **No save/load yet.**
- **Bindings.** `MultiOutputKriging` is available in C++ and in every binding
  (Python, R, Octave/MATLAB, Julia). The scikit-learn estimators of
  `pylibkriging.sklearn` do not wrap it.

## References

- Higdon, D., Gattiker, J., Williams, B., & Rightley, M. (2008). *Computer
  model calibration using high-dimensional output*. JASA, 103(482), 570–583
  (the `"pca"` model).
- Gu, M., & Berger, J. O. (2016). *Parallel partial Gaussian process
  emulation for computer models with massive output*. Annals of Applied
  Statistics, 10(3), 1317–1347 (the `"shared"` model; R package
  `RobustGaSP::ppgasp`).
- Conti, S., & O'Hagan, A. (2010). *Bayesian emulation of complex
  multi-output and dynamic computer models*. Journal of Statistical Planning
  and Inference, 140(3), 640–651 (the separable `"separable"` model).
- Álvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). *Kernels for
  vector-valued functions: a review*. Foundations and Trends in Machine
  Learning, 4(3), 195–266 (ICM, LMC, autokrigeability).
- Dubrule, O. (1983). *Cross validation of kriging in a unique
  neighborhood*. Mathematical Geology, 15(6), 687–699 (closed-form LOO).
- The design notes, literature review and comparison of 18 packages are in
  [todo/multi-output/ANALYSIS.md](../../todo/multi-output/ANALYSIS.md).
