---
name: libkriging
description: Use whenever writing or reviewing code that fits, predicts, or simulates a Gaussian-process / Kriging model with libKriging, in C++, Python (pylibkriging), R (rlibkriging), Julia (jlibkriging), Octave, or MATLAB. Covers which class to use for a given problem (plain GP, noisy data, mixed/categorical inputs, large n), and which kernel/trend/objective/optimizer options to pass. Trigger on mentions of Kriging, GP regression, surrogate model, emulator, NestedKriging, WarpKriging, MLPKriging, Vecchia, Nystrom (method="vecchia"/"nystrom", or the deprecated LLVecchia/LLNystrom objectives), or any of the libKriging bindings above.
---

# libKriging usage

libKriging is a C++ Gaussian-process (Kriging) library with equivalent
bindings in Python, R, Julia, Octave and MATLAB. This skill tells an
assistant *which* class and options fit a given modelling problem; for the
exact call syntax in one language, open `references/<language>.md`.

All bindings expose the same concepts under the same names (`kernel`,
`regmodel`/`trend`, `objective`, `optim`, `noise`, `parameters`), so the
decision guide below is language-agnostic — only the syntax differs.

## 1. Which class?

Ask, in order:

1. **Is the response noise-free (deterministic simulator, e.g. a CFD/FEM
   code)?**
   → `Kriging` (interpolates the data exactly). See
   [Kriging.md](../../docs/math/Kriging.md) for the model/objectives math.

2. **Is the response noisy?**
   → still `Kriging`, not a separate class (see
   [Noise.md](../../docs/math/Noise.md) for the math):
   - Known per-observation noise variance → pass it as `noise` (a vector).
     The noise of new points (`update(y_u, noise_u, X_u)`, required for this
     model) and `simulate`'s `with_noise` are variances too (same for
     `WarpKriging` / `MLPKriging`'s C++ `simulate(..., with_noise)`).
   - Unknown, but homogeneous noise → pass `noise="nugget"` (estimates one
     shared nugget term).
   - **Do not suggest `NuggetKriging` or `NoiseKriging`.** These used to be
     separate classes; both have been merged into `Kriging`'s `noise`
     argument. The old class names only still exist to *load* models saved
     by older libKriging versions (see `load()` / `load_kriging()`), never
     to fit new ones.

3. **Do inputs include categorical, ordinal, or otherwise non-linearly-scaled
   variables, or do you want a learned input transform (Box-Cox, embeddings,
   a small monotone network, …) instead of hand-picked features?**
   → `WarpKriging`. It fits a per-variable warp jointly with the GP
   hyperparameters by maximum likelihood; the public API mirrors `Kriging`
   (`fit`, `predict`, `simulate`, `update`, `summary`, `logLikelihood`). Its
   `noise` is a per-observation variance vector only (there is no
   `"nugget"` mode), and it always fits with the `"LL"` objective.

4. **Do you want a single deep/joint feature map across *all* inputs
   (deep-kernel learning), rather than one warp per variable?**
   → `MLPKriging`, a thin facade over `WarpKriging` with a joint MLP feature
   map. Prefer plain `WarpKriging` when a per-variable warp is enough —
   it's cheaper to fit and easier to interpret.

5. **Is `n` (number of observations) too large for an O(n³) fit — say,
   more than a few thousand points?** (See
   [Scalability.md](../../docs/math/Scalability.md) for the full method
   inventory and combinability notes.)
   → Keep `Kriging` (not `WarpKriging` / `MLPKriging`: they accept
   `method` but only implement `"exact"`), any `objective`
   (`"LL"` by default; `"LOO"` and `"LMP"` have approximate versions too,
   noise-free only; `"LL"` also with a noise model),
   but switch the fit `method` to one of the two scalable approximations
   (O(n·m³) / O(n·k²) instead of O(n³); each approximates the
   criterion, so don't expect bit-identical results to the exact fit):
   - **Low-to-moderate dimension, spatially local structure** →
     `method="vecchia"` or `"vecchia(m)"` (default `m=30`
     conditioning neighbors). Degrades in higher dimension because nearest
     neighbors become less informative (recommended for d ≲ 5). Works
     with noisy data too (`noise="nugget"` or known noise variances:
     "response" Vecchia, accurate while the noise is small to moderate
     relative to the signal variance; increase `m` or check against an
     exact fit on a subsample when the noise dominates). See
     [Vecchia.md](../../docs/math/Vecchia.md).
   - **Higher dimension, or no reliable local/spatial structure** →
     `method="nystrom"` or `"nystrom(k)"` (default `k=50`
     landmarks): a global low-rank approximation, dimension-robust in the
     same way `NestedKriging`'s `NK` aggregation is, but a single model
     rather than a partition. Works with noisy data too (same `noise=`
     argument as the exact fit; `simulate` with `method="nystrom"` draws
     the latent process, no `with_noise`; a heterogeneous model's `update`
     needs the new points' `noise`). See
     [Nystrom.md](../../docs/math/Nystrom.md).
   - The method is explicit at every later step, never inferred from the
     fit: `predict` / `simulate` / `logLikelihood` / `leaveOneOut` /
     `leaveOneOutVec` / `logMargPost` take
     `method="exact"` (default), `"vecchia"` / `"vecchia(m)"` or
     `"nystrom"`. A Vecchia or Nystrom fit holds no exact factorization,
     so pass `method="vecchia"` / `"nystrom"` there (or `k.method()` as
     is) — `"exact"` raises; approximate methods give mean / stdev only.
     For the exact model at the fitted θ, make a separate exact fit with
     `optim="none"` and that `theta` (`is_theta_estim` false). `update`
     and `save` / `load` keep the method. See the table in
     [bindings/README.md](../../bindings/README.md).
   - `objective="LLVecchia(m)"` / `"LLNystrom(k)"` are **deprecated**
     aliases of `objective="LL"` with `method="vecchia(m)"` /
     `"nystrom(k)"` — write the `method=` form in new code.

6. **Is `n` large enough (~10⁴–10⁶) that even Vecchia/Nystrom are too
   slow, and the design can be partitioned / prediction can be
   parallelized?**
   → `NestedKriging`: splits `(X, y)` into groups, fits one submodel per
   group (`Kriging` by default, `WarpKriging` if a warp spec is given),
   unifies hyperparameters, then aggregates predictions;
   `method="vecchia(m)"` / `"nystrom(k)"` estimates the common prior with
   one global approximate fit instead. See §3 for the
   aggregation choice. Current restrictions: no nugget/noise channel,
   `normalize` unsupported — mention these
   if a user's request would hit them. `NestedKriging` is exposed in
   `pylibkriging` (alongside `Kriging`/`WarpKriging`/`MLPKriging`), plus
   the other bindings (C++/R/Julia/Octave-MATLAB).

Don't reach for `NestedKriging`, Vecchia or Nystrom by default — for the common case
(n in the hundreds to low thousands), plain `Kriging` with default options
is both simpler and, for NestedKriging's NK aggregation, actually a
*fallback target* (it converges to it as group size grows).

## 2. Shared fit/constructor options

| Option | Values | Guidance |
|---|---|---|
| `kernel` / `covType` | `"gauss"`, `"exp"`, `"matern3_2"`, `"matern5_2"` (`"whitenoise"` exists but is an internal building block, not a modelling choice) | `"matern5_2"` is the sane general-purpose default (smoother than Matérn 3/2, less rigid than Gaussian, which tends to numerical ill-conditioning). Use `"gauss"` only if the underlying function is known to be very smooth/analytic. See [Kernels.md](../../docs/math/Kernels.md) for formulas. |
| `regmodel` / trend | `"constant"`, `"linear"`, `"interactive"`, `"quadratic"` (`"none"` = zero mean) | `"constant"` (ordinary kriging) is the default and usually the right start. Move to `"linear"` if the response has an obvious global trend the GP should not have to explain via short-range correlation. Avoid `"quadratic"`/`"interactive"` in high dimension — parameter count grows fast and can overfit the trend, starving the covariance part. |
| `objective` | `"LL"` (log-likelihood, default), `"LOO"` (leave-one-out, [details](../../docs/math/LOO.md)), `"LMP"` (log marginal posterior, [details](../../docs/math/LMP.md)) | `"LL"` by default. `"LOO"` is a reasonable alternative when you specifically care about predictive accuracy at the design points rather than the full likelihood. `"LMP"` is a good alternative with few observations, where `"LL"` can drift θ into a degenerate (too small/too large) range. |
| `method` | `"exact"` (default), `"vecchia"` / `"vecchia(m)"`, `"nystrom"` / `"nystrom(k)"` | How the covariance is evaluated, for the fit and afterwards. Approximate methods only for scaling, with any `objective` (see §1.5; the approximate `"LOO"` is the leave-one-out of the approximate joint law, not of the local Vecchia predictor); noise models (both methods) only with `objective="LL"`, `"LOO"` / `"LMP"` noise-free — they approximate the criterion, so don't use them on small problems expecting identical results to the exact fit. |
| `optim` | `"BFGS"` (default), `"BFGSk"` for k random restarts (e.g. `"BFGS10"`), `"none"` | Use `"none"` only when supplying fixed/known hyperparameters via `parameters` (e.g. reusing a fit, or a controlled experiment). For a difficult/multimodal likelihood (many inputs, clustered design), multistart (`"BFGS10"`+) is cheap insurance against a bad local optimum — recommend it over blindly trusting a single `"BFGS"` run when the user reports an unstable or suspicious fit. |
| `normalize` | boolean, default off | Turn on when input dimensions have very different scales/units — it rescales `X`/`y` to `[0,1]` internally, which helps the optimizer's bounds and starting values. On `Kriging` everything public stays in the original scale of `X`/`y`: `parameters` in, `theta()`/`sigma2()`/`nugget()`/`noise()`/`beta()` and the `*Fun(theta)` objectives (values and gradients) out, so they round-trip and match a `normalize=false` fit (`X()`/`y()` are the internal normalized data). Not on `WarpKriging`/`MLPKriging` (warped feature space); not supported yet on `NestedKriging`. |
| `noise` | vector, `"nugget"`, or absent | See §1.2. |
| `parameters` | dict/struct with `sigma2`, `theta`, `beta`, plus per-argument `is_*_estim` flags | Use to seed the optimizer (multistart with multiple `theta` rows) or to fix a hyperparameter (`is_theta_estim=false`) rather than estimate it. |

## 3. `NestedKriging` aggregation

| Aggregation | Cost | Notes |
|---|---|---|
| `PoE`, `gPoE`, `BCM`, `rBCM` | cheap: O(q·n²/p) | Precision-weighted products of experts. Fast, but predictive variances are not fully consistent (can be over/under-confident, especially `PoE`/`BCM` at group boundaries). |
| `NK` (default) | O(q·n²) worst case | Kriges the submodel predictors themselves as noisy observations of the true GP — it *is* a proper kriging predictor: interpolates the data and gives consistent variances. Requires a **`"constant"` trend** (simple-kriging theory); the PoE family works with any trend. |

Default to `NK` unless the user is explicitly optimizing for raw prediction
speed at large `q` and can tolerate the PoE family's known variance
inconsistency.

## 4. `WarpKriging` per-variable warp specs

One spec string per input column, e.g. `{"kumaraswamy", "categorical(5,2)", "none"}`:

| Spec | Use for | Params | Details |
|---|---|---|---|
| `none` | leave the variable as-is | 0 | — |
| `affine` | rescale/shift a continuous variable | 2 | [Warping-Affine.md](../../docs/math/Warping-Affine.md) |
| `boxcox` | stabilize variance / skew on a positive continuous variable | 1 | [Warping-BoxCox.md](../../docs/math/Warping-BoxCox.md) |
| `kumaraswamy` | flexible monotone warp of a variable already in [0,1] | 2 | [Warping-Kumaraswamy.md](../../docs/math/Warping-Kumaraswamy.md) |
| `neural_mono(H)` | more flexible monotone warp, H hidden units (default `H=8`) | 3H+1 | [Warping-NeuralMono.md](../../docs/math/Warping-NeuralMono.md) |
| `knots(K)` / `knots(t1:...:tK)` | piecewise-linear warp, non-stationarity localized to specific regions (DiceKriging-style) | K+1 | [Warping-Knots.md](../../docs/math/Warping-Knots.md) |
| `mlp(h1:h2,q,act)` | unconstrained (non-monotone), possibly multi-dim, per-variable warp | depends on layers | [Warping-MLP.md](../../docs/math/Warping-MLP.md) |
| `categorical(L,q)` / `categorical(["a","b",...], q)` | unordered categorical with L levels, embedded in ℝ^q | L·q | [Warping-Categorical.md](../../docs/math/Warping-Categorical.md) |
| `ordinal(L)` / `ordinal(["low","med","high"])` | ordered categorical | L−1 | [Warping-Ordinal.md](../../docs/math/Warping-Ordinal.md) |

For a genuinely deep/joint transform across several continuous variables at
once, use `MLPKriging` (`mlp_joint(h1:h2,q,act)`-style joint map, see
[Warping-MLPJoint.md](../../docs/math/Warping-MLPJoint.md)) instead of
stacking per-variable warps. Valid `act`/`activation` values are `"selu"`
(default), `"relu"`, `"tanh"`, `"sigmoid"`, `"elu"`. If a user reports an
unstable or stuck fit with a single-start optimizer (`optim="BFGS"`, no
restarts), suggest `"tanh"` over the default `"selu"`: SELU's non-smooth
kink at `z=0` can make the likelihood surface locally jagged for a
gradient-based optimizer — the analytic gradient is still correct, this is
an optimization-landscape issue, not a bug.

## 5. Where to look for exact syntax

- `references/cpp.md` — C++ (the reference API; every other binding mirrors it)
- `references/python.md` — pylibkriging
- `references/r.md` — rlibkriging
- `references/julia.md` — jlibkriging
- `references/octave-matlab.md` — mLibKriging (Octave and MATLAB share the same `.m` classes)

For worked, end-to-end examples — fitting the same test function with
libKriging and a competitor package, then mimicking one of the
competitor's specific features (DiceKriging's `knots` warp, RobustGaSP's
LMP objective, SMT's KPLS reduction, OpenTURNS's joint conditional
simulation, GPy's inducing points, scikit-learn's `WhiteKernel`, GPflow's
HMC hyperparameters, GaussianProcesses.jl's composable kernels, GPyTorch's
iterative/GPU scalability vs. `method="nystrom(k)"`) — see the
notebooks under `docs/comparisons/libKriging_vs_*.ipynb`. Each ends with an
argument-correspondence table between libKriging and the competitor's API,
and is a good source to check exact option names against if this file and
the reference files disagree.
