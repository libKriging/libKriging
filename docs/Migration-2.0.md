# Migrating from libKriging 1.x to 2.0

This guide lists every change of a default, of a fitted result or of an API between the 1.2.2
release and 2.0, in the order of the **Breaking changes** section of
[CHANGELOG.md](../CHANGELOG.md) (the section numbers §1–§27 are the same). For each change: what
changed, how to get the 1.x behaviour back where there is a way, the code before / after, and
which results move.

Conventions:
- "1.x" means the 1.2.2 release. Some 2.0 entries of the changelog describe intermediate states
  of the 2.0 development cycle (e.g. a `"BFGS"` default of `MLPKriging`, `calibration_folds` 3,
  a long-range restart): they were never released and are not migration steps from 1.2.2.
- The snippets assume `X` (`n × d`, rows = observations), `y` (length `n`) and `Xnew` (`q × d`).
  In Python: `import numpy as np, pylibkriging as lk`; in R: `library(rlibkriging)`; in C++:
  `#include "libKriging/Kriging.hpp"` (and the other headers named), namespace `libKriging` for
  `WarpKriging` / `MLPKriging`. Lines commented `# 1.2.2` show the former call.
- "No way back" means that no setting reproduces 1.2.2 exactly; reinstalling 1.2.2 is then the
  only way to get the former numbers.
- Some 1.2.x APIs were removed **without a released deprecation period**: they were documented and
  not deprecated in 1.2.2, the deprecation existed only during the 2.0 cycle. They are marked
  *no deprecation in 1.2.x*.

## Checklist

| If your 1.x code... | then in 2.0 | § |
|---|---|---|
| relies on the default optimizer of `Kriging` / `MLPKriging` / `WarpKriging` | results change; pass `optim` explicitly | 1–3 |
| uses the predicted stdev or simulations of `MLPKriging` / `mlp_joint` | they are calibrated; `calibration_folds = "0"` for the 1.x variance | 4 |
| compares fitted `theta` / `logLikelihood()` / predictions with stored 1.x numbers | expect differences, see §5–§12 | 5–12 |
| calls `Optim.set_thread_pool_size` / `set_thread_start_delay_ms` (or R `rlibkriging:::optim_*` for them) | use `Parallel` / `parallel_*` | 13 |
| sets `LK_*` environment variables | check the warnings at load time | 15 |
| uses `objective = "LLVecchia(m)"` / `"LLNystrom(k)"` | `objective = "LL", method = "vecchia(m)"` / `"nystrom(k)"`, and `method=` in `predict` / `simulate` | 16, 17 |
| fits `Kriging` with `normalize = true` and reads its parameters | they are in the original scale of `X` / `y` | 18 |
| indexes Python outputs as `[:, 0]` | outputs are 1-D | 19 |
| passes `with_noise` to `simulate` | variances, not standard deviations | 20 |
| unpacks `logLikelihoodFun` into three values | two values | 22 |
| links the C++ library or a binding built against 1.x | rebuild (SONAME 2, CMake `SameMajorVersion`) | 26 |
| fits `WarpKriging` with `mlp_joint` | pass `optim = "BFGS4"` (known inconsistency) | 27 |

Saved models: files written by libKriging 1.2 load in 2.0 (including those of an
`"LLVecchia(m)"` / `"LLNystrom(k)"` fit, read as `method = "vecchia(m)"` / `"nystrom(k)"`, and
the former `NuggetKriging` / `NoiseKriging` types). A loaded model keeps its stored settings
(optimizer, `max_iter_adam`, `adam_lr`, `max_iter_bfgs`, number of calibration folds); a 1.2 file of
a heterogeneous-noise `Kriging` with `normalize = true` predicts the same, but `noise()` now
reports the stored noise × `scaleY²` (§18). Objects serialized by an older rlibkriging with
`saveRDS` cannot be restored: use `$save()` / `load.*()`.

---

## Defaults and fitted results

### 1. `Kriging`: default optimizer `"BFGS4"`

**What changed** (T3, #443). The default `optim` of `Kriging` (C++ constructor and `fit`, Python
`Kriging` / `KrigingRegressor`, R `Kriging()` / `fit()`, Julia, Octave / MATLAB) is `"BFGS4"`: 4
starts run in parallel and the best is kept, instead of the single start `"BFGS"`. On the T7
benchmark base (4080 data sets) a single start ends more than 10 log-likelihood units below the
best optimum found on 18 % of the sets, `"BFGS4"` on 5 %, `"BFGS10"` on 1 %; `"BFGS4"` costs 1.6–3.8
times the CPU time of a single start on one core and 0.3–2.6 times its wall time on 4 cores.

**Results that move**: every fit without an explicit `optim` (theta, sigma2, `logLikelihood()`,
predictions), usually towards a better optimum. `WarpKriging` (§3) and `NestedKriging` keep
`"BFGS"`.

**Back to 1.x**: `optim = "BFGS"`. The single start of 2.0 is that of 1.2.2 except for the
restart rule (§9) and the changes of §5 and §10.

```cpp
// 1.2.2: Kriging k(y, X, "gauss");                      // single start
Kriging k(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS");
```
```python
# 1.2.2: k = lk.Kriging(y, X, "gauss")                    # single start
k = lk.Kriging(y, X, "gauss", optim="BFGS")
k_default = lk.Kriging(y, X, "gauss")                     # "BFGS4" in 2.0
assert k_default.optim() == "BFGS4"
```
```r
# 1.2.2: k <- Kriging(y, X, "gauss")                      # single start
k <- Kriging(y, X, "gauss", optim = "BFGS")
stopifnot(Kriging(y, X, "gauss")$optim() == "BFGS4")
```
Julia `Kriging(y, X, "gauss"; optim="BFGS")`; Octave / MATLAB
`Kriging(y, X, "gauss", "constant", false, "BFGS")`.

### 2. `MLPKriging`: default optimizer `"BFGS4"`

**What changed** (#467, #489). 1.2.2: `"BFGS+Adam"` (bi-level: L-BFGS-B on theta inside 10 Adam
steps of size 1e-3, which left the network close to its random initialization, #382). 2.0:
`"BFGS4"`, 4 joint L-BFGS-B starts (network weights and theta) in parallel. During the 2.0 cycle
the default was first `"BFGS"` (#382), a single joint start: test Q² < 0.5 on 68.5 % of the
600 cases of the MLP benchmark against 29.3 % for `"BFGS4"` (`docs/dev/rationalization/mlp-default-467.md`).

**Results that move**: every `MLPKriging` fit without an explicit `optim`: weights, theta, mean,
variance factor (§4), `logLikelihood()`.

**Back to 1.x**: only approximately. `"BFGS+Adam"` itself changed in 2.0 (Adam defaults, inner
restarts, early stop: §3), and 1.2.2 had no variance calibration (§4). The closest call restores
the 1.2 Adam knobs and disables the calibration:

```cpp
// 1.2.2: libKriging::MLPKriging m(y, X, {16, 8}, 2, "selu", "gauss");
libKriging::MLPKriging m(y, X, {16, 8}, 2, "selu", "gauss", Trend::RegressionModel::Constant, false,
                         "BFGS+Adam", "LL",
                         {{"max_iter_adam", "10"}, {"adam_lr", "0.001"}, {"calibration_folds", "0"}});
```
```python
# 1.2.2: m = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation="selu", kernel="gauss")
m_1x = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation="selu", kernel="gauss",
                     optim="BFGS+Adam",
                     parameters={"max_iter_adam": "10", "adam_lr": "0.001", "calibration_folds": "0"})
m = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation="selu", kernel="gauss")  # "BFGS4"
```
```r
# 1.2.2: m <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2, activation = "selu", kernel = "gauss")
m_1x <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2, activation = "selu", kernel = "gauss",
                   optim = "BFGS+Adam",
                   parameters = list(max_iter_adam = "10", adam_lr = "0.001", calibration_folds = "0"))
m <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2, activation = "selu", kernel = "gauss")
```
This is not recommended: the 1.2.2 default mostly returned the untrained network. For a hard fit
use `"BFGS10"`.

### 3. `WarpKriging`: default optimizer `"BFGS"`, and `"BFGS+Adam"` changed

**What changed** (#382, #418).
- Default `optim`: `"BFGS"` (one joint L-BFGS-B on the warp parameters and theta) instead of
  `"BFGS+Adam"`. `NestedKriging` (warped or not) keeps `"BFGS"`.
- `"BFGS+Adam"`, when asked for: `max_iter_adam` 10 → 200, `adam_lr` 1e-3 → 3e-2, early stop after
  20 steps without gain; the first inner theta L-BFGS-B restarts as in `Kriging`; the Adam loop
  goes back to its best point with half the step at a plateau.
- Joint `"BFGS"`: iteration budget 20 per parameter (capped at 1000, at least
  `Optim::max_iteration`), a run hitting the budget is resumed (a joint MLP keeps
  `Optim::max_iteration` per call, #407); `max_iter_bfgs` defaults to 0 (automatic).
- `update(refit = true)` of a model fitted with `"BFGS"` / `"BFGSk"` re-runs the joint L-BFGS-B
  (warm start) instead of the `"BFGS+Adam"` path.
- Multistart (`"BFGSk"`): starts 2..k re-draw the MLP weights; theta starts are drawn
  log-uniformly from the `Random` generator (`Random::reset_seed` applies).

**Results that move**: every `WarpKriging` fit without an explicit `optim`, and fits with an
explicit `"BFGS+Adam"`; slower but better optima (e.g. Branin, n = 30, `hidden_dims=[8]`: LL −80
instead of −172, ~0.2 s instead of ~5 ms).

**Back to 1.x**: approximately, `optim = "BFGS+Adam"` with `max_iter_adam = "10"`,
`adam_lr = "0.001"` (and `max_iter_bfgs = "100"`, the 1.2 inner budget); the restart and
bound changes (§9, §10) stay.

```python
# 1.2.2: w = lk.WarpKriging(y, X, ["kumaraswamy", "none"], "gauss")       # "BFGS+Adam"
w_1x = lk.WarpKriging(y, X, ["kumaraswamy", "none"], "gauss", optim="BFGS+Adam",
                      parameters={"max_iter_adam": "10", "adam_lr": "0.001", "max_iter_bfgs": "100"})
w = lk.WarpKriging(y, X, ["kumaraswamy", "none"], "gauss")                # "BFGS" in 2.0
assert w.optim() == "BFGS"
```
```r
w_1x <- WarpKriging(y, X, c("kumaraswamy", "none"), "gauss", optim = "BFGS+Adam",
                    parameters = list(max_iter_adam = "10", adam_lr = "0.001", max_iter_bfgs = "100"))
w <- WarpKriging(y, X, c("kumaraswamy", "none"), "gauss")
```
```cpp
libKriging::WarpKriging w(y, X, {"kumaraswamy", "none"}, "gauss", Trend::RegressionModel::Constant, false,
                          "BFGS+Adam", "LL",
                          {{"max_iter_adam", "10"}, {"adam_lr", "0.001"}, {"max_iter_bfgs", "100"}});
```
With an `mlp_joint` warping, see §27.

### 4. `MLPKriging` / `mlp_joint`: calibrated predictive variance

**What changed** (#407, T8 #448, #489). New in 2.0: after the fit, a K-fold cross-validation
scales sigma2 for `predict`, `simulate` and `update_simulate` by `c = mean(e²/s²)` (held-out
residuals over their predicted variances). K = `calibration_folds`, 5 by default (3 was tried
during the cycle; 1.2.2 had no calibration, so 5 is not "the 1.x value"). `c` is not clamped to
>= 1. It is reported by `summary()` (and `var_scale()` in C++), saved with the model, and not
applied with per-observation noise or when n < 2K. Without it, the 95 % intervals of a
many-weights network covered 1–2 % of test points on `Branin(Wx)` in 10D.

**Results that move**: predicted stdev, simulations, `update_simulate` paths of `MLPKriging` and
of `WarpKriging` with `mlp_joint`; the fit costs K more single starts (run in parallel). The
mean, `sigma2()` and `logLikelihood()` are not affected by the calibration.

**Back to 1.x** (variance only): `calibration_folds = "0"`.

```python
m0 = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation="selu", kernel="gauss",
                   parameters={"calibration_folds": "0"})   # no calibration, as 1.2.2
```
```r
m0 <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2, activation = "selu", kernel = "gauss",
                 parameters = list(calibration_folds = "0"))
```
```cpp
libKriging::MLPKriging m0(y, X, {16, 8}, 2, "selu", "gauss", Trend::RegressionModel::Constant, false,
                          "BFGS4", "LL", {{"calibration_folds", "0"}});
```

### 5. Search jitter of noise-free `gauss` / `matern5_2` fits

**What changed** (#415). During the optimization, noise-free fits add a fixed jitter to the
diagonal of the correlation matrix: 1e-9 for `gauss`, 1e-10 for `matern5_2`, none for
`matern3_2` / `exp`. Without it the stepwise numerical nugget of a failing Cholesky made the
objective discontinuous and a single start ended at an optimum depending on the BLAS, its thread
count or the CPU (18 % of 240 `gauss` cases; 1.7 % with the jitter). The model used by `predict` is
exact (§7). The jitter is saved with the model (`chol_jitter()`, `model_chol_jitter()`).

**Results that move**: theta, `logLikelihood()` (it is the objective at the jitter) and
predictions of noise-free `gauss` / `matern5_2` fits of `Kriging`, `WarpKriging`, `MLPKriging`,
`NestedKriging`; within a few % in accuracy, except very smooth functions (Branin, n = 50: median
test RMSE 5e-4 → 0.058).

**Back to 1.x**: set the jitter to 0 before fitting.

```cpp
LinearAlgebra::set_chol_jitter(0.0);   // #include "libKriging/LinearAlgebra.hpp"
```
```python
lk.LinearAlgebra.set_chol_jitter(0.0)
k = lk.Kriging(y, X, "gauss", optim="BFGS")
lk.LinearAlgebra.set_chol_jitter(-1.0)   # back to the per-kernel default
```
```r
linalg_set_chol_jitter(0)
k <- Kriging(y, X, "gauss", optim = "BFGS")
linalg_set_chol_jitter(-1)
```
Or `LK_CHOL_JITTER=0` in the environment for a whole run; Julia `set_chol_jitter(0.0)`, Octave /
MATLAB `LinearAlgebra.set_chol_jitter(0)`.

### 6. One jitter rule: `logLikelihood()` is always the fit objective

**What changed** (#465, #479). In every class, `logLikelihood()` (and `leaveOneOut()` /
`logMargPost()` for `Kriging`) equals its objective function at the committed parameters:
`logLikelihood() == logLikelihoodFun(theta())`, after `fit`, `update` and `load`. `Kriging` and
`NestedKriging` already did. For `WarpKriging` / `MLPKriging`, after `update(refit = false)` and
for a file saved without `objective_ll`, `logLikelihood()` used to return the log-likelihood of the
committed exact model, another quantity (up to 907 units from the objective for `MLPKriging`).

**Results that move**: only that value; fits, predictions and the value after `fit` /
`update(refit = true)` / `load` with `objective_ll` are unchanged.

**Back to 1.x**: none, the exact model's value is not exposed (it is dominated by rounding where
the correlation matrix is near singular). Code that compares models by `logLikelihood()` should
compare models of the same kernel and noise model.

### 7. Exact committed model of `WarpKriging` / `MLPKriging`, variance floor

**What changed** (#428). The model used by `predict`, `simulate` and `update` of `WarpKriging` /
`MLPKriging` is factorized without the 1e-10 it kept on its diagonal (as `Kriging`'s). `predict`
floors the variance at 4 eps × sigma2 instead of clamping it to 0, in every class: the stdev is
never 0 (no infinite NLPD) and its derivative is 0 where the floor applies (was NaN).

**Results that move**: predictions of noise-free `WarpKriging` / `MLPKriging` models (they
interpolate the data more closely); predicted stdev at the design points (about 3e-8 × sigma,
the square root of the floor, instead of 0). **Back to 1.x**: none.

### 8. Noise-free committed model checked by its interpolation residual

**What changed** (#496, #498). The Cholesky retry added a nugget when the rcond estimate of the
factor was below 1e-18, a false alarm on most exact noise-free models of `Kriging` /
`WarpKriging` (they then stopped interpolating, error up to 4.8e-3 sd(y)). The noise-free committed
model is now factorized without that test and kept when its interpolation residual is <= 1e-5,
otherwise factorized again with the test (the protection is kept for `MLPKriging`, whose learned
features can make the matrix genuinely ill-conditioned). The test is skipped where the diagonal
already bounds rcond (search jitter, nugget, noise). `NestedKriging` keeps the rcond test in its
group factors.

**Results that move**: predictions of near-singular noise-free fits (34 of 388 measured
`Kriging` / `WarpKriging` fits, RMSE better by > 10 % on 22, never worse by > 5 %), theta of
`Kriging` at n 1000 (13 of 40 fits, LL never lower), a few nugget fits (3 of 560, LL −30 to +5);
the variance of `gauss` fits, already overconfident (#425), lost the widening of the retry nugget
on 5 fits of 444. **Back to 1.x**: none (`LinearAlgebra::check_chol_rcond(false)` removes the test
everywhere, which is not the 1.x rule either).

### 9. Restarts no longer cascade

**What changed** (T2, #442). The L-BFGS-B restart criteria of `Kriging`, the `WarpKriging` joint
BFGS and `AdamBFGS` are one component; a restart started only because the previous run was worse
does not trigger another one by being worse again (1.2.2: up to `max_restart` = 10 restarts, each
from a point closer to the lower bound). Restart points are clamped to the bounds.

**Results that move**: fits that went through such a cascade end with fewer restarts and may end
at a slightly different point (4 of 1728 measured `Kriging` fits, |ΔLL| <= 1.1e-5; 216
`WarpKriging` fits unchanged). **Back to 1.x**: none; `Optim::set_max_restart(n)` (new) bounds the
restarts.

### 10. Other changes of the optimization path

No setting brings 1.2.2 back for these; they move the results of the fits concerned.
- **Theta starts drawn log-uniformly** in their bounds for `Kriging` / `WarpKriging` multistarts
  (#418), not uniformly (90 % of the uniform starts were in the flat long-range region);
  `WarpKriging` draws from the `Random` generator (`Random::reset_seed` / the `seed` applies), not
  Armadillo's. The joint MLP keeps the uniform draw.
- **Nugget fits** (`noise = "nugget"`) warm-start their first BFGS start from the noise-free fit
  (alpha = 0.99) instead of a random start (#383).
- **Monotone warps** (`boxcox`, `kumaraswamy`, `neural_mono`, `knots`): `kumaraswamy` a, b in
  [1/3, 3], and a penalty against squeezing a sub-range of the inputs onto one point (#406);
  `logLikelihood()` stays the plain LL.
- **Fixed `theta`**: `is_theta_estim = false` with a `theta` keeps it fixed with `"BFGS*"` too
  (#388; 1.2.2 re-estimated it silently unless `optim = "none"`).
- **Nystrom** (`method = "nystrom(k)"`): block-Cholesky factorization, stable at the landmark rows
  (#410); the log-likelihood and mean no longer depend on rounding.
- **Rounding**: theta gradients summed in one order (T1, #441), covariance / gradient loops on
  OpenMP threads from n = 200 (Linux, MSVC) or 800, objective evaluations at the thread budget
  (#473): last-bit differences, which an ill-conditioned problem can amplify into another optimum.

### 11. `sigma2` / `nugget` exactly 0 on data in the span of the trend

**What changed** (#450, and the degenerate-data fix of the same cycle). When `y` lies in the span
of the trend (constant `y` with a constant trend, affine `y` with a linear trend, ...), the
residual is 0 for every theta: the fit skips the optimization, commits the least-squares trend at
the given theta (or a deterministic default) with sigma2 = 0 and nugget = 0 exactly. 1.2.2 drove
theta to a bound and failed ("All 1 optimization attempts failed"), or stored the rounding
residual (2.4e-32 ...). With a known heterogeneous noise this case raises an explicit error.

**Results that move**: these fits succeed, with a zero predicted variance. **Back to 1.x**: none
(the former outcome was an error).

### 12. Thread budget, multistart in parallel, forked children

**What changed** (#415, T5 #445, T6 #446, #460, #473). One budget (`Parallel::set_num_threads_percent`,
`LK_NUM_THREADS_PERCENT`; default all physical cores, under the affinity mask and cgroup quotas)
bounds the multistart pool, the OpenMP loops and the BLAS together. `Kriging` multistart starts
run in parallel (1.2.2: one after the other, each with ncpu/k threads). Concurrent calls from user
threads split the budget; Python releases the GIL during `fit` / `predict` / `simulate` /
`update`. A forked child (R `parallel::mclapply`, Python `multiprocessing` with `fork`) gets a
budget of 1 thread, whatever the OpenMP runtime, until `Parallel::set_num_threads_percent` is
called in the child.

**Results that move**: times; values in the last bits where the BLAS thread count differs (a fit
is bit-reproducible for a given BLAS thread count). **Back to 1.x** for a forked child that should
use several threads: call the setter in the child.

```python
lk.Parallel.set_num_threads_percent(0.5)   # half of the physical cores
n_threads = lk.Parallel.get_num_threads()
```
```r
parallel_set_num_threads_percent(0.5)
# in a forked worker that should use 2 threads:
# parallel::mclapply(1:2, function(i) { parallel_set_num_threads_percent(2 / parallel::detectCores()); ... })
```

## API

### 13. Thread settings moved from `Optim` to `Parallel`

**What changed** (#468, #500), *no deprecation in 1.2.x*. The thread settings are in `Parallel`
only, with the same names. Of those, 1.2.2 had `set/get_thread_pool_size` and
`set/get_thread_start_delay_ms` in `Optim` (C++, Python, Octave / MATLAB, R internal
`rlibkriging:::optim_*`; not Julia); `set_num_threads_percent`, `get_num_threads_percent`,
`get_num_threads`, `set/get_blas_threads_min_n` were added during the 2.0 cycle and moved before
the release. The setters validate their argument: `set_thread_pool_size(n)` with n < 0 raises
(1.2.2: any n <= 0 meant `std::thread::hardware_concurrency()`, the logical CPUs; 2.0: 0 means the
thread budget, the physical cores by default), `set_thread_start_delay_ms(ms < 0)` and
`set_blas_threads_min_n(n < -1)` raise.

**Back to 1.x**: same calls on `Parallel`; for the former automatic pool on SMT machines, pass the
logical CPU count explicitly.

```cpp
// 1.2.2: Optim::set_thread_pool_size(4);
#include "libKriging/Parallel.hpp"
Parallel::set_thread_pool_size(4);
```
```python
# 1.2.2: lk.Optim.set_thread_pool_size(4)
lk.Parallel.set_thread_pool_size(4)
assert lk.Parallel.get_thread_pool_size() == 4
lk.Parallel.set_thread_pool_size(0)   # automatic: the thread budget
```
```r
# 1.2.2: rlibkriging:::optim_set_thread_pool_size(4)
parallel_set_thread_pool_size(4)
stopifnot(parallel_get_thread_pool_size() == 4)
parallel_set_thread_pool_size(0)
```
Julia `jlibkriging.Parallel.set_thread_pool_size(4)`; Octave / MATLAB
`Parallel.set_thread_pool_size(4)` (was `Optim.set_thread_pool_size(4)`).

### 14. Multistart start delay: 0 ms by default

**What changed** (#468, #500). The first start of `WarpKriging` / `MLPKriging` worker k waited
k × 10 ms (against start-up races of an older pool); the default is now 0. 800 multistart fits
were bit-identical with delays 0 to 10 ms; small multistart fits are faster (WarpKriging n 60
`BFGS8`: 26 instead of 42 ms).

**Results that move**: none (times only). **Back to 1.x**:
`Parallel.set_thread_start_delay_ms(10)` (Python / Octave), `parallel_set_thread_start_delay_ms(10)`
(R), `Parallel::set_thread_start_delay_ms(10)` (C++), or `LK_THREAD_START_DELAY_MS=10`.

### 15. `LK_*` environment variables

**What changed** (#468, #500). The 15 variables read by the library are listed in
[Settings.md](Settings.md), each with its setter and getter in every binding. They give the
default only, read once when the library is loaded; a setter call wins. An invalid value is now
ignored with a line `[WARNING] LK_X="..." ignored (expected ...), default ... used`, where 1.2.2
ignored it silently or took a prefix (`LK_MAX_ITERATION=3.7` gave 3, `"12abc"` gave 12).
Booleans accept `1/true/yes/on` and `0/false/no/off` in any case (1.2.2: only `1` / `true` /
`TRUE` / `True` were true, so `LK_REPARAMETRIZE=yes` turned it off). The variables of the
long-range restart and two developer diagnostics are ignored (they were never in a release).

**Back to 1.x**: write whole, valid values; read the warnings at load time.

```sh
# 1.2.2: LK_MAX_ITERATION=3.7 meant 3; 2.0: warning, default 20
LK_MAX_ITERATION=3 python3 my_script.py
```
```python
lk.Optim.set_max_iteration(30)   # a setter call overrides the environment
assert lk.Optim.get_max_iteration() == 30
lk.Optim.set_max_iteration(20)
```
```r
optim_set_max_iteration(30)
stopifnot(optim_get_max_iteration() == 30)
optim_set_max_iteration(20)
```

### 16. Removed APIs

**What changed** (#469, #499).
- `objective = "LLVecchia(m)"` / `"LLVecchia"` and `"LLNystrom(k)"` / `"LLNystrom"`, *no
  deprecation in 1.2.x* (they were the 1.2 way to ask for an approximate fit, in `Kriging` and
  `NestedKriging`): a fit with them raises (C++ `std::invalid_argument`, Python `ValueError`, an
  error elsewhere) with a message naming the replacement, e.g.
  `objective "LLVecchia(7)" was removed in libKriging 2.0; use objective="LL", method="vecchia(7)"`.
  Save files storing them still load.
- C++ `Kriging::set_vecchia_exact_commit(bool)` / `vecchia_exact_commit()`, *no deprecation in
  1.2.x* (documented as C++-only in 1.2.2): a Vecchia fit always keeps no exact factorization
  (§17).
- R `simulate(k, ..., with_nugget = b)`, *no deprecation in 1.2.x* (a silent alias of
  `with_noise`): now an error naming `with_noise`. The Julia `with_nugget` keyword of `simulate` is
  unchanged.
- Julia `get_X`, `get_y`, `get_theta`, ..., `get_warping`, `get_hidden_dims` and `is_normalize`,
  deprecated in 1.2.2 (`Base.depwarn`): `UndefVarError`; use the names without `get_`
  (`theta(k)`, `warping(wk)`, ...) and `normalize(k)`.
- C++ `Kriging::set_deprecation_handler`, `deprecated_objective_message` (2.0 cycle only).

**Back to 1.x**: none; use the replacements.

```cpp
// 1.2.2: Kriging k(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(10)");
Kriging k(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS", "LL", Kriging::Parameters{}, "vecchia(10)");
// 1.2.2: k.set_vecchia_exact_commit(true);  -> for the exact model at the Vecchia estimate:
Kriging::Parameters p;
p.theta = arma::mat(k.theta().t());
p.is_theta_estim = false;
Kriging k_exact(y, X, "gauss", Trend::RegressionModel::Constant, false, "none", "LL", p, "exact");
```
```python
# 1.2.2: k = lk.Kriging(y, X, "gauss", objective="LLVecchia(10)")      # 2.0: ValueError
k = lk.Kriging(y, X, "gauss", objective="LL", method="vecchia(10)")
mean, stdev, *_ = k.predict(Xnew, return_stdev=True, method="vecchia")
# 1.2.2: k = lk.Kriging(y, X, "gauss", objective="LLNystrom(15)")
k_ny = lk.Kriging(y, X, "gauss", objective="LL", method="nystrom(15)")
# exact model at the Vecchia estimate (the 1.2.2 default of set_vecchia_exact_commit):
k_exact = lk.Kriging(y, X, "gauss", optim="none",
                     parameters={"theta": np.atleast_2d(k.theta()), "is_theta_estim": False})
```
```r
# 1.2.2: k <- Kriging(y, X, "gauss", objective = "LLVecchia(10)")       # 2.0: error
k <- Kriging(y, X, "gauss", objective = "LL", method = "vecchia(10)")
p <- predict(k, Xnew, return_stdev = TRUE, method = "vecchia")
k_ny <- Kriging(y, X, "gauss", objective = "LL", method = "nystrom(15)")
kn <- Kriging(y, X, "gauss", noise = "nugget")
# 1.2.2: simulate(kn, nsim = 5, seed = 1, x = Xnew, with_nugget = FALSE)  # 2.0: error
s <- simulate(kn, nsim = 5, seed = 1, x = Xnew, with_noise = FALSE)
```
```julia
# 1.2.2: get_theta(k), is_normalize(k)
theta(k); normalize(k)
```

### 17. Approximate methods explicit at every step

**What changed** (#375). `objective` is only the criterion (`"LL"`, `"LOO"`, `"LMP"`); the
covariance evaluation is a separate `method`: `"exact"` (default), `"vecchia"` / `"vecchia(m)"`
(m = 30 by default) or `"nystrom"` / `"nystrom(k)"` (k = 50), in `fit` and in every later call
(`predict`, `simulate`, `logLikelihood`, `leaveOneOut`, `logMargPost`, ...), never inferred. In
1.2.2, `predict` / `simulate` switched silently to the Nystrom approximation after an
`"LLNystrom"` fit; `method = "exact"` (the default) on a Vecchia or Nystrom fit now raises an
error naming the method. A Vecchia fit no longer performs the final exact O(n³) factorization,
and its beta / sigma2 come from the Vecchia likelihood (1.2.2: from the exact one). The
approximate methods now exist for `"LOO"` / `"LMP"` (noise-free) and with the noise models (`"LL"`).

**Results that move**: beta / sigma2 / predictions of Vecchia fits (Vecchia-profiled instead of
exact); code predicting after an approximate fit without `method` raises. **Back to 1.x**: for the
exact model at the approximate estimate, a separate fit with `optim = "none"` and the fitted
theta (§16).

```python
k = lk.Kriging(y, X, "gauss", method="nystrom(15)")
# 1.2.2: mean = k.predict(Xnew)[0]          # silently Nystrom after an LLNystrom fit
mean, stdev, *_ = k.predict(Xnew, return_stdev=True, method="nystrom")
assert k.method() == "nystrom(15)"
```
```r
k <- Kriging(y, X, "gauss", method = "nystrom(15)")
p <- predict(k, Xnew, return_stdev = TRUE, method = "nystrom")
```
C++ `predict(X, stdev, cov, deriv, "nystrom")`, `simulate(nsim, seed, X, "nystrom", will_update)`;
Octave / MATLAB: trailing `method` argument.

### 18. `Kriging` with `normalize = true`: original scale everywhere

**What changed** (#379). Every parameter accessor and objective function of a `Kriging` fitted
with `normalize = true` is in the original scale of `X` and `y`, like the `parameters` given to
`fit`: `theta()` (× `scaleX`), `sigma2()`, `nugget()`, `noise()` (× `scaleY²`), `beta()` (exact
affine change of basis), `covMat()`, `summary()`, and `logLikelihoodFun`, `leaveOneOutFun`,
`logMargPostFun`, `*VecchiaFun`, `*NystromFun`, `leaveOneOutVec` (original-scale theta in,
original-scale values and gradients out). 1.2.2 returned the internal normalized values. A fixed
`parameters.beta` and a heterogeneous `noise` are now transformed exactly. `normalize = false` is
unchanged, bit for bit; `X()`, `y()`, `F()`, `T()`, `M()`, `z()` stay internal. `WarpKriging` /
`MLPKriging` are unchanged.

**Back to 1.x** (to read the former normalized values):

```python
k = lk.Kriging(y, X, "gauss", normalize=True)
theta_1x = k.theta() / k.scaleX()          # 1.2.2: k.theta()
sigma2_1x = k.sigma2() / k.scaleY() ** 2   # 1.2.2: k.sigma2()
```
```r
k <- Kriging(y, X, "gauss", normalize = TRUE)
theta_1x <- as.numeric(k$theta()) / as.numeric(k$scaleX())
sigma2_1x <- k$sigma2() / k$scaleY()^2
```

### 19. Python: vector outputs are 1-D

**What changed**. Every vector output of `Kriging`, `WarpKriging` and `MLPKriging` is a 1-D array
of shape `(n,)` instead of `(n, 1)` (`(1, d)` for `centerX` / `scaleX`): `predict` mean and stdev,
`theta`, `beta`, `y`, `z`, `noise`, `warp_params`, gradients, `leaveOneOutVec`, `subsetOfData`.
Matrices (`X`, `F`, covariances, derivatives, simulations) are unchanged.

```python
mean, stdev, *_ = lk.Kriging(y, X, "gauss").predict(Xnew, return_stdev=True)
# 1.2.2: mean[:, 0]
assert mean.shape == (Xnew.shape[0],)
```
`ravel()` / `reshape(-1)` work in both versions.

### 20. `simulate(..., with_noise)` takes variances

**What changed**. For `Kriging` (exact and Vecchia `simulate`, the noise re-added by
`update_simulate`), `WarpKriging` and `MLPKriging`, `with_noise` holds observation-noise
**variances**, like the `noise` of `fit` / `update`; 1.2.2 took standard deviations. Negative or
NaN values raise. R: a numeric scalar `with_noise` (e.g. `0.3`) is a variance for every point,
where 1.2.2 took it as the logical nugget flag `TRUE`.

**Back to 1.x**: square the standard deviations.

```python
noise_var = np.full(len(y), 0.01)
kh = lk.Kriging(y, X, "gauss", noise=noise_var)
sd_new = np.full(Xnew.shape[0], 0.1)
# 1.2.2: kh.simulate(10, 123, Xnew, with_noise=sd_new)
s = kh.simulate(10, 123, Xnew, with_noise=sd_new ** 2)
```
```r
kh <- Kriging(y, X, "gauss", noise = rep(0.01, length(y)))
sd_new <- rep(0.1, nrow(Xnew))
s <- simulate(kh, nsim = 10, seed = 123, x = Xnew, with_noise = sd_new^2)
```

### 21. Python: `Kriging.simulate` on a nugget model includes the nugget

**What changed**. Without `with_noise`, the draws of a `noise = "nugget"` model include the
nugget (as in R and Julia), except with `method = "nystrom"`. 1.2.2 simulated the latent process.

```python
kn = lk.Kriging(y, X, "gauss", noise="nugget")
# 1.2.2: kn.simulate(10, 123, Xnew)                       # latent process
s_latent = kn.simulate(10, 123, Xnew, with_noise=False)
```

### 22. `logLikelihoodFun` without Hessian

**What changed**. The Hessian (never computed: an empty matrix) is gone from C++ and every
binding. Python returns `(ll, grad)` (was `(ll, grad, hess)`), with arguments
`(theta, return_grad=False, bench=False)` for `Kriging` (`want_hess` removed: a third positional
argument is now `bench`) and `(theta, return_grad=False)` for `WarpKriging` / `MLPKriging`. R: no
`return_hess` argument nor `logLikelihoodHess` output. C++ `WarpKriging` / `MLPKriging`
`logLikelihoodFun(theta, return_grad)` return a pair. Octave / MATLAB:
`[ll, llgrad] = logLikelihoodFun(theta, return_grad[, bench])`. Julia: `(ll=, grad=)`.

```python
k = lk.Kriging(y, X, "gauss")
# 1.2.2: ll, grad, hess = k.logLikelihoodFun(k.theta(), True, False)
ll, grad = k.logLikelihoodFun(k.theta(), return_grad=True)
```
```r
k <- Kriging(y, X, "gauss")
# 1.2.2: logLikelihoodFun(k, t(k$theta()), return_grad = TRUE, return_hess = FALSE)
r <- logLikelihoodFun(k, t(k$theta()), return_grad = TRUE)   # one row per theta
```
```cpp
libKriging::WarpKriging w(y, X, {"kumaraswamy", "none"}, "gauss");
// 1.2.2: auto [ll, grad, hess] = w.logLikelihoodFun(theta, true, false);
auto [ll, grad] = w.logLikelihoodFun(w.theta(), true);
```

### 23. Errors instead of silent or unclear results

**What changed**. These calls raise (C++ `std::invalid_argument`, Python `ValueError`, an error
elsewhere) where 1.2.2 returned something else:
- `leaveOneOut(method)`, `logMargPost(method)`, `leaveOneOutVec(theta, method)` with
  `"vecchia"` / `"nystrom"` on a nugget / heterogeneous model (1.2.2: the noise-free approximate
  criterion);
- `update(y_u, X_u)` and `update_simulate(y_u, X_u)` on a heterogeneous model (1.2.2: a dimension
  error, or conditioning on noise-free observations): pass `noise_u`;
- `fit` with a negative, NaN or infinite noise variance (#292);
- `WarpKriging::update` / `update_simulate` of a model fitted with noise and an empty `noise_u`:
  `std::invalid_argument` (was `std::runtime_error`), as negative / NaN `with_noise`.

```python
kh = lk.Kriging(y, X, "gauss", noise=np.full(len(y), 0.01))
# 1.2.2: kh.update(y[:2], X[:2])
kh.update(y[:2], X[:2], refit=False, noise_u=np.full(2, 0.01))
```
```r
kh <- Kriging(y, X, "gauss", noise = rep(0.01, length(y)))
update(kh, y[1:2], X_u = X[1:2, , drop = FALSE], noise_u = rep(0.01, 2), refit = FALSE)
```

### 24. Python 3.10 to 3.14

**What changed** (#386, #389). `python_requires >= 3.10`; wheels for CPython 3.10–3.14 (1.2.2:
3.9–3.12, plus 3.7 on Windows). A source distribution is published too (`pip install pylibkriging`
builds from source where no wheel matches). **Back to 1.x**: none, use Python >= 3.10.

### 25. Octave / MATLAB: integer results are `int32`

**What changed** (#468). Integer results (`Optim.get_max_iteration()`, `Parallel.get_*`, ...) are
`int32` arrays (they were `uint64` with only 4 bytes written, so a negative value read as
4294967295). Integer arguments accept a plain double holding an integer (`10`) as well as
`int32(10)`.

```matlab
% 1.2.2: isa(Optim.get_max_iteration(), 'uint64')
assert(isinteger(Optim.get_max_iteration()))
```

### 26. Version 2: shared library, CMake package, R package

**What changed** at the version bump (#471): the shared library is `libKriging.so.2`
(SONAME = major version), the CMake package is `SameMajorVersion`-compatible, the R package is
`2.0-0`, the Python package `2.0.0`.

**Consequences**: programs and bindings linked against `libKriging.so.1` must be rebuilt;
`find_package(libKriging 1.2 CONFIG)` does not accept 2.0 (write `find_package(libKriging 2.0
CONFIG)`, or no version); pin `pylibkriging<2` / `rlibkriging` 1.2-2 to stay on 1.x.

```cmake
# 1.x: find_package(libKriging 1.2 CONFIG REQUIRED)
find_package(libKriging 2.0 CONFIG REQUIRED)
target_link_libraries(app PRIVATE libKriging::Kriging)
```

### 27. Known inconsistency: `WarpKriging` with `mlp_joint` keeps `"BFGS"`

`MLPKriging` is a facade over `WarpKriging` with one `mlp_joint` warping: both run the same
joint-MLP code. `MLPKriging` defaults to `"BFGS4"` (§2), but `WarpKriging` keeps `"BFGS"` for every
warping, `mlp_joint` included, so the same model fitted through the two classes gets a different
optimizer by default. The other warpings were not measured with `"BFGS4"`; the maintainer
decided on 2026-11-27 (#464) to document this in 2.0 rather than change it.

**Recommendation**: pass `optim = "BFGS4"` (or more) with an `mlp_joint` warping, or use
`MLPKriging`.

```cpp
libKriging::WarpKriging w(y, X, {"mlp_joint(16:8,2,selu)"}, "gauss", Trend::RegressionModel::Constant, false, "BFGS4");
```
Julia `WarpKriging(y, X, ["mlp_joint(16:8,2,selu)"], "gauss"; optim="BFGS4")`; Octave / MATLAB
`WarpKriging(y, X, {"mlp_joint(16:8,2,selu)"}, "gauss", "constant", false, "BFGS4")`.

In Python and R, `WarpKriging` with a single `mlp_joint` spec currently fails when `X` has more
than one column ("mlp_joint must be the only warp spec"): their wrappers recycle a single warping
spec to one per column (`_normalize_warping` in `pylibkriging/__init__.py`, `rep(warping, d)` in
`WarpKrigingClass.R`), already in 1.2.2. Use `MLPKriging` there, which takes the same model and
defaults to `"BFGS4"`:

```python
m = lk.MLPKriging(y, X, hidden_dims=[16, 8], d_out=2, activation="selu", kernel="gauss")   # "BFGS4"
# = WarpKriging(y, X, ["mlp_joint(16:8,2,selu)"], "gauss", optim="BFGS4") in C++ / Julia / Octave
```
```r
m <- MLPKriging(y, X, hidden_dims = c(16, 8), d_out = 2, activation = "selu", kernel = "gauss")
```

---

See also: [CHANGELOG.md](../CHANGELOG.md), [Settings.md](Settings.md) (settings and `LK_*`
variables), [docs/dev/CholJitter.md](dev/CholJitter.md) (jitter and Cholesky retry),
[docs/dev/parallel.md](dev/parallel.md) (threads), [docs/math/Warping-MLPJoint.md](math/Warping-MLPJoint.md)
(`MLPKriging` / `mlp_joint`), and the skill [skills/libkriging/SKILL.md](../skills/libkriging/SKILL.md).
