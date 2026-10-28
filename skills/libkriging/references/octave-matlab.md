# libKriging — Octave / MATLAB (mLibKriging)

Octave and MATLAB share the same `.m` classes (`Kriging`, `WarpKriging`,
`MLPKriging`, `NestedKriging`), backed by the `mLibKriging` mex function.
Arguments are **positional**, not named — order matters and there is no
keyword-argument fallback. See `SKILL.md` in this directory for *which*
class/options to pick.

`X` is `n × d` (rows = observations), `y` an `n × 1` column vector.

## Kriging (noise-free or noisy)

```matlab
% k = Kriging(y, X, kernel, regmodel, normalize, optim, objective, parameters, noise_model, noise, method)
%   noise_model: "none" (default), "nugget" or "heterogeneous" (then noise = variance vector)
%   method: "exact" (default) | "vecchia" | "vecchia(m)" | "nystrom" | "nystrom(k)" (see Large designs)
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL");
k_nug = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget");
% Unfitted model, to configure before fit:
% k0 = Kriging("matern5_2"); k0.fit(y, X, "constant", false, "BFGS", "LL");
%   fit(y, X, regmodel, normalize, optim, objective, parameters, method)
%   (heterogeneous noise: fit(y, noise, X, regmodel, normalize, optim, objective, parameters, method))

% Or all-default:
k = Kriging(y, X, "matern5_2");

[p_mean, p_stdev] = k.predict(Xnew, true, false, false);
%   optional 5th argument: method "exact" (default) | "vecchia" | "vecchia(m)" | "nystrom" (see Large designs)
%   k.method() returns the fit method: "exact", "vecchia(m)" or "nystrom(k)"
%                             (X, return_stdev, return_cov, return_deriv)
s = k.simulate(int32(10), int32(123), Xnew, true);   % size(Xnew,1) x nsim
%              (nsim, seed, X, will_update)
s_u = k.update_simulate(y_u, X_u);   % condition those paths on new observations
k.update(y_u, X_u, true);   % (y_u, X_u, refit)
% update_simulate dispatches on its argument count: (y_u, X_u) or
% (y_u, noise_u, X_u); a heterogeneous model needs the latter ((y_u, X_u)
% raises, as update). fit / update drop the last simulate(..., true)
% heterogeneous noise model: with_noise / noise_u are noise VARIANCES, like the
% fit's noise; update without noise_u raises an error
%   s = k.simulate(int32(10), int32(123), Xnew, noise_var_new, false);   % (nsim, seed, X, with_noise, will_update)
%   k.update(y_u, noise_var_u, X_u, true);   % (y_u, noise_u, X_u, refit)
% [ll, llgrad] = k.logLikelihoodFun(theta, return_grad, bench)

k.save("k.json");              % JSON, any libKriging class
k = load_kriging("k.json");    % auto-detects the class

ll = k.logLikelihood();    % a mex method returning a value needs an output variable
loo = k.leaveOneOut();        % or k.leaveOneOut("vecchia") / ("nystrom") after such a fit
lmp = k.logMargPost();        % idem
```

Optional starting/fixed hyperparameters go through `Params(...)`, e.g.
`Kriging(y, X, "gauss", "constant", false, "BFGS", "LL", Params("is_sigma2_estim", true))`.

Do **not** call `NuggetKriging(...)`/`NoiseKriging(...)` for new models —
libKriging's noise handling is unified into `Kriging`'s trailing `noise`
argument (see `SKILL.md` §1.2); older class names remain only for loading
legacy saved models via `load_kriging(...)`.

## WarpKriging

```matlab
% k = WarpKriging(y, X, warping, kernel, regmodel, normalize, optim, objective, parameters, noise, method)
%   method: only "exact" is implemented (same for MLPKriging)
k = WarpKriging(y, X, {"kumaraswamy", "categorical(3,2)"}, "matern5_2");
[p_mean, p_stdev] = k.predict(Xnew, true);
s = k.simulate(int32(10), int32(123), Xnew, true, noise_var_new);   % (nsim, seed, X, will_update, with_noise)
%   with_noise: noise VARIANCES (column vector, one per row of Xnew, or a scalar;
%   [] / omitted: latent process); same for MLPKriging
s_u = k.update_simulate(y_u, X_u);   % (y_u, X_u [, noise_u]): noise_u required iff fitted with noise
```
`warping` is a cell array with one spec string per column of `X` (see
`SKILL.md` §4). `noise` is a variance vector (no `"nugget"` mode) and `objective`
is always `"LL"` (any other value is ignored).

## MLPKriging

```matlab
% k = MLPKriging(y, X, hidden_dims, d_out, activation, kernel, regmodel, normalize, ...)
k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss", "constant", true);
```
Valid `activation` values: `"selu"` (default), `"relu"`, `"tanh"`,
`"sigmoid"`, `"elu"`. Prefer `"tanh"` over `"selu"` if a single-start fit
looks unstable — SELU's kink at `z = 0` can make the likelihood surface
locally jagged for a gradient-based optimizer (the gradient itself is
correct; this is an optimization-landscape issue, not a bug).

## Large designs

```matlab
% Pre-fit reduction: keep n_max representative rows (k-means centroids snapped
% to real observations). Returns 1-based row indices (column vector).
idx = Kriging.subsetOfData(X, int32(2000), "kmeans", int32(123));
k = Kriging(y(idx), X(idx, :), "matern5_2");

% Or keep every point and approximate the likelihood (objective "LL"; vecchia also with
% noise_model "nugget" / "heterogeneous", e.g. Kriging(y, X, k, "constant", false, "BFGS", "LL", [],
% "nugget", [], "vecchia(30)"), then simulate(nsim, seed, X, with_nugget, will_update, "vecchia")):
% method is the 11th constructor argument ([] for unused parameters / noise)
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], "vecchia(30)");   % d <~ 5
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], "nystrom(50)");   % higher d
mth = k.method()   % "nystrom(50)"; k.objective() is "LL"
r = k.nystrom_rank()   % 50 (0 if the model was not fitted with method "nystrom")
m = k.vecchia_neighbors()   % 30 for a "vecchia(30)" fit (0 otherwise)
```
An approximate fit keeps no exact factorization, and the method is explicit at every step
(`"exact"` by default, never inferred from the fit):
```matlab
kn = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], "nystrom(50)");
[m, s] = kn.predict(Xnew, true, false, false, "nystrom");   % "exact" would raise
sims = kn.simulate(int32(10), int32(123), Xnew, false, "nystrom");
ll = kn.logLikelihood("nystrom");                            % the fitted objective
% noisy data: Nystrom (not Vecchia) supports both noise models
knn = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget", [], "nystrom(50)");
knh = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "heterogeneous", noise_var, "nystrom(50)");
kv = Kriging("matern5_2");
kv.fit(y, X, "constant", false, "BFGS", "LL", [], "vecchia(30)");
[m, s] = kv.predict(Xnew, true, false, false, "vecchia");    % or kv.method()
kv.update(y_u, X_u, true);                                   % Vecchia refit, still approximate
% the exact model at the Vecchia estimate: a separate exact fit, theta fixed
th = kv.theta();
ke = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", ...
             Params("theta", th(:)', "is_theta_estim", false));
[m, s, c] = ke.predict(Xnew, true, true, false);             % exact, with covariance
```
The Vecchia predictor (`"vecchia"`) is also usable after an exact fit. Approximate methods
return mean / stdev only (`return_cov`, `return_deriv`, `will_update` raise).
Objectives `"LLVecchia(m)"` / `"LLNystrom(k)"` are **deprecated** aliases of `"LL"` with
method `"vecchia(m)"` / `"nystrom(k)"`.
`predictVecchia`, `predictNystrom`, `simulateVecchia` and `simulateNystrom` (behind `method`) exist in C++ only.

## NestedKriging

```matlab
% nk = NestedKriging(y, X, kernel, nb_groups, aggregation, partition, seed, regmodel, optim, objective, parameters, warping, method)
%   method (13th; 9th of nk.fit): "exact" (default) | "vecchia(m)" | "nystrom(k)" (common prior from one global
%   approximate fit, objective "LL", no warping); nk.method() / nk.objective() report it
nk = NestedKriging(y, X, "matern5_2", 8);  % aggregation="NK", partition="kmeans" by default
[p_mean, p_stdev] = nk.predict(Xnew, true);

% Explicit aggregation choice:
nk = NestedKriging(y, X, "matern5_2", 8, "PoE");
```
`aggregation = "NK"` (the default) requires the `regmodel` in position 8 to
be `"constant"` (also the default) — see `SKILL.md` §3. No `noise`
argument, no `normalize` support. `nk.save("nk.json")` /
`NestedKriging.load("nk.json")` (or `load_kriging`) work as for the other classes.

## Common pitfalls to flag in review

- `NuggetKriging(...)`/`NoiseKriging(...)` calls for new fits.
- Getting positional argument order wrong — these bindings have **no**
  name-based argument matching; double-check against the signatures above
  rather than assuming Python/R-style keyword calls translate directly.
- Passing `nsim`/`seed` as non-integer doubles (`10.5`) in `simulate(...)`:
  plain integer-valued doubles are accepted, but releases before that
  change need `int32(...)`.
- `NestedKriging(..., "NK", ...)` (5th positional arg) combined with a
  non-`"constant"` `regmodel` (8th positional arg).

## See also

No dedicated comparison notebook exists yet for Octave/MATLAB in
`docs/comparisons/` (unlike R/Python/Julia). If one is requested, the
strongest candidate competitors are the SUMO/ooDACE toolbox and the DACE
toolbox (both Kriging-focused MATLAB toolboxes with a comparable
positional-argument-style API), following the same pattern as
`libKriging_vs_DiceKriging.ipynb`: fit the competitor, mimic it with
libKriging, then highlight one competitor-specific feature (e.g. DACE's
regression/correlation model grid) against a matching libKriging option.
