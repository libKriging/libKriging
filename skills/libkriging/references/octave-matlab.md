# libKriging — Octave / MATLAB (mLibKriging)

Octave and MATLAB share the same `.m` classes (`Kriging`, `WarpKriging`,
`MLPKriging`, `NestedKriging`, `MultiOutputKriging`), backed by the `mLibKriging` mex function.
Arguments are **positional**, not named — order matters and there is no
keyword-argument fallback. See `SKILL.md` in this directory for *which*
class/options to pick.

`X` is `n × d` (rows = observations), `y` an `n × 1` column vector.

## Kriging (noise-free or noisy)

```matlab
% k = Kriging(y, X, kernel, regmodel, normalize, optim, objective, parameters, noise_model, noise)
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL");

% Or all-default:
k = Kriging(y, X, "matern5_2");

[p_mean, p_stdev] = k.predict(Xnew, true, false, false);
%                             (X, return_stdev, return_cov, return_deriv)
s = k.simulate(int32(10), int32(123), Xnew, false);
%              (nsim, seed, X, will_update)
k.update(y_u, X_u, true);   % (y_u, X_u, refit)

ll = k.logLikelihood();   % assign the result: a bare `k.logLikelihood();`
loo = k.leaveOneOut();    % is called with nargout = 0 and fails
lmp = k.logMargPost();
```

Optional starting/fixed hyperparameters go through `Params(...)`, e.g.
`Kriging(y, X, "gauss", "constant", false, "BFGS", "LL", Params("is_sigma2_estim", true))`.

Do **not** call `NuggetKriging(...)`/`NoiseKriging(...)` for new models —
libKriging's noise handling is unified into `Kriging`'s trailing
`noise_model` (`"none"`, `"nugget"` or `"heterogeneous"`) and `noise`
(variance vector, with `"heterogeneous"`) arguments (see `SKILL.md` §1.2),
e.g. `Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", Params(), "nugget")`;
older class names remain only for loading legacy saved models via
`load_kriging(...)`.

## WarpKriging

```matlab
% k = WarpKriging(y, X, warping, kernel, regmodel, normalize, optim, objective, parameters, noise)
k = WarpKriging(y, X, {"kumaraswamy", "categorical(3,2)"}, "matern5_2");
[p_mean, p_stdev] = k.predict(Xnew, true);
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

% Or keep every point and approximate the objective (noise-free Kriging only)
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLVecchia(30)");   % d <~ 5
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLNystrom(50)");   % higher d
r = k.nystrom_rank()   % 50 (0 if the model was not fitted with LLNystrom)
```
`predict` is the only prediction entry point from Octave/MATLAB:
`predictVecchia`, `predictNystrom`, `simulateNystrom` and
`set_vecchia_exact_commit` (the "light" Vecchia mode) exist in C++ only.

## NestedKriging

```matlab
% nk = NestedKriging(y, X, kernel, nb_groups, aggregation, partition, seed, regmodel, optim, objective, parameters, warping)
nk = NestedKriging(y, X, "matern5_2", 8);  % aggregation="NK", partition="kmeans" by default
[p_mean, p_stdev] = nk.predict(Xnew);   % stdev is computed when a 2nd output is requested

% Explicit aggregation choice:
nk = NestedKriging(y, X, "matern5_2", 8, "PoE");
```
`aggregation = "NK"` (the default) requires the `regmodel` in position 8 to
be `"constant"` (also the default) — see `SKILL.md` §3. No `noise`
argument, no `normalize` support, no save/load yet on `NestedKriging`.

## MultiOutputKriging

```matlab
% Y is n x q (one column per output), X is n x d, same rows
% k = MultiOutputKriging(Y, X, kernel, output_model, regmodel, normalize, optim, objective, parameters, output_coordinates)
k = MultiOutputKriging(Y, X, "matern5_2", "pca(0.999)");   % or "shared", "separable", "separable(matern5_2)"
[m, s, c, dm] = k.predict(Xnew, true, true, true);   % m, s: m x q; c: mq x mq; dm: m x d x q
sims = k.simulate(int32(100), int32(1), Xnew, true); % m x q x nsim
upd = k.update_simulate(Y_u, X_u);
k.update(Y_u, X_u, false);
km = k.component(1);                                  % "pca": copy of latent Kriging 1 (1-based)

sep = MultiOutputKriging(Y, X, "matern5_2", "separable");
[Cx, Sigma] = sep.predictCovFactors(Xnew);            % cov of sep.predict == kron(Sigma, Cx)

% Unfitted object, then fit with a fixed theta
k = MultiOutputKriging("matern5_2", "shared");
k.fit(Y, X, "constant", false, "none", "LL", Params("theta", 0.3 * ones(1, size(X, 2))));

% curves: Sigma = sigma2 R_t(phi), a matern 5/2 kernel over the time steps t (q may exceed n)
sk = MultiOutputKriging("matern5_2", "separable(matern5_2)");
sk.set_output_coordinates(t(:));
sk.fit(Y, X);
phi = sk.output_theta();                              % logLikelihoodFun([sk.theta(); phi]) takes both

k.save("mo.json");                                    % JSON
k2 = MultiOutputKriging.load("mo.json");              % or load_kriging("mo.json")
```
`parameters` only takes `theta`, `is_theta_estim` and `output_theta`.
`"shared"`/`"separable"` accept `objective` `"LL"` or `"LOO"`,
`"separable(<kernel>)"` only `"LL"`. No noise yet.

## Common pitfalls to flag in review

- `NuggetKriging(...)`/`NoiseKriging(...)` calls for new fits.
- Getting positional argument order wrong — these bindings have **no**
  name-based argument matching; double-check against the signatures above
  rather than assuming Python/R-style keyword calls translate directly.
- Passing `nsim`/`seed` as plain doubles instead of `int32(...)` in
  `simulate(...)`.
- `NestedKriging(..., "NK", ...)` (5th positional arg) combined with a
  non-`"constant"` `regmodel` (8th positional arg).
- Calling a getter as a bare statement (`k.logLikelihood();`): the mex is
  then called with no output and fails with "Output requires exactly 1
  arguments". Assign the result (`ll = k.logLikelihood();`).

## See also

No dedicated comparison notebook exists yet for Octave/MATLAB in
`docs/comparisons/` (unlike R/Python/Julia). If one is requested, the
strongest candidate competitors are the SUMO/ooDACE toolbox and the DACE
toolbox (both Kriging-focused MATLAB toolboxes with a comparable
positional-argument-style API), following the same pattern as
`libKriging_vs_DiceKriging.ipynb`: fit the competitor, mimic it with
libKriging, then highlight one competitor-specific feature (e.g. DACE's
regression/correlation model grid) against a matching libKriging option.
