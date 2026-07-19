# libKriging — Octave / MATLAB (mLibKriging)

Octave and MATLAB share the same `.m` classes (`Kriging`, `WarpKriging`,
`MLPKriging`, `NestedKriging`), backed by the `mLibKriging` mex function.
Arguments are **positional**, not named — order matters and there is no
keyword-argument fallback. See `SKILL.md` in this directory for *which*
class/options to pick.

`X` is `n × d` (rows = observations), `y` an `n × 1` column vector.

## Kriging (noise-free or noisy)

```matlab
% k = Kriging(y, X, kernel, regmodel, normalize, optim, objective, parameters, noise)
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL");

% Or all-default:
k = Kriging(y, X, "matern5_2");

[p_mean, p_stdev] = k.predict(Xnew, true, false, false);
%                             (X, return_stdev, return_cov, return_deriv)
s = k.simulate(int32(10), int32(123), Xnew, false);
%              (nsim, seed, X, will_update)
k.update(y_u, X_u, true);   % (y_u, X_u, refit)

k.logLikelihood();
k.leaveOneOut();
k.logMargPost();
```

Optional starting/fixed hyperparameters go through `Params(...)`, e.g.
`Kriging(y, X, "gauss", "constant", false, "BFGS", "LL", Params("is_sigma2_estim", true))`.

Do **not** call `NuggetKriging(...)`/`NoiseKriging(...)` for new models —
libKriging's noise handling is unified into `Kriging`'s trailing `noise`
argument (see `SKILL.md` §1.2); older class names remain only for loading
legacy saved models via `load_kriging(...)`.

## WarpKriging

```matlab
% k = WarpKriging(y, X, warping, kernel, regmodel, normalize, optim, objective, parameters, noise)
k = WarpKriging(y, X, {"kumaraswamy", "categorical(3,2)"}, "matern5_2");
[p_mean, p_stdev] = k.predict(Xnew, true);
```
`warping` is a cell array with one spec string per column of `X` (see
`SKILL.md` §4).

## MLPKriging

```matlab
% k = MLPKriging(y, X, hidden_dims, d_out, activation, kernel, regmodel, normalize, ...)
k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss", "constant", true);
```

## NestedKriging

```matlab
% nk = NestedKriging(y, X, kernel, nb_groups, aggregation, partition, seed, regmodel, optim, objective, parameters, warping)
nk = NestedKriging(y, X, "matern5_2", 8);  % aggregation="NK", partition="kmeans" by default
[p_mean, p_stdev] = nk.predict(Xnew, true);

% Explicit aggregation choice:
nk = NestedKriging(y, X, "matern5_2", 8, "PoE");
```
`aggregation = "NK"` (the default) requires the `regmodel` in position 8 to
be `"constant"` (also the default) — see `SKILL.md` §3. No `noise`
argument, no `normalize` support, no save/load yet on `NestedKriging`
(v1.1).

## Common pitfalls to flag in review

- `NuggetKriging(...)`/`NoiseKriging(...)` calls for new fits.
- Getting positional argument order wrong — these bindings have **no**
  name-based argument matching; double-check against the signatures above
  rather than assuming Python/R-style keyword calls translate directly.
- Passing `nsim`/`seed` as plain doubles instead of `int32(...)` in
  `simulate(...)`.
- `NestedKriging(..., "NK", ...)` (5th positional arg) combined with a
  non-`"constant"` `regmodel` (8th positional arg).
