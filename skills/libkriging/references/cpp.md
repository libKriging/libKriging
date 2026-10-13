# libKriging — C++

Headers: `libKriging/Kriging.hpp`, `WarpKriging.hpp`, `MLPKriging.hpp`,
`NestedKriging.hpp`, `Trend.hpp` (regression models), `Covariance.hpp`
(kernels), `Optim.hpp`.

See `SKILL.md` in this directory for *which* class/options to pick; this
file gives the exact call syntax.

## Kriging (noise-free or noisy)

```cpp
#include "libKriging/Kriging.hpp"

// Constructor fits immediately.
Kriging model(y, X, "matern5_2",
              Trend::RegressionModel::Constant,  // regmodel
              /*normalize=*/false,
              "BFGS",                              // optim
              "LL",                                // objective
              /*parameters=*/{},
              "exact");                            // method (default)

// Or default-construct, then fit (equivalent):
Kriging model2("matern5_2");
model2.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});

// Heterogeneous known noise: separate fit() overload taking `noise` before X.
arma::vec noise_var = ...; // one variance per observation
model.fit(y, noise_var, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
// For an unknown, homogeneous nugget instead, construct/fit with:
Kriging nugget_model("matern5_2", Kriging::NoiseModel::Nugget);
nugget_model.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
```

`Trend::RegressionModel` is an enum: `None, Constant, Linear, Interactive,
Quadratic` (`Trend::fromString("constant")` also works if you have a
string).

```cpp
auto [mean, stdev, cov, mean_deriv, stdev_deriv]
    = model.predict(Xnew, /*return_stdev=*/true, /*return_cov=*/false, /*return_deriv=*/false);
arma::mat sims = model.simulate(/*nsim=*/10, /*seed=*/123, Xnew, /*will_update=*/true);  // Xnew.n_rows x nsim
// exact by default; explicit method overloads: predict(X, stdev, cov, deriv, method),
// simulate(nsim, seed, X, method, will_update), logLikelihood(method),
// method = "exact" | "vecchia" | "vecchia(m)" | "nystrom" | "nystrom(k)" (the fitted k)
arma::mat sims_u = model.update_simulate(y_new, X_new);  // condition those paths on new observations
model.update(y_new, X_new, /*refit=*/true);

model.save("k.json");                        // JSON
Kriging model_loaded = Kriging::load("k.json");

double ll  = model.logLikelihood();
auto [ll2, grad] = model.logLikelihoodFun(theta, /*return_grad=*/true, /*bench=*/false);
```

Vecchia approximation: same class, `objective="LL"`, trailing `method`:
```cpp
model.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "vecchia(30)");
model.method();   // "vecchia(30)"; objective() is "LL"
// no exact factorization is kept: the method is explicit at each step,
// "exact" (the default) raises; same as predictVecchia(Xnew, true)
auto [m_v, s_v, c_v, dm_v, ds_v] = model.predict(Xnew, true, false, false, "vecchia");
auto [mean, stdev] = model.predictVecchia(Xnew, /*return_stdev=*/true);
// sequential conditional simulation on m nearest (observed or already simulated) points
arma::mat sims = model.simulateVecchia(/*nsim=*/10, /*seed=*/123, Xnew /*, m=0 -> vecchia_neighbors() or 30 */);
double ll_v = model.logLikelihood(model.method());   // = logLikelihood("vecchia")
```

Nystrom approximation: same class, `objective="LL"`, trailing `method`:
```cpp
model.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(50)");
auto [mean, stdev] = model.predictNystrom(Xnew, /*return_stdev=*/true);   // = predict(..., "nystrom")
arma::mat sims = model.simulateNystrom(/*nsim=*/10, /*seed=*/123, Xnew);  // = simulate(..., "nystrom")
// no exact factorization after a Nystrom fit: predict(Xnew, ...) (method "exact") raises
double ll_nys = model.logLikelihood("nystrom");
```

Exact model at the Vecchia / Nystrom estimate (separate exact fit, θ fixed):
```cpp
Kriging::Parameters params;
params.theta = arma::mat(model.theta().t());
params.is_theta_estim = false;
Kriging exact(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params);
```
Approximate methods: any objective (`"LL"`, `"LOO"`, `"LMP"`) with
`NoiseModel::None`; `"vecchia(m)"` with `objective="LL"` also supports
the Nugget / Heterogeneous `NoiseModel` (response Vecchia); other
combinations raise `std::invalid_argument`. `leaveOneOut(method)`,
`leaveOneOutVec(theta, method)`, `logMargPost(method)` evaluate the approximate
criteria (C++ also `leaveOneOutVecchiaFun` / `logMargPostNystromFun` etc. at a
given theta, with analytic gradient); `update` keeps the method (a
Heterogeneous model needs `update(y_u, noise_u, X_u, refit)`), `save` /
`load` too. Noise-model Vecchia simulation: `simulate(nsim, seed, X_n,
with_nugget, will_update, "vecchia")` (Nugget) /
`simulate(nsim, seed, X_n, with_noise, will_update, "vecchia")`
(Heterogeneous), or `simulateVecchia(nsim, seed, X_n, m, with_nugget,
with_noise)`.
`Kriging::fit_objective_method(objective, method)` returns the canonical
(objective, method) pair. Deprecated: objective `"LLVecchia(m)"` /
`"LLNystrom(k)"` (aliases of `"LL"` + `"vecchia(m)"` / `"nystrom(k)"`).

Pre-fit reduction and accessors:
```cpp
// n_max representative rows (k-means centroids snapped to real observations);
// returns sorted 0-based row indices.
arma::uvec idx = Kriging::subsetOfData(X, /*n_max=*/2000, /*method=*/"kmeans", /*seed=*/123);
Kriging small(y.elem(idx), X.rows(idx), "matern5_2");

model.method();         // "exact", "vecchia(m)" or "nystrom(k)"
model.nystrom_rank();   // rank k of a Nystrom fit, 0 otherwise
model.vecchia_neighbors();   // m of a Vecchia fit, 0 otherwise
model.is_vecchia_light(); model.is_nystrom_light();   // C++ only: no exact factorization held
// Deprecated: set_vecchia_exact_commit(false) is a no-op, (true) throws;
// vecchia_exact_commit() returns false (a Vecchia fit is always light).
```

## WarpKriging

```cpp
#include "libKriging/WarpKriging.hpp"

// WarpKriging and MLPKriging live in namespace libKriging
libKriging::WarpKriging model(y, X, {"kumaraswamy", "categorical(5,2)", "none"}, "gauss");  // builds and fits
model.fit(y, X);  // refit on new data (warpings fixed at construction)
auto [mean, stdev, cov, mean_deriv, stdev_deriv] = model.predict(Xnew, true, false, false);
```
One spec string per column of `X`, in column order (see `SKILL.md` §4 for
the spec vocabulary). `WarpKriging::fit` ignores its `objective` argument
(always `"LL"`), takes a trailing `method` of which only `"exact"` is
implemented (same for `MLPKriging`), and its noise is a per-observation variance vector (there is
no nugget mode).

## MLPKriging

```cpp
#include "libKriging/MLPKriging.hpp"
// Facade over WarpKriging({"mlp_joint(...)"} , kernel); construct/fit the
// same way as Kriging/WarpKriging — see MLPKriging.hpp for the exact
// Parameters struct (theta / warp_params seeds) if you need to warm-start.
```

## NestedKriging

```cpp
#include "libKriging/NestedKriging.hpp"

NestedKriging model(y, X, "matern5_2", nb_groups,
                    NestedKriging::Aggregation::NK,
                    NestedKriging::Partition::KMeans,
                    /*seed=*/123,
                    Trend::RegressionModel::Constant);
// trailing args: optim, objective, parameters, warping, method ("exact" default;
// "vecchia(m)" / "nystrom(k)" = common prior from one global approximate fit)
auto [mean, stdev] = model.predict(Xnew, /*return_stdev=*/true);
model.method();   // and model.objective()
```
`Aggregation` is `PoE, gPoE, BCM, rBCM, NK` — see `SKILL.md` §3. Remember:
`NK` requires `Trend::RegressionModel::Constant`; no `normalize`, no
noise/nugget channel. `nk.save(file)` / `NestedKriging::load(file)`
round-trip the fitted model (`KrigingLoader::describe` reports
`KrigingType::NestedKriging`).

## Common pitfalls to flag in review

- Passing a `NuggetKriging`/`NoiseKriging` construction pattern from an old
  example — merged into `Kriging`'s `noise` argument, see `SKILL.md` §1.2.
- Using `NestedKriging` with `Aggregation::NK` and a non-`Constant` trend —
  will fail at runtime, not compile time.
- Calling `fit`/`predict` with `X` laid out as observations-in-columns
  instead of features-in-columns — libKriging matrices are `n × d`
  (row = observation), consistent across all bindings.

## See also

There is no C++-specific comparison notebook (comparing against another
C++ Kriging library), but every worked example under `docs/comparisons/`
(`libKriging_vs_DiceKriging.ipynb`, `_RobustGaSP`, `_SMT`, `_OpenTURNS`,
`_GPy`, `_sklearn`, `_GPflow`, `_GaussianProcessesJL`) exercises this same
core C++ API through its R/Python/Julia bindings — read them for
end-to-end usage patterns (fit → predict → simulate → save/load) even
when writing pure C++ code.
