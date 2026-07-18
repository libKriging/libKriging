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
              /*parameters=*/{});

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
auto [mean, stdev] = model.predict(Xnew, /*return_stdev=*/true,
                                    /*return_cov=*/false, /*return_deriv=*/false);
arma::mat sims = model.simulate(/*nsim=*/10, /*seed=*/123, Xnew);
model.update(y_new, X_new, /*refit=*/true);

double ll  = model.logLikelihood();
auto [ll2, grad] = model.logLikelihoodFun(theta, /*return_grad=*/true, /*bench=*/false);
```

Vecchia approximation: same class, just change `objective`:
```cpp
model.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "VLL(30)", {});
```

## WarpKriging

```cpp
#include "libKriging/WarpKriging.hpp"

WarpKriging model(y, X, {"kumaraswamy", "categorical(5,2)", "none"}, "gauss");
model.fit(y, X, {"kumaraswamy", "categorical(5,2)", "none"});
auto [mean, stdev] = model.predict(Xnew, true, false, false);
```
One spec string per column of `X`, in column order (see `SKILL.md` §4 for
the spec vocabulary).

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
                     Trend::RegressionModel::Constant,
                     NestedKriging::Aggregation::NK);
auto [mean, stdev] = model.predict(Xnew, /*return_stdev=*/true);
```
`Aggregation` is `PoE, gPoE, BCM, rBCM, NK` — see `SKILL.md` §3. Remember:
`NK` requires `Trend::RegressionModel::Constant`; no `normalize`, no
noise/nugget channel, no save/load yet (as of v1.1).

## Common pitfalls to flag in review

- Passing a `NuggetKriging`/`NoiseKriging` construction pattern from an old
  example — merged into `Kriging`'s `noise` argument, see `SKILL.md` §1.2.
- Using `NestedKriging` with `Aggregation::NK` and a non-`Constant` trend —
  will fail at runtime, not compile time.
- Calling `fit`/`predict` with `X` laid out as observations-in-columns
  instead of features-in-columns — libKriging matrices are `n × d`
  (row = observation), consistent across all bindings.
