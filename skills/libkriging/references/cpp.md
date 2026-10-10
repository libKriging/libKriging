# libKriging — C++

Headers: `libKriging/Kriging.hpp`, `WarpKriging.hpp`, `MLPKriging.hpp`,
`NestedKriging.hpp`, `MultiOutputKriging.hpp`, `Trend.hpp` (regression models), `Covariance.hpp`
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
// predict() always returns a 5-tuple; the flags only control what is computed
auto [mean, stdev, cov, mean_deriv, stdev_deriv]
    = model.predict(Xnew, /*return_stdev=*/true, /*return_cov=*/false, /*return_deriv=*/false);
arma::mat sims = model.simulate(/*nsim=*/10, /*seed=*/123, Xnew);
model.update(y_new, X_new, /*refit=*/true);

double ll  = model.logLikelihood();
auto [ll2, grad] = model.logLikelihoodFun(theta, /*return_grad=*/true, /*bench=*/false);
```

Vecchia approximation: same class, just change `objective`:
```cpp
model.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLVecchia(30)", {});
auto [mean, stdev] = model.predictVecchia(Xnew, /*return_stdev=*/true);
```

Nystrom approximation: same class, just change `objective`:
```cpp
model.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLNystrom(50)", {});
auto [mean, stdev] = model.predictNystrom(Xnew, /*return_stdev=*/true);
arma::mat sims = model.simulateNystrom(/*nsim=*/10, /*seed=*/123, Xnew);
```

Pre-fit reduction and accessors:
```cpp
// n_max representative rows (k-means centroids snapped to real observations);
// returns sorted 0-based row indices.
arma::uvec idx = Kriging::subsetOfData(X, /*n_max=*/2000, /*method=*/"kmeans", /*seed=*/123);
Kriging small(y.elem(idx), X.rows(idx), "matern5_2");

model.nystrom_rank();   // rank k of an LLNystrom fit, 0 otherwise
model.vecchia_neighbors();   // m of an LLVecchia fit, 0 otherwise
// Factorization-free "light" Vecchia mode: call BEFORE fit(..., "LLVecchia(m)");
// predict() then routes to predictVecchia (no cov/deriv, simulate, update or save).
model.set_vecchia_exact_commit(false);
```

## WarpKriging

```cpp
#include "libKriging/WarpKriging.hpp"

WarpKriging model(y, X, {"kumaraswamy", "categorical(5,2)", "none"}, "gauss");
model.fit(y_new, X_new);   // refit on new data: the warping is set by the constructor
auto [mean, stdev, cov, mean_deriv, stdev_deriv] = model.predict(Xnew, true, false, false);
```
One spec string per column of `X`, in column order (see `SKILL.md` §4 for
the spec vocabulary). `WarpKriging::fit` ignores its `objective` argument
(always `"LL"`), and its noise is a per-observation variance vector (there is
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
                    NestedKriging::Aggregation::NK,      // default
                    NestedKriging::Partition::KMeans,    // default
                    /*seed=*/123,
                    Trend::RegressionModel::Constant);
auto [mean, stdev] = model.predict(Xnew, /*return_stdev=*/true);
```
`Aggregation` is `PoE, gPoE, BCM, rBCM, NK` — see `SKILL.md` §3. Remember:
`NK` requires `Trend::RegressionModel::Constant`; no `normalize`, no
noise/nugget channel, no save/load yet.

## MultiOutputKriging

```cpp
#include "libKriging/MultiOutputKriging.hpp"

// Y is n x q (one column per output), X is n x d, same rows
MultiOutputKriging model(Y, X, "matern5_2", "pca(0.999)",   // or "shared", "separable", "separable(matern5_2)"
                         Trend::RegressionModel::Constant,
                         /*normalize=*/false, "BFGS", "LL");
auto [mean, stdev, cov, mean_deriv] = model.predict(Xnew, true, false, false);  // mean, stdev: m x q
arma::cube sims = model.simulate(nsim, seed, Xnew, /*will_update=*/true);       // m x q x nsim
arma::cube upd = model.update_simulate(Y_u, X_u);
model.update(Y_u, X_u, /*refit=*/false);
model.save("mo.json");
MultiOutputKriging loaded = MultiOutputKriging::load("mo.json");  // simulate again before update_simulate

// curves: Σ = σ² R_t(φ), a kernel over the output coordinates t (q × d_t, required)
MultiOutputKriging sk("matern5_2", "separable(matern5_2)");
sk.set_output_coordinates(t);
sk.fit(Y, X);
arma::vec phi = sk.output_theta();  // logLikelihoodFun(join_cols(theta, phi)) takes both
```
`MultiOutputKriging::Parameters` has only `theta` (rows = starting points),
`is_theta_estim` and `output_theta` (φ, `"separable(<kernel>)"` only). `cov` is over `vec(Y_n)` (the m points of output 1, then
output 2, …). `"shared"`/`"separable"` take `objective` `"LL"` or `"LOO"`,
`"separable(<kernel>)"` only `"LL"`;
`"pca"` forwards it to each latent `Kriging` (`component(k)`, 0-based).
`predictCovFactors(Xnew)` gives the Kronecker factors `(C_x, Σ)` without the
dense `mq × mq` matrix. No noise/nugget yet. See `SKILL.md` §1.7.

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
