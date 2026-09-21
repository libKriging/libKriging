# Comparisons with other packages

One notebook per package, all fitting the same Branin 2D function. Each notebook has two parts:

1. **Mimicry**: reproduce with libKriging a default fit equivalent to the other package's (kernel, trend, maximum
   likelihood) and check that the two agree.
2. **Package-specific feature**: the feature the other package is known for, and how to obtain the same thing with
   libKriging.

Each notebook ends with an argument-correspondence table between libKriging and the package's API.

| Notebook | Package | libKriging binding | Package-specific feature |
|---|---|---|---|
| [libKriging_vs_sklearn.ipynb](libKriging_vs_sklearn.ipynb) | scikit-learn `GaussianProcessRegressor` | Python | noise estimated with a `WhiteKernel` term, vs `noise="nugget"` |
| [libKriging_vs_GPy.ipynb](libKriging_vs_GPy.ipynb) | GPy | Python | sparse / inducing-point approximation (`SparseGPRegression`) |
| [libKriging_vs_GPflow.ipynb](libKriging_vs_GPflow.ipynb) | GPflow | Python | full-Bayesian hyperparameters by HMC, vs `objective="LMP"` |
| [libKriging_vs_GPyTorch.ipynb](libKriging_vs_GPyTorch.ipynb) | GPyTorch | Python | scaling exact inference with iterative linear algebra, vs `LLNystrom` |
| [libKriging_vs_SMT.ipynb](libKriging_vs_SMT.ipynb) | SMT | Python | KPLS dimension reduction |
| [libKriging_vs_OpenTURNS.ipynb](libKriging_vs_OpenTURNS.ipynb) | OpenTURNS | Python | joint conditional simulation of sample paths |
| [libKriging_vs_DiceKriging.ipynb](libKriging_vs_DiceKriging.ipynb) | DiceKriging | R | the `knots` warping, vs `WarpKriging(..., "knots(K)")` |
| [libKriging_vs_RobustGaSP.ipynb](libKriging_vs_RobustGaSP.ipynb) | RobustGaSP | R | robust marginal-posterior-mode estimation, vs `objective="LMP"` |
| [libKriging_vs_GaussianProcessesJL.ipynb](libKriging_vs_GaussianProcessesJL.ipynb) | GaussianProcesses.jl | Julia | composable kernels (`k1 + k2`, `k1 * k2`) |
| [libKriging_vs_STK.ipynb](libKriging_vs_STK.ipynb) | STK | Octave | conditional sample paths (`stk_generate_samplepaths`), vs `simulate` |

The cross-package speed and accuracy benchmark, run in CI, is in [../../bench/comparison](../../bench/comparison).
