# Mathematical background

One page per model, objective, scalability method and input warping of libKriging. Each page states the model, the
`libKriging` option that selects it, its cost and current limitations, and its references. The bibliography of the
whole project is in [../dev/References.md](../dev/References.md).

## Model and fit

| Page | Content |
|---|---|
| [Kriging.md](Kriging.md) | the base Gaussian process model, the default `LL` objective, prediction |
| [Kernels.md](Kernels.md) | covariance kernels (`kernel` / `covType`) |
| [Noise.md](Noise.md) | noise models (`noise_model` / `noise`: none, nugget, heterogeneous) |
| [LOO.md](LOO.md) | the `objective="LOO"` criterion (leave-one-out) |
| [LMP.md](LMP.md) | the `objective="LMP"` criterion (log-marginal-posterior) |
| [Update.md](Update.md) | incremental `update`, `simulate` and `update_simulate` |

## Large designs

Start with [Scalability.md](Scalability.md): it compares the methods and says which one to pick. Each method has a page
and a notebook that runs it against exact Cholesky.

| Method | Page | Notebook |
|---|---|---|
| `objective="LLVecchia(m)"` | [Vecchia.md](Vecchia.md) | [llvecchia_vs_cholesky.ipynb](llvecchia_vs_cholesky.ipynb) |
| `objective="LLNystrom(k)"` | [Nystrom.md](Nystrom.md) | [llnystrom_vs_cholesky.ipynb](llnystrom_vs_cholesky.ipynb) |
| `NestedKriging` | [Nested.md](Nested.md) | [nested_vs_cholesky.ipynb](nested_vs_cholesky.ipynb) |
| `subsetOfData` | [SubsetOfData.md](SubsetOfData.md) | [subsetofdata_vs_cholesky.ipynb](subsetofdata_vs_cholesky.ipynb) |

## Input warpings (`WarpKriging`, `MLPKriging`)

One warping per input column, chosen by a spec string such as `kumaraswamy` or `categorical(5,2)`.

| Spec | Page |
|---|---|
| `affine` | [Warping-Affine.md](Warping-Affine.md) |
| `boxcox` | [Warping-BoxCox.md](Warping-BoxCox.md) |
| `kumaraswamy` | [Warping-Kumaraswamy.md](Warping-Kumaraswamy.md) |
| `neural_mono(H)` | [Warping-NeuralMono.md](Warping-NeuralMono.md) |
| `knots(K)` / `knots(t1:…:tK)` | [Warping-Knots.md](Warping-Knots.md) |
| `mlp(h1:h2,d_out,act)` | [Warping-MLP.md](Warping-MLP.md) |
| `mlp_joint(h1:h2,d_out,act)` (`MLPKriging`) | [Warping-MLPJoint.md](Warping-MLPJoint.md) |
| `categorical(L,q)` | [Warping-Categorical.md](Warping-Categorical.md) |
| `ordinal(L)` | [Warping-Ordinal.md](Warping-Ordinal.md) |

## See also

- [../comparisons](../comparisons): the same problems solved with other Kriging / Gaussian process packages.
- [../../bindings/README.md](../../bindings/README.md): the method reference of each binding and the worked notebooks.
