# libKriging Example Notebooks

Jupyter notebooks demonstrating the various kriging models in libKriging.

## Standard Kriging

| Notebook | Description |
|----------|-------------|
| [kriging_branin2d_py.ipynb](kriging_branin2d_py.ipynb) | Kriging on 2D Branin function (Python) |
| [kriging_branin2d_r.ipynb](kriging_branin2d_r.ipynb) | Kriging on 2D Branin function (R) |
| [nuggetkriging_branin2d_py.ipynb](nuggetkriging_branin2d_py.ipynb) | NuggetKriging on 2D Branin (Python) |
| [nuggetkriging_branin2d_r.ipynb](nuggetkriging_branin2d_r.ipynb) | NuggetKriging on 2D Branin (R) |
| [noisekriging_branin2d_py.ipynb](noisekriging_branin2d_py.ipynb) | NoiseKriging on 2D Branin (Python) |
| [noisekriging_branin2d_r.ipynb](noisekriging_branin2d_r.ipynb) | NoiseKriging on 2D Branin (R) |

## WarpKriging (Input Warping)

WarpKriging extends Kriging with per-variable input transformations.
Each input dimension can be independently warped before the GP kernel is evaluated.

| Notebook | Warping Type | Description |
|----------|-------------|-------------|
| [warpkriging_none_branin2d_py.ipynb](warpkriging_none_branin2d_py.ipynb) | `none` | Baseline (identity warp) |
| [warpkriging_affine_branin2d_py.ipynb](warpkriging_affine_branin2d_py.ipynb) | `affine` | Linear w(x) = a·x + b |
| [warpkriging_boxcox_branin2d_py.ipynb](warpkriging_boxcox_branin2d_py.ipynb) | `boxcox` | Box-Cox transform |
| [warpkriging_kumaraswamy_branin2d_py.ipynb](warpkriging_kumaraswamy_branin2d_py.ipynb) | `kumaraswamy` | Kumaraswamy CDF on [0,1] |
| [warpkriging_neural_mono_branin2d_py.ipynb](warpkriging_neural_mono_branin2d_py.ipynb) | `neural_mono` | Monotone neural network |
| [warpkriging_mlp_branin2d_py.ipynb](warpkriging_mlp_branin2d_py.ipynb) | `mlp` | Multi-layer perceptron |
| [warpkriging_mlp_joint_branin2d_py.ipynb](warpkriging_mlp_joint_branin2d_py.ipynb) | `mlp_joint` | Joint MLP (Deep Kernel Learning) |
| [warpkriging_categorical_branin2d_py.ipynb](warpkriging_categorical_branin2d_py.ipynb) | `categorical` | Learned embeddings for discrete levels |
| [warpkriging_ordinal_branin2d_py.ipynb](warpkriging_ordinal_branin2d_py.ipynb) | `ordinal` | Ordered positions for ordinal data |
