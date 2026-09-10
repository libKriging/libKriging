"""Shared, seeded LHS designs for the GPU cross-package comparison, at the
training sizes the CPU `../comparison` benchmark stops below (n >= 2000).

Same conventions as `../comparison/make_datasets.py` (identical conditioning
points for every package, no-header CSVs at full float precision). Only the
**designs** are written here -- one set serves the whole sweep. The shared
reference hyperparameters are computed by `run_gpu.py` from each design:
`theta_i = theta_frac * (max_i - min_i)` (a fraction of the observed input
range, swept over `--theta-frac`) and `sigma2 = var(y)`. Both GPyTorch and
libKriging run at this **fixed, shared theta** so the benchmark measures
each package's matrix-free CG linear algebra, not its optimizer -- same
argument as `docs/comparisons/libKriging_vs_GPyTorch.ipynb` sec 4.

`theta_frac` controls the conditioning of the (pure-interpolation) kernel
matrix: small -> short correlation -> well-conditioned R -> CG converges
fast; large -> long correlation -> ill-conditioned R -> CG runs many
iterations (near the MLE optimum of these smooth surfaces R's condition
number reaches ~1e16 and *neither* side's CG converges -- see ANALYSIS.md).

Interpolation only (no nugget / no observation noise), matching the CPU
`../comparison` benchmark: libKriging's `LLIterative` is
`NoiseModel::None`-only anyway.

Layout:
  data/<func>/X_test.csv, y_test.csv                  (common test set)
  data/<func>/n<N>/rep<K>/X_train.csv, y_train.csv
"""
import argparse
import os

import numpy as np
from scipy.stats import qmc

from functions import DOMAINS, FUNCTIONS, SINE_SUM_DIM

CASES = {  # function (dimension) -> training sizes (all > 1000)
    "sine_sum": [2000, 5000],   # d=4, smooth periodic
    "hartmann3": [2000, 5000],  # d=3
    "hartmann6": [2000, 5000],  # d=6
    "borehole": [2000, 5000],   # d=8, raw physical units
}
QUICK_CASES = {"sine_sum": [2000]}
N_TEST = 2000
TEST_SEED = 20260909
JITTER = 1e-10
# The reference length-scale (theta_frac * input range) and sigma2 = var(y)
# are computed by run_gpu.py from the design itself -- one dataset serves
# every --theta-frac / --kernel of the sweep, nothing per-hyperparameter is
# written here.


def lhs(n, dom, seed):
    d = dom.shape[0]
    u = qmc.LatinHypercube(d=d, seed=seed).random(n)
    return qmc.scale(u, dom[:, 0], dom[:, 1])


def save(path, arr):
    np.savetxt(path, np.atleast_2d(arr.T).T, delimiter=",", fmt="%.17g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    cases = QUICK_CASES if args.quick else CASES
    for func, sizes in cases.items():
        f, dom = FUNCTIONS[func], DOMAINS[func]
        fdir = os.path.join(args.out, func)
        os.makedirs(fdir, exist_ok=True)
        Xt = lhs(N_TEST, dom, TEST_SEED)
        save(os.path.join(fdir, "X_test.csv"), Xt)
        save(os.path.join(fdir, "y_test.csv"), f(Xt))
        for n in sizes:
            for rep in range(args.repeats):
                rdir = os.path.join(fdir, f"n{n}", f"rep{rep}")
                os.makedirs(rdir, exist_ok=True)
                X = lhs(n, dom, seed=1000 * n + rep)  # deterministic per (n, rep)
                y = f(X)
                save(os.path.join(rdir, "X_train.csv"), X)
                save(os.path.join(rdir, "y_train.csv"), y)
                print(f"[make_datasets] {func} (d={dom.shape[0]}) n={n} rep{rep}", flush=True)
    print(f"[make_datasets] done: cases={ {k: v for k, v in cases.items()} } "
          f"repeats={args.repeats} N_TEST={N_TEST}")


if __name__ == "__main__":
    main()
