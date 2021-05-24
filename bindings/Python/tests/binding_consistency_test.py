import pylibkriging as lk
import numpy as np
import pytest

# prefer pathlib over os.path when you can
# https://towardsdatascience.com/dont-use-python-os-library-any-more-when-pathlib-can-do-141fefb6bdb5
# https://docs.python.org/3/library/pathlib.html
from pathlib import Path
import sys

tolerance = 1e-12


def find_dir():
    path = Path.cwd()
    found = False
    # while (! is.null(path) and !found):
    while not found:
        testpath = path / ".git" / ".." / "tests" / "references"
        if testpath.exists():
            return testpath
        else:
            parent = path.parent
            if parent == path:
                print("Cannot find reference test directory", file=sys.stderr)
                sys.exit(1)
            path = parent


def relative_error(x, y):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm > 0 or y_norm > 0:
        diff_norm = np.linalg.norm(x - y)
        return diff_norm / max(x_norm, y_norm)
    else:
        return 0


@pytest.fixture(scope="module", autouse=True)
def check_find_dir():
    print("Reference directory=", find_dir())


def test_data1():
    refpath = find_dir()
    prefix = "data1-scal"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_llo = refpath / f"{prefix}-result-leaveOneOut.csv"
    file_llograd = refpath / f"{prefix}-result-leaveOneOutGrad.csv"
    llo_ref = np.genfromtxt(file_llo, delimiter=',')
    llograd_ref = np.genfromtxt(file_llograd, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_lll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_lllgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    lll_ref = np.genfromtxt(file_lll, delimiter=',')
    lllgrad_ref = np.genfromtxt(file_lllgrad, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec

    kernel = "gauss"
    r = lk.Kriging2(kernel)  # FIXME prefer Kriging (not Kriging2 is possible)
    r.fit(y, X, lk.RegressionModel.Constant, False, "BFGS", "LL", lk.Parameters())
    x = 0.3 * np.ones(np.shape(X)[1])

    llo, llograd = r.leaveOneOut(x, True)
    assert relative_error(llo, llo_ref) < tolerance
    assert relative_error(llograd, llograd_ref) < tolerance

    lll, lllgrad, lllhess = r.logLikelihood(x, True, False)
    assert relative_error(lll, lll_ref) < tolerance
    assert relative_error(lllgrad, lllgrad_ref) < tolerance


@pytest.mark.parametrize("i", np.arange(1, 11))
def test_data2(i):
    refpath = find_dir()
    prefix = f"data2-grad-{i}"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',')
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_lll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_lllgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    lll_ref = np.genfromtxt(file_lll, delimiter=',')
    lllgrad_ref = np.genfromtxt(file_lllgrad, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec

    kernel = "gauss"
    r = lk.Kriging(kernel)  # FIXME prefer Kriging (not Kriging2 is possible)
    r.fit(y, X, lk.RegressionModel.Constant, False, "BFGS", "LL", lk.Parameters())

    x = 0.3 * np.ones(np.shape(X)[1])

    lll, lllgrad, lllhess = r.logLikelihood(x, True, False)
    assert relative_error(lll, lll_ref) < tolerance
    assert relative_error(lllgrad, lllgrad_ref) < tolerance
