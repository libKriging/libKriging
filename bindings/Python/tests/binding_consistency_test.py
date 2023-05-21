import pylibkriging as lk
import numpy as np
import pytest
from find_dir import find_reference_dir

tolerance = 1e-12


def relative_error(x, y):
    # FIXME check compatible vector sizes
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm > 0 or y_norm > 0:
        diff_norm = np.linalg.norm(x - y)
        return diff_norm / max(x_norm, y_norm)
    else:
        return 0


@pytest.fixture(scope="module", autouse=True)
def check_find_dir():
    print("Reference directory=", find_reference_dir())


def test_data1():
    refpath = find_reference_dir()
    prefix = "data1-scal"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_loo = refpath / f"{prefix}-result-leaveOneOut.csv"
    file_loograd = refpath / f"{prefix}-result-leaveOneOutGrad.csv"
    loo_ref = np.genfromtxt(file_loo, delimiter=',')
    loograd_ref = np.genfromtxt(file_loograd, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_ll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_llgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    ll_ref = np.genfromtxt(file_ll, delimiter=',')
    llgrad_ref = np.genfromtxt(file_llgrad, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec

    kernel = "gauss"
    r = lk.Kriging(y, X, kernel, "constant", False, "BFGS", "LL")
    x = 0.3 * np.ones(np.shape(X)[1])

    loo, loograd = r.leaveOneOutFun(x, True)
    assert relative_error(loo, loo_ref) < tolerance
    assert relative_error(loograd, loograd_ref) < tolerance

    ll, llgrad, llhess = r.logLikelihoodFun(x, True, False)
    assert relative_error(ll, ll_ref) < tolerance
    assert relative_error(llgrad, llgrad_ref) < tolerance


@pytest.mark.parametrize("i", np.arange(1, 11))
def test_data2(i):
    refpath = find_reference_dir()
    prefix = f"data2-grad-{i}"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',')
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_ll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_llgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    ll_ref = np.genfromtxt(file_ll, delimiter=',')
    llgrad_ref = np.genfromtxt(file_llgrad, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec

    kernel = "gauss"
    r = lk.Kriging(y, X, kernel)  # use defaults: lk.RegressionModel.Constant, False, "BFGS", "LL", lk.Parameters())

    x = 0.3 * np.ones(np.shape(X)[1])

    ll, llgrad, llhess = r.logLikelihoodFun(x, True, False)
    assert relative_error(ll, ll_ref) < tolerance
    assert relative_error(llgrad, llgrad_ref) < tolerance


@pytest.mark.parametrize("i", np.arange(1, 11))
def test_data2_with_save_reload(i):
    refpath = find_reference_dir()
    prefix = f"data2-grad-{i}"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',')
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_ll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_llgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    ll_ref = np.genfromtxt(file_ll, delimiter=',')
    llgrad_ref = np.genfromtxt(file_llgrad, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec

    kernel = "gauss"
    r = lk.Kriging(y, X, kernel)  # use defaults: lk.RegressionModel.Constant, False, "BFGS", "LL", lk.Parameters())

    r.save("test_data2_with_save_reload.h5");
    r = lk.Kriging.load("test_data2_with_save_reload.h5");

    x = 0.3 * np.ones(np.shape(X)[1])

    ll, llgrad, llhess = r.logLikelihoodFun(x, True, False)
    assert relative_error(ll, ll_ref) < tolerance
    assert relative_error(llgrad, llgrad_ref) < tolerance
