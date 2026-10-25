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
    # same shapes only: numpy broadcasting would silently compare (n,) to (n, 1)
    assert np.shape(x) == np.shape(y), f"shape mismatch: {np.shape(x)} vs {np.shape(y)}"
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


@pytest.fixture(scope="module", autouse=True)
def exact_objective():
    # The references evaluate the exact LL / LOO formulas: no fixed Cholesky
    # jitter (gauss uses 1e-9 by default in the fit objective)
    saved = lk.LinearAlgebra.get_chol_jitter()
    lk.LinearAlgebra.set_chol_jitter(0)
    yield
    lk.LinearAlgebra.set_chol_jitter(saved)


def test_chol_jitter_setting():
    saved = lk.LinearAlgebra.get_chol_jitter()
    lk.LinearAlgebra.set_chol_jitter(1e-9)
    assert lk.LinearAlgebra.get_chol_jitter() == 1e-9
    lk.LinearAlgebra.set_chol_jitter(saved)
    assert lk.LinearAlgebra.get_chol_jitter() == saved


def test_model_chol_jitter_save_load(tmp_path):
    """Jitters of a fitted model (fit / committed), kept by save / load"""
    saved = lk.LinearAlgebra.get_chol_jitter()
    lk.LinearAlgebra.set_chol_jitter(-1)  # automatic per kernel (this module sets 0)
    try:
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(6 * X[:, 0])
        models = [
            (lk.Kriging(y, X, "gauss"), lk.Kriging.load),
            (lk.WarpKriging(y, X, ["kumaraswamy"], "gauss"), lk.WarpKriging.load),
            (lk.MLPKriging(y, X, hidden_dims=[4], d_out=1, kernel="gauss", parameters={"max_iter_adam": "20"}),
             lk.MLPKriging.load),
        ]
        for k, load in models:
            assert k.chol_jitter() > 0  # gauss, noise-free
            assert 0 <= k.model_chol_jitter() <= k.chol_jitter()
            f = str(tmp_path / "model.json")
            k.save(f)
            k2 = load(f)
            assert k2.chol_jitter() == k.chol_jitter()
            assert k2.model_chol_jitter() == k.model_chol_jitter()
        assert models[0][0].model_chol_jitter() == 0  # Kriging: exact committed model
    finally:
        lk.LinearAlgebra.set_chol_jitter(saved)


def test_linalg_settings():
    saved = lk.LinearAlgebra.chol_rcond_checked()
    lk.LinearAlgebra.check_chol_rcond(False)
    assert not lk.LinearAlgebra.chol_rcond_checked()
    lk.LinearAlgebra.check_chol_rcond(True)
    assert lk.LinearAlgebra.chol_rcond_checked()
    lk.LinearAlgebra.check_chol_rcond(saved)
    lk.LinearAlgebra.set_chol_warning(False)


def test_arguments_as_in_r_julia():
    # argument names aligned with R / Julia (tools/checks/binding_api)
    X = np.linspace(0, 1, 12).reshape(-1, 1)
    y = np.sin(6 * X[:, 0])
    k = lk.Kriging(y, X, "gauss")
    t = k.theta()[:, 0] if k.theta().ndim == 2 else k.theta()
    ll, grad = k.logLikelihoodFun(t, return_grad=True, bench=True)
    assert ll == pytest.approx(k.logLikelihoodFun(t)[0])
    assert grad.size == 1
    assert k.leaveOneOutFun(theta=t, return_grad=False, bench=True)[0] == pytest.approx(k.leaveOneOutFun(t)[0])
    assert k.logMargPostFun(theta=t, return_grad=False, bench=True)[0] == pytest.approx(k.logMargPostFun(t)[0])

    wk = lk.WarpKriging(y, X, ["none"], "gauss")
    tw = np.atleast_1d(np.asarray(wk.theta()).ravel())
    ll, grad = wk.logLikelihoodFun(tw, return_grad=True)
    assert grad.size == tw.size
    with pytest.raises(TypeError):
        wk.simulate(nsim=2, seed=1)  # X required
    assert wk.simulate(nsim=2, seed=1, X=X[:3]).shape == (3, 2)

    # nugget model: simulate includes the nugget by default, as in R / Julia
    kn = lk.Kriging(y, X, "gauss", noise="nugget")
    Xn = np.array([[0.25], [0.75]])
    assert np.array_equal(kn.simulate(3, 7, Xn), kn.simulate(3, 7, Xn, with_noise=True))
    assert not np.array_equal(kn.simulate(3, 7, Xn), kn.simulate(3, 7, Xn, with_noise=False))


def test_data1():
    refpath = find_dir()
    prefix = "data1-scal"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_loo = refpath / f"{prefix}-result-leaveOneOut.csv"
    file_loograd = refpath / f"{prefix}-result-leaveOneOutGrad.csv"
    loo_ref = np.genfromtxt(file_loo, delimiter=',')
    loograd_ref = np.atleast_1d(np.genfromtxt(file_loograd, delimiter=','))
    file_ll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_llgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    ll_ref = np.genfromtxt(file_ll, delimiter=',')
    llgrad_ref = np.atleast_1d(np.genfromtxt(file_llgrad, delimiter=','))

    kernel = "gauss"
    r = lk.Kriging(y, X, kernel, "constant", False, "BFGS", "LL")
    x = 0.3 * np.ones(np.shape(X)[1])

    loo, loograd = r.leaveOneOutFun(x, True)
    assert relative_error(loo, loo_ref) < tolerance
    assert relative_error(loograd, loograd_ref) < tolerance

    ll, llgrad = r.logLikelihoodFun(x, True, False)
    assert relative_error(ll, ll_ref) < tolerance
    assert relative_error(llgrad, llgrad_ref) < tolerance


@pytest.mark.parametrize("i", np.arange(1, 11))
def test_data2(i):
    refpath = find_dir()
    prefix = f"data2-grad-{i}"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',')
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_ll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_llgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    ll_ref = np.genfromtxt(file_ll, delimiter=',')
    llgrad_ref = np.atleast_1d(np.genfromtxt(file_llgrad, delimiter=','))

    kernel = "gauss"
    r = lk.Kriging(y, X, kernel)  # use defaults: lk.RegressionModel.Constant, False, "BFGS", "LL", lk.Parameters())

    x = 0.3 * np.ones(np.shape(X)[1])

    ll, llgrad = r.logLikelihoodFun(x, True, False)
    assert relative_error(ll, ll_ref) < tolerance
    assert relative_error(llgrad, llgrad_ref) < tolerance


@pytest.mark.parametrize("i", np.arange(1, 11))
def test_data2_with_save_reload(i):
    refpath = find_dir()
    prefix = f"data2-grad-{i}"
    filex = refpath / f"{prefix}-X.csv"
    filey = refpath / f"{prefix}-y.csv"
    X = np.genfromtxt(filex, delimiter=',')
    y = np.genfromtxt(filey, delimiter=',').reshape(-1, 1)  # FIXME should be a col vec
    file_ll = refpath / f"{prefix}-result-logLikelihood.csv"
    file_llgrad = refpath / f"{prefix}-result-logLikelihoodGrad.csv"
    ll_ref = np.genfromtxt(file_ll, delimiter=',')
    llgrad_ref = np.atleast_1d(np.genfromtxt(file_llgrad, delimiter=','))

    kernel = "gauss"
    r = lk.Kriging(y, X, kernel)  # use defaults: lk.RegressionModel.Constant, False, "BFGS", "LL", lk.Parameters())

    r.save("test_data2_with_save_reload.json");
    r = lk.Kriging.load("test_data2_with_save_reload.json");

    x = 0.3 * np.ones(np.shape(X)[1])

    ll, llgrad = r.logLikelihoodFun(x, True, False)
    assert relative_error(ll, ll_ref) < tolerance
    assert relative_error(llgrad, llgrad_ref) < tolerance
