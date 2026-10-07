"""MultiOutputKriging ("pca", "shared" and "separable") — mirrors todo/multi-output/draft/example_python.py."""
import numpy as np
import pytest

import pylibkriging as lk

t = np.linspace(0.0, 10.0, 200)


def code(x):
    f, a = 0.5 + 1.5 * x[0], 0.1 + 0.5 * x[1]
    return np.exp(-a * t) * np.cos(2 * np.pi * f * t / 5)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(40, 2))
    Y = np.array([code(x) for x in X])
    Xnew = rng.uniform(size=(5, 2))
    Ynew = np.array([code(x) for x in Xnew])
    return X, Y, Xnew, Ynew


@pytest.fixture(scope="module")
def pca(data):
    X, Y, _, _ = data
    return lk.MultiOutputKriging(Y, X, "matern5_2", output_model="pca(0.99)", regmodel="constant", normalize=True)


def test_pca_basis(pca, data):
    X, Y, _, _ = data
    K = pca.nb_components()
    assert 2 <= K < 40
    assert pca.output_model() == "pca(0.99)"
    assert pca.nb_outputs() == 200
    assert pca.pca_basis().shape == (200, K)
    assert pca.pca_explained().shape == (K,)
    assert pca.pca_explained()[-1] >= 0.99
    assert pca.pca_residual().shape == Y.shape
    np.testing.assert_allclose(pca.Y(), Y)
    np.testing.assert_allclose(pca.X(), X)
    assert pca.component(0).theta().size == 2
    assert "MultiOutputKriging" in repr(pca)


def test_predict_shapes_and_accuracy(pca, data):
    _, _, Xnew, Ynew = data
    mean, stdev, cov, deriv = pca.predict(Xnew, return_stdev=True, return_cov=True, return_deriv=True)
    assert mean.shape == (5, 200)
    assert stdev.shape == (5, 200)
    assert cov.shape == (1000, 1000)
    assert deriv.shape == (5, 2, 200)
    # vec(Y_n) ordering: output-major, column-major
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), stdev.ravel(order="F"), rtol=1e-8, atol=1e-12)
    assert np.sqrt(np.mean((mean - Ynew) ** 2)) < 0.1 * Ynew.std()
    mean2, stdev2, cov2, deriv2 = pca.predict(Xnew)
    np.testing.assert_allclose(mean2, mean)
    assert cov2.size == 0 and deriv2.size == 0


def test_simulate(pca, data):
    _, _, Xnew, _ = data
    sims = pca.simulate(nsim=1000, seed=123, X=Xnew)
    assert sims.shape == (5, 200, 1000)
    np.testing.assert_array_equal(sims, pca.simulate(nsim=1000, seed=123, X=Xnew))
    mean, stdev, _, _ = pca.predict(Xnew)
    assert np.abs(sims.mean(axis=2) - mean).max() < 5 * stdev.max() / np.sqrt(1000)
    trough = sims.min(axis=1)  # functional of the trajectory
    q = np.quantile(trough, [0.05, 0.5, 0.95], axis=1)
    assert q.shape == (3, 5) and np.all(q[0] <= q[2])


def test_loo_and_model_choice(data):
    X, Y, _, _ = data
    loos = {spec: lk.MultiOutputKriging(Y, X, "matern5_2", output_model=spec).leaveOneOut()
            for spec in ("pca(3)", "pca(0.999)", "shared")}
    assert loos["pca(0.999)"] < loos["pca(3)"]
    assert loos["shared"] < loos["pca(3)"]
    m = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="pca(3)")
    loo_mean, loo_sd = m.leaveOneOutMat()
    assert loo_mean.shape == Y.shape and loo_sd.shape == Y.shape


def test_update(data):
    X, Y, Xnew, _ = data
    rng = np.random.default_rng(1)
    X_u = rng.uniform(size=(3, 2))
    Y_u = np.array([code(x) for x in X_u])

    m = lk.MultiOutputKriging(Y, X, "matern3_2", output_model="pca(0.999)")
    m.simulate(nsim=200, seed=5, X=Xnew, will_update=True)
    assert m.update_simulate(Y_u, X_u).shape == (5, 200, 200)

    basis = m.pca_basis()
    m.update(Y_u, X_u, refit=False)
    assert m.X().shape == (43, 2)
    np.testing.assert_array_equal(m.pca_basis(), basis)

    m.update(Y_u, X_u, refit=True)
    assert m.Y().shape == (46, 200)


def test_single_output_matches_kriging(data):
    X, Y, Xnew, _ = data
    y1 = Y[:, 50]
    a = lk.MultiOutputKriging(y1, X, "matern5_2", output_model="pca")  # 1-D Y = one output
    b = lk.Kriging(y1, X, "matern5_2")
    np.testing.assert_allclose(a.component(0).theta(), b.theta(), rtol=1e-6)
    m1, s1, _, _ = a.predict(Xnew)
    m2, s2, *_ = b.predict(Xnew, True, False, False)
    np.testing.assert_allclose(m1.ravel(), m2.ravel(), atol=1e-6)
    np.testing.assert_allclose(s1.ravel(), s2.ravel(), atol=1e-6)


def test_errors(data):
    X, Y, _, _ = data
    with pytest.raises(ValueError):
        lk.MultiOutputKriging("gauss", "pca(1.5)")
    with pytest.raises(RuntimeError, match="not implemented"):
        lk.MultiOutputKriging(Y, X, "matern5_2", output_model="separable(matern5_2)", output_coordinates=t)
    with pytest.raises(ValueError, match="output_coordinates"):
        lk.MultiOutputKriging(Y, X, "matern5_2", output_coordinates=t[:10])
    with pytest.raises(ValueError, match="unsupported parameter"):
        lk.MultiOutputKriging(Y, X, "matern5_2", parameters={"sigma2": 1.0})


# ----------------------------------------------------------------------------- shared


@pytest.fixture(scope="module")
def shared(data):
    X, Y, _, _ = data
    return lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared", normalize=True)


def test_shared_parameters(shared, data):
    X, Y, Xnew, Ynew = data
    assert shared.output_model() == "shared"
    assert shared.theta().shape == (2,)
    assert shared.sigma2().shape == (200,)
    assert shared.beta().shape == (1, 200)
    assert shared.sigma2()[0] == 0  # t = 0: every curve equals 1, constant output
    assert shared.nb_components() == 0
    ll, grad = shared.logLikelihoodFun(shared.theta(), return_grad=True)
    assert ll == pytest.approx(shared.logLikelihood())
    assert np.abs(grad).max() < 1e-2 * abs(ll)


def test_shared_predict_simulate(shared, data):
    _, _, Xnew, Ynew = data
    mean, stdev, cov, deriv = shared.predict(Xnew, return_stdev=True, return_cov=True, return_deriv=True)
    assert mean.shape == (5, 200) and stdev.shape == (5, 200)
    assert cov.shape == (1000, 1000) and deriv.shape == (5, 2, 200)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), stdev.ravel(order="F"), rtol=1e-8, atol=1e-12)
    assert np.abs(cov[:5, 5:10]).max() == 0  # outputs independent given theta
    assert np.sqrt(np.mean((mean - Ynew) ** 2)) < 0.15 * Ynew.std()

    sims = shared.simulate(nsim=2000, seed=3, X=Xnew)
    assert sims.shape == (5, 200, 2000)
    act = stdev > 1e-3 * stdev.max()
    np.testing.assert_allclose(sims.mean(axis=2)[act], mean[act], atol=0.1 * stdev.max())
    np.testing.assert_allclose(sims.std(axis=2)[act], stdev[act], rtol=0.1)


def test_shared_update_simulate(data):
    X, Y, Xnew, _ = data
    rng = np.random.default_rng(4)
    X_u = rng.uniform(size=(3, 2))
    Y_u = np.array([code(x) for x in X_u])
    m = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared")
    m.simulate(nsim=3000, seed=9, X=Xnew, will_update=True)
    up = m.update_simulate(Y_u, X_u)
    assert up.shape == (5, 200, 3000)
    ref = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared")
    ref.update(Y_u, X_u, refit=False)
    mean, sd, _, _ = ref.predict(Xnew)
    s2, s2_ref = m.sigma2(), ref.sigma2()
    sd = sd * np.sqrt(np.divide(s2, s2_ref, out=np.ones_like(s2), where=s2_ref > 0))  # update_simulate keeps sigma2
    act = sd > 1e-3 * sd.max()
    np.testing.assert_allclose(up.mean(axis=2)[act], mean[act], atol=0.1 * sd.max())
    np.testing.assert_allclose(up.std(axis=2)[act], sd[act], rtol=0.1)


def test_shared_loo_objective(data):
    X, Y, _, _ = data
    loo = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared", objective="LOO")
    ll = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared")
    assert loo.objective() == "LOO"
    f, g = loo.leaveOneOutFun(loo.theta(), return_grad=True)
    assert g.shape == (2,)
    assert f <= loo.leaveOneOutFun(ll.theta())[0] * (1 + 1e-6)


def test_shared_single_output_matches_kriging(data):
    X, Y, Xnew, _ = data
    y1 = Y[:, 50]
    a = lk.MultiOutputKriging(y1, X, "matern5_2", output_model="shared")
    b = lk.Kriging(y1, X, "matern5_2")
    np.testing.assert_allclose(a.theta(), b.theta().ravel(), rtol=1e-5)
    np.testing.assert_allclose(a.sigma2()[0], b.sigma2(), rtol=1e-5)
    m1, s1, _, _ = a.predict(Xnew)
    m2, s2, *_ = b.predict(Xnew, True, False, False)
    np.testing.assert_allclose(m1.ravel(), m2.ravel(), atol=1e-6)
    np.testing.assert_allclose(s1.ravel(), s2.ravel(), atol=1e-6)


def test_shared_update(data):
    X, Y, _, _ = data
    rng = np.random.default_rng(2)
    X_u = rng.uniform(size=(3, 2))
    Y_u = np.array([code(x) for x in X_u])
    m = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared")
    theta = m.theta()
    m.update(Y_u, X_u, refit=False)
    np.testing.assert_array_equal(m.theta(), theta)
    np.testing.assert_allclose(m.predict(X_u)[0], Y_u, atol=1e-6)


# ----------------------------------------------------------------------------- separable


def few_outputs(X):
    return np.column_stack([np.sin(6 * X[:, 0]) + X[:, 1],
                            np.sin(6 * X[:, 0]) - 2 * np.cos(5 * X[:, 1]),
                            X[:, 0] * X[:, 1] + 0.5 * np.cos(7 * X[:, 1]),
                            3 + np.cos(4 * X[:, 0] + 2 * X[:, 1])])


def test_separable():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(30, 2))
    Y = few_outputs(X)
    Xnew = rng.uniform(size=(6, 2))
    sep = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="separable")
    sh = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared")
    S = sep.output_cov()
    assert S.shape == (4, 4)
    np.testing.assert_allclose(S, S.T)
    np.testing.assert_allclose(np.diag(S), sep.sigma2())
    assert sep.logLikelihoodFun(sh.theta())[0] >= sh.logLikelihood()

    mean, sd, cov, _ = sep.predict(Xnew, return_stdev=True, return_cov=True)
    Cx, Sraw = sep.predictCovFactors(Xnew)
    assert Cx.shape == (6, 6) and Sraw.shape == (4, 4)
    np.testing.assert_allclose(cov, np.kron(Sraw, Cx), atol=1e-10)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), sd.ravel(order="F"), rtol=1e-8)
    # same mean as "shared" at the same theta (autokrigeability)
    sh_same = lk.MultiOutputKriging(Y, X, "matern5_2", output_model="shared", optim="none",
                                    parameters={"theta": sep.theta()[None, :]})
    np.testing.assert_allclose(mean, sh_same.predict(Xnew)[0], atol=1e-10)

    sims = sep.simulate(nsim=20000, seed=1, X=Xnew[:1])
    np.testing.assert_allclose(np.cov(sims[0]), Sraw * Cx[0, 0], atol=0.05 * np.diag(Sraw * Cx[0, 0]).max())


def test_separable_restrictions(data):
    X, Y, _, _ = data
    with pytest.raises(ValueError, match="n - p >= q"):
        lk.MultiOutputKriging(Y, X, "matern5_2", output_model="separable")  # q = 200 > n - p = 39
