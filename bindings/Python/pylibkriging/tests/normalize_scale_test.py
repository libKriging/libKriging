import numpy as np
import pytest

import pylibkriging as lk

# With normalize=True, Kriging reads its parameters in the original scale of
# X and y, and gives every parameter accessor / objective function in that
# same original scale (round trip, and agreement with normalize=False).


def make_data(n=25):
    rng = np.random.default_rng(4321)
    U = rng.uniform(size=(n, 2))
    X = np.column_stack([10.0 + 20.0 * U[:, 0], -2.0 + 5.0 * U[:, 1]])
    y = 100.0 + 50.0 * (np.sin(5 * U[:, 0]) * np.cos(3 * U[:, 1]) + 0.5 * U[:, 0] * U[:, 1])
    noise = 0.5 + 0.5 * rng.uniform(size=n)
    return X, y, noise


@pytest.mark.parametrize("noise", [None, "nugget", "vector"])
@pytest.mark.parametrize("regmodel,nbeta", [("constant", 1), ("linear", 3), ("quadratic", 6)])
def test_fixed_parameters_round_trip(noise, regmodel, nbeta):
    X, y, nvec = make_data()
    params = {"theta": np.array([[7.0, 1.3]]), "is_theta_estim": False,
              "sigma2": 321.0, "is_sigma2_estim": False,
              "beta": np.linspace(-3.0, 5.0, nbeta), "is_beta_estim": False}
    if noise == "nugget":
        params.update({"nugget": 12.5, "is_nugget_estim": False})
    nz = nvec if noise == "vector" else noise
    k = lk.Kriging(y, X, "matern5_2", regmodel=regmodel, normalize=True, optim="none",
                   parameters=params, noise=nz)
    np.testing.assert_allclose(k.theta(), [7.0, 1.3], rtol=1e-12)
    np.testing.assert_allclose(k.sigma2(), 321.0, rtol=1e-12)
    np.testing.assert_allclose(k.beta(), np.linspace(-3.0, 5.0, nbeta), rtol=1e-12, atol=1e-12)
    if noise == "nugget":
        np.testing.assert_allclose(k.nugget(), 12.5, rtol=1e-12)
    if noise == "vector":
        np.testing.assert_allclose(k.noise(), nvec, rtol=1e-12)


@pytest.mark.parametrize("regmodel", ["constant", "linear"])
def test_normalize_true_matches_false(regmodel):
    X, y, _ = make_data()
    params = {"theta": np.array([[9.0, 2.1]]), "is_theta_estim": False}
    kn = lk.Kriging(y, X, "matern5_2", regmodel=regmodel, normalize=True, optim="none", parameters=params)
    ko = lk.Kriging(y, X, "matern5_2", regmodel=regmodel, normalize=False, optim="none", parameters=params)
    np.testing.assert_allclose(kn.theta(), ko.theta(), rtol=1e-10)
    np.testing.assert_allclose(kn.sigma2(), ko.sigma2(), rtol=1e-8)
    np.testing.assert_allclose(kn.beta(), ko.beta(), rtol=1e-8)
    np.testing.assert_allclose(kn.logLikelihood(), ko.logLikelihood(), rtol=1e-10)
    np.testing.assert_allclose(kn.leaveOneOut(), ko.leaveOneOut(), rtol=1e-8)
    np.testing.assert_allclose(kn.logMargPost(), ko.logMargPost(), rtol=1e-10)
    t = ko.theta()
    lln, gn, _ = kn.logLikelihoodFun(t, return_grad=True)
    llo, go, _ = ko.logLikelihoodFun(t, return_grad=True)
    np.testing.assert_allclose(lln, llo, rtol=1e-10)
    np.testing.assert_allclose(gn, go, rtol=1e-6)
    np.testing.assert_allclose(kn.leaveOneOutFun(t, True)[1], ko.leaveOneOutFun(t, True)[1], rtol=1e-6)
    np.testing.assert_allclose(kn.logMargPostFun(t, True)[1], ko.logMargPostFun(t, True)[1], rtol=1e-6)


def test_objective_at_theta_and_gradient():
    X, y, _ = make_data()
    k = lk.Kriging(y, X, "matern5_2", regmodel="constant", normalize=True, optim="BFGS")
    t = k.theta()
    np.testing.assert_allclose(k.logLikelihoodFun(t)[0], k.logLikelihood(), rtol=1e-10)
    np.testing.assert_allclose(k.leaveOneOutFun(t, False)[0], k.leaveOneOut(), rtol=1e-10)
    np.testing.assert_allclose(k.logMargPostFun(t, False)[0], k.logMargPost(), rtol=1e-10)
    t1 = t * np.array([1.3, 0.8])
    _, g, _ = k.logLikelihoodFun(t1, return_grad=True)
    fd = np.zeros(2)
    for i in range(2):
        h = 1e-5 * t1[i]
        tp, tm = t1.copy(), t1.copy()
        tp[i] += h
        tm[i] -= h
        fd[i] = (k.logLikelihoodFun(tp)[0] - k.logLikelihoodFun(tm)[0]) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-4)
