"""
Test suite for newly added features: covMat, model(), and Optim class
"""
import pylibkriging as lk
import numpy as np
import pytest


class TestCovMat:
    """Test covariance matrix computation"""
    
    def test_covMat_basic(self):
        """Test basic covMat functionality"""
        np.random.seed(123)
        n = 20
        X = np.random.uniform(size=(n, 2))
        y = np.sin(X[:, 0]) + np.cos(X[:, 1])
        
        # Fit model
        k = lk.Kriging(y.reshape(-1, 1), X, "matern3_2")
        
        # Test covMat computation
        X1 = np.random.uniform(size=(5, 2))
        X2 = np.random.uniform(size=(10, 2))
        
        cov = k.covMat(X1, X2)
        
        # Check dimensions
        assert cov.shape == (5, 10), f"Expected shape (5, 10), got {cov.shape}"
        
        # Check symmetry when X1 == X2
        cov_sym = k.covMat(X1, X1)
        assert cov_sym.shape == (5, 5)
        assert np.allclose(cov_sym, cov_sym.T), "Covariance should be symmetric"
        
        # Covariance should be positive semi-definite
        eigenvals = np.linalg.eigvals(cov_sym)
        assert np.all(eigenvals >= -1e-10), "Covariance should be positive semi-definite"
    
    def test_covMat_all_classes(self):
        """Test covMat for all Kriging noise modes"""
        np.random.seed(456)
        n = 15
        X = np.random.uniform(size=(n, 1))
        y = np.sin(3 * X).flatten()
        noise = 0.01 * np.ones(n)
        
        X_test = np.random.uniform(size=(5, 1))
        
        # Test Kriging (no noise)
        k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
        cov1 = k1.covMat(X_test, X_test)
        assert cov1.shape == (5, 5)
        
        # Test Kriging with heterogeneous noise
        k2 = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise=noise.reshape(-1, 1))
        cov2 = k2.covMat(X_test, X_test)
        assert cov2.shape == (5, 5)
        
        # Test Kriging with nugget
        k3 = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise="nugget")
        cov3 = k3.covMat(X_test, X_test)
        assert cov3.shape == (5, 5)


class TestModel:
    """Test model() method for parameter introspection"""
    
    def test_model_basic(self):
        """Test basic model() functionality"""
        np.random.seed(789)
        n = 10
        X = np.random.uniform(size=(n, 1))
        y = np.exp(X).flatten()
        
        k = lk.Kriging(y.reshape(-1, 1), X, "matern5_2", "linear", True, "BFGS", "LL")
        
        # Get model parameters
        params = k.model()
        
        # Check that all expected keys are present
        expected_keys = ['kernel', 'optim', 'objective', 'theta', 'is_theta_estim',
                        'sigma2', 'is_sigma2_estim', 'X', 'centerX', 'scaleX',
                        'y', 'centerY', 'scaleY', 'normalize', 'regmodel',
                        'beta', 'is_beta_estim', 'F', 'T', 'M', 'z']
        
        for key in expected_keys:
            assert key in params, f"Missing key: {key}"
        
        # Check types and values
        assert params['kernel'] == 'matern5_2'
        assert params['optim'] == 'BFGS'
        assert params['objective'] == 'LL'
        assert params['normalize'] == True
        assert params['regmodel'] == 'linear'
        
        # Check array shapes: X is a matrix, vectors are 1-D
        assert params['X'].shape == (n, 1)
        assert params['y'].shape == (n,)
        assert len(params['theta']) > 0
        assert len(params['beta']) > 0
    
    def test_model_noise_kriging(self):
        """Test model() for Kriging with heterogeneous noise includes noise field"""
        np.random.seed(321)
        n = 12
        X = np.random.uniform(size=(n, 2))
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        noise = 0.05 * np.ones(n)
        
        k = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise=noise.reshape(-1, 1))
        params = k.model()
        
        # Heterogeneous noise Kriging should have 'noise' field
        assert 'noise' in params, "Heterogeneous noise model should have 'noise' field"
        assert params['noise'].shape == (n,)
        assert params['noise_model'] == 'heterogeneous'
    
    def test_model_nugget_kriging(self):
        """Test model() for Kriging with nugget includes nugget fields"""
        np.random.seed(654)
        n = 15
        X = np.random.uniform(size=(n, 1))
        y = X.flatten() ** 2
        
        k = lk.Kriging(y.reshape(-1, 1), X, "matern3_2", noise="nugget")
        params = k.model()
        
        # Nugget Kriging should have 'nugget' and 'is_nugget_estim' fields
        assert 'nugget' in params, "Nugget model should have 'nugget' field"
        assert 'is_nugget_estim' in params, "Nugget model should have 'is_nugget_estim' field"
        assert isinstance(params['nugget'], (int, float))
        assert isinstance(params['is_nugget_estim'], bool)
        assert params['noise_model'] == 'nugget'


class TestOptim:
    """Test Optim class static configuration methods"""
    
    def test_optim_reparametrization(self):
        """Test reparametrization settings"""
        # Save original state
        orig_state = lk.Optim.is_reparametrized()
        
        try:
            # Test setter and getter
            lk.Optim.use_reparametrize(True)
            assert lk.Optim.is_reparametrized() == True
            
            lk.Optim.use_reparametrize(False)
            assert lk.Optim.is_reparametrized() == False
        finally:
            # Restore original state
            lk.Optim.use_reparametrize(orig_state)
    
    def test_optim_theta_bounds(self):
        """Test theta bound factor settings"""
        # Save original values
        orig_lower = lk.Optim.get_theta_lower_factor()
        orig_upper = lk.Optim.get_theta_upper_factor()
        
        try:
            # Test lower factor
            lk.Optim.set_theta_lower_factor(0.05)
            assert abs(lk.Optim.get_theta_lower_factor() - 0.05) < 1e-10
            
            # Test upper factor
            lk.Optim.set_theta_upper_factor(15.0)
            assert abs(lk.Optim.get_theta_upper_factor() - 15.0) < 1e-10
        finally:
            # Restore original values
            lk.Optim.set_theta_lower_factor(orig_lower)
            lk.Optim.set_theta_upper_factor(orig_upper)
    
    def test_optim_variogram_bounds(self):
        """Test variogram bounds heuristic"""
        orig_state = lk.Optim.variogram_bounds_heuristic_used()
        
        try:
            lk.Optim.use_variogram_bounds_heuristic(True)
            assert lk.Optim.variogram_bounds_heuristic_used() == True
            
            lk.Optim.use_variogram_bounds_heuristic(False)
            assert lk.Optim.variogram_bounds_heuristic_used() == False
        finally:
            lk.Optim.use_variogram_bounds_heuristic(orig_state)
    
    def test_optim_log_level(self):
        """Test log level settings"""
        orig_level = lk.Optim.get_log_level()
        
        try:
            for level in [0, 1, 2, 3]:
                lk.Optim.set_log_level(level)
                assert lk.Optim.get_log_level() == level
        finally:
            lk.Optim.set_log_level(orig_level)
    
    def test_optim_max_iteration(self):
        """Test max iteration settings"""
        orig_max = lk.Optim.get_max_iteration()
        
        try:
            lk.Optim.set_max_iteration(500)
            assert lk.Optim.get_max_iteration() == 500
            
            lk.Optim.set_max_iteration(1000)
            assert lk.Optim.get_max_iteration() == 1000
        finally:
            lk.Optim.set_max_iteration(orig_max)
    
    def test_optim_tolerances(self):
        """Test tolerance settings"""
        orig_grad = lk.Optim.get_gradient_tolerance()
        orig_obj = lk.Optim.get_objective_rel_tolerance()
        
        try:
            lk.Optim.set_gradient_tolerance(1e-6)
            assert abs(lk.Optim.get_gradient_tolerance() - 1e-6) < 1e-15
            
            lk.Optim.set_objective_rel_tolerance(1e-8)
            assert abs(lk.Optim.get_objective_rel_tolerance() - 1e-8) < 1e-15
        finally:
            lk.Optim.set_gradient_tolerance(orig_grad)
            lk.Optim.set_objective_rel_tolerance(orig_obj)
    
    def test_optim_thread_settings(self):
        """Test thread configuration"""
        orig_delay = lk.Optim.get_thread_start_delay_ms()
        orig_pool = lk.Optim.get_thread_pool_size()
        
        try:
            lk.Optim.set_thread_start_delay_ms(20)
            assert lk.Optim.get_thread_start_delay_ms() == 20
            
            lk.Optim.set_thread_pool_size(4)
            assert lk.Optim.get_thread_pool_size() == 4
        finally:
            lk.Optim.set_thread_start_delay_ms(orig_delay)
            lk.Optim.set_thread_pool_size(orig_pool)

    def test_optim_thread_budget(self):
        """Thread budget: share of the cores, resolved thread count"""
        orig = lk.Optim.get_num_threads_percent()
        try:
            lk.Optim.set_num_threads_percent(100)
            full = lk.Optim.get_num_threads()
            assert full >= 1
            lk.Optim.set_num_threads_percent(50)
            assert lk.Optim.get_num_threads_percent() == 50
            assert 1 <= lk.Optim.get_num_threads() <= full
            lk.Optim.set_num_threads_percent(0.5)  # ratio: same as 50 %
            assert 1 <= lk.Optim.get_num_threads() <= full
            # a fit runs with a reduced budget
            X = np.random.RandomState(1).uniform(size=(20, 1))
            k = lk.Kriging(np.sin(6 * X[:, 0]), X, "matern3_2")
            assert np.isfinite(k.logLikelihood())
        finally:
            lk.Optim.set_num_threads_percent(orig)
    
    def test_optim_affects_optimization(self):
        """Test that Optim settings actually affect model fitting"""
        np.random.seed(999)
        n = 25
        X = np.random.uniform(size=(n, 1))
        y = np.sin(5 * X).flatten()
        
        # Save original settings
        orig_max_iter = lk.Optim.get_max_iteration()
        
        try:
            # Fit with low iteration limit
            lk.Optim.set_max_iteration(5)
            k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
            ll1 = k1.logLikelihood()
            
            # Fit with higher iteration limit
            lk.Optim.set_max_iteration(100)
            k2 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
            ll2 = k2.logLikelihood()
            
            # More iterations should generally give better or equal log-likelihood
            # (may not be strictly greater due to convergence)
            print(f"LogLikelihood with 5 iters: {ll1}, with 100 iters: {ll2}")
            assert ll2 >= ll1 - 1e-6, "More iterations should not worsen the fit significantly"
            
        finally:
            lk.Optim.set_max_iteration(orig_max_iter)


class TestConsistency:
    """Test consistency across different Kriging noise modes"""
    
    def test_all_modes_have_covMat(self):
        """Ensure all Kriging noise modes have covMat method"""
        np.random.seed(111)
        n = 10
        X = np.random.uniform(size=(n, 1))
        y = X.flatten()
        noise = 0.01 * np.ones(n)
        
        k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
        k2 = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise=noise.reshape(-1, 1))
        k3 = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise="nugget")
        
        X_test = np.random.uniform(size=(3, 1))
        
        # All should work
        cov1 = k1.covMat(X_test, X_test)
        cov2 = k2.covMat(X_test, X_test)
        cov3 = k3.covMat(X_test, X_test)
        
        assert cov1.shape == (3, 3)
        assert cov2.shape == (3, 3)
        assert cov3.shape == (3, 3)
    
    def test_all_modes_have_model(self):
        """Ensure all Kriging noise modes have model method returning dicts"""
        np.random.seed(222)
        n = 10
        X = np.random.uniform(size=(n, 1))
        y = X.flatten()
        noise = 0.01 * np.ones(n)
        
        k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
        k2 = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise=noise.reshape(-1, 1))
        k3 = lk.Kriging(y.reshape(-1, 1), X, "gauss", noise="nugget")
        
        # All should return dicts
        m1 = k1.model()
        m2 = k2.model()
        m3 = k3.model()
        
        assert isinstance(m1, dict)
        assert isinstance(m2, dict)
        assert isinstance(m3, dict)
        
        # Check noise_model field
        assert m1['noise_model'] == 'none'
        assert m2['noise_model'] == 'heterogeneous'
        assert m3['noise_model'] == 'nugget'
        
        # Check mode-specific fields
        assert 'noise' in m2
        assert 'nugget' in m3
        assert 'is_nugget_estim' in m3
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_vecchia_neighbors_and_nystrom_rank():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(30, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    assert lk.Kriging(y, X, "matern5_2", method="vecchia(7)").vecchia_neighbors() == 7
    assert lk.Kriging(y, X, "matern5_2", method="nystrom(5)").nystrom_rank() == 5
    k = lk.Kriging(y, X, "matern5_2")
    assert k.vecchia_neighbors() == 0
    assert k.nystrom_rank() == 0
    assert k.method() == "exact"


def test_fit_method_and_deprecated_objectives():
    rng = np.random.default_rng(9)
    X = rng.uniform(size=(40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    k = lk.Kriging(y, X, "matern5_2", method="vecchia")
    assert k.objective() == "LL" and k.method() == "vecchia(30)"
    assert lk.Kriging(y, X, "matern5_2", method="nystrom").method() == "nystrom(50)"
    assert k.model()["method"] == "vecchia(30)"
    # deprecated aliases: objective "LLVecchia(m)" / "LLNystrom(k)", with a
    # native DeprecationWarning (an error under a "error" warnings filter)
    with pytest.warns(DeprecationWarning, match=r'objective "LLVecchia\(7\)" is deprecated'):
        ka = lk.Kriging(y, X, "matern5_2", objective="LLVecchia(7)")
    assert ka.objective() == "LL" and ka.method() == "vecchia(7)"
    with pytest.warns(DeprecationWarning, match='method="nystrom\\(5\\)"'):
        kb = lk.Kriging(y, X, "matern5_2", objective="LLNystrom(5)")
    assert kb.objective() == "LL" and kb.method() == "nystrom(5)"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            lk.Kriging(y, X, "matern5_2", objective="LLVecchia(7)")
        lk.Kriging(y, X, "matern5_2", method="vecchia(7)")  # no warning
    # LOO / LMP have approximate versions too (without noise model); with
    # objective LL, both approximate methods support the noise models
    assert lk.Kriging(y, X, "matern5_2", objective="LOO", method="vecchia(10)").objective() == "LOO"
    with pytest.raises(ValueError):
        lk.Kriging(y, X, "matern5_2", objective="LOO", method="vecchia", noise="nugget")
    with pytest.raises(ValueError):
        lk.Kriging(y, X, "matern5_2", objective="LMP", method="nystrom(10)", noise="nugget")
    assert lk.Kriging(y, X, "matern5_2", method="nystrom(10)", noise="nugget").method() == "nystrom(10)"
    assert lk.Kriging(y, X, "matern5_2", method="vecchia", noise="nugget").method() == "vecchia(30)"
    with pytest.raises(ValueError):
        lk.Kriging(y, X, "matern5_2", method="Vecchia")
    with pytest.raises(ValueError):  # conflict: raised before any warning
        lk.Kriging(y, X, "matern5_2", objective="LLVecchia(7)", method="nystrom")


@pytest.mark.parametrize("method", ["vecchia(10)", "nystrom(20)"])
def test_loo_lmp_objectives_with_approximate_method(method, tmp_path):
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(60, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    am = method[:7]

    kl = lk.Kriging(y, X, "matern5_2", objective="LOO", method=method)
    assert kl.objective() == "LOO" and kl.method() == method
    loo = kl.leaveOneOut(method=am)
    assert np.isfinite(loo)
    assert kl.leaveOneOut(method=method) == loo
    assert np.isfinite(kl.leaveOneOut())  # exact LOO at the approximate theta
    mean, stdev = kl.leaveOneOutVec(kl.theta(), method=am)
    assert mean.shape == (60,) and np.all(stdev > 0)
    np.testing.assert_allclose(np.mean((mean - y) ** 2), loo, rtol=1e-8)
    with pytest.raises(ValueError):
        kl.leaveOneOut(method="unknown")

    kp = lk.Kriging(y, X, "matern5_2", objective="LMP", method=method)
    assert kp.objective() == "LMP" and kp.method() == method
    lmp = kp.logMargPost(method=am)
    assert np.isfinite(lmp)
    fname = str(tmp_path / "approx_lmp_py.json")
    kp.save(fname)
    kpl = lk.Kriging.load(fname)
    assert kpl.objective() == "LMP" and kpl.method() == method
    np.testing.assert_allclose(kpl.logMargPost(method=am), lmp, rtol=1e-10)

    # with m = n - 1 (k = n) the approximate criteria are the exact ones
    full = "vecchia(59)" if am == "vecchia" else "nystrom(60)"
    ke = lk.Kriging(y, X, "matern5_2", optim="none", objective="LOO", method=full,
                    parameters={"theta": np.array([[0.3, 0.3]]), "is_theta_estim": False})
    np.testing.assert_allclose(ke.leaveOneOut(method=am), ke.leaveOneOut(), rtol=1e-2)
    np.testing.assert_allclose(ke.logMargPost(method=am), ke.logMargPost(), rtol=1e-3)


def test_vecchia_fit(tmp_path):
    rng = np.random.default_rng(6)
    X = rng.uniform(size=(60, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    Xn = rng.uniform(size=(7, 2))

    k = lk.Kriging("matern5_2")
    k.fit(y, X, "constant", False, "BFGS", "LL", {}, method="vecchia(10)")
    assert k.method() == "vecchia(10)"

    # explicit method: no implicit switch from the exact predictor
    with pytest.raises(RuntimeError):
        k.predict(Xn, True, False, False)
    with pytest.raises(RuntimeError):
        k.simulate(4, 123, Xn)
    mean, stdev, _, _, _ = k.predict(Xn, True, False, False, method="vecchia")
    assert mean.shape == (7,) and stdev.shape == (7,)
    assert np.all(np.isfinite(mean)) and np.all(stdev >= 0)
    assert k.simulate(4, 123, Xn, method="vecchia").shape == (7, 4)
    assert k.simulate(4, 123, Xn, method="vecchia(5)").shape == (7, 4)
    with pytest.raises(ValueError):
        k.predict(Xn, True, True, False, method="vecchia")  # return_cov
    with pytest.raises(ValueError):
        k.predict(Xn, method="unknown")
    assert np.isfinite(k.logLikelihood(method="vecchia"))
    assert k.logLikelihood(method=k.method()) == k.logLikelihood(method="vecchia")

    # update (Vecchia refit) and save / load
    Xu = rng.uniform(size=(5, 2))
    yu = np.sin(3 * Xu[:, 0]) + Xu[:, 1]
    k.update(yu, Xu, False)
    assert k.method() == "vecchia(10)" and k.X().shape == (65, 2)
    m_u = k.predict(Xu, method="vecchia")[0]
    assert np.max(np.abs(m_u - yu)) < 1e-3
    fname = str(tmp_path / "vecchia_py.json")
    k.save(fname)
    kl = lk.Kriging.load(fname)
    assert kl.method() == "vecchia(10)" and kl.vecchia_neighbors() == 10
    np.testing.assert_allclose(kl.predict(Xn, method="vecchia")[0], k.predict(Xn, method="vecchia")[0],
                               atol=1e-12)

    # the exact model at the Vecchia estimate is a separate exact fit
    ke = lk.Kriging(y, X, "matern5_2", optim="none",
                    parameters={"theta": np.atleast_2d(k.theta()), "is_theta_estim": False})
    assert ke.method() == "exact"
    ke.predict(Xn, True, True, False)
    ke.predict(Xn, method="vecchia")
    assert ke.logLikelihood() == ke.logLikelihood(method="exact")


def test_vecchia_noise_models(tmp_path):
    rng = np.random.default_rng(11)
    n = 80
    X = rng.uniform(size=(n, 2))
    noise = 0.005 + 0.02 * X[:, 0]
    y = np.sin(3 * X[:, 0]) + X[:, 1] + np.sqrt(noise) * rng.standard_normal(n)
    Xn = rng.uniform(size=(6, 2))

    # Nugget: response Vecchia fit, light (no exact factorization)
    k = lk.Kriging(y, X, "matern5_2", noise="nugget", method="vecchia(15)")
    assert k.method() == "vecchia(15)"
    assert k.nugget() > 0 and k.sigma2() > 0
    with pytest.raises(RuntimeError):
        k.predict(Xn)
    mean, stdev, _, _, _ = k.predict(Xn, method="vecchia")
    assert np.all(np.isfinite(mean)) and np.all(stdev > 0)
    assert np.isfinite(k.logLikelihood(method="vecchia"))
    s = k.simulate(5, 123, Xn, method="vecchia")
    assert s.shape == (6, 5)
    s_nug = k.simulate(5, 123, Xn, with_noise=True, method="vecchia")
    assert s_nug.shape == (6, 5) and not np.array_equal(s, s_nug)
    with pytest.raises(ValueError):
        k.simulate(5, 123, Xn, with_noise=True, method="nystrom")
    k.update(y[:5] + 0.01, X[:5] + 0.01, False)
    assert k.X().shape == (n + 5, 2)
    fname = str(tmp_path / "vecchia_nugget_py.json")
    k.save(fname)
    kl = lk.Kriging.load(fname)
    assert kl.method() == "vecchia(15)"
    np.testing.assert_allclose(kl.predict(Xn, method="vecchia")[0], k.predict(Xn, method="vecchia")[0],
                               atol=1e-12)

    # Heterogeneous (known noise variances)
    kh = lk.Kriging(y, X, "matern5_2", noise=noise, method="vecchia(15)")
    assert kh.method() == "vecchia(15)"
    mean_h, stdev_h, _, _, _ = kh.predict(Xn, method="vecchia")
    assert np.all(np.isfinite(mean_h)) and np.all(stdev_h > 0)
    assert kh.simulate(5, 123, Xn, with_noise=np.full(6, 0.1), method="vecchia").shape == (6, 5)
    kh.update(y[:5], X[:5] + 0.01, False, noise[:5])
    assert kh.X().shape == (n + 5, 2)

    # same theta as the exact fit, roughly
    ke = lk.Kriging(y, X, "matern5_2", noise=noise)
    assert np.max(np.abs(np.log(kh.theta().ravel()) - np.log(ke.theta().ravel()))) < 1.0


def test_nystrom_fit_explicit_method():
    rng = np.random.default_rng(7)
    X = rng.uniform(size=(60, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    Xn = rng.uniform(size=(7, 2))
    k = lk.Kriging(y, X, "matern5_2", method="nystrom(8)")
    assert k.method() == "nystrom(8)"
    with pytest.raises(RuntimeError):
        k.predict(Xn)
    mean, stdev, _, _, _ = k.predict(Xn, method="nystrom")
    assert mean.shape == (7,) and np.all(stdev >= 0)
    np.testing.assert_array_equal(k.predict(Xn, method=k.method())[0], mean)
    with pytest.raises(ValueError):
        k.predict(Xn, method="nystrom(3)")  # not the fitted rank
    assert k.simulate(3, 1, Xn, method="nystrom").shape == (7, 3)
    assert np.isfinite(k.logLikelihood(method="nystrom"))
    with pytest.raises(RuntimeError):
        k.logLikelihood(method="vecchia")  # not a Vecchia fit


def test_nystrom_fit_noise_models(tmp_path):
    rng = np.random.default_rng(11)
    n = 80
    X = rng.uniform(size=(n, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] + 0.05 * rng.standard_normal(n)
    Xn = rng.uniform(size=(7, 2))
    noise = np.full(n, 0.0025)

    # nugget: the noise variance is estimated
    kn = lk.Kriging(y, X, "matern5_2", noise="nugget", method="nystrom(20)")
    assert kn.method() == "nystrom(20)" and kn.noise_model() == "nugget"
    assert kn.nugget() > 0
    with pytest.raises(RuntimeError):
        kn.predict(Xn)  # no exact factorization
    mean, stdev, _, _, _ = kn.predict(Xn, method="nystrom")
    assert mean.shape == (7,) and np.all(stdev > 0)
    assert kn.simulate(3, 1, Xn, method="nystrom").shape == (7, 3)
    assert np.isfinite(kn.logLikelihood(method="nystrom"))
    kn.update(y[:5] + 0.01, X[:5] + 0.01, refit=False)
    assert kn.X().shape == (n + 5, 2)

    # heterogeneous: known noise vector
    kh = lk.Kriging(y, X, "matern5_2", noise=noise, method="nystrom(20)")
    assert kh.method() == "nystrom(20)" and kh.noise_model() == "heterogeneous"
    mean_h, stdev_h, _, _, _ = kh.predict(Xn, method="nystrom")
    assert np.all(np.isfinite(mean_h)) and np.all(stdev_h >= 0)
    assert kh.simulate(3, 1, Xn, method="nystrom").shape == (7, 3)
    f = str(tmp_path / "nys_hetero.json")
    kh.save(f)
    kh2 = lk.Kriging.load(f)
    np.testing.assert_array_equal(kh2.predict(Xn, method="nystrom")[0], mean_h)
    kh.update(y[:5], X[:5] + 0.01, refit=False, noise_u=np.full(5, 0.0025))
    assert kh.X().shape == (n + 5, 2) and kh.method() == "nystrom(20)"

    # close to full rank, the Nystrom fit agrees with the exact one
    ke = lk.Kriging(y, X, "matern5_2", noise="nugget")
    kf = lk.Kriging(y, X, "matern5_2", noise="nugget", method=f"nystrom({n})")
    me = ke.predict(Xn)[0]
    mf = kf.predict(Xn, method="nystrom")[0]
    assert np.max(np.abs(me - mf)) < 0.1 * np.std(y)

    # Nystrom simulates the latent process only: with_noise raises with it
    assert kn.simulate(3, 1, Xn, with_noise=False, method="nystrom").shape == (7, 3)
    with pytest.raises(ValueError):
        kn.simulate(3, 1, Xn, with_noise=True, method="nystrom")
    assert kh.simulate(3, 1, Xn, with_noise=np.zeros(7), method="nystrom").shape == (7, 3)
    with pytest.raises(ValueError):
        kh.simulate(3, 1, Xn, with_noise=np.full(7, 0.1), method="nystrom")


def test_nested_kriging_method():
    rng = np.random.default_rng(10)
    X = rng.uniform(size=(200, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    nk = lk.NestedKriging(y, X, "matern5_2", 3, method="vecchia(15)")
    assert nk.objective() == "LL" and nk.method() == "vecchia(15)"
    mean, _ = nk.predict(X[:5], True)
    assert np.max(np.abs(mean - y[:5])) < 1e-3
    assert lk.NestedKriging(y, X, "matern5_2", 3).method() == "exact"


def test_sklearn_regressor_method():
    pytest.importorskip("sklearn")
    from pylibkriging.sklearn import KrigingRegressor
    rng = np.random.default_rng(8)
    X = rng.uniform(size=(40, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    reg = KrigingRegressor(method="nystrom(8)").fit(X, y)
    assert reg.model_.method() == "nystrom(8)"
    assert reg.predict(X[:3]).shape == (3,)
    assert reg.sample_y(X[:3], n_samples=2).shape == (3, 2)


def test_heterogeneous_noise_api():
    """with_noise holds noise variances; update without noise_u raises;
    approximate LOO / LMP raise on noise models"""
    rng = np.random.default_rng(5)
    n = 30
    X = rng.uniform(size=(n, 2))
    noise = np.full(n, 0.01)
    y = np.sin(3 * X[:, 0]) + np.cos(5 * X[:, 1]) + np.sqrt(noise) * rng.standard_normal(n)
    Xn = rng.uniform(size=(3, 2))
    kh = lk.Kriging(y, X, "matern5_2", noise=noise)

    # same seed: the latent draws are identical, the difference is the noise
    var_n = np.array([0.05, 0.2, 0.5])
    for method in ("exact", "vecchia"):
        s_noisy = kh.simulate(20000, 7, Xn, with_noise=var_n, method=method)
        s_latent = kh.simulate(20000, 7, Xn, with_noise=np.zeros(3), method=method)
        emp_var = np.var(s_noisy - s_latent, axis=1, ddof=1)
        np.testing.assert_allclose(emp_var, var_n, rtol=0.05)
    with pytest.raises(ValueError):
        kh.simulate(10, 7, Xn, with_noise=-var_n)

    # update needs the noise of the new points
    with pytest.raises(ValueError, match="heterogeneous noise model: use update"):
        kh.update(y[:2], X[:2] + 0.01, False)
    assert kh.X().shape == (n, 2)
    kh.update(y[:2], X[:2] + 0.01, False, noise[:2])
    assert kh.X().shape == (n + 2, 2)

    # update_simulate needs the noise of the new points too; the failed call
    # changes nothing
    Xu = X[:2] + 0.02
    kh.simulate(10, 7, Xn, will_update=True)
    with pytest.raises(ValueError, match="heterogeneous noise model: use update_simulate"):
        kh.update_simulate(y[:2], Xu)
    u1 = kh.update_simulate(y[:2], Xu, noise[:2])
    with pytest.raises(ValueError):
        kh.update_simulate(y[:2], Xu)
    np.testing.assert_allclose(kh.update_simulate(y[:2], Xu, noise[:2]), u1, atol=1e-12)

    # approximate LOO / LMP: noise model None only
    kv = lk.Kriging(y, X, "matern5_2", noise="nugget", method="vecchia(10)")
    assert np.isfinite(kv.logLikelihood(method="vecchia"))
    with pytest.raises(ValueError, match="not implemented for Nugget/Heterogeneous"):
        kv.leaveOneOut(method="vecchia")
    with pytest.raises(ValueError):
        kv.logMargPost(method="vecchia")


def test_fit_and_update_drop_the_last_simulate():
    """simulate(will_update=True) -> update / fit -> update_simulate raises
    (the stored paths were built for the previous model); a new simulate works"""
    rng = np.random.default_rng(11)
    X = rng.uniform(size=(15, 2))
    y = np.sin(3 * X[:, 0]) + np.cos(5 * X[:, 1])
    Xn = rng.uniform(size=(4, 2))
    Xu = rng.uniform(size=(3, 2))
    yu = np.sin(3 * Xu[:, 0]) + np.cos(5 * Xu[:, 1])
    k = lk.Kriging(y, X, "gauss")
    k.simulate(10, 1, Xn, will_update=True)
    k.update_simulate(yu, Xu)
    k.update(yu, Xu, False)
    with pytest.raises(RuntimeError, match="No previous simulation data available"):
        k.update_simulate(yu, Xu)
    k.simulate(10, 1, Xn, will_update=True)
    assert k.update_simulate(yu, Xu).shape == (4, 10)
    k.fit(y, X, "constant", False, "BFGS", "LL")
    with pytest.raises(RuntimeError, match="No previous simulation data available"):
        k.update_simulate(yu, Xu)

    wk = lk.WarpKriging(y, X, ["affine", "affine"], "gauss", parameters={"max_iter_adam": "20"})
    wk.simulate(10, 1, Xn, will_update=True)
    wk.update(yu, Xu, refit=False)
    with pytest.raises(RuntimeError, match="No previous simulation data available"):
        wk.update_simulate(yu, Xu)
