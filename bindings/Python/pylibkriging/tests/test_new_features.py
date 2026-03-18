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
        """Test covMat for all Kriging classes"""
        np.random.seed(456)
        n = 15
        X = np.random.uniform(size=(n, 1))
        y = np.sin(3 * X).flatten()
        noise = 0.01 * np.ones(n)
        
        X_test = np.random.uniform(size=(5, 1))
        
        # Test Kriging
        k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
        cov1 = k1.covMat(X_test, X_test)
        assert cov1.shape == (5, 5)
        
        # Test NoiseKriging
        k2 = lk.NoiseKriging(y.reshape(-1, 1), noise.reshape(-1, 1), X, "gauss")
        cov2 = k2.covMat(X_test, X_test)
        assert cov2.shape == (5, 5)
        
        # Test NuggetKriging
        k3 = lk.NuggetKriging(y.reshape(-1, 1), X, "gauss")
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
        
        # Check array shapes
        assert params['X'].shape == (n, 1)
        assert params['y'].shape == (n,)
        assert len(params['theta']) > 0
        assert len(params['beta']) > 0
    
    def test_model_noise_kriging(self):
        """Test model() for NoiseKriging includes noise field"""
        np.random.seed(321)
        n = 12
        X = np.random.uniform(size=(n, 2))
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])
        noise = 0.05 * np.ones(n)
        
        k = lk.NoiseKriging(y.reshape(-1, 1), noise.reshape(-1, 1), X, "gauss")
        params = k.model()
        
        # NoiseKriging should have 'noise' field
        assert 'noise' in params, "NoiseKriging model should have 'noise' field"
        assert params['noise'].shape == (n,)
    
    def test_model_nugget_kriging(self):
        """Test model() for NuggetKriging includes nugget fields"""
        np.random.seed(654)
        n = 15
        X = np.random.uniform(size=(n, 1))
        y = X.flatten() ** 2
        
        k = lk.NuggetKriging(y.reshape(-1, 1), X, "matern3_2")
        params = k.model()
        
        # NuggetKriging should have 'nugget' and 'is_nugget_estim' fields
        assert 'nugget' in params, "NuggetKriging model should have 'nugget' field"
        assert 'is_nugget_estim' in params, "NuggetKriging model should have 'is_nugget_estim' field"
        assert isinstance(params['nugget'], (int, float))
        assert isinstance(params['is_nugget_estim'], bool)


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
    """Test consistency across different Kriging classes"""
    
    def test_all_classes_have_covMat(self):
        """Ensure all Kriging classes have covMat method"""
        np.random.seed(111)
        n = 10
        X = np.random.uniform(size=(n, 1))
        y = X.flatten()
        noise = 0.01 * np.ones(n)
        
        k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
        k2 = lk.NoiseKriging(y.reshape(-1, 1), noise.reshape(-1, 1), X, "gauss")
        k3 = lk.NuggetKriging(y.reshape(-1, 1), X, "gauss")
        
        X_test = np.random.uniform(size=(3, 1))
        
        assert hasattr(k1, 'covMat')
        assert hasattr(k2, 'covMat')
        assert hasattr(k3, 'covMat')
        
        # All should work
        cov1 = k1.covMat(X_test, X_test)
        cov2 = k2.covMat(X_test, X_test)
        cov3 = k3.covMat(X_test, X_test)
        
        assert cov1.shape == (3, 3)
        assert cov2.shape == (3, 3)
        assert cov3.shape == (3, 3)
    
    def test_all_classes_have_model(self):
        """Ensure all Kriging classes have model method"""
        np.random.seed(222)
        n = 10
        X = np.random.uniform(size=(n, 1))
        y = X.flatten()
        noise = 0.01 * np.ones(n)
        
        k1 = lk.Kriging(y.reshape(-1, 1), X, "gauss")
        k2 = lk.NoiseKriging(y.reshape(-1, 1), noise.reshape(-1, 1), X, "gauss")
        k3 = lk.NuggetKriging(y.reshape(-1, 1), X, "gauss")
        
        assert hasattr(k1, 'model')
        assert hasattr(k2, 'model')
        assert hasattr(k3, 'model')
        
        # All should return dicts
        m1 = k1.model()
        m2 = k2.model()
        m3 = k3.model()
        
        assert isinstance(m1, dict)
        assert isinstance(m2, dict)
        assert isinstance(m3, dict)
        
        # Check class-specific fields
        assert 'noise' in m2
        assert 'nugget' in m3
        assert 'is_nugget_estim' in m3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
