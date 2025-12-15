% Test suite for newly added features: covMat, model(), and Optim class
% Run this test with: octave test_new_features.m

function test_new_features()
    fprintf('=== Testing New Features ===\n\n');
    
    test_covMat_basic();
    test_covMat_all_classes();
    test_model_basic();
    test_model_noise_kriging();
    test_model_nugget_kriging();
    test_optim_reparametrization();
    test_optim_theta_bounds();
    test_optim_variogram_bounds();
    test_optim_log_level();
    test_optim_max_iteration();
    test_optim_tolerances();
    test_optim_thread_settings();
    test_all_classes_have_covMat();
    test_all_classes_have_model();
    
    fprintf('\n=== All Tests Passed! ===\n');
end


%% Test covMat functionality
function test_covMat_basic()
    fprintf('Test: covMat basic functionality...');
    
    rand('seed', 123);
    n = 20;
    X = rand(n, 2);
    y = sin(X(:, 1)) + cos(X(:, 2));
    
    % Fit model
    k = Kriging(y, X, 'matern3_2');
    
    % Test covMat computation
    X1 = rand(5, 2);
    X2 = rand(10, 2);
    
    cov = k.covMat(X1, X2);
    
    % Check dimensions
    assert(all(size(cov) == [5, 10]), 'Expected size [5, 10]');
    
    % Check symmetry when X1 == X2
    cov_sym = k.covMat(X1, X1);
    assert(all(size(cov_sym) == [5, 5]));
    assert(max(max(abs(cov_sym - cov_sym'))) < 1e-10, 'Covariance should be symmetric');
    
    % Covariance should be positive semi-definite
    eigenvals = eig(cov_sym);
    assert(all(eigenvals >= -1e-10), 'Covariance should be positive semi-definite');
    
    fprintf(' PASSED\n');
end


function test_covMat_all_classes()
    fprintf('Test: covMat for all Kriging classes...');
    
    rand('seed', 456);
    n = 15;
    X = rand(n, 1);
    y = sin(3 * X);
    noise = 0.01 * ones(n, 1);
    
    X_test = rand(5, 1);
    
    % Test Kriging
    k1 = Kriging(y, X, 'gauss');
    cov1 = k1.covMat(X_test, X_test);
    assert(all(size(cov1) == [5, 5]));
    
    % Test NoiseKriging
    k2 = NoiseKriging(y, noise, X, 'gauss');
    cov2 = k2.covMat(X_test, X_test);
    assert(all(size(cov2) == [5, 5]));
    
    % Test NuggetKriging
    k3 = NuggetKriging(y, X, 'gauss');
    cov3 = k3.covMat(X_test, X_test);
    assert(all(size(cov3) == [5, 5]));
    
    fprintf(' PASSED\n');
end


%% Test model() functionality
function test_model_basic()
    fprintf('Test: model() basic functionality...');
    
    rand('seed', 789);
    n = 10;
    X = rand(n, 1);
    y = exp(X);
    
    k = Kriging(y, X, 'matern5_2', 'linear', true, 'BFGS', 'LL');
    
    % Get model parameters
    params = k.model();
    
    % Check that all expected fields are present
    expected_fields = {'kernel', 'optim', 'objective', 'theta', 'is_theta_estim', ...
                       'sigma2', 'is_sigma2_estim', 'X', 'centerX', 'scaleX', ...
                       'y', 'centerY', 'scaleY', 'normalize', 'regmodel', ...
                       'beta', 'is_beta_estim', 'F', 'T', 'M', 'z'};
    
    for i = 1:length(expected_fields)
        assert(isfield(params, expected_fields{i}), ...
               sprintf('Missing field: %s', expected_fields{i}));
    end
    
    % Check types and values
    assert(strcmp(params.kernel, 'matern5_2'));
    assert(strcmp(params.optim, 'BFGS'));
    assert(strcmp(params.objective, 'LL'));
    assert(params.normalize == true);
    assert(strcmp(params.regmodel, 'linear'));
    
    % Check array shapes
    assert(all(size(params.X) == [n, 1]));
    assert(length(params.y) == n);
    assert(length(params.theta) > 0);
    assert(length(params.beta) > 0);
    
    fprintf(' PASSED\n');
end


function test_model_noise_kriging()
    fprintf('Test: model() for NoiseKriging...');
    
    rand('seed', 321);
    n = 12;
    X = rand(n, 2);
    y = sin(X(:, 1)) .* cos(X(:, 2));
    noise = 0.05 * ones(n, 1);
    
    k = NoiseKriging(y, noise, X, 'gauss');
    params = k.model();
    
    % NoiseKriging should have 'noise' field
    assert(isfield(params, 'noise'), 'NoiseKriging model should have noise field');
    assert(length(params.noise) == n);
    
    fprintf(' PASSED\n');
end


function test_model_nugget_kriging()
    fprintf('Test: model() for NuggetKriging...');
    
    rand('seed', 654);
    n = 15;
    X = rand(n, 1);
    y = X .^ 2;
    
    k = NuggetKriging(y, X, 'matern3_2');
    params = k.model();
    
    % NuggetKriging should have 'nugget' and 'is_nugget_estim' fields
    assert(isfield(params, 'nugget'), 'NuggetKriging model should have nugget field');
    assert(isfield(params, 'is_nugget_estim'), ...
           'NuggetKriging model should have is_nugget_estim field');
    assert(isnumeric(params.nugget));
    assert(islogical(params.is_nugget_estim));
    
    fprintf(' PASSED\n');
end


%% Test Optim class functionality
function test_optim_reparametrization()
    fprintf('Test: Optim reparametrization...');
    
    % Save original state
    orig_state = Optim.is_reparametrized();
    
    try
        % Test setter and getter
        Optim.use_reparametrize(true);
        assert(Optim.is_reparametrized() == true);
        
        Optim.use_reparametrize(false);
        assert(Optim.is_reparametrized() == false);
    catch err
        % Restore original state even on error
        Optim.use_reparametrize(orig_state);
        rethrow(err);
    end
    
    % Restore original state
    Optim.use_reparametrize(orig_state);
    
    fprintf(' PASSED\n');
end


function test_optim_theta_bounds()
    fprintf('Test: Optim theta bounds...');
    
    % Save original values
    orig_lower = Optim.get_theta_lower_factor();
    orig_upper = Optim.get_theta_upper_factor();
    
    try
        % Test lower factor
        Optim.set_theta_lower_factor(0.05);
        assert(abs(Optim.get_theta_lower_factor() - 0.05) < 1e-10);
        
        % Test upper factor
        Optim.set_theta_upper_factor(15.0);
        assert(abs(Optim.get_theta_upper_factor() - 15.0) < 1e-10);
    catch err
        Optim.set_theta_lower_factor(orig_lower);
        Optim.set_theta_upper_factor(orig_upper);
        rethrow(err);
    end
    
    % Restore original values
    Optim.set_theta_lower_factor(orig_lower);
    Optim.set_theta_upper_factor(orig_upper);
    
    fprintf(' PASSED\n');
end


function test_optim_variogram_bounds()
    fprintf('Test: Optim variogram bounds...');
    
    orig_state = Optim.variogram_bounds_heuristic_used();
    
    try
        Optim.use_variogram_bounds_heuristic(true);
        assert(Optim.variogram_bounds_heuristic_used() == true);
        
        Optim.use_variogram_bounds_heuristic(false);
        assert(Optim.variogram_bounds_heuristic_used() == false);
    catch err
        Optim.use_variogram_bounds_heuristic(orig_state);
        rethrow(err);
    end
    
    Optim.use_variogram_bounds_heuristic(orig_state);
    
    fprintf(' PASSED\n');
end


function test_optim_log_level()
    fprintf('Test: Optim log level...');
    
    orig_level = Optim.get_log_level();
    
    try
        for level = [0, 1, 2, 3]
            Optim.set_log_level(level);
            assert(Optim.get_log_level() == level);
        end
    catch err
        Optim.set_log_level(orig_level);
        rethrow(err);
    end
    
    Optim.set_log_level(orig_level);
    
    fprintf(' PASSED\n');
end


function test_optim_max_iteration()
    fprintf('Test: Optim max iteration...');
    
    orig_max = Optim.get_max_iteration();
    
    try
        Optim.set_max_iteration(500);
        assert(Optim.get_max_iteration() == 500);
        
        Optim.set_max_iteration(1000);
        assert(Optim.get_max_iteration() == 1000);
    catch err
        Optim.set_max_iteration(orig_max);
        rethrow(err);
    end
    
    Optim.set_max_iteration(orig_max);
    
    fprintf(' PASSED\n');
end


function test_optim_tolerances()
    fprintf('Test: Optim tolerances...');
    
    orig_grad = Optim.get_gradient_tolerance();
    orig_obj = Optim.get_objective_rel_tolerance();
    
    try
        Optim.set_gradient_tolerance(1e-6);
        assert(abs(Optim.get_gradient_tolerance() - 1e-6) < 1e-15);
        
        Optim.set_objective_rel_tolerance(1e-8);
        assert(abs(Optim.get_objective_rel_tolerance() - 1e-8) < 1e-15);
    catch err
        Optim.set_gradient_tolerance(orig_grad);
        Optim.set_objective_rel_tolerance(orig_obj);
        rethrow(err);
    end
    
    Optim.set_gradient_tolerance(orig_grad);
    Optim.set_objective_rel_tolerance(orig_obj);
    
    fprintf(' PASSED\n');
end


function test_optim_thread_settings()
    fprintf('Test: Optim thread settings...');
    
    orig_delay = Optim.get_thread_start_delay_ms();
    orig_pool = Optim.get_thread_pool_size();
    
    try
        Optim.set_thread_start_delay_ms(20);
        assert(Optim.get_thread_start_delay_ms() == 20);
        
        Optim.set_thread_pool_size(4);
        assert(Optim.get_thread_pool_size() == 4);
    catch err
        Optim.set_thread_start_delay_ms(orig_delay);
        Optim.set_thread_pool_size(orig_pool);
        rethrow(err);
    end
    
    Optim.set_thread_start_delay_ms(orig_delay);
    Optim.set_thread_pool_size(orig_pool);
    
    fprintf(' PASSED\n');
end


%% Test consistency across classes
function test_all_classes_have_covMat()
    fprintf('Test: All classes have covMat...');
    
    rand('seed', 111);
    n = 10;
    X = rand(n, 1);
    y = X;
    noise = 0.01 * ones(n, 1);
    
    k1 = Kriging(y, X, 'gauss');
    k2 = NoiseKriging(y, noise, X, 'gauss');
    k3 = NuggetKriging(y, X, 'gauss');
    
    X_test = rand(3, 1);
    
    % All should work
    cov1 = k1.covMat(X_test, X_test);
    cov2 = k2.covMat(X_test, X_test);
    cov3 = k3.covMat(X_test, X_test);
    
    assert(all(size(cov1) == [3, 3]));
    assert(all(size(cov2) == [3, 3]));
    assert(all(size(cov3) == [3, 3]));
    
    fprintf(' PASSED\n');
end


function test_all_classes_have_model()
    fprintf('Test: All classes have model...');
    
    rand('seed', 222);
    n = 10;
    X = rand(n, 1);
    y = X;
    noise = 0.01 * ones(n, 1);
    
    k1 = Kriging(y, X, 'gauss');
    k2 = NoiseKriging(y, noise, X, 'gauss');
    k3 = NuggetKriging(y, X, 'gauss');
    
    % All should return structs
    m1 = k1.model();
    m2 = k2.model();
    m3 = k3.model();
    
    assert(isstruct(m1));
    assert(isstruct(m2));
    assert(isstruct(m3));
    
    % Check class-specific fields
    assert(isfield(m2, 'noise'));
    assert(isfield(m3, 'nugget'));
    assert(isfield(m3, 'is_nugget_estim'));
    
    fprintf(' PASSED\n');
end


% Run all tests
test_new_features();
