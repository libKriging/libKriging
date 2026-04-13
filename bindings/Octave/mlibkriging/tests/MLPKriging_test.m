%% Tests for MLPKriging Octave binding.
%% Mirrors bindings/Python/pylibkriging/tests/MLPKriging_test.py
%% and bindings/R/rlibkriging/tests/testthat/test-MLPKriging.R.
1;  % mark this file as a script

f1d = @(x) 1 - 0.5 * (sin(12 * x) ./ (1 + x) + 2 * cos(7 * x) .* x.^5 + 0.7);
branin = @(x1, x2) (x2 - 5.1 / (4 * pi^2) .* x1.^2 + 5.0 / pi .* x1 - 6.0).^2 ...
                   + 10 * (1 - 1.0 / (8 * pi)) .* cos(x1) + 10;

n_failed = 0;

% -----------------------------------------------------------------------
% Test 1: 1D fit / predict / accessors (gauss)
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 10)';
    y = f1d(X);

    k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss", "constant", true);
    s = k.summary();
    assert(ischar(s) && length(s) > 0);

    [mean_p, stdev_p] = k.predict(X, true, false, false);
    assert(length(mean_p) == 10);
    assert(all(isfinite(mean_p)));

    x_pred = linspace(0.01, 0.99, 50)';
    [mean2, ~] = k.predict(x_pred, true, false, false);
    rmse = sqrt(mean((mean2 - f1d(x_pred)).^2));
    assert(isfinite(rmse));

    assert(strcmp(k.kernel(), "gauss"));
    assert(strcmp(k.activation(), "selu"));
    hd = k.hidden_dims();
    assert(length(hd) == 2 && hd(1) == 16 && hd(2) == 8);
    assert(k.feature_dim() == 2);
    assert(k.sigma2() > 0);
    assert(isfinite(k.logLikelihood()));
    fprintf("  Test 1 MLP 1D OK (RMSE=%.4f)\n", rmse);
catch err
    fprintf("  Test 1 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 2: Branin 2D matern5_2 + simulate + update
% -----------------------------------------------------------------------
try
    rand("state", 77);
    n = 20;
    X = rand(n, 2);
    y = branin(X(:, 1) * 15 - 5, X(:, 2) * 15);

    k = MLPKriging(y, X, [32, 16], 3, "selu", "matern5_2", "constant", true);

    X_test = rand(10, 2);
    [mean_p, stdev_p] = k.predict(X_test, true, false, false);
    assert(length(mean_p) == 10);
    assert(all(isfinite(mean_p)));
    assert(all(isfinite(stdev_p)));

    sims = k.simulate(int32(20), int32(42), X_test);
    assert(size(sims, 1) == 10);
    assert(size(sims, 2) == 20);
    assert(all(isfinite(sims(:))));

    rand("state", 1234);
    X_new = rand(3, 2);
    y_new = branin(X_new(:, 1) * 15 - 5, X_new(:, 2) * 15);
    k.update(y_new, X_new);

    [mean2, ~] = k.predict(X_test, true, false, false);
    assert(length(mean2) == 10);
    fprintf("  Test 2 MLP Branin 2D OK\n");
catch err
    fprintf("  Test 2 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 3: predict with deriv
% -----------------------------------------------------------------------
try
    rand("state", 88);
    n = 20;
    X = rand(n, 2);
    y = sin(3 * X(:, 1)) + cos(4 * X(:, 2)) + 0.5 * X(:, 1) .* X(:, 2);

    k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss");
    X_new = rand(5, 2);
    [mean_p, stdev_p, cov_p, mean_deriv, stdev_deriv] = k.predict(X_new, true, false, true);
    assert(size(mean_deriv, 1) == 5);
    assert(size(mean_deriv, 2) == 2);
    assert(size(stdev_deriv, 1) == 5);
    assert(size(stdev_deriv, 2) == 2);
    assert(all(isfinite(mean_deriv(:))));
    assert(all(isfinite(stdev_deriv(:))));
    fprintf("  Test 3 MLP predict-deriv OK\n");
catch err
    fprintf("  Test 3 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 4: logLikelihoodFun + gradient
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 10)';
    y = f1d(X);
    k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss");

    th = k.theta();
    ll_val = k.logLikelihood();
    assert(isfinite(ll_val));

    [ll, grad] = k.logLikelihoodFun(th, true, false);
    assert(isfinite(ll));
    assert(all(isfinite(grad)));

    [ll2, ~] = k.logLikelihoodFun(th, false, false);
    assert(isfinite(ll2));
    fprintf("  Test 4 MLP logLikelihoodFun OK\n");
catch err
    fprintf("  Test 4 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 5: is_fitted
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);
    k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss");
    assert(k.is_fitted());
    fprintf("  Test 5 MLP is_fitted OK\n");
catch err
    fprintf("  Test 5 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 6: Getters (X, y)
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);
    k = MLPKriging(y, X, [16, 8], 2, "selu", "gauss");
    Xr = k.X();
    assert(size(Xr, 1) == 8);
    yr = k.y();
    assert(length(yr) == 8);
    assert(all(k.theta() > 0));
    assert(k.sigma2() > 0);
    fprintf("  Test 6 MLP Getters OK\n");
catch err
    fprintf("  Test 6 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

if n_failed > 0
    error("MLPKriging tests: %d failed", n_failed);
end
fprintf("MLPKriging tests: all 6 passed\n");
