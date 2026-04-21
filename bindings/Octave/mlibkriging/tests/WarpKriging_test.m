%% Tests for WarpKriging Octave binding.
%% Mirrors bindings/Python/pylibkriging/tests/WarpKriging_test.py
%% and bindings/R/rlibkriging/tests/testthat/test-WarpKriging.R.
1;  % mark this file as a script

f1d = @(x) 1 - 0.5 * (sin(12 * x) ./ (1 + x) + 2 * cos(7 * x) .* x.^5 + 0.7);
branin = @(x1, x2) (x2 - 5.1 / (4 * pi^2) .* x1.^2 + 5.0 / pi .* x1 - 6.0).^2 ...
                   + 10 * (1 - 1.0 / (8 * pi)) .* cos(x1) + 10;

n_failed = 0;

% -----------------------------------------------------------------------
% Test 1: Kumaraswamy warping (1D continuous)
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);

    k = WarpKriging(y, X, {"kumaraswamy"}, "gauss");
    s = k.summary();
    assert(ischar(s) && length(s) > 0);

    [mean_p, stdev_p] = k.predict(X, true, false, false);
    assert(length(mean_p) == 8);
    assert(all(isfinite(mean_p)));

    x_pred = linspace(0.01, 0.99, 50)';
    [mean2, ~] = k.predict(x_pred, true, false, false);
    rmse = sqrt(mean((mean2 - f1d(x_pred)).^2));
    assert(isfinite(rmse));
    fprintf("  Test 1 Kumaraswamy OK (RMSE=%.4f)\n", rmse);
catch err
    fprintf("  Test 1 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 2: Categorical embedding
% -----------------------------------------------------------------------
try
    mu = [1.0; 5.0; 3.0];
    rand("state", 42);
    n = 15;
    levels = mod(0:(n-1), 3)';
    X = levels;
    y = mu(levels + 1) + 0.1 * randn(n, 1);

    k = WarpKriging(y, X, {"categorical(3,2)"}, "gauss");
    X_test = (0:2)';
    [mean_p, ~] = k.predict(X_test, true, false, false);
    assert(length(mean_p) == 3);
    assert(all(isfinite(mean_p)));
    fprintf("  Test 2 Categorical OK\n");
catch err
    fprintf("  Test 2 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 3: Mixed continuous + categorical
% -----------------------------------------------------------------------
try
    offset = [1.0; 2.0; 0.5];
    rand("state", 99);
    n = 30;
    x1 = rand(n, 1);
    x2 = mod(0:(n-1), 3)';
    X = [x1, x2];
    y = sin(2 * pi * x1) .* offset(x2 + 1);

    k = WarpKriging(y, X, {"kumaraswamy", "categorical(3,2)"}, "matern5_2");
    xc = linspace(0.01, 0.99, 20)';
    for cat_idx = 0:2
        X_test = [xc, cat_idx * ones(20, 1)];
        [mean_p, ~] = k.predict(X_test, true, false, false);
        assert(length(mean_p) == 20);
        assert(all(isfinite(mean_p)));
    end
    fprintf("  Test 3 Mixed OK\n");
catch err
    fprintf("  Test 3 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 4: Ordinal warping
% -----------------------------------------------------------------------
try
    L = 5;
    rand("state", 7);
    n = 20;
    levels = mod(0:(n-1), L)';
    X = levels;
    y = levels.^2 + 0.1 * randn(n, 1);

    k = WarpKriging(y, X, {"ordinal(5)"}, "gauss");
    X_test = (0:(L-1))';
    [mean_p, ~] = k.predict(X_test, true, false, false);
    assert(length(mean_p) == L);
    assert(all(isfinite(mean_p)));
    fprintf("  Test 4 Ordinal OK\n");
catch err
    fprintf("  Test 4 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 5: NeuralMono
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 10)';
    y = f1d(X);
    k = WarpKriging(y, X, {"neural_mono(8)"}, "gauss");
    [mean_p, ~] = k.predict(X, true, false, false);
    assert(all(isfinite(mean_p)));
    fprintf("  Test 5 NeuralMono OK\n");
catch err
    fprintf("  Test 5 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 6: MLP warping 1D
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 10)';
    y = f1d(X);
    k = WarpKriging(y, X, {"mlp(16:8,3,selu)"}, "gauss");
    x_pred = linspace(0.01, 0.99, 50)';
    [mean_p, ~] = k.predict(x_pred, true, false, false);
    rmse = sqrt(mean((mean_p - f1d(x_pred)).^2));
    assert(isfinite(rmse));
    fprintf("  Test 6 MLP 1D OK (RMSE=%.4f)\n", rmse);
catch err
    fprintf("  Test 6 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 7: MLP + categorical mixed
% -----------------------------------------------------------------------
try
    offset = [1.0; 2.0; 0.5];
    rand("state", 99);
    n = 30;
    x1 = rand(n, 1);
    x2 = mod(0:(n-1), 3)';
    X = [x1, x2];
    y = sin(2 * pi * x1) .* offset(x2 + 1);

    k = WarpKriging(y, X, {"mlp(16:8,2,tanh)", "categorical(3,2)"}, "matern5_2");
    xc = linspace(0.01, 0.99, 20)';
    for cat_idx = 0:2
        X_test = [xc, cat_idx * ones(20, 1)];
        [mean_p, ~] = k.predict(X_test, true, false, false);
        assert(all(isfinite(mean_p)));
    end
    fprintf("  Test 7 MLP+Cat OK\n");
catch err
    fprintf("  Test 7 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 8: simulate mixed
% -----------------------------------------------------------------------
try
    rand("state", 42);
    n = 20;
    x1 = rand(n, 1);
    x2 = mod(0:(n-1), 2)';
    X = [x1, x2];
    mult = [1.0; 3.0];
    y = sin(2 * pi * x1) .* mult(x2 + 1);

    k = WarpKriging(y, X, {"affine", "categorical(2,2)"}, "gauss");
    X_sim = [linspace(0.1, 0.9, 10)', zeros(10, 1)];
    sims = k.simulate(int32(30), int32(123), X_sim);
    assert(size(sims, 1) == 10);
    assert(size(sims, 2) == 30);
    assert(all(isfinite(sims(:))));
    fprintf("  Test 8 Simulate mixed OK\n");
catch err
    fprintf("  Test 8 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 9: update
% -----------------------------------------------------------------------
try
    X0 = [0.1, 0.0; 0.5, 1.0; 0.9, 0.0];
    y0 = [1.0; 3.0; 0.5];
    k = WarpKriging(y0, X0, {"none", "categorical(2,1)"}, "gauss");
    X_new = [0.3, 1.0; 0.7, 0.0];
    k.update([2.0; 1.5], X_new);
    [mean_p, ~] = k.predict(X0, true, false, false);
    assert(all(isfinite(mean_p)));
    fprintf("  Test 9 Update OK\n");
catch err
    fprintf("  Test 9 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 10: Branin 2D with per-variable MLP
% -----------------------------------------------------------------------
try
    rand("state", 77);
    n = 25;
    X = rand(n, 2);
    y = branin(X(:, 1) * 15 - 5, X(:, 2) * 15);

    k = WarpKriging(y, X, {"mlp(16:8,2,selu)", "mlp(16:8,2,selu)"}, ...
                    "matern5_2", "constant", true);
    rand("state", 88);
    X_test = rand(15, 2);
    [mean_p, stdev_p] = k.predict(X_test, true, false, false);
    assert(length(mean_p) == 15);
    assert(all(isfinite(stdev_p)));

    sims = k.simulate(int32(20), int32(42), X_test);
    assert(size(sims, 1) == 15);
    assert(size(sims, 2) == 20);
    fprintf("  Test 10 Branin MLP OK\n");
catch err
    fprintf("  Test 10 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 11: None vs Kumaraswamy vs MLP comparison
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 12)';
    y = f1d(X);
    xp = linspace(0.01, 0.99, 50)';
    ytrue = f1d(xp);

    k_none = WarpKriging(y, X, {"none"}, "gauss");
    k_kuma = WarpKriging(y, X, {"kumaraswamy"}, "gauss");
    k_mlp  = WarpKriging(y, X, {"mlp(16:8,2,selu)"}, "gauss");

    [m_none, ~] = k_none.predict(xp, true, false, false);
    [m_kuma, ~] = k_kuma.predict(xp, true, false, false);
    [m_mlp, ~]  = k_mlp.predict(xp, true, false, false);
    for mm = {m_none, m_kuma, m_mlp}
        rmse = sqrt(mean((mm{1} - ytrue).^2));
        assert(isfinite(rmse));
    end
    assert(isfinite(k_none.logLikelihood()));
    assert(isfinite(k_kuma.logLikelihood()));
    assert(isfinite(k_mlp.logLikelihood()));
    fprintf("  Test 11 Comparison OK\n");
catch err
    fprintf("  Test 11 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 12: logLikelihoodFun + FD gradient
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);
    k = WarpKriging(y, X, {"affine"}, "gauss", "constant", false, "none");

    % Evaluate at a fixed benign theta (not the optimum) so FD is well-conditioned.
    th = 0.3 * ones(size(k.theta()));
    [ll, grad] = k.logLikelihoodFun(th, true, false);
    assert(isfinite(ll));
    assert(all(isfinite(grad)));

    h = 1e-5;
    grad_num = zeros(size(th));
    for i = 1:length(th)
        tp = th; tm = th;
        tp(i) = tp(i) + h;
        tm(i) = tm(i) - h;
        [lp, ~] = k.logLikelihoodFun(tp, false, false);
        [lm, ~] = k.logLikelihoodFun(tm, false, false);
        grad_num(i) = (lp - lm) / (2 * h);
    end
    rel = norm(grad(:) - grad_num(:)) / (norm(grad_num(:)) + 1e-12);
    assert(rel < 1e-4, sprintf("grad FD rel error %.3g", rel));
    fprintf("  Test 12 logLikelihoodFun FD OK (rel=%.3g)\n", rel);
catch err
    fprintf("  Test 12 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 13: Accessors / summary
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 6)';
    y = f1d(X);
    k = WarpKriging(y, X, {"boxcox"}, "matern3_2");

    s = k.summary();
    assert(ischar(s) && length(s) > 0);
    th = k.theta();
    assert(length(th) > 0 && all(th > 0));
    assert(k.sigma2() > 0);
    assert(strcmp(k.kernel(), "matern3_2"));
    assert(isfinite(k.logLikelihood()));
    ws = k.warping();
    assert(iscell(ws) && length(ws) == 1 && strcmp(ws{1}, "boxcox"));
    assert(k.is_fitted());
    fprintf("  Test 13 Accessors OK\n");
catch err
    fprintf("  Test 13 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 14: predict with derivatives
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);
    k = WarpKriging(y, X, {"affine"}, "gauss");
    X_new = linspace(0.1, 0.9, 5)';
    [mean_p, stdev_p, cov_p, mean_deriv, stdev_deriv] = k.predict(X_new, true, false, true);
    assert(size(mean_deriv, 1) == 5);
    assert(size(mean_deriv, 2) == 1);
    assert(size(stdev_deriv, 1) == 5);
    assert(size(stdev_deriv, 2) == 1);
    assert(all(isfinite(mean_deriv(:))));
    assert(all(isfinite(stdev_deriv(:))));
    fprintf("  Test 14 Predict-deriv OK\n");
catch err
    fprintf("  Test 14 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 15: Getters (X, y, feature_dim)
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);
    k = WarpKriging(y, X, {"affine"}, "gauss");
    Xr = k.X();
    assert(size(Xr, 1) == 8);
    assert(size(Xr, 2) == 1);
    yr = k.y();
    assert(length(yr) == 8);
    assert(k.feature_dim() > 0);
    fprintf("  Test 15 Getters OK\n");
catch err
    fprintf("  Test 15 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 16: Copy
% -----------------------------------------------------------------------
try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);
    k = WarpKriging(y, X, {"affine"}, "gauss");
    k2 = k.copy();
    assert(k2.is_fitted());
    assert(strcmp(k2.kernel(), k.kernel()));
    X_test = linspace(0.1, 0.9, 5)';
    [m1, ~] = k.predict(X_test, true, false, false);
    [m2, ~] = k2.predict(X_test, true, false, false);
    assert(max(abs(m1 - m2)) < 1e-10);
    fprintf("  Test 16 Copy OK\n");
catch err
    fprintf("  Test 16 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

if n_failed > 0
    error("WarpKriging tests: %d failed", n_failed);
end
fprintf("WarpKriging tests: all 16 passed\n");
