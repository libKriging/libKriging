%% Tests for MultiOutputKriging Octave binding.
%% Mirrors bindings/Python/pylibkriging/tests/MultiOutputKriging_test.py
%% and bindings/R/rlibkriging/tests/testthat/test-MultiOutputKriging.R.
1;  % mark this file as a script

% Damped oscillation sampled at q = 30 time steps, two inputs
t_out = linspace(0.5, 10, 30);
code = @(X) exp(-(0.1 + 0.5 * X(:, 2)) * t_out) .* cos(2 * pi * (0.5 + 1.5 * X(:, 1)) * t_out / 5);

rand("seed", 1);
X = rand(40, 2);
Y = code(X);
Xt = rand(10, 2);
Yt = code(Xt);

n_failed = 0;

% -----------------------------------------------------------------------
% Test 1: pca, shapes and accuracy
% -----------------------------------------------------------------------
try
    k = MultiOutputKriging(Y, X, "matern5_2", "pca(0.999)");
    assert(k.nb_outputs() == 30);
    K = k.nb_components();
    assert(isequal(size(k.pca_basis()), [30, K]));
    assert(strcmp(k.output_model(), "pca(0.999)"));
    [m, s, c, d] = k.predict(Xt, true, true, true);
    assert(isequal(size(m), [10, 30]));
    assert(isequal(size(s), [10, 30]));
    assert(isequal(size(c), [300, 300]));
    assert(isequal(size(d), [10, 2, 30]));
    assert(max(abs(sqrt(diag(c)) - s(:))) < 1e-8);
    rmse = sqrt(mean((m(:) - Yt(:)).^2));
    assert(rmse < 0.5 * std(Y(:)));
    km = k.component(1);
    assert(size(km.X(), 1) == 40);
    fprintf("  Test 1 pca OK (K=%d, RMSE=%.2e)\n", K, rmse);
catch err
    fprintf("  Test 1 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 2: shared with one output is Kriging
% -----------------------------------------------------------------------
try
    y = Y(:, 5);
    km = Kriging(y, X, "matern5_2");
    k = MultiOutputKriging(y, X, "matern5_2", "shared");
    assert(max(abs(k.theta() - km.theta())) < 1e-6 * max(km.theta()));
    assert(abs(k.logLikelihood() - km.logLikelihood()) < 1e-8 * abs(km.logLikelihood()));
    [m1, s1] = km.predict(Xt, true, false, false);
    [m2, s2] = k.predict(Xt, true, false, false);
    assert(max(abs(m1 - m2)) < 1e-6 * std(y));
    assert(max(abs(s1 - s2)) < 1e-6 * std(y));
    fprintf("  Test 2 shared q=1 OK\n");
catch err
    fprintf("  Test 2 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 3: shared, gradient of logLikelihoodFun
% -----------------------------------------------------------------------
try
    k = MultiOutputKriging(Y, X, "matern5_2", "shared");
    th = k.theta() * 1.3;
    [ll, g] = k.logLikelihoodFun(th, true);
    h = 1e-4;  % smaller steps are dominated by the rounding noise of the LL
    for i = 1:2
        e = zeros(2, 1); e(i) = h;
        fd = (k.logLikelihoodFun(th + e) - k.logLikelihoodFun(th - e)) / (2 * h);
        assert(abs(g(i) - fd) < 1e-4 * max(1, abs(fd)));
    end
    fprintf("  Test 3 shared gradient OK\n");
catch err
    fprintf("  Test 3 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 4: separable, covariance factors and joint simulations
% -----------------------------------------------------------------------
try
    Y3 = Y(:, [3, 10, 20]);
    k = MultiOutputKriging(Y3, X, "matern5_2", "separable");
    S = k.output_cov();
    assert(isequal(size(S), [3, 3]));
    [Cx, Sig] = k.predictCovFactors(Xt);
    [m, s, c] = k.predict(Xt, true, true, false);
    assert(max(max(abs(kron(Sig, Cx) - c))) < 1e-8 * max(abs(c(:))));
    sims = k.simulate(int32(2000), int32(3), Xt);
    assert(isequal(size(sims), [10, 3, 2000]));
    emp = mean(sims, 3);
    assert(max(abs(emp(:) - m(:)) ./ max(s(:), 1e-12)) < 15 / sqrt(2000));
    fprintf("  Test 4 separable OK\n");
catch err
    fprintf("  Test 4 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 5: update and update_simulate
% -----------------------------------------------------------------------
try
    models = {"pca(0.999)", "shared"};
    for i = 1:numel(models)
        k = MultiOutputKriging(Y, X, "matern5_2", models{i});
        Xu = Xt(1:3, :); Yu = Yt(1:3, :); xs = Xt(4:10, :);
        s0 = k.simulate(int32(1000), int32(5), xs, true);
        s1 = k.update_simulate(Yu, Xu);
        assert(isequal(size(s1), size(s0)));
        k.update(Yu, Xu, false);
        assert(size(k.X(), 1) == 43);
        [m, s] = k.predict(xs, true, false, false);
        emp = mean(s1, 3);
        assert(max(abs(emp(:) - m(:))) < 5 * max(s(:)) / sqrt(1000) + 1e-8);
    end
    fprintf("  Test 5 update / update_simulate OK\n");
catch err
    fprintf("  Test 5 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 6: leave-one-out
% -----------------------------------------------------------------------
try
    k = MultiOutputKriging(Y, X, "matern5_2", "shared");
    [lm, ls] = k.leaveOneOutMat();
    assert(isequal(size(lm), size(Y)));
    assert(abs(k.leaveOneOut() - mean((Y(:) - lm(:)).^2)) < 1e-10);
    fprintf("  Test 6 leaveOneOut OK\n");
catch err
    fprintf("  Test 6 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

% -----------------------------------------------------------------------
% Test 7: unfitted constructor then fit, fixed theta
% -----------------------------------------------------------------------
try
    k = MultiOutputKriging("matern5_2", "shared");
    k.fit(Y, X, "constant", false, "none", "LL", Params("theta", [0.3, 0.4]));
    assert(max(abs(k.theta() - [0.3; 0.4])) < 1e-12);
    fprintf("  Test 7 fit with fixed theta OK\n");
catch err
    fprintf("  Test 7 FAILED: %s\n", err.message);
    n_failed = n_failed + 1;
end

if n_failed > 0
    error("MultiOutputKriging tests: %d failed", n_failed);
end
fprintf("MultiOutputKriging tests: all 7 passed\n");
