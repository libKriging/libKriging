% Test suite for gradient-enhanced kriging: Kriging(..., dydX) / fit(..., dydX)
% Run this test with: octave test_gradient_kriging.m

function test_gradient_kriging()
    fprintf('=== Testing gradient-enhanced kriging (dydX) ===\n\n');

    test_dydX_interpolates_values_and_gradients();
    test_dydX_empty_is_value_only_fit();
    test_dydX_beats_value_only_out_of_sample();
    test_fit_dydX_clears_on_later_fit_without_dydX();
    test_dydX_rejects_non_differentiable_kernel();
    test_dydX_rejects_wrong_shape();

    fprintf('\n=== All Tests Passed! ===\n');
end


function [X, y, dy] = make_design(n, seed)
    rand('seed', seed);
    X = rand(n, 2);
    y = sin(3*X(:,1)) + cos(5*X(:,2));
    dy = [3*cos(3*X(:,1)), -5*sin(5*X(:,2))];
end


function test_dydX_interpolates_values_and_gradients()
    fprintf('Test: dydX interpolates values and gradients...');

    [X, y, dy] = make_design(20, 1);
    k = Kriging(y, X, 'gauss', 'constant', false, 'BFGS', 'LL', [], 'none', [], dy);

    d = k.dy();
    assert(all(size(d) == [20, 2]), 'Expected dy() size [20, 2]');

    [mean_, stdev_, cov_, mean_deriv_, stdev_deriv_] = k.predict(X, true, false, true);
    assert(max(abs(mean_ - y)) < 1e-4, 'mean should interpolate y');
    assert(max(max(abs(mean_deriv_ - dy))) < 1e-3, 'mean_deriv should interpolate dy');

    fprintf(' PASSED\n');
end


function test_dydX_empty_is_value_only_fit()
    fprintf('Test: dydX=[] is a value-only fit...');

    [X, y, dy] = make_design(20, 1);
    k = Kriging(y, X, 'gauss');
    assert(isempty(k.dy()), 'dy() should be empty without dydX');

    fprintf(' PASSED\n');
end


function test_dydX_beats_value_only_out_of_sample()
    fprintf('Test: dydX beats a value-only fit out of sample...');

    [X, y, dy] = make_design(15, 72);
    [Xt, yt, ~] = make_design(200, 720);

    k_plain = Kriging(y, X, 'gauss');
    k_grad = Kriging(y, X, 'gauss', 'constant', false, 'BFGS', 'LL', [], 'none', [], dy);

    mean_plain = k_plain.predict(Xt, false, false, false);
    mean_grad = k_grad.predict(Xt, false, false, false);

    rmse_plain = sqrt(mean((mean_plain - yt).^2));
    rmse_grad = sqrt(mean((mean_grad - yt).^2));
    assert(rmse_grad < rmse_plain, 'gradient-enhanced fit should beat value-only fit');

    fprintf(' PASSED\n');
end


function test_fit_dydX_clears_on_later_fit_without_dydX()
    fprintf('Test: fit(...) without dydX clears previous gradient observations...');

    [X, y, dy] = make_design(20, 1);
    k = Kriging(y, X, 'gauss', 'constant', false, 'BFGS', 'LL', [], 'none', [], dy);
    assert(~isempty(k.dy()), 'dy() should be non-empty after fit with dydX');

    k.fit(y, X);
    assert(isempty(k.dy()), 'dy() should be cleared after fit without dydX');

    fprintf(' PASSED\n');
end


function test_dydX_rejects_non_differentiable_kernel()
    fprintf('Test: dydX rejects a non-differentiable kernel...');

    [X, y, dy] = make_design(10, 1);
    threw = false;
    try
        Kriging(y, X, 'exp', 'constant', false, 'BFGS', 'LL', [], 'none', [], dy);
    catch
        threw = true;
    end
    assert(threw, 'expected an error for dydX with kernel=exp');

    fprintf(' PASSED\n');
end


function test_dydX_rejects_wrong_shape()
    fprintf('Test: dydX rejects a wrongly shaped matrix...');

    [X, y, dy] = make_design(10, 1);
    threw = false;
    try
        Kriging(y, X, 'gauss', 'constant', false, 'BFGS', 'LL', [], 'none', [], dy(:, 1));
    catch
        threw = true;
    end
    assert(threw, 'expected an error for a wrongly shaped dydX');

    fprintf(' PASSED\n');
end
