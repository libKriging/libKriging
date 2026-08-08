% predictCG: matrix-free conjugate-gradient prediction, mirroring the
% C++/R/Julia/Python predictCG test suites' test function and fixed-theta
% setup (see docs/math/PredictCG.md for why theta is fixed: a free BFGS
% fit is known to drift theta toward a near-singular correlation matrix on
% this noise-free deterministic function, unrelated to predictCG itself).

rand("seed", 123);
n = 60;
X = rand(n, 2);
y = sin(3 * X(:,1)) + cos(5 * X(:,2)) + X(:,1) .* X(:,2);

k = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", ...
            struct("theta", [0.3 0.3], "sigma2", 1.0));

Xt = rand(20, 2);

% mean/stdev match exact predict at a moderate theta
[m_ex, s_ex] = k.predict(Xt, true, false, false);
[m_cg, s_cg] = k.predictCG(Xt, true);
sdy = std(y);
assert(max(abs(m_ex - m_cg)) < 0.05 * sdy);
assert(max(abs(s_ex - s_cg)) < 0.05 * sdy);

% defaults to mean only
m_only = k.predictCG(Xt);
assert(length(m_only) == size(Xt, 1));

% interpolates the training data
[m_train, s_train] = k.predictCG(X, true);
assert(max(abs(m_train - y)) < 0.05 * sdy);
assert(max(s_train) < 0.05 * sdy);

% rejects a negative max_iter
threw = false;
try
    k.predictCG(Xt, false, -1);
catch err
    threw = true;
end
assert(threw);

disp("KrigingPredictCG_test: all assertions passed");
