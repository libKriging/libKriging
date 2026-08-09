% predictCG: matrix-free conjugate-gradient prediction, mirroring the
% C++/R/Julia/Python predictCG test suites' test function and fixed-theta
% setup (see docs/math/PredictCG.md for why theta is fixed: a free BFGS
% fit is known to drift theta toward a near-singular correlation matrix on
% this noise-free deterministic function, unrelated to predictCG itself).

rand("seed", 123);
n = 60;
X = rand(n, 2);
y = sin(3 * X(:,1)) + cos(5 * X(:,2)) + X(:,1) .* X(:,2);

% NB: the constructor's `parameters` slot expects a Params object (built
% via key-value pairs), not a native Octave struct() -- passing a struct
% is silently ignored, which used to make this whole test file fail with
% "Theta should be given" before ever reaching predictCG.
p = Params("theta", [0.3 0.3], "sigma2", 1.0);
k = Kriging(y, X, "matern5_2", "constant", false, "none", "LL", p);

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
% NB: the mex layer's `int` args (max_iter, precond_rank, n_max, ...) are
% read via a converter requiring an actual int32-typed mxArray -- plain
% Octave numeric literals are double and must be wrapped in int32(...).
threw = false;
try
    k.predictCG(Xt, false, int32(-1));
catch err
    threw = true;
end
assert(threw);

% Nystrom-preconditioned CG still matches exact predict
[m_pc, s_pc] = k.predictCG(Xt, true, int32(0), 1e-8, true, int32(20));
assert(max(abs(m_ex - m_pc)) < 0.05 * sdy);
assert(max(abs(s_ex - s_pc)) < 0.05 * sdy);

% rejects a negative precond_rank
threw = false;
try
    k.predictCG(Xt, false, int32(0), 1e-8, true, int32(-1));
catch err
    threw = true;
end
assert(threw);

% subsetOfData returns n_max sorted 1-based indices
idx = Kriging.subsetOfData(X, int32(20));
assert(numel(idx) == 20);
assert(isequal(idx, sort(idx)));
assert(min(idx) >= 1);
assert(max(idx) <= n);

% subsetOfData is a no-op when n_max covers all rows
idx_all = Kriging.subsetOfData(X, int32(n));
assert(isequal(idx_all, (1:n)'));

disp("KrigingPredictCG_test: all assertions passed");
