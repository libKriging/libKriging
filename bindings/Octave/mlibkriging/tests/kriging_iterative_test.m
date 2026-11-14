1;  % mark this file as a script

% Method "iterative(...)": matrix-free fit / predict (CG, SLQ log-determinant),
% objective "LL" and noise model None only; Gpu and LinearAlgebra settings.

n = 40;
X = [mod(0.3819660 * (1:n)', 1), mod(0.7548777 * (1:n)', 1)];  % Kronecker design
y = sin(3 * X(:, 1)) + X(:, 2);
Xn = [0.15 0.3; 0.45 0.6; 0.75 0.2];

% predict method "iterative" on an exact fit: same predictor, solved by CG
ke = Kriging(y, X, "matern5_2");
[m_ex, s_ex] = ke.predict(Xn, true, false, false);
[m_cg, s_cg] = ke.predict(Xn, true, false, false, "iterative(30,0,20,10,1e-12)");
assert(max(abs(m_ex - m_cg)) < 1e-5 * std(y));
assert(max(abs(s_ex - s_cg)) < 1e-4 * std(y));

% iterative fit: no exact factorization, explicit method at every step
k = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], "iterative(8)");
assert(strcmp(k.method(), "iterative(8)"));
[m, s] = k.predict(Xn, true, false, false, "iterative");
assert(all(isfinite(m)) && all(isfinite(s)));
assert(isfinite(k.logLikelihood("iterative")));
failed = false;
try
  k.predict(Xn, true, false, false);  % exact: no factorization
catch
  failed = true;
end
assert(failed);

% restrictions: LOO / LMP, noise models
failed = false;
try
  Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LOO", [], "none", [], "iterative");
catch
  failed = true;
end
assert(failed);

% WarpKriging / MLPKriging: predict(..., method)
wk = WarpKriging(y, X, {"kumaraswamy", "kumaraswamy"}, "matern5_2");
[mw_ex, sw_ex] = wk.predict(Xn, true, false, false);
[mw_cg, sw_cg] = wk.predict(Xn, true, false, false, "iterative(30,0,20,10,1e-12)");
assert(max(abs(mw_ex - mw_cg)) < 1e-4 * std(y));

% settings
assert(ischar(Gpu.compiled_backends()));
assert(any(strcmp(Gpu.active_backend(), {"none", "cuda", "hip", "sycl", "metal"})));
e = Gpu.get_enabled();
Gpu.set_enabled(false);
assert(! Gpu.get_enabled());
assert(strcmp(Gpu.active_backend(), "none"));
Gpu.set_enabled(e);
mb = Gpu.get_dense_max_mb();
Gpu.set_dense_max_mb(0);
assert(Gpu.get_dense_max_mb() == 0);
Gpu.set_dense_max_mb(mb);
hmb = LinearAlgebra.get_iterative_dense_max_mb();
LinearAlgebra.set_iterative_dense_max_mb(0);
assert(LinearAlgebra.get_iterative_dense_max_mb() == 0);
LinearAlgebra.set_iterative_dense_max_mb(hmb);
LinearAlgebra.set_cg_warning(true);
LinearAlgebra.set_cg_auto_precond(true);
