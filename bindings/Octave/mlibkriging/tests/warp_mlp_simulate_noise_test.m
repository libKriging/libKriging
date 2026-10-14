1;  % mark this file as a script

% WarpKriging / MLPKriging simulate(nsim, seed, X, will_update, with_noise):
% with_noise holds observation-noise VARIANCES (as Kriging); WarpKriging
% update_simulate with 3 and 4 arguments (with a categorical / cell X too)

f1d = @(x) 1 - 0.5 * (sin(12 * x) ./ (1 + x) + 2 * cos(7 * x) .* x.^5 + 0.7);
X = linspace(0.01, 0.99, 10)';
y = f1d(X);
Xn = [0.15; 0.45; 0.85];
var_n = [0.02; 0.1; 0.3];
nsim = int32(5000);

function check_with_noise(k, Xn, var_n, nsim)
  % default call unchanged (no noise); [] means no noise too. Same seed:
  % identical latent draws, the difference is the added noise only
  s0 = k.simulate(nsim, int32(7), Xn);
  assert(size(s0), [3, double(nsim)]);
  assert(k.simulate(nsim, int32(7), Xn, false), s0);
  assert(k.simulate(nsim, int32(7), Xn, false, []), s0);
  s1 = k.simulate(nsim, int32(7), Xn, false, var_n);
  assert(abs(var(s1 - s0, 0, 2) ./ var_n - 1) < 0.15);
  % a scalar is a variance common to every point
  s2 = k.simulate(nsim, int32(8), Xn, false, 0.2);
  s20 = k.simulate(nsim, int32(8), Xn, false);
  assert(abs(var(s2 - s20, 0, 2) / 0.2 - 1) < 0.15);
  % invalid variances raise
  failed = false;
  try
    s_ = k.simulate(int32(10), int32(7), Xn, false, -var_n);
  catch err
    failed = true;
    assert(~isempty(strfind(err.message, "non-negative noise variances")), err.message);
  end
  assert(failed, "negative with_noise should raise");
  failed = false;
  try
    s_ = k.simulate(int32(10), int32(7), Xn, false, [0.1; NaN; 0.1]);
  catch
    failed = true;
  end
  assert(failed, "NaN with_noise should raise");
end

kw = WarpKriging(y, X, {"kumaraswamy"}, "gauss", "constant", false, "Adam", "LL", struct("max_iter_adam", "100"));
check_with_noise(kw, Xn, var_n, nsim);
km = MLPKriging(y, X, [8], 2, "tanh", "gauss", "constant", false, "BFGS+Adam", "LL", struct("max_iter_adam", "100"));
check_with_noise(km, Xn, var_n, nsim);

% will_update=true: update_simulate adds fresh N(0, with_noise) draws again
Xu = [0.3; 0.7];
yu = f1d(Xu);
s_ = kw.simulate(nsim, int32(10), Xn, true, var_n);
u1 = kw.update_simulate(yu, Xu);
s_ = kw.simulate(nsim, int32(10), Xn, true);
u0 = kw.update_simulate(yu, Xu);
assert(size(u1), [3, double(nsim)]);
assert(abs(var(u1 - u0, 0, 2) ./ var_n - 1) < 0.15);
s_ = km.simulate(nsim, int32(10), Xn, true, var_n);
assert(size(km.update_simulate(yu, Xu)), [3, double(nsim)]);

% WarpKriging fit with noise: update_simulate(y_u, X_u, noise_u) (4 args);
% the 3-argument form raises (noise_u required)
noise = 1e-3 * ones(10, 1);
kn = WarpKriging(y, X, {"affine"}, "gauss", "constant", false, "BFGS", "LL", [], noise);
s_ = kn.simulate(int32(50), int32(3), Xn, true);
assert(size(kn.update_simulate(yu, Xu, 1e-3 * ones(2, 1))), [3, 50]);
failed = false;
try
  kn.update_simulate(yu, Xu);
catch
  failed = true;
end
assert(failed, "update_simulate without noise_u should raise on a noisy WarpKriging");

% categorical (cell) X_u: 3 and 4 argument forms are both forwarded
Xc = {[0.1; 0.5; 0.9; 0.3; 0.7; 0.2], {"a"; "b"; "a"; "b"; "a"; "b"}};
yc = [1.0; 3.0; 0.5; 2.5; 0.8; 2.9];
kc = WarpKriging(yc, Xc, {"none", "categorical(2,1)"}, "gauss");
Xnc = {[0.4; 0.6], {"a"; "b"}};
Xuc = {[0.25], {"b"}};
s_ = kc.simulate(int32(20), int32(1), Xnc, true);
assert(size(kc.update_simulate(2.0, Xuc)), [2, 20]);
kcn = WarpKriging(yc, Xc, {"none", "categorical(2,1)"}, "gauss", "constant", false, "BFGS", "LL", [], 1e-3 * ones(6, 1));
s_ = kcn.simulate(int32(20), int32(1), Xnc, true, 0.1);
assert(size(kcn.update_simulate(2.0, Xuc, 1e-3)), [2, 20]);
