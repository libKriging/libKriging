1;  % mark this file as a script

% logLikelihoodFun argument mapping, simulate with_noise variances, update
% without noise on a heterogeneous model, approximate LOO / LMP on noise models

n = 30;
X = [mod(0.3819660 * (1:n)', 1), mod(0.7548777 * (1:n)', 1)];
y = sin(3 * X(:, 1)) + cos(5 * X(:, 2));
Xn = [0.2 0.3; 0.5 0.5; 0.8 0.1];

% logLikelihoodFun(theta, return_grad, return_hess, bench): Kriging has no
% hessian (empty, as in Python / R), bench only prints timings
k = Kriging(y, X, "matern5_2");
th = [0.4; 0.6];
[ll, g] = k.logLikelihoodFun(th, true);
[ll2, g2, h] = k.logLikelihoodFun(th, true, true);
assert(ll2, ll, -1e-14);
assert(g2, g, -1e-14);
assert(isempty(h));
[ll3, g3] = k.logLikelihoodFun(th, true, false, true);  % bench=true
assert(ll3, ll, -1e-14);
assert(g3, g, -1e-14);
ll4 = k.logLikelihoodFun(th, false, false, false);
assert(ll4, ll, -1e-14);
failed = false;
try
  [ll5, g5] = k.logLikelihoodFun(th, true, true);  % return_hess without its output
catch
  failed = true;
end
assert(failed, "return_hess without the hessian output should raise");

% heterogeneous model: simulate with_noise holds noise VARIANCES
noise = 0.01 * ones(n, 1);
kh = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "heterogeneous", noise);
var_n = [0.05; 0.2; 0.5];
nsim = int32(20000);
s_noisy = kh.simulate(nsim, int32(7), Xn, var_n, false);
s_latent = kh.simulate(nsim, int32(7), Xn, zeros(3, 1), false);
emp_var = var(s_noisy - s_latent, 0, 2);
assert(abs(emp_var ./ var_n - 1) < 0.05);

% update without noise on a heterogeneous model: clear error, no change
failed = false;
try
  kh.update(y(1:2), X(1:2, :) + 0.01, false);
catch err
  failed = true;
  assert(~isempty(strfind(err.message, "heterogeneous noise model: use update(y_u, noise_u, X_u)")), err.message);
end
assert(failed, "update(y, X) should raise on a heterogeneous model");
assert(size(kh.X(), 1) == n);
kh.update(y(1:2), noise(1:2), X(1:2, :) + 0.01, false);
assert(size(kh.X(), 1) == n + 2);

% approximate LOO / LMP are noise model None only
kv = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget", [], "vecchia(10)");
assert(isfinite(kv.logLikelihood("vecchia")));
failed = false;
try
  loo_ = kv.leaveOneOut("vecchia");
catch err
  failed = true;
  assert(~isempty(strfind(err.message, "not implemented for Nugget/Heterogeneous noise models")), err.message);
end
assert(failed, "leaveOneOut(vecchia) should raise on a nugget model");
failed = false;
try
  lmp_ = kv.logMargPost("vecchia");
catch err
  failed = true;
  assert(~isempty(strfind(err.message, "not implemented for Nugget/Heterogeneous noise models")), err.message);
end
assert(failed, "logMargPost(vecchia) should raise on a nugget model");

% update_simulate dispatches on the number of arguments, whatever the noise
% model: on a heterogeneous model the 3-arg form (y_u, X_u) raises (the noise
% of the update points is needed, as for update), the 4-arg form works
Xu = [0.35 0.65; 0.75 0.25];
yu = sin(3 * Xu(:, 1)) + cos(5 * Xu(:, 2));
nsim_u = int32(5000);
s_ = kh.simulate(nsim_u, int32(11), Xn, var_n, true);
failed = false;
try
  u3_ = kh.update_simulate(yu, Xu);
catch err
  failed = true;
  assert(~isempty(strfind(err.message, "heterogeneous noise model: use update_simulate(y_u, noise_u, X_u)")), err.message);
end
assert(failed, "update_simulate(y_u, X_u) should raise on a heterogeneous model");
u4 = kh.update_simulate(yu, 0.01 * ones(2, 1), Xu);  % 4-arg form
assert(size(u4), [3, double(nsim_u)]);
assert(all(isfinite(u4(:))));
% the 4-arg form adds the with_noise draws of the last simulate again: the
% spread of u4 exceeds the one after a latent simulate by about var_n
s_ = kh.simulate(nsim_u, int32(11), Xn, zeros(3, 1), true);
u4_latent = kh.update_simulate(yu, 0.01 * ones(2, 1), Xu);
assert(all(var(u4, 0, 2) > var(u4_latent, 0, 2)));
% fit / update drop the last simulate: update_simulate raises until a new one
kh.update(yu, 0.01 * ones(2, 1), Xu, false);
failed = false;
try
  u_ = kh.update_simulate(yu, 0.01 * ones(2, 1), Xu);
catch err
  failed = true;
  assert(~isempty(strfind(err.message, "No previous simulation data available")), err.message);
end
assert(failed, "update_simulate after update should raise");
% on a model without noise, the 3-arg form is the only one
s_ = k.simulate(int32(10), int32(1), Xn, true);
assert(size(k.update_simulate(yu, Xu)), [3, 10]);
