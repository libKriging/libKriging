1;  % mark this file as a script

% With normalize=true, Kriging reads its parameters in the original scale of
% X and y and gives every accessor / objective function in that same scale:
% the given parameters round-trip, and agree with a normalize=false model.

n = 25;
U = [mod(0.3819660 * (1:n)', 1), mod(0.7548777 * (1:n)', 1)];
X = [10 + 20 * U(:, 1), -2 + 5 * U(:, 2)];
y = 100 + 50 * (sin(5 * U(:, 1)) .* cos(3 * U(:, 2)) + 0.5 * U(:, 1) .* U(:, 2));

p = Params("theta", [7.0 1.3], "is_theta_estim", false, ...
           "sigma2", 321.0, "is_sigma2_estim", false, ...
           "beta", [0.5; -2.0; 3.0], "is_beta_estim", false);
k = Kriging(y, X, "matern5_2", "linear", true, "none", "LL", p);
assert(k.theta(), [7.0; 1.3], -1e-12);
assert(k.sigma2(), 321.0, -1e-12);
assert(k.beta(), [0.5; -2.0; 3.0], -1e-12);

ko = Kriging(y, X, "matern5_2", "linear", false, "none", "LL", p);
assert(k.logLikelihood(), ko.logLikelihood(), -1e-10);
[lln, gn] = k.logLikelihoodFun([9.0; 2.1], true);
[llo, go] = ko.logLikelihoodFun([9.0; 2.1], true);
assert(lln, llo, -1e-10);
assert(gn, go, -1e-6);

% nugget: given sigma2 / nugget round-trip too
pn = Params("theta", [7.0 1.3], "is_theta_estim", false, ...
            "sigma2", 321.0, "is_sigma2_estim", false, ...
            "nugget", 12.5, "is_nugget_estim", false);
kn = Kriging(y, X, "matern5_2", "constant", true, "none", "LL", pn, "nugget");
assert(kn.nugget(), 12.5, -1e-12);
assert(kn.sigma2(), 321.0, -1e-12);
