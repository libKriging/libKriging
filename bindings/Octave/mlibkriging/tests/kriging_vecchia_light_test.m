1;  % mark this file as a script

% set_vecchia_exact_commit(false) + LLVecchia(m): "light" fit, no exact
% factorization; predict / simulate go through the Vecchia predictor /
% simulator, return_cov and save are refused.

n = 60;
X = [mod(0.3819660 * (1:n)', 1), mod(0.7548777 * (1:n)', 1)];  % Kronecker design
y = sin(3 * X(:, 1)) + X(:, 2);
Xn = [0.15 0.3; 0.45 0.6; 0.75 0.2];

k = Kriging("matern5_2");
assert(k.vecchia_exact_commit());
k.set_vecchia_exact_commit(false);
assert(~k.vecchia_exact_commit());
k.fit(y, X, "constant", false, "BFGS", "LLVecchia(10)");
assert(k.is_vecchia_light());
assert(~k.is_nystrom_light());

[m, s] = k.predict(Xn, true, false, false);
assert(numel(m) == 3);
assert(all(isfinite(m)));
assert(all(s >= 0));
sims = k.simulate(int32(4), int32(123), Xn, false);
assert(isequal(size(sims), [3, 4]));

failed = false;
try
  k.predict(Xn, true, true, false);
catch
  failed = true;
end
assert(failed, "return_cov should be refused on a light Vecchia fit");

failed = false;
try
  k.save([tempname() ".json"]);
catch
  failed = true;
end
assert(failed, "save should be refused on a light Vecchia fit");

k.set_vecchia_exact_commit(true);
k.fit(y, X, "constant", false, "BFGS", "LLVecchia(10)");
assert(~k.is_vecchia_light());

kn = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLNystrom(8)");
assert(kn.is_nystrom_light());
assert(~Kriging(y, X, "matern5_2").is_nystrom_light());
