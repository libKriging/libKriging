1;  % mark this file as a script

% Explicit methods: set_vecchia_exact_commit(false) + LLVecchia(m) gives a
% "light" fit (no exact factorization); the exact methods then raise and
% method "vecchia" is used explicitly. LLNystrom fits need method "nystrom".

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

% no implicit switch from the exact methods
failed = false;
try
  [m_, s_] = k.predict(Xn, true, false, false);
catch
  failed = true;
end
assert(failed, "exact predict should raise on a light fit");
failed = false;
try
  k.simulate(int32(2), int32(1), Xn, false);
catch
  failed = true;
end
assert(failed, "exact simulate should raise on a light fit");

[m, s] = k.predict(Xn, true, false, false, "vecchia");
assert(numel(m) == 3);
assert(all(isfinite(m)));
assert(all(s >= 0));
sims = k.simulate(int32(4), int32(123), Xn, false, "vecchia");
assert(isequal(size(sims), [3, 4]));
assert(isequal(size(k.simulate(int32(4), int32(123), Xn, false, "vecchia(5)")), [3, 4]));
failed = false;
try
  [m_, s_, c_] = k.predict(Xn, true, true, false, "vecchia");
catch
  failed = true;
end
assert(failed, "return_cov needs method exact");
failed = false;
try
  [m_, s_] = k.predict(Xn, true, false, false, "unknown");
catch
  failed = true;
end
assert(failed, "unknown method should raise");
assert(isfinite(k.logLikelihood("vecchia")));

% update (Vecchia refit) and save / load on a light fit
Xu = [0.12 0.81; 0.52 0.33];
yu = sin(3 * Xu(:, 1)) + Xu(:, 2);
k.update(yu, Xu, false);
assert(k.is_vecchia_light());
assert(size(k.X(), 1) == n + 2);
assert(max(abs(k.predict(Xu, false, false, false, "vecchia") - yu)) < 1e-3);
f = [tempname() ".json"];
k.save(f);
kl = Kriging.load(f);
assert(kl.is_vecchia_light());
assert(kl.vecchia_neighbors() == 10);
assert(max(abs(kl.predict(Xn, false, false, false, "vecchia") - k.predict(Xn, false, false, false, "vecchia"))) < 1e-12);
delete(f);

% refit with the exact commit: exact and Vecchia methods available
k.set_vecchia_exact_commit(true);
k.fit(y, X, "constant", false, "BFGS", "LLVecchia(10)");
assert(~k.is_vecchia_light());
[m, s, c] = k.predict(Xn, true, true, false);
assert(isequal(size(c), [3, 3]));
assert(numel(k.predict(Xn, false, false, false, "vecchia")) == 3);
assert(k.logLikelihood() == k.logLikelihood("exact"));

% Nystrom fits need method "nystrom"
kn = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLNystrom(8)");
assert(kn.is_nystrom_light());
failed = false;
try
  [m_, s_] = kn.predict(Xn, true, false, false);
catch
  failed = true;
end
assert(failed, "exact predict should raise on a Nystrom fit");
assert(numel(kn.predict(Xn, false, false, false, "nystrom")) == 3);
assert(isequal(size(kn.simulate(int32(3), int32(1), Xn, false, "nystrom")), [3, 3]));
assert(isfinite(kn.logLikelihood("nystrom")));
failed = false;
try
  kn.logLikelihood("vecchia");
catch
  failed = true;
end
assert(failed, "Vecchia likelihood needs an LLVecchia fit");
assert(~Kriging(y, X, "matern5_2").is_nystrom_light());
