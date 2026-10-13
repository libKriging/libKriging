1;  % mark this file as a script

% Fit method: "exact" (default), "vecchia(m)" or "nystrom(k)". An approximate
% fit keeps no exact factorization: the exact methods then raise and the
% approximate method is passed explicitly at every step.

n = 60;
X = [mod(0.3819660 * (1:n)', 1), mod(0.7548777 * (1:n)', 1)];  % Kronecker design
y = sin(3 * X(:, 1)) + X(:, 2);
Xn = [0.15 0.3; 0.45 0.6; 0.75 0.2];

ke = Kriging(y, X, "matern5_2");
assert(strcmp(ke.method(), "exact"));

% fit(y, X, regmodel, normalize, optim, objective, parameters, method)
k = Kriging("matern5_2");
k.fit(y, X, "constant", false, "BFGS", "LL", [], "vecchia(10)");
assert(strcmp(k.objective(), "LL"));
assert(strcmp(k.method(), "vecchia(10)"));
mdl = k.model();
assert(strcmp(mdl.method, "vecchia(10)"));

% deprecated aliases: objective "LLVecchia(m)" / "LLNystrom(k)"
lastwarn("");
ka = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLVecchia(7)");
[wmsg, wid] = lastwarn();
assert(strcmp(wid, "mLibKriging:deprecated"));
assert(! isempty(strfind(wmsg, 'method="vecchia(7)"')));
lastwarn("");
kno = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], "vecchia(7)");
assert(isempty(lastwarn()));
assert(strcmp(ka.objective(), "LL"));
assert(strcmp(ka.method(), "vecchia(7)"));
failed = false;
try
  kx = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LOO", [], "none", [], "vecchia");
catch
  failed = true;
end
assert(failed, "only objective LL has an approximate version");

% no implicit switch from the exact methods
failed = false;
try
  [m_, s_] = k.predict(Xn, true, false, false);
catch
  failed = true;
end
assert(failed, "exact predict should raise on a Vecchia fit");
failed = false;
try
  k.simulate(int32(2), int32(1), Xn, false);
catch
  failed = true;
end
assert(failed, "exact simulate should raise on a Vecchia fit");

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
assert(k.logLikelihood(k.method()) == k.logLikelihood("vecchia"));

% update (Vecchia refit) and save / load
Xu = [0.12 0.81; 0.52 0.33];
yu = sin(3 * Xu(:, 1)) + Xu(:, 2);
k.update(yu, Xu, false);
assert(strcmp(k.method(), "vecchia(10)"));
assert(size(k.X(), 1) == n + 2);
assert(max(abs(k.predict(Xu, false, false, false, "vecchia") - yu)) < 1e-3);
f = [tempname() ".json"];
k.save(f);
kl = Kriging.load(f);
assert(strcmp(kl.method(), "vecchia(10)"));
assert(kl.vecchia_neighbors() == 10);
assert(max(abs(kl.predict(Xn, false, false, false, "vecchia") - k.predict(Xn, false, false, false, "vecchia"))) < 1e-12);
delete(f);

% an exact fit: exact and Vecchia predictors available
k.fit(y, X, "constant", false, "BFGS", "LL", [], "exact");
assert(strcmp(k.method(), "exact"));
[m, s, c] = k.predict(Xn, true, true, false);
assert(isequal(size(c), [3, 3]));
assert(numel(k.predict(Xn, false, false, false, "vecchia")) == 3);
assert(k.logLikelihood() == k.logLikelihood("exact"));

% Nystrom fits need method "nystrom"
kn = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LL", [], "none", [], "nystrom(8)");
assert(strcmp(kn.method(), "nystrom(8)"));
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
assert(failed, "Vecchia likelihood needs a Vecchia fit");

% NestedKriging: common prior from one global Vecchia fit
Xl = [mod(0.3819660 * (1:200)', 1), mod(0.7548777 * (1:200)', 1)];
yl = sin(3 * Xl(:, 1)) + Xl(:, 2);
nk = NestedKriging(yl, Xl, "matern5_2", 3, "NK", "kmeans", 123, "constant", "BFGS", "LL", [], {}, "vecchia(15)");
assert(strcmp(nk.objective(), "LL"));
assert(strcmp(nk.method(), "vecchia(15)"));

% method "vecchia(m)" with noise models (response Vecchia)
% Kriging(y, X, kernel, regmodel, normalize, optim, objective, parameters, noise_model, noise, method)
noise = 0.005 + 0.02 * X(:, 1);
yn = y + sqrt(noise) .* sin(37 * (1:n)');  % deterministic pseudo-noise
kg = Kriging(yn, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget", [], "vecchia(15)");
assert(strcmp(kg.method(), "vecchia(15)"));
failed = false;
try
  [m_, s_] = kg.predict(Xn, true, false, false);
catch
  failed = true;
end
assert(failed, "exact predict should raise on a Vecchia nugget fit");
[mg, sg] = kg.predict(Xn, true, false, false, "vecchia");
assert(all(isfinite(mg)));
assert(all(sg > 0));
assert(isfinite(kg.logLikelihood("vecchia")));
% simulate(nsim, seed, X_n, with_nugget, will_update, method)
assert(isequal(size(kg.simulate(int32(4), int32(123), Xn, true, false, "vecchia")), [3, 4]));
assert(isequal(size(kg.simulate(int32(4), int32(123), Xn, false, false, "vecchia(10)")), [3, 4]));
kg.update(yn(1:3), X(1:3, :) + 0.01, false);
assert(size(kg.X(), 1) == n + 3);

kh = Kriging(yn, X, "matern5_2", "constant", false, "BFGS", "LL", [], "heterogeneous", noise, "vecchia(15)");
assert(strcmp(kh.method(), "vecchia(15)"));
[mh, sh] = kh.predict(Xn, true, false, false, "vecchia");
assert(all(isfinite(mh)));
assert(all(sh > 0));
% simulate(nsim, seed, X_n, with_noise, will_update, method)
assert(isequal(size(kh.simulate(int32(4), int32(123), Xn, [0.1; 0.1; 0.1], false, "vecchia")), [3, 4]));
kh.update(yn(1:3), noise(1:3), X(1:3, :) + 0.01, false);
assert(size(kh.X(), 1) == n + 3);
failed = false;
try
  kx = Kriging(yn, X, "matern5_2", "constant", false, "BFGS", "LL", [], "nugget", [], "nystrom(8)");
catch
  failed = true;
end
assert(failed, "method nystrom has no noise model");
