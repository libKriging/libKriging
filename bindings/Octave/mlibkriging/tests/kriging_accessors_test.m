1;  % mark this file as a script

% vecchia_neighbors / nystrom_rank report the size of the approximation
% (0 when the model was not fitted with the matching objective).

f1d = @(x) 1 - 0.5 * (sin(12 * x) ./ (1 + x) + 2 * cos(7 * x) .* x.^5 + 0.7);
X = linspace(0.01, 0.99, 20)';
y = f1d(X);

kv = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLVecchia(5)");
assert(kv.vecchia_neighbors() == 5);
kn = Kriging(y, X, "matern5_2", "constant", false, "BFGS", "LLNystrom(4)");
assert(kn.nystrom_rank() == 4);
k = Kriging(y, X, "matern5_2");
assert(k.vecchia_neighbors() == 0);
assert(k.nystrom_rank() == 0);
