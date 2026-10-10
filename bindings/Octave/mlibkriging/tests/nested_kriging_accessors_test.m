1;  % mark this file as a script

% NestedKriging accessors and tuning setters, aligned with Python / R / Julia.

X = [linspace(0.01, 0.99, 40)', mod((1:40)' * 0.37, 1)];
y = sin(3 * X(:, 1)) + X(:, 2);
nk = NestedKriging(y, X, "gauss", 4);

assert(isequal(size(nk.X()), size(X)));
assert(max(max(abs(nk.X() - X))) == 0);
assert(max(abs(nk.y() - y)) == 0);

g = nk.groups();
assert(iscell(g) && numel(g) == 4);
all_idx = sort(vertcat(g{:}));
assert(isequal(all_idx(:), (1:40)'));  % 1-based partition of the rows

assert(isempty(nk.warping()));

nk.set_predict_chunk(7);
[m1, s1] = nk.predict(X(1:10, :));
nk.set_predict_chunk(128);
[m2, s2] = nk.predict(X(1:10, :));
assert(max(abs(m1 - m2)) < 1e-10);  % chunking does not change the result
assert(max(abs(s1 - s2)) < 1e-10);

nk.set_warp_subsample(500);  % accepted (used by warped fits)

nkw = NestedKriging(y, X, "gauss", 2, "NK", "kmeans", 123, "constant", "BFGS", "LL", Params(), {"kumaraswamy", "kumaraswamy"});
w = nkw.warping();
assert(iscell(w) && numel(w) == 2 && strcmp(w{1}, "kumaraswamy"));
