1;  % mark this file as a script

% Kriging.copy returns an independent Kriging object (not a raw reference),
% like WarpKriging.copy / MLPKriging.copy.

f1d = @(x) 1 - 0.5 * (sin(12 * x) ./ (1 + x) + 2 * cos(7 * x) .* x.^5 + 0.7);
X = linspace(0.01, 0.99, 8)';
y = f1d(X);
X_test = linspace(0.1, 0.9, 5)';

k = Kriging(y, X, "gauss");
k2 = k.copy();
assert(isa(k2, "Kriging"));
assert(k2.ref ~= k.ref);
assert(strcmp(k2.kernel(), k.kernel()));
assert(max(abs(k2.theta() - k.theta())) == 0);
[m1, s1] = k.predict(X_test, true, false, false);
[m2, s2] = k2.predict(X_test, true, false, false);
assert(max(abs(m1 - m2)) == 0);
assert(max(abs(s1 - s2)) == 0);

% independent: deleting the original leaves the copy usable
clear k;
[m3, s3] = k2.predict(X_test, true, false, false);
assert(max(abs(m3 - m2)) == 0);
