1;  % mark this file as a script

% Integer arguments given as plain doubles, the default numeric type (#435):
% nsim / seed of simulate, n_max of subsetOfData, as well as int32.

X = [0.05 0.2; 0.15 0.9; 0.3 0.4; 0.45 0.7; 0.55 0.1; 0.65 0.8; 0.75 0.3; 0.85 0.6; 0.92 0.5; 0.97 0.15];
y = sin(3 * X(:, 1)) + X(:, 2);
Xn = [0.25 0.5; 0.5 0.5; 0.7 0.2];

k = Kriging(y, X, "gauss");

s_double = k.simulate(5, 123, Xn, false);
s_int32 = k.simulate(int32(5), int32(123), Xn, false);
assert(isequal(size(s_double), [3, 5]));
assert(isequal(s_double, s_int32));

idx = Kriging.subsetOfData(X, 4);
assert(numel(idx) == 4);
assert(isequal(idx, Kriging.subsetOfData(X, int32(4))));

% a non-integer or negative value is still an error
failed = false;
try
    k.simulate(2.5, 123, Xn, false);
catch
    failed = true;
end
assert(failed);

failed = false;
try
    k.simulate(5, 1.5, Xn, false);
catch
    failed = true;
end
assert(failed);
