% clear all
mLibKriging("help")

function y = f(x)
  y = prod(sin((x-0.5).^2), dim=2); 
endfunction

n = 40;
m = 3;
X = randn(n, m);
y = f(X);
km = Kriging(y, X, "gauss")
[y_pred, _stderr, _cov] = km.predict(X, true, true);
disp(km.describeModel())