% clear all
y = randn (20,1);
X = randn (20,1);
X2 = randn (20,1);
a=LinearRegression();
a.fit(y,X);
[y2,stderr] = a.predict(X2);
delete(a);