% clear all
% addpath("mLibKriging")
mLibKriging("help")

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0

X = [0.0;0.2;0.5;0.8;1.0];
f = @(x) 1-1/2.*(sin(12*x)./(1+x)+2*cos(7.*x).*x.^5+0.7)
y = f(X);
% k_m = Kriging(y, X, "gauss"); % without optional and parameters
k_m = Kriging(y, X, "gauss", "constant", false, "BFGS", "LL", Params("is_sigma2_estim", true))
disp(k_m.summary());

% session
x = reshape(0:(1/99):1,100,1);
[p_mean, p_stdev] = k_m.predict(x, true, false);
if (isOctave)
    h = figure(1, 'Visible','off'); % no display
else
    h = figure(1);       
    h.Visible = 'off'; % no display
end
hold on;
plot(x,f(x));
scatter(X,f(X));

plot(x,p_mean,'b')
poly = fill([x; flip(x)], [(p_mean-2*p_stdev); flip(p_mean+2*p_stdev)],'b');
set( poly, 'facealpha', 0.2);

hold off;
try
    saveas(h, 'mplot1.png'); % plot to file
catch
    fprintf("Cannot export plot\n")
end
% close(h);

s = k_m.simulate(int32(10),int32(123), x);

if (isOctave)
    h = figure(2, 'Visible','off'); % no display
else
    h = figure(1);       
    h.Visible = 'off'; % no display
end
hold on;
plot(x,f(x));
scatter(X,f(X));
for i=1:10
   plot(x,s(:,i),'b');
end
hold off;
try
    saveas(h, 'mplot2.png'); % plot to file
catch
    fprintf("Cannot export plot\n")
end
% close(h);

Xn = [0.3;0.4];
yn = f(Xn);
disp(k_m.summary());
k_m.update(yn, Xn)
disp(k_m.summary());
