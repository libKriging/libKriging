% clear all
% addpath("mLibKriging")
# mLibKriging("help")

X = [0.0;0.2;0.5;0.8;1.0];
f = @(x) 1-1/2.*(sin(12*x)./(1+x)+2*cos(7.*x).*x.^5+0.7)
y = f(X);
k_m = Kriging(y, X, "gauss");
disp(k_m.describeModel());

% session
x = reshape(0:(1/99):1,100,1);
[p_mean, p_stdev] = k_m.predict(x, true, false);
h = figure(1, 'Visible','off'); % no display
hold on;
plot(x,f(x));
scatter(X,f(X));

plot(x,p_mean,'b')
poly = fill([x; flip(x)], [(p_mean-2*p_stdev); flip(p_mean+2*p_stdev)],'b');
set( poly, 'facealpha', 0.2);

hold off;
try
    print(h,'-dpng','mplot1.png'); % plot to file
catch
    printf("Cannot export plot\n")
end_try_catch
% close(h);

s = k_m.simulate(int32(10),int32(123), x);

h = figure(2, 'Visible','off'); % no display
hold on;
plot(x,f(x));
scatter(X,f(X));
for i=1:10
   plot(x,s(:,i),'b');
endfor
hold off;
try
    print(h,'-dpng','mplot2.png'); % plot to file
catch
    printf("Cannot export plot\n")
end_try_catch    
% close(h);

Xn = [0.3;0.4];
yn = f(Xn);
disp(k_m.describeModel());
k_m.update(yn, Xn, false)
disp(k_m.describeModel());
