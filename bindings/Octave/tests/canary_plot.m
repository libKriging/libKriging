% clear all
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0

X = [0.0;0.2;0.5;0.8;1.0];
f = @(x) 1-1/2.*(sin(12*x)./(1+x)+2*cos(7.*x).*x.^5+0.7)
y = f(X);

% session
x = reshape(0:(1/99):1,100,1);
if (length(getenv("GITHUB_ACTION"))>0)
    disp('in GITHUB_ACTION: skip plotting');
else
    if (isOctave)
        h = figure(1, 'Visible','off'); % no display
    else
        h = figure(1);       
        h.Visible = 'off'; % no display
    end
    hold on;
    plot(x,f(x));
    scatter(X,f(X));
end
