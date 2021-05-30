% clear all

nrow = 200
ncol = 2
nrow2 = 10

try 
    printf("\nTry bad usage #1\n")
    a=LinearRegression(1); % no argument required
catch exception
    if (~strcmp(exception.identifier,'mLibKriging:badArgs'))
        printf("Unexpected test exception %s\n\n",exception.identifier)
        rethrow(exception)
    else
        printf("Exception well caught %s\n",exception.identifier)
     end
end

% right has been caught, follows with good args
printf("\nNow try right usage\n")
a=LinearRegression(); 

X = randn(nrow, ncol);
hidden_coef = randn(ncol, 1)
noise_amplitude = 1e-5;
y = X*hidden_coef + noise_amplitude*randn(nrow,1);
% y = randn (nrow, 1);
X2=randn(nrow2,ncol);

try 
    printf("\nTry bad usage #2\n")
    badX = randn (nrow+2,ncol);
    a.fit(y,badX); % incompatibles X y sizes
catch exception
    if (~strcmp(exception.identifier,'mLibKriging:kernelException'))
        printf("Unexpected test exception %s\n\n",exception.identifier)
        rethrow(exception)
    else
        printf("Exception well caught %s\n",exception.identifier)
     end
end

% right has been caught, follows with good args
printf("\nNow try right usage\n")
a.fit(y,X);

try 
    printf("\nTry bad usage #3\n")
    badX = randn (nrow,ncol+1);
    [y_pred,stderr] = a.predict(badX);
catch exception
    if (~strcmp(exception.identifier,'mLibKriging:kernelException'))
        printf("Unexpected test exception %s\n\n",exception.identifier)
        rethrow(exception)
    else
        printf("Exception well caught %s\n",exception.identifier)
     end
end

try 
    printf("\nTry bad usage #4\n")
    [y_pred,stderr, dummy] = a.predict(X2)
catch exception
    if (~strcmp(exception.identifier,'mLibKriging:badArgs'))
        printf("Unexpected test exception %s\n\n",exception.identifier)
        rethrow(exception)
     else
        printf("Exception well caught %s\n",exception.identifier)
    end
end

printf("\nNow try right usage\n")
[y_pred,stderr] = a.predict(X2);

printf("\nCheck error (must be below %g)\n", noise_amplitude)
err = norm(X2*hidden_coef - y_pred, p=Inf)

if (err > noise_amplitude)
    error('error to large : %g > %g', err, noise_amplitude);
end