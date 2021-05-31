% clear all

tolerance = 1e-12
refpath=find_dir();
printf("Reference directory=%s\n", refpath);

% test data 1
try
    prefix = "data1-scal";
    filex = [ refpath filesep prefix "-X.csv" ];
    filey = [ refpath filesep prefix "-y.csv" ];
    X = dlmread(filex, ",");
    y = dlmread(filey, ","); 

    file_loo = [ refpath filesep prefix "-result-leaveOneOut.csv" ]
    file_loograd = [ refpath filesep prefix "-result-leaveOneOutGrad.csv" ]
    loo_ref = dlmread(file_loo, delimiter=',')
    loograd_ref = dlmread(file_loograd, delimiter=',')
    file_ll = [ refpath filesep prefix "-result-logLikelihood.csv" ];
    file_llgrad = [ refpath filesep prefix "-result-logLikelihoodGrad.csv" ];
    ll_ref = dlmread(file_ll, delimiter=',');
    llgrad_ref = dlmread(file_llgrad, delimiter=',');
    
    kernel = "gauss";
    k_m = Kriging(y, X, kernel, "constant", false, "BFGS", "LL") % lk.Parameters() % FIXME parameters arg not mapped
    x = 0.3 * ones(size(X)(2), 1);
    [loo, loograd] = k_m.leaveOneOut(x, true);
    assert(relative_error(loo, loo_ref) < tolerance)
    assert(relative_error(loograd, loograd_ref) < tolerance)
    
    [ll, llgrad] = k_m.logLikelihood(x, true, false); % flags are optional; then results are driven by the output
    assert(relative_error(ll, ll_ref) < tolerance)
    assert(relative_error(llgrad, llgrad_ref) < tolerance)
    
catch exception
    printf("Exception caught %s : \n%s\n",exception.identifier, exception.message);
    rethrow(exception);
end


% test data 2
for i = 1:10
    try
        prefix = [ "data2-grad-" int2str(i) ];
        filex = [ refpath filesep prefix "-X.csv" ]; 
        filey = [ refpath filesep prefix "-y.csv" ];
        X = dlmread(filex, ",");
        y = dlmread(filey, ","); 
    
        file_ll = [ refpath filesep prefix "-result-logLikelihood.csv" ];
        file_llgrad = [ refpath filesep prefix "-result-logLikelihoodGrad.csv" ];
        ll_ref = dlmread(file_ll, delimiter=',');
        llgrad_ref = dlmread(file_llgrad, delimiter=',');
        llgrad_ref = transpose(llgrad_ref); % has been read as a row vector
        
        kernel = "gauss";
        k_m = k_m = Kriging(y, X, kernel) % use all default formal parameters
        x = 0.3 * ones(size(X)(2), 1);
        
        [ll, llgrad] = k_m.logLikelihood(x, true, false);
        assert(relative_error(ll, ll_ref) < tolerance)
        assert(relative_error(llgrad, llgrad_ref) < tolerance)
        
    catch exception
        printf("Exception caught %s : \n%s\n",exception.identifier, exception.message);
        rethrow(exception);
    end
end