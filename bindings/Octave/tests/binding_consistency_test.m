% clear all

tolerance = 1e-12
refpath=find_dir();
fprintf("Reference directory=%s\n", refpath);

% test data 1
try
    prefix = "data1-scal";
    filex = fullfile(refpath, sprintf("%s-X.csv", prefix));
    filey = fullfile(refpath, sprintf("%s-y.csv", prefix));
    X = dlmread(filex, ",");
    y = dlmread(filey, ","); 

    file_loo = fullfile(refpath, sprintf("%s-result-leaveOneOut.csv", prefix));
    file_loograd = fullfile(refpath, sprintf("%s-result-leaveOneOutGrad.csv", prefix));
    loo_ref = dlmread(file_loo, ',')
    loograd_ref = dlmread(file_loograd, ',')
    file_ll = fullfile(refpath, sprintf("%s-result-logLikelihood.csv", prefix));
    file_llgrad = fullfile(refpath, sprintf("%s-result-logLikelihoodGrad.csv", prefix));
    ll_ref = dlmread(file_ll, ',');
    llgrad_ref = dlmread(file_llgrad, ',');

    kernel = "gauss";
    k_m = Kriging(y, X, kernel, "constant", false, "BFGS", "LL") % lk.Parameters() % FIXME parameters arg not mapped
    s = size(X);
    x = 0.3 * ones(s(2), 1);
    [loo, loograd] = k_m.leaveOneOutFun(x, true);
    assert(relative_error(loo, loo_ref) < tolerance)
    assert(relative_error(loograd, loograd_ref) < tolerance)

    [ll, llgrad] = k_m.logLikelihoodFun(x, true, false); % flags are optional; then results are driven by the output
    assert(relative_error(ll, ll_ref) < tolerance)
    assert(relative_error(llgrad, llgrad_ref) < tolerance)

catch exception
    fprintf("Exception caught %s : \n%s\n",exception.identifier, exception.message);
    rethrow(exception);
end


% test data 2
for i = 1:10
    try
        prefix = sprintf("data2-grad-%s", int2str(i));
        filex = fullfile(refpath, sprintf("%s-X.csv", prefix));; 
        filey = fullfile(refpath, sprintf("%s-y.csv", prefix));;
        X = dlmread(filex, ",");
        y = dlmread(filey, ","); 

        file_ll = fullfile(refpath, sprintf("%s-result-logLikelihood.csv", prefix));;
        file_llgrad = fullfile(refpath, sprintf("%s-result-logLikelihoodGrad.csv", prefix));;
        ll_ref = dlmread(file_ll, ',');
        llgrad_ref = dlmread(file_llgrad, ',');
        llgrad_ref = transpose(llgrad_ref); % has been read as a row vector

        kernel = "gauss";
        k_m = Kriging(y, X, kernel) % use all default formal parameters
        s = size(X);
        x = 0.3 * ones(s(2), 1);

        [ll, llgrad] = k_m.logLikelihoodFun(x, true, false);
        assert(relative_error(ll, ll_ref) < tolerance)
        assert(relative_error(llgrad, llgrad_ref) < tolerance)

    catch exception
        fprintf("Exception caught %s : \n%s\n",exception.identifier, exception.message);
        rethrow(exception);
    end
end