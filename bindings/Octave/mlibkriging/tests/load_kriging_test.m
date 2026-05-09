1;  % mark this file as a script

f1d = @(x) 1 - 0.5 * (sin(12 * x) ./ (1 + x) + 2 * cos(7 * x) .* x.^5 + 0.7);

cleanup_files = {"load_kriging_k.json", "load_kriging_wk.json", "load_kriging_mlp.json"};
for i = 1:numel(cleanup_files)
    if exist(cleanup_files{i}, "file")
        unlink(cleanup_files{i});
    end
end

try
    X = linspace(0.01, 0.99, 8)';
    y = f1d(X);

    k = Kriging(y, X, "gauss");
    k.save("load_kriging_k.json");
    assert(strcmp(class_saved("load_kriging_k.json"), "Kriging"));
    k2 = load_kriging("load_kriging_k.json");
    assert(isa(k2, "Kriging"));

    wk = WarpKriging(y, X, {"kumaraswamy"}, "gauss");
    wk.save("load_kriging_wk.json");
    assert(strcmp(class_saved("load_kriging_wk.json"), "WarpKriging"));
    wk2 = load_kriging("load_kriging_wk.json");
    assert(isa(wk2, "WarpKriging"));

    mk = MLPKriging(y, X, [8 4], 2, "selu", "gauss");
    mk.save("load_kriging_mlp.json");
    assert(strcmp(class_saved("load_kriging_mlp.json"), "MLPKriging"));
    mk2 = load_kriging("load_kriging_mlp.json");
    assert(isa(mk2, "MLPKriging"));
catch err
    for i = 1:numel(cleanup_files)
        if exist(cleanup_files{i}, "file")
            unlink(cleanup_files{i});
        end
    end
    rethrow(err);
end

for i = 1:numel(cleanup_files)
    if exist(cleanup_files{i}, "file")
        unlink(cleanup_files{i});
    end
end
