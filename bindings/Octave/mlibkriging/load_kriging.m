function obj = load_kriging(filename)
% Load any libKriging model from file, auto-detecting its class.
    kclass = class_saved(filename);
    switch kclass
        case 'Kriging'
            obj = Kriging.load(filename);
        case 'WarpKriging'
            obj = WarpKriging.load(filename);
        case 'MLPKriging'
            obj = MLPKriging.load(filename);
        otherwise
            error('Unknown Kriging class in file: %s', filename);
    end
end
