function kclass = class_saved(filename)
% Detect the saved libKriging model class from a JSON file.
    kclass = mLibKriging("class_saved", filename);
end
