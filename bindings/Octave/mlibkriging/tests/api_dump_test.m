% Lists the names exposed by mLibKriging, one "<scope>\t<name>" line each,
% for tools/checks/binding_api.py, which compares them to bindings/api.yaml.
% Written to $LK_API_DUMP_DIR/octave.txt when that variable is set (ctest sets
% it to <build>/binding-api), only checked for non-emptiness otherwise.

scopes = {'Kriging', 'WarpKriging', 'MLPKriging', 'NestedKriging', 'Optim'};
% methods inherited from the handle superclass are not part of the API
inherited = {};
try
    inherited = methods('handle');
catch
end

lines = {};
% module: the functions and classes installed next to Kriging.m
lib_dir = fileparts(which('Kriging'));
files = dir(fullfile(lib_dir, '*.m'));
for i = 1:numel(files)
    [~, name] = fileparts(files(i).name);
    lines{end+1} = sprintf('module\t%s', name);
end
for i = 1:numel(scopes)
    names = methods(scopes{i});
    n = 0;
    for j = 1:numel(names)
        if strcmp(names{j}, scopes{i}) || any(strcmp(names{j}, inherited))
            continue
        end
        lines{end+1} = sprintf('%s\t%s', scopes{i}, names{j});
        n = n + 1;
    end
    assert(n > 0, sprintf('no method found for %s', scopes{i}));
end

out_dir = getenv('LK_API_DUMP_DIR');
if ~isempty(out_dir)
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    fid = fopen(fullfile(out_dir, 'octave.txt'), 'w');
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);
end
