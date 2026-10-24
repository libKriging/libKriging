% Lists the names exposed by mLibKriging, one "<scope>\t<name>" line each,
% for check.py, which compares them to bindings/api.yaml.
% Usage: octave --path <mLibKriging dirs> --eval "dump_octave('<output file>')"
function dump_octave(out_file)
    scopes = {'Kriging', 'WarpKriging', 'MLPKriging', 'NestedKriging', 'Optim'};
    % methods inherited from the handle superclass are not part of the API
    inherited = methods('handle');

    lines = {};
    % module: the functions and classes installed next to Kriging.m
    files = dir(fullfile(fileparts(which('Kriging')), '*.m'));
    for i = 1:numel(files)
        [~, name] = fileparts(files(i).name);
        lines{end+1} = sprintf('module\t%s', name);
    end
    for i = 1:numel(scopes)
        names = methods(scopes{i});
        for j = 1:numel(names)
            if ~strcmp(names{j}, scopes{i}) && ~any(strcmp(names{j}, inherited))
                lines{end+1} = sprintf('%s\t%s', scopes{i}, names{j});
            end
        end
    end

    fid = fopen(out_file, 'w');
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);
end
