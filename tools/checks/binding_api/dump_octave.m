% Lists the names exposed by mLibKriging, one "<scope>\t<name>" line each,
% for compare.py, which compares the bindings with each other.
% Usage: octave --path <mLibKriging dirs> --eval "dump_octave('<output file>')"
function dump_octave(out_file)
    scopes = {'Kriging', 'WarpKriging', 'MLPKriging', 'NestedKriging', 'Optim', 'LinearAlgebra'};
    % methods inherited from the handle superclass are not part of the API
    inherited = methods('handle');

    lines = {};
    for i = 1:numel(scopes)
        names = methods(scopes{i});
        hidden = non_public(scopes{i});
        for j = 1:numel(names)
            if ~strcmp(names{j}, scopes{i}) && ~any(strcmp(names{j}, inherited)) ...
                    && ~any(strcmp(names{j}, hidden))
                lines{end+1} = sprintf('%s\t%s', scopes{i}, names{j});
            end
        end
    end

    fid = fopen(out_file, 'w');
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);
end

% methods with Access other than public: Octave 6 methods() lists them
function names = non_public(scope)
    names = {};
    ml = meta.class.fromName(scope).MethodList;
    for k = 1:numel(ml)
        if iscell(ml)
            m = ml{k};
        else
            m = ml(k);
        end
        if ~ischar(m.Access) || ~strcmp(m.Access, 'public')
            names{end+1} = m.Name;
        end
    end
end
