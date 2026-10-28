% Smoke calls of mLibKriging (issue #435): each family of methods is called
% once on a tiny model (n = 10, d = 2) with the argument types a user would
% pass (a plain double, never int32), and the settings of Optim /
% LinearAlgebra are checked by a get / set / get round trip
% (smoke_setters.tsv). Prints one "ok" / "FAIL" line per call; returns the
% number of failures.
% Usage: octave-cli --path <mLibKriging dirs> --eval "n = smoke_octave('<tsv dir>', '<markdown file>'); exit(n > 0)"
function nfail = smoke_octave(tsv_dir, md_file)
    global SMOKE_FAILURES SMOKE_CALLS
    SMOKE_FAILURES = {};
    SMOKE_CALLS = 0;

    fid = fopen(fullfile(tsv_dir, 'smoke_setters.tsv'));
    line = fgetl(fid);
    while ischar(line)
        if ~isempty(line) && line(1) ~= '#'
            f = strsplit(line, "\t");
            smoke_check0([f{1} '.' f{2}], @() setter_roundtrip(f{:}));
        end
        line = fgetl(fid);
    end
    fclose(fid);

    rand('seed', 1);
    X = rand(10, 2);
    y = sin(3 * X(:, 1)) + X(:, 2);
    Xn = rand(3, 2);
    Xu = rand(2, 2);
    yu = [0.5; 0.7];
    tmp = tempname();
    mkdir(tmp);

    k = [];
    labels = {'Kriging', 'WarpKriging', 'MLPKriging', 'NestedKriging'};
    makers = {@() Kriging(y, X, 'gauss'), ...
              @() WarpKriging(y, X, {'none', 'none'}, 'gauss'), ...
              @() MLPKriging(y, X, [2], 2, 'selu', 'gauss'), ...
              @() NestedKriging(y, X, 'gauss', 2)};
    skip = {'fit', 'save', 'load', 'update', 'update_simulate', 'simulate', 'copy', 'delete', 'disp'};
    for i = 1:numel(labels)
        label = labels{i};
        m = [];
        try
            m = makers{i}();
            smoke_check0([label ' fit'], @() true);
        catch e
            smoke_check0([label ' fit'], @() error('%s', e.message));
            continue
        end
        if strcmp(label, 'Kriging'), k = m; end
        % zero-argument methods, whatever their name
        names = setdiff(methods(label), [skip, {label}, methods('handle')']);
        for j = 1:numel(names)
            smoke_check([label '.' names{j} '()'], @() zero_arg(m, names{j}));
        end
        smoke_check([label '.predict'], @() m.predict(Xn));
        if strcmp(label, 'NestedKriging'), continue, end
        smoke_check([label '.simulate'], @() m.simulate(2, 123, Xn, false));
        smoke_check0([label '.update'], @() m.copy().update(yu, Xu, true));
        smoke_check([label '.update_simulate'], @() update_simulate_after_simulate(m, Xn, yu, Xu));
        path = fullfile(tmp, [label '.h5']);
        smoke_check([label '.save/load'], @() save_load(m, path, label));
    end

    if ~isempty(k)
        smoke_check('Kriging.logLikelihood', @() k.logLikelihood());
        smoke_check('Kriging.leaveOneOut', @() k.leaveOneOut());
        smoke_check('Kriging.logMargPost', @() k.logMargPost());
        smoke_check('Kriging.covMat', @() k.covMat(Xn, Xn));
        smoke_check('Kriging.logLikelihoodFun', @() two_out(@() k.logLikelihoodFun([0.5; 0.5], true, false)));
        smoke_check('Kriging.leaveOneOutFun', @() two_out(@() k.leaveOneOutFun([0.5; 0.5], true)));
        smoke_check('Kriging.leaveOneOutVec', @() two_out(@() k.leaveOneOutVec([0.5; 0.5])));
        smoke_check('Kriging.subsetOfData', @() Kriging.subsetOfData(X, 5));
    end

    nfail = numel(SMOKE_FAILURES);
    printf('\n%d call(s), %d failure(s)\n', SMOKE_CALLS, nfail);
    if nargin > 1
        fid = fopen(md_file, 'a');
        fprintf(fid, '\n## Smoke calls: Octave\n\n%d call(s), **%d failure(s)**\n\n', SMOKE_CALLS, nfail);
        for i = 1:nfail
            fprintf(fid, '- `%s`\n', SMOKE_FAILURES{i});
        end
        fclose(fid);
    end
end

% fn returns a value: called with one output, as the methods of the binding need
function smoke_check(label, fn)
    smoke_run(label, @() fn(), 1);
end

% fn returns nothing (a setter, a check)
function smoke_check0(label, fn)
    smoke_run(label, fn, 0);
end

function smoke_run(label, fn, nout)
    global SMOKE_FAILURES SMOKE_CALLS
    SMOKE_CALLS = SMOKE_CALLS + 1;
    try
        if nout > 0
            r = fn();
        else
            fn();
        end
        printf('ok    %s\n', label);
    catch e
        SMOKE_FAILURES{end+1} = label;
        msg = strsplit(e.message, "\n"){1};
        printf('FAIL  %s: %s\n', label, msg);
    end
end

function ok = zero_arg(m, name)
    ok = true;
    try
        eval(sprintf('r = m.%s();', name));
    catch e
        % a method needing arguments (or private) is not a getter
        if isempty(regexp(e.message, 'input|argument|Invalid call|nargin|undefined|out of bound|private access', 'once'))
            rethrow(e);
        end
    end
end

% a method with two outputs
function ok = two_out(fn)
    [a, b] = fn();
    ok = true;
end

function ok = save_load(m, path, label)
    m.save(path);
    obj = eval([label '.load(path)']);
    ok = true;
end

% update_simulate needs the simulation to update
function r = update_simulate_after_simulate(m, Xn, yu, Xu)
    c = m.copy();
    % int32 here: the plain-double simulate is checked by the .simulate call above
    s = c.simulate(int32(2), int32(123), Xn, true);
    r = c.update_simulate(yu, Xu);
end

function setter_roundtrip(scope, setter, getter, kind, value)
    if ~ismember(setter, methods(scope))
        error('%s.%s not exposed', scope, setter);
    end
    % static methods called by name: eval of "Scope.method(literal)"
    call = @(name, arg) eval(sprintf('%s.%s(%s);', scope, name, arg));
    fetch = @(name) eval(sprintf('%s.%s()', scope, name));
    if strcmp(getter, '-')
        call(setter, ifelse_str(strcmpi(value, 'true'), 'true', 'false'));
        return
    end
    old = double(fetch(getter));
    switch kind
        case {'int', 'float'}
            new = str2double(value);   % a plain double, as a user types it
        case 'bool'
            new = double(~old);
    end
    if strcmp(kind, 'bool')
        arg = ifelse_str(new, 'true', 'false');
        old_arg = ifelse_str(old, 'true', 'false');
    else
        arg = sprintf('%.17g', new);
        old_arg = sprintf('%.17g', old);
    end
    call(setter, arg);
    got = double(fetch(getter));
    call(setter, old_arg);
    if abs(got - new) > 1e-12 * max(1, abs(new))
        error('set %g then get %g', new, got);
    end
    back = double(fetch(getter));
    if abs(back - old) > 1e-12 * max(1, abs(old))
        error('restore %g then get %g', old, back);
    end
end

function s = ifelse_str(cond, a, b)
    if cond, s = a; else, s = b; end
end
