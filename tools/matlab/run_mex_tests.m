function run_mex_tests(build_dir)
% RUN_MEX_TESTS  Run the mLibKriging MEX tests inside the current MATLAB session.
%
%   run_mex_tests(build_dir)
%
% The CTest entries "Matlab/*" start one `matlab -batch` process per test. On
% GitHub-hosted runners of a public repository, MATLAB is licensed only inside
% the matlab-actions/run-* actions, not in a plain shell step, so CI runs the
% same tests here, in the session opened by matlab-actions/run-command.
%
% The test list, and the tests expected to fail (WILL_FAIL), are read from
% bindings/Octave/mlibkriging/tests/CMakeLists.txt so that CTest and this
% runner share one definition. Errors on failure (non-zero exit of the action).

    root_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    src_dir = fullfile(root_dir, 'bindings', 'Octave', 'mlibkriging');
    test_dir = fullfile(src_dir, 'tests');

    % The directory holding the compiled MEX file is CTest's WORKING_DIRECTORY
    mex_file = dir(fullfile(build_dir, '**', ['mLibKriging.' mexext]));
    mex_file = mex_file(~contains({mex_file.folder}, [filesep 'installed']));
    if isempty(mex_file)
        error('run_mex_tests:noMex', 'mLibKriging.%s not found under %s', mexext, build_dir);
    end
    work_dir = mex_file(1).folder;

    tests = read_test_list(fullfile(test_dir, 'CMakeLists.txt'));
    if isempty(tests)
        error('run_mex_tests:noTest', 'no add_mex_test entry found in %s', test_dir);
    end

    % Octave's assert (comparison forms used by the tests), MATLAB only
    addpath(fullfile(root_dir, 'tools', 'matlab', 'octave_compat'));
    warning('off', 'MATLAB:dispatcher:nameConflict');
    addpath(work_dir);
    addpath(src_dir);
    addpath(test_dir);
    old_dir = cd(work_dir);
    cleanup = onCleanup(@() cd(old_dir));

    failures = {};
    for i = 1:numel(tests)
        t = tests(i);
        fprintf('\n==== Matlab/%s (%s)%s\n', t.name, t.file, ternary(t.will_fail, ' [WILL_FAIL]', ''));
        tic;
        threw = run_one(t.script);
        passed = (threw == t.will_fail);
        fprintf('==== Matlab/%s: %s (%.1f s)\n', t.name, ternary(passed, 'Passed', '***Failed'), toc);
        if ~passed
            failures{end+1} = t.name; %#ok<AGROW>
        end
        close all force;
        cd(work_dir);
    end

    fprintf('\n%d/%d tests passed\n', numel(tests) - numel(failures), numel(tests));
    if ~isempty(failures)
        error('run_mex_tests:failed', 'Failed tests: %s', strjoin(failures, ', '));
    end
end

function threw = run_one(script)
    % own workspace for each test script; threw is set after run() so that a
    % `clear` in the script cannot remove it
    try
        run(script);
        threw = false;
    catch err
        threw = true;
        disp('An exception has been thrown during the execution');
        disp(getReport(err, 'extended'));
    end
end

function tests = read_test_list(cmake_file)
    txt = fileread(cmake_file);
    lines = regexp(txt, 'add_mex_test\(([^)]*)\)', 'tokens');
    tests = struct('name', {}, 'file', {}, 'script', {}, 'will_fail', {});
    for i = 1:numel(lines)
        args = lines{i}{1};
        name = regexp(args, 'NAME\s+(\S+)', 'tokens', 'once');
        file = regexp(args, 'FILENAME\s+(\S+)', 'tokens', 'once');
        if isempty(name) || isempty(file)
            error('run_mex_tests:parse', 'cannot parse add_mex_test(%s)', args);
        end
        [~, script] = fileparts(file{1});
        tests(end+1) = struct('name', name{1}, 'file', file{1}, 'script', script, ...
                              'will_fail', ~isempty(regexp(args, '\<WILL_FAIL\>', 'once'))); %#ok<AGROW>
    end
end

function r = ternary(c, a, b)
    if c
        r = a;
    else
        r = b;
    end
end
