function assert(cond, varargin)
% ASSERT  Octave's assert, for the mLibKriging tests run in MATLAB.
%
%   assert(cond)                     error if cond is not all true
%   assert(cond, errmsg, ...)        same, with an sprintf-formatted message
%   assert(observed, expected)       error unless observed equals expected
%   assert(observed, expected, tol)  numeric comparison: tol > 0 absolute,
%                                    tol < 0 relative (|tol| * |expected|)
%
% The tests in bindings/Octave/mlibkriging/tests use the comparison forms,
% which MATLAB's assert lacks (it only takes a condition and a message).
% tools/matlab/run_mex_tests.m puts this directory on the MATLAB path only;
% Octave keeps its own assert. Calls of the condition form behave as
% MATLAB's assert.

    if nargin == 1 || (nargin > 1 && islogical(cond) && (ischar(varargin{1}) || isstring(varargin{1})))
        if isempty(cond) || ~all(cond(:))
            if nargin == 1
                error('assert (cond) failed');
            end
            msg = char(varargin{1});
            if numel(varargin) > 1 && ~isempty(regexp(msg, '^\w+(:\w+)+$', 'once'))
                error(msg, char(varargin{2}), varargin{3:end}); % MATLAB form: assert(cond, errID, msg, ...)
            end
            error('Octave:assert', msg, varargin{2:end});
        end
        return
    end

    observed = cond;
    expected = varargin{1};
    if nargin > 2
        tol = varargin{2};
    else
        tol = 0;
    end

    if ischar(expected) || isstring(expected)
        if ~(ischar(observed) || isstring(observed))
            error('Octave:assert', 'ASSERT errors: observed is a %s, expected ''%s''', class(observed), char(expected));
        end
        if ~strcmp(char(observed), char(expected))
            error('Octave:assert', 'ASSERT errors: observed ''%s'', expected ''%s''', ...
                  char(observed), char(expected));
        end
        return
    end

    if iscell(expected) || isstruct(expected)
        if ~isequaln(observed, expected)
            error('Octave:assert', 'ASSERT errors: observed and expected %s differ', class(expected));
        end
        return
    end

    if ~isequal(size(observed), size(expected))
        error('Octave:assert', 'ASSERT errors: dimensions mismatch, observed %s, expected %s', ...
              mat2str(size(observed)), mat2str(size(expected)));
    end
    o = double(observed(:));
    e = double(expected(:));
    nan_o = isnan(o);
    if ~isequal(nan_o, isnan(e))
        error('Octave:assert', 'ASSERT errors: NaN positions differ');
    end
    o = o(~nan_o);
    e = e(~nan_o);
    err = abs(o - e);
    if tol == 0
        bad = o ~= e;
    elseif tol > 0
        bad = err > tol;
    else
        bad = err > abs(tol) * abs(e);
    end
    % equal infinities are equal (Inf - Inf is NaN)
    bad = bad & ~(isinf(o) & o == e);
    if any(bad)
        [worst, i] = max(err .* bad);
        error('Octave:assert', ['ASSERT errors: %d of %d values out of tolerance %g, ', ...
              'worst observed %.17g, expected %.17g (|diff| %g)'], ...
              nnz(bad), numel(bad), tol, o(i), e(i), worst);
    end
end
