classdef WarpKriging < handle
    % WarpKriging  Kriging with per-variable input warping
    %
    %   wk = WarpKriging(y, X, warping, kernel)
    %   wk = WarpKriging(y, X, warping, kernel, regmodel, normalize, optim, objective)
    %
    %   warping is a cell array of strings, e.g. {"kumaraswamy", "none", "categorical(5,2)"}
    %
    %   X can be a numeric matrix or a cell array of columns.  When a column is
    %   a cell array of strings (e.g. {"red";"blue";"red"}), it is automatically
    %   encoded as integers 0..L-1 and the warping spec is rewritten to include
    %   level names.
    %
    %   See also Kriging

    properties
        ref
    end

    methods (Static, Access = private)
        function [X_num, warping_out] = encode_string_columns(X, warping)
            % Detect cell-of-string columns, encode to integers, rewrite warping.
            if iscell(X)
                n = numel(X{1});
                d = numel(X);
            else
                X_num = double(X);
                warping_out = warping;
                return;
            end

            if ischar(warping)
                warping = {warping};
            end
            warping_out = warping;
            X_num = zeros(n, d);

            for j = 1:d
                col = X{j};
                if iscell(col) && ischar(col{1})
                    % String column: build sorted label list
                    labels = sort(unique(col));
                    for i = 1:n
                        idx = find(strcmp(labels, col{i}));
                        X_num(i, j) = idx - 1;  % 0-based
                    end

                    % Rewrite warping spec
                    spec = strtrim(warping_out{j});
                    names_str = ['[' strjoin(cellfun(@(s) ['"' s '"'], labels, 'UniformOutput', false), ',') ']'];

                    if strncmpi(spec, 'categorical', 11)
                        embed_dim = 2;
                        tok = regexp(spec, '\(([^)]*)\)', 'tokens');
                        if ~isempty(tok) && ~isempty(tok{1}{1})
                            parts = strsplit(tok{1}{1}, ',');
                            if numel(parts) >= 2
                                embed_dim = str2double(parts{end});
                            end
                        end
                        warping_out{j} = ['categorical(' names_str ',' num2str(embed_dim) ')'];
                    elseif strncmpi(spec, 'ordinal', 7)
                        warping_out{j} = ['ordinal(' names_str ')'];
                    else
                        error('Column %d contains strings but warping spec ''%s'' is not categorical or ordinal', j, spec);
                    end
                else
                    X_num(:, j) = double(col(:));
                end
            end
        end

        function X_out = encode_X_from_warping(X, warping_specs)
            % Encode X using existing warping specs (for predict/simulate/update).
            if iscell(X)
                has_str = false;
                for j = 1:numel(X)
                    if iscell(X{j}) && ischar(X{j}{1})
                        has_str = true;
                        break;
                    end
                end
                if has_str
                    [X_out, ~] = WarpKriging.encode_string_columns(X, warping_specs);
                else
                    n = numel(X{1});
                    d = numel(X);
                    X_out = zeros(n, d);
                    for j = 1:d
                        X_out(:, j) = double(X{j}(:));
                    end
                end
            else
                X_out = double(X);
            end
        end
    end

    methods
        function obj = WarpKriging(varargin)
            if nargin == 2 && ischar(varargin{1}) && strcmp(varargin{1}, '__ref__')
                obj.ref = varargin{2};
            else
                % Check if X (arg 2) has string columns
                if nargin >= 3 && iscell(varargin{2})
                    [X_enc, warping_enc] = WarpKriging.encode_string_columns(varargin{2}, varargin{3});
                    args = [{varargin{1}, X_enc, warping_enc}, varargin(4:end)];
                    obj.ref = mLibKriging("WarpKriging::new", args{:});
                else
                    obj.ref = mLibKriging("WarpKriging::new", varargin{:});
                end
            end
        end

        function k2 = copy(obj)
            ref_copy = mLibKriging("WarpKriging::copy", obj.ref);
            k2 = WarpKriging('__ref__', ref_copy);
        end

        function delete(obj, varargin)
            if ~isempty(obj.ref)
                obj.ref = mLibKriging("WarpKriging::delete", obj.ref, varargin{:});
            end
        end

        function fit(obj, varargin)
            % fit(obj, y, X, ...)
            if nargin >= 3 && iscell(varargin{2})
                ws = mLibKriging("WarpKriging::warping", obj.ref);
                X_enc = WarpKriging.encode_X_from_warping(varargin{2}, ws);
                args = [{varargin{1}, X_enc}, varargin(3:end)];
                mLibKriging("WarpKriging::fit", obj.ref, args{:});
            else
                mLibKriging("WarpKriging::fit", obj.ref, varargin{:});
            end
        end

        function varargout = predict(obj, varargin)
            % predict(obj, X, ...)
            if nargin >= 2 && iscell(varargin{1})
                ws = mLibKriging("WarpKriging::warping", obj.ref);
                X_enc = WarpKriging.encode_X_from_warping(varargin{1}, ws);
                args = [{X_enc}, varargin(2:end)];
                [varargout{1:nargout}] = mLibKriging("WarpKriging::predict", obj.ref, args{:});
            else
                [varargout{1:nargout}] = mLibKriging("WarpKriging::predict", obj.ref, varargin{:});
            end
        end

        function varargout = simulate(obj, varargin)
            % simulate(obj, nsim, seed, X, ...)
            if nargin >= 4 && iscell(varargin{3})
                ws = mLibKriging("WarpKriging::warping", obj.ref);
                X_enc = WarpKriging.encode_X_from_warping(varargin{3}, ws);
                args = [varargin(1:2), {X_enc}, varargin(4:end)];
                [varargout{1:nargout}] = mLibKriging("WarpKriging::simulate", obj.ref, args{:});
            else
                [varargout{1:nargout}] = mLibKriging("WarpKriging::simulate", obj.ref, varargin{:});
            end
        end

        function varargout = update_simulate(obj, varargin)
            % update_simulate(obj, y_u, X_u)
            if nargin >= 3 && iscell(varargin{2})
                ws = mLibKriging("WarpKriging::warping", obj.ref);
                X_enc = WarpKriging.encode_X_from_warping(varargin{2}, ws);
                [varargout{1:nargout}] = mLibKriging("WarpKriging::update_simulate", obj.ref, varargin{1}, X_enc);
            else
                [varargout{1:nargout}] = mLibKriging("WarpKriging::update_simulate", obj.ref, varargin{:});
            end
        end

        function varargout = update(obj, varargin)
            % update(obj, y_u, X_u, ...)
            if nargin >= 3 && iscell(varargin{2})
                ws = mLibKriging("WarpKriging::warping", obj.ref);
                X_enc = WarpKriging.encode_X_from_warping(varargin{2}, ws);
                args = [{varargin{1}, X_enc}, varargin(3:end)];
                [varargout{1:nargout}] = mLibKriging("WarpKriging::update", obj.ref, args{:});
            else
                [varargout{1:nargout}] = mLibKriging("WarpKriging::update", obj.ref, varargin{:});
            end
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::summary", obj.ref, varargin{:});
        end

        function varargout = logLikelihoodFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::logLikelihoodFun", obj.ref, varargin{:});
        end

        function varargout = logLikelihood(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::logLikelihood", obj.ref, varargin{:});
        end

        function varargout = kernel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::kernel", obj.ref, varargin{:});
        end

        function varargout = X(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::X", obj.ref, varargin{:});
        end

        function varargout = y(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::y", obj.ref, varargin{:});
        end

        function varargout = theta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::theta", obj.ref, varargin{:});
        end

        function varargout = sigma2(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::sigma2", obj.ref, varargin{:});
        end

        function varargout = is_fitted(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::is_fitted", obj.ref, varargin{:});
        end

        function varargout = feature_dim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::feature_dim", obj.ref, varargin{:});
        end

        function varargout = warping(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::warping", obj.ref, varargin{:});
        end

        function varargout = save(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::save", obj.ref, varargin{:});
        end
    end

    methods (Static = true)
        function obj = load(varargin)
            ref = mLibKriging("WarpKriging::load", varargin{:});
            obj = WarpKriging('__ref__', ref);
        end
    end
end
