classdef MLPKriging < handle
    % MLPKriging  Kriging with a joint MLP feature map (Deep Kernel Learning).
    %
    %   mk = MLPKriging(y, X, hidden_dims, d_out, activation, kernel)
    %   mk = MLPKriging(y, X, hidden_dims, d_out, activation, kernel, ...
    %                   regmodel, normalize, optim, objective)
    %
    %   hidden_dims is a numeric vector of hidden layer sizes, e.g. [16 8].
    %
    %   See also WarpKriging, Kriging, NuggetKriging, NoiseKriging

    properties
        ref
    end

    methods
        function obj = MLPKriging(varargin)
            if nargin == 2 && ischar(varargin{1}) && strcmp(varargin{1}, '__ref__')
                obj.ref = varargin{2};
            else
                obj.ref = mLibKriging("MLPKriging::new", varargin{:});
            end
        end

        function delete(obj, varargin)
            if ~isempty(obj.ref)
                obj.ref = mLibKriging("MLPKriging::delete", obj.ref, varargin{:});
            end
        end

        function fit(obj, varargin)
            mLibKriging("MLPKriging::fit", obj.ref, varargin{:});
        end

        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::update", obj.ref, varargin{:});
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::summary", obj.ref, varargin{:});
        end

        function varargout = logLikelihoodFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::logLikelihoodFun", obj.ref, varargin{:});
        end

        function varargout = logLikelihood(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::logLikelihood", obj.ref, varargin{:});
        end

        function varargout = kernel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::kernel", obj.ref, varargin{:});
        end

        function varargout = X(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::X", obj.ref, varargin{:});
        end

        function varargout = y(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::y", obj.ref, varargin{:});
        end

        function varargout = theta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::theta", obj.ref, varargin{:});
        end

        function varargout = sigma2(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::sigma2", obj.ref, varargin{:});
        end

        function varargout = is_fitted(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::is_fitted", obj.ref, varargin{:});
        end

        function varargout = feature_dim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::feature_dim", obj.ref, varargin{:});
        end

        function varargout = hidden_dims(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::hidden_dims", obj.ref, varargin{:});
        end

        function varargout = activation(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::activation", obj.ref, varargin{:});
        end

        function k2 = copy(obj)
            ref_copy = mLibKriging("MLPKriging::copy", obj.ref);
            k2 = MLPKriging('__ref__', ref_copy);
        end

        function varargout = save(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MLPKriging::save", obj.ref, varargin{:});
        end
    end

    methods (Static = true)
        function obj = load(varargin)
            ref = mLibKriging("MLPKriging::load", varargin{:});
            obj = MLPKriging('__ref__', ref);
        end
    end
end
