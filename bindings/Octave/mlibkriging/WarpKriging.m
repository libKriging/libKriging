classdef WarpKriging < handle
    % WarpKriging  Kriging with per-variable input warping
    %
    %   wk = WarpKriging(y, X, warping, kernel)
    %   wk = WarpKriging(y, X, warping, kernel, regmodel, normalize, optim, objective)
    %
    %   warping is a cell array of strings, e.g. {"kumaraswamy", "none", "categorical(5,2)"}
    %
    %   See also Kriging, NuggetKriging, NoiseKriging

    properties
        ref
    end

    methods
        function obj = WarpKriging(varargin)
            obj.ref = mLibKriging("WarpKriging::new", varargin{:});
        end

        function varargout = copy(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::copy", obj.ref, varargin{:});
        end

        function delete(obj, varargin)
            if ~isempty(obj.ref)
                obj.ref = mLibKriging("WarpKriging::delete", obj.ref, varargin{:});
            end
        end

        function fit(obj, varargin)
            mLibKriging("WarpKriging::fit", obj.ref, varargin{:});
        end

        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("WarpKriging::update", obj.ref, varargin{:});
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
    end
end
