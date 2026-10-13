classdef NestedKriging < handle
    % Divide-and-conquer Kriging for large designs (see libKriging NestedKriging).
    % k = NestedKriging(y, X, kernel, nb_groups, [aggregation], [partition], [seed],
    %                   [regmodel], [optim], [objective], [parameters], [warping_cell], [method])
    % aggregation: "NK" (optimal, default) | "PoE" | "gPoE" | "BCM" | "rBCM"
    % objective  : "LL" (default) | "LOO" | "LMP"
    % method     : "exact" (default) | "vecchia(m)" | "nystrom(k)" (common prior
    %              from one global Vecchia / Nystrom fit; objective "LL", no warping)
    properties
        ref
    end

    methods
        function obj = NestedKriging(varargin)
            if nargin == 2 && ischar(varargin{1}) && strcmp(varargin{1}, '__ref__')
                obj.ref = varargin{2};
            else
                obj.ref = mLibKriging("NestedKriging::new", varargin{:});
            end
        end

        function delete(obj, varargin)
            if ~isempty(obj.ref)
                obj.ref = mLibKriging("NestedKriging::delete", obj.ref, varargin{:});
            end
        end

        function fit(obj, varargin)
            mLibKriging("NestedKriging::fit", obj.ref, varargin{:});
        end

        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::predict", obj.ref, varargin{:});
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::summary", obj.ref, varargin{:});
        end

        function varargout = kernel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::kernel", obj.ref, varargin{:});
        end

        function varargout = aggregation(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::aggregation", obj.ref, varargin{:});
        end

        function varargout = nb_groups(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::nb_groups", obj.ref, varargin{:});
        end

        function varargout = theta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::theta", obj.ref, varargin{:});
        end

        function varargout = sigma2(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::sigma2", obj.ref, varargin{:});
        end

        function varargout = beta0(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::beta0", obj.ref, varargin{:});
        end

        function disp(obj, varargin)
            disp(obj.summary());
        end

        function varargout = X(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::X", obj.ref, varargin{:});
        end

        function varargout = y(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::y", obj.ref, varargin{:});
        end

        function varargout = groups(obj, varargin)
            % cell array of 1-based row indices (one column vector per group)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::groups", obj.ref, varargin{:});
        end

        function varargout = warping(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::warping", obj.ref, varargin{:});
        end

        function varargout = objective(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::objective", obj.ref, varargin{:});
        end

        function varargout = method(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::method", obj.ref, varargin{:});
        end

        function set_predict_chunk(obj, chunk)
            mLibKriging("NestedKriging::set_predict_chunk", obj.ref, int32(chunk));
        end

        function set_warp_subsample(obj, m)
            mLibKriging("NestedKriging::set_warp_subsample", obj.ref, int32(m));
        end

        function varargout = save(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NestedKriging::save", obj.ref, varargin{:});
        end
    end

    methods (Static = true)
        function obj = load(varargin)
            ref = mLibKriging("NestedKriging::load", varargin{:});
            obj = NestedKriging('__ref__', ref);
        end
    end
end
