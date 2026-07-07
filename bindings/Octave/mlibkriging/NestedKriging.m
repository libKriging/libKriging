classdef NestedKriging < handle
    % Divide-and-conquer Kriging for large designs (see libKriging NestedKriging).
    % k = NestedKriging(y, X, kernel, nb_groups, [aggregation], [partition], [seed],
    %                   [regmodel], [optim], [objective], [parameters], [warping_cell])
    % aggregation: "NK" (optimal, default) | "PoE" | "gPoE" | "BCM" | "rBCM"
    % objective  : "LL" (default) | "VLL(m)" (common prior from one global Vecchia fit)
    properties
        ref
    end

    methods
        function obj = NestedKriging(varargin)
            obj.ref = mLibKriging("NestedKriging::new", varargin{:});
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
    end
end
