classdef NuggetKriging < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = NuggetKriging(varargin)
            % printf("New NuggetKriging\n");
            obj.ref = mLibKriging("NuggetKriging::new", varargin{:});
        end
        
        function delete(obj, varargin)
            % disp(["ObjectRef = ", num2str(obj.ref)])
            % destroy the mex backend
            if ~isempty(obj.ref)
                % printf("Delete NuggetKriging\n")
                obj.ref = mLibKriging("NuggetKriging::delete", obj.ref, varargin{:})
            end
        end
        
        function fit(obj, varargin)
            mLibKriging("NuggetKriging::fit", obj.ref, varargin{:});
        end
        
        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::update", obj.ref, varargin{:});
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::summary", obj.ref, varargin{:});
        end

        function varargout = logLikelihood(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::logLikelihood", obj.ref, varargin{:});
        end

        function varargout = logMargPost(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::logMargPost", obj.ref, varargin{:});
        end

        function varargout = kernel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::kernel", obj.ref, varargin{:});
        end

        function varargout = optim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::optim", obj.ref, varargin{:});
        end

        function varargout = objective(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::objective", obj.ref, varargin{:});
        end

        function varargout = X(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::X", obj.ref, varargin{:});
        end

        function varargout = centerX(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::centerX", obj.ref, varargin{:});
        end

        function varargout = scaleX(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::scaleX", obj.ref, varargin{:});
        end

        function varargout = y(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::y", obj.ref, varargin{:});
        end

        function varargout = centerY(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::centerY", obj.ref, varargin{:});
        end

        function varargout = scaleY(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::scaleY", obj.ref, varargin{:});
        end

        function varargout = regmodel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::regmodel", obj.ref, varargin{:});
        end

        function varargout = F(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::F", obj.ref, varargin{:});
        end

        function varargout = T(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::T", obj.ref, varargin{:});
        end

        function varargout = M(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::M", obj.ref, varargin{:});
        end

        function varargout = z(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::z", obj.ref, varargin{:});
        end

        function varargout = beta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::beta", obj.ref, varargin{:});
        end

        function varargout = is_beta_estim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::is_beta_estim", obj.ref, varargin{:});
        end

        function varargout = theta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::theta", obj.ref, varargin{:});
        end

        function varargout = is_theta_estim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::is_theta_estim", obj.ref, varargin{:});
        end

        function varargout = sigma2(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::sigma2", obj.ref, varargin{:});
        end

        function varargout = is_sigma2_estim (obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::is_sigma2_estim ", obj.ref, varargin{:});
        end

        function varargout = nugget(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::nugget", obj.ref, varargin{:});
        end

        function varargout = is_nugget_estim (obj, varargin)
            [varargout{1:nargout}] = mLibKriging("NuggetKriging::is_nugget_estim ", obj.ref, varargin{:});
        end

    end
end
