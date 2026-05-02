classdef NuggetKriging < handle
    % DEPRECATED: Use Kriging(..., "nugget") instead.
    % This class is a compatibility wrapper that delegates to the unified Kriging class.
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = NuggetKriging(varargin)
            warning('NuggetKriging is deprecated. Use Kriging(..., "nugget") instead.');
            % NuggetKriging(y, X, kernel, ...) -> Kriging(y, X, kernel, ..., "nugget")
            % Find where to insert the noise_model arg (after parameters or after the last provided optional)
            % Original args: y, X, kernel, [regmodel], [normalize], [optim], [objective], [parameters]
            % New args:      y, X, kernel, [regmodel], [normalize], [optim], [objective], [parameters], "nugget"
            args = varargin;
            args{end+1} = 'nugget';
            obj.ref = mLibKriging("Kriging::new", args{:});
        end

        function varargout = copy(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::copy", obj.ref, varargin{:});
        end

        function delete(obj, varargin)
            if ~isempty(obj.ref)
                obj.ref = mLibKriging("Kriging::delete", obj.ref, varargin{:});
            end
        end
        
        function fit(obj, varargin)
            mLibKriging("Kriging::fit", obj.ref, varargin{:});
        end
        
        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::update", obj.ref, varargin{:});
        end

        function varargout = update_simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::update_simulate", obj.ref, varargin{:});
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::summary", obj.ref, varargin{:});
        end

        function varargout = save(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::save", obj.ref, varargin{:});
        end

        function varargout = logLikelihoodFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::logLikelihoodFun", obj.ref, varargin{:});
        end

        function varargout = logLikelihood(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::logLikelihood", obj.ref, varargin{:});
        end
        
        function varargout = logMargPostFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::logMargPostFun", obj.ref, varargin{:});
        end

        function varargout = logMargPost(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::logMargPost", obj.ref, varargin{:});
        end

        function varargout = covMat(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::covMat", obj.ref, varargin{:});
        end

        function varargout = model(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::model", obj.ref, varargin{:});
        end

        function varargout = kernel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::kernel", obj.ref, varargin{:});
        end

        function varargout = optim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::optim", obj.ref, varargin{:});
        end

        function varargout = objective(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::objective", obj.ref, varargin{:});
        end

        function varargout = X(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::X", obj.ref, varargin{:});
        end

        function varargout = centerX(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::centerX", obj.ref, varargin{:});
        end

        function varargout = scaleX(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::scaleX", obj.ref, varargin{:});
        end

        function varargout = y(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::y", obj.ref, varargin{:});
        end

        function varargout = centerY(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::centerY", obj.ref, varargin{:});
        end

        function varargout = scaleY(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::scaleY", obj.ref, varargin{:});
        end

        function varargout = normalize(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::normalize", obj.ref, varargin{:});
        end

        function varargout = regmodel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::regmodel", obj.ref, varargin{:});
        end

        function varargout = F(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::F", obj.ref, varargin{:});
        end

        function varargout = T(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::T", obj.ref, varargin{:});
        end

        function varargout = M(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::M", obj.ref, varargin{:});
        end

        function varargout = z(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::z", obj.ref, varargin{:});
        end

        function varargout = beta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::beta", obj.ref, varargin{:});
        end

        function varargout = is_beta_estim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::is_beta_estim", obj.ref, varargin{:});
        end

        function varargout = theta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::theta", obj.ref, varargin{:});
        end

        function varargout = is_theta_estim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::is_theta_estim", obj.ref, varargin{:});
        end

        function varargout = sigma2(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::sigma2", obj.ref, varargin{:});
        end

        function varargout = is_sigma2_estim (obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::is_sigma2_estim ", obj.ref, varargin{:});
        end

        function varargout = nugget(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::nugget", obj.ref, varargin{:});
        end

        function varargout = is_nugget_estim (obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::is_nugget_estim", obj.ref, varargin{:});
        end
    end

    methods (Static = true)
        function obj = load(varargin)
            obj = Kriging([1], [1], "gauss") % TODO should find a more straightforward default ctor
            obj.ref = mLibKriging("Kriging::load", varargin{:});
        end
    end
end
