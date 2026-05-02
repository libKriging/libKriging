classdef NoiseKriging < handle
    % DEPRECATED: Use Kriging(y, X, kernel, ..., "heterogeneous", noise) instead.
    % This class is a compatibility wrapper that delegates to the unified Kriging class.
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = NoiseKriging(varargin)
            warning('NoiseKriging is deprecated. Use Kriging(y, X, kernel, ..., "heterogeneous", noise) instead.');
            % NoiseKriging(y, noise, X, kernel, [regmodel], [normalize], [optim], [objective], [parameters])
            % -> Kriging(y, X, kernel, [regmodel], [normalize], [optim], [objective], [parameters], "heterogeneous", noise)
            if nargin < 4
                error('NoiseKriging requires at least 4 arguments: y, noise, X, kernel');
            end
            y_val = varargin{1};
            noise_val = varargin{2};
            X_val = varargin{3};
            kernel_val = varargin{4};
            % Remap remaining optional args (indices 5..end in original become 5..end in new)
            rest = varargin(5:end);
            % Pad rest to 4 elements (regmodel, normalize, optim, objective) if needed
            % to ensure noise_model lands at position 9 and noise at position 10
            while length(rest) < 4
                rest{end+1} = [];
            end
            % Build new arg list: y, X, kernel, regmodel, normalize, optim, objective, [], "heterogeneous", noise
            args = {y_val, X_val, kernel_val, rest{:}, [], 'heterogeneous', noise_val};
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

        function varargout = noise(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::noise", obj.ref, varargin{:});
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
    end

    methods (Static = true)
        function obj = load(varargin)
            obj = Kriging([1], [1], "gauss") % TODO should find a more straightforward default ctor
            obj.ref = mLibKriging("Kriging::load", varargin{:});
        end
    end
end
