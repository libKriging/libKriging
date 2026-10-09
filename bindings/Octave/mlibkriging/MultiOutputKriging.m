classdef MultiOutputKriging < handle
    % Multi-output Kriging on an isotopic design (see libKriging MultiOutputKriging).
    % k = MultiOutputKriging(Y, X, kernel, [output_model], [regmodel], [normalize], [optim],
    %                        [objective], [parameters], [output_coordinates])
    % k = MultiOutputKriging(kernel, [output_model])   % not fitted, then k.fit(Y, X, ...)
    % Y is n x q (one column per output), X is n x d.
    % output_model: "pca" | "pca(K)" | "pca(v)" (Karhunen-Loeve, default "pca" = "pca(0.99)")
    %             | "shared" (one theta, outputs independent) | "separable" (ICM, free q x q Sigma)
    % parameters  : Params("theta", theta0, "is_theta_estim", false)
    % [mean, stdev, cov, mean_deriv] = k.predict(X_n)  -> m x q, m x q, mq x mq, m x d x q
    % sims = k.simulate(int32(nsim), int32(seed), X_n, [will_update])  -> m x q x nsim
    properties
        ref
    end

    methods
        function obj = MultiOutputKriging(varargin)
            if nargin == 2 && ischar(varargin{1}) && strcmp(varargin{1}, '__ref__')
                obj.ref = varargin{2};
            elseif nargin >= 1 && (ischar(varargin{1}) || isstring(varargin{1}))
                obj.ref = mLibKriging("MultiOutputKriging::new_empty", varargin{:});
            else
                obj.ref = mLibKriging("MultiOutputKriging::new", varargin{:});
            end
        end

        function delete(obj, varargin)
            if ~isempty(obj.ref)
                obj.ref = mLibKriging("MultiOutputKriging::delete", obj.ref, varargin{:});
            end
        end

        function fit(obj, varargin)
            mLibKriging("MultiOutputKriging::fit", obj.ref, varargin{:});
        end

        function set_output_coordinates(obj, varargin)
            mLibKriging("MultiOutputKriging::set_output_coordinates", obj.ref, varargin{:});
        end

        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update_simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::update_simulate", obj.ref, varargin{:});
        end

        function update(obj, varargin)
            mLibKriging("MultiOutputKriging::update", obj.ref, varargin{:});
        end

        function varargout = leaveOneOutMat(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::leaveOneOutMat", obj.ref, varargin{:});
        end

        function varargout = leaveOneOut(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::leaveOneOut", obj.ref, varargin{:});
        end

        function varargout = logLikelihood(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::logLikelihood", obj.ref, varargin{:});
        end

        function varargout = logLikelihoodFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::logLikelihoodFun", obj.ref, varargin{:});
        end

        function varargout = leaveOneOutFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::leaveOneOutFun", obj.ref, varargin{:});
        end

        function varargout = predictCovFactors(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::predictCovFactors", obj.ref, varargin{:});
        end

        % Copy of latent Kriging number k (1-based), as a Kriging object
        function km = component(obj, k)
            km = Kriging('__ref__', mLibKriging("MultiOutputKriging::component", obj.ref, k));
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::summary", obj.ref, varargin{:});
        end

        function varargout = kernel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::kernel", obj.ref, varargin{:});
        end

        function varargout = output_model(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::output_model", obj.ref, varargin{:});
        end

        function varargout = nb_outputs(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::nb_outputs", obj.ref, varargin{:});
        end

        function varargout = X(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::X", obj.ref, varargin{:});
        end

        function varargout = Y(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::Y", obj.ref, varargin{:});
        end

        function varargout = output_coordinates(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::output_coordinates", obj.ref, varargin{:});
        end

        function varargout = regmodel(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::regmodel", obj.ref, varargin{:});
        end

        function varargout = normalize(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::normalize", obj.ref, varargin{:});
        end

        function varargout = optim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::optim", obj.ref, varargin{:});
        end

        function varargout = objective(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::objective", obj.ref, varargin{:});
        end

        function varargout = centerY(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::centerY", obj.ref, varargin{:});
        end

        function varargout = scaleY(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::scaleY", obj.ref, varargin{:});
        end

        function varargout = theta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::theta", obj.ref, varargin{:});
        end

        function varargout = sigma2(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::sigma2", obj.ref, varargin{:});
        end

        function varargout = beta(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::beta", obj.ref, varargin{:});
        end

        function varargout = output_cov(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::output_cov", obj.ref, varargin{:});
        end

        function varargout = nb_components(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::nb_components", obj.ref, varargin{:});
        end

        function varargout = pca_basis(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::pca_basis", obj.ref, varargin{:});
        end

        function varargout = pca_explained(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::pca_explained", obj.ref, varargin{:});
        end

        function varargout = pca_residual(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("MultiOutputKriging::pca_residual", obj.ref, varargin{:});
        end

        function disp(obj, varargin)
            disp(obj.summary());
        end
    end
end
