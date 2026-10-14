classdef Kriging < handle
    % attributes doc: https://fr.mathworks.com/help/matlab/matlab_oop/property-attributes.html
    properties %(NonCopyable, Transient, Access = protected, Hidden) 
        ref
    end

    methods
        function obj = Kriging(varargin)
            % fprintf("New Kriging\n");
            if nargin == 2 && ischar(varargin{1}) && strcmp(varargin{1}, '__ref__')
                obj.ref = varargin{2};  % wrap an existing mex reference (see load)
            else
                obj.ref = mLibKriging("Kriging::new", varargin{:});
            end
        end

        function k2 = copy(obj)
            ref_copy = mLibKriging("Kriging::copy", obj.ref);
            k2 = Kriging('__ref__', ref_copy);
        end
        
        function delete(obj, varargin)
            % disp(["ObjectRef = ", num2str(obj.ref)])
            % destroy the mex backend
            if ~isempty(obj.ref)
                % fprintf("Delete Kriging\n")
                obj.ref = mLibKriging("Kriging::delete", obj.ref, varargin{:});
            end
        end
        
        % fit(y, X, regmodel, normalize, optim, objective, parameters [, method])
        % (heterogeneous noise: fit(y, noise, X, ...)); constructor:
        % Kriging(y, X, kernel, regmodel, normalize, optim, objective,
        % parameters, noise_model, noise [, method]). method: "exact"
        % (default), "vecchia" / "vecchia(m)" (Vecchia approximation, m
        % neighbors, default 30) or "nystrom" / "nystrom(k)" (rank-k
        % Nystrom, default 50); an approximate fit keeps no exact
        % factorization; objectives "LL", "LOO" and "LMP" all have
        % approximate versions (noise-free models), "LL" also with the
        % noise models (both methods).
        % "LLVecchia(m)" / "LLNystrom(k)" objectives are deprecated aliases.
        function fit(obj, varargin)
            mLibKriging("Kriging::fit", obj.ref, varargin{:});
        end
        
        % predict(X, return_stdev, return_cov, return_deriv [, method]) and
        % simulate(nsim, seed, X, will_update [, method]): method "exact"
        % (default, needs a method "exact" fit), "vecchia" / "vecchia(m)"
        % (local Vecchia, usable after any fit) or "nystrom" (needs a
        % method "nystrom" fit); nothing is chosen implicitly.
        function varargout = predict(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::predict", obj.ref, varargin{:});
        end

        function varargout = simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::simulate", obj.ref, varargin{:});
        end

        function varargout = update(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::update", obj.ref, varargin{:});
        end

        % update_simulate(y_u, X_u) or update_simulate(y_u, noise_u, X_u)
        % (after simulate(..., will_update=true)), chosen by the number of
        % arguments whatever the noise model. With noise_u (noise variances
        % of the update points) the update points are noisy observations and
        % the with_noise draws of the last simulate are added again. A
        % heterogeneous model needs noise_u: update_simulate(y_u, X_u)
        % raises on it (as update(y_u, X_u)). fit / update drop the last
        % simulate: call simulate(..., will_update=true) again.
        function varargout = update_simulate(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::update_simulate", obj.ref, varargin{:});
        end

        function varargout = summary(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::summary", obj.ref, varargin{:});
        end

        function varargout = save(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::save", obj.ref, varargin{:});
        end

        function varargout = leaveOneOutFun(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::leaveOneOutFun", obj.ref, varargin{:});
        end

        function varargout = leaveOneOutVec(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::leaveOneOutVec", obj.ref, varargin{:});
        end

        function varargout = leaveOneOut(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::leaveOneOut", obj.ref, varargin{:});
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

        function varargout = nystrom_rank(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::nystrom_rank", obj.ref, varargin{:});
        end

        function varargout = vecchia_neighbors(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::vecchia_neighbors", obj.ref, varargin{:});
        end

        % fit method: "exact", "vecchia(m)" or "nystrom(k)"
        function varargout = method(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::method", obj.ref, varargin{:});
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

        function varargout = noise_model(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::noise_model", obj.ref, varargin{:});
        end

        function varargout = nugget(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::nugget", obj.ref, varargin{:});
        end

        function varargout = is_nugget_estim(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::is_nugget_estim", obj.ref, varargin{:});
        end

        function varargout = noise(obj, varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::noise", obj.ref, varargin{:});
        end

    end

    methods (Static = true)
        function obj = load(varargin)
            ref = mLibKriging("Kriging::load", varargin{:});
            obj = Kriging('__ref__', ref);
        end

        % Subset-of-data pre-fit reduction: select n_max rows of X
        % (k-means centroids snapped to the nearest real point, or a
        % uniform random subsample), returned as sorted 1-based
        % row-indices into X. Args: (X, n_max, method="kmeans", seed=123).
        function varargout = subsetOfData(varargin)
            [varargout{1:nargout}] = mLibKriging("Kriging::subsetOfData", varargin{:});
        end
    end
end
