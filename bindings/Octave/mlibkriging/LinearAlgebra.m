classdef LinearAlgebra
    % LinearAlgebra - Settings of the Cholesky factorizations of libKriging
    %
    % Static methods; they modify global settings.
    %   chol_jitter: fixed jitter on the correlation diagonal of noise-free
    %     fits (< 0: automatic per kernel, the default; 0: disabled, former
    %     behaviour; > 0: this value for every kernel)
    %   num_nugget: numerical nugget of the Cholesky fallback (default 1e-10)
    %   chol_warning: warn when the fallback adds a numerical nugget
    %   check_chol_rcond: check the reciprocal condition number of the
    %     factorizations
    % Self-test of the linear algebra backend: see linearalgebra_check.

    methods (Static)
        function val = get_chol_jitter()
            val = mLibKriging("LinearAlgebra::get_chol_jitter");
        end

        function set_chol_jitter(val)
            mLibKriging("LinearAlgebra::set_chol_jitter", val);
        end

        function val = get_num_nugget()
            val = mLibKriging("LinearAlgebra::get_num_nugget");
        end

        function set_num_nugget(val)
            mLibKriging("LinearAlgebra::set_num_nugget", val);
        end

        function set_chol_warning(warn)
            mLibKriging("LinearAlgebra::set_chol_warning", logical(warn));
        end

        function check_chol_rcond(check)
            mLibKriging("LinearAlgebra::check_chol_rcond", logical(check));
        end

        function val = chol_rcond_checked()
            val = mLibKriging("LinearAlgebra::chol_rcond_checked");
        end

        % Iterative method (method "iterative(...)"): warning of a CG solve
        % stopped before its tolerance, automatic Nystrom preconditioner of
        % its predictor, host-memory budget (MiB) of the dense correlation
        % matrix (0: matrix-free)
        function set_cg_warning(warn)
            mLibKriging("LinearAlgebra::set_cg_warning", logical(warn));
        end

        function set_cg_auto_precond(on)
            mLibKriging("LinearAlgebra::set_cg_auto_precond", logical(on));
        end

        function val = get_iterative_dense_max_mb()
            val = mLibKriging("LinearAlgebra::get_iterative_dense_max_mb");
        end

        function set_iterative_dense_max_mb(val)
            mLibKriging("LinearAlgebra::set_iterative_dense_max_mb", int32(val));
        end
    end
end
