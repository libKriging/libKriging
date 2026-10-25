classdef LinearAlgebra
    % LinearAlgebra - Settings of the Cholesky factorizations of libKriging
    %
    % Static methods; they modify global settings.
    %   chol_jitter: fixed jitter on the correlation diagonal of noise-free
    %     fits (< 0: automatic per kernel, the default; 0: disabled, former
    %     behaviour; > 0: this value for every kernel)
    %   num_nugget: numerical nugget of the Cholesky fallback (default 1e-10)
    %   check_blas: self-test of the BLAS / LAPACK routines libKriging relies
    %     on, from one thread then n_threads concurrent threads (default 0:
    %     hardware concurrency); [ok, library, report] = LinearAlgebra.check_blas()

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

        function [ok, library, report] = check_blas(n_threads)
            if nargin < 1
                n_threads = 0;
            end
            [ok, library, report] = mLibKriging("LinearAlgebra::check_blas", int32(n_threads));
        end
    end
end
