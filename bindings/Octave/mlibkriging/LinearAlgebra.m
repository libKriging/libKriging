classdef LinearAlgebra
    % LinearAlgebra - Settings of the Cholesky factorizations of libKriging
    %
    % Static methods; they modify global settings.
    %   chol_jitter: fixed jitter on the correlation diagonal of noise-free
    %     fits (< 0: automatic per kernel, the default; 0: disabled, former
    %     behaviour; > 0: this value for every kernel)
    %   num_nugget: numerical nugget of the Cholesky fallback (default 1e-10)

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
    end
end
