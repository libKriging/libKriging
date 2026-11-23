classdef Gpu
    % Gpu - GPU backend of the iterative method of libKriging (method
    % "iterative(...)"; Gpu.hpp, docs/dev/AllCMakeOptions.md). Static methods;
    % they modify global settings, to set before the calls. A CPU-only build
    % (the default packages) reports no backend and its settings have no
    % effect.

    methods (Static)
        % Read only: compiled backends ("cuda,hip..." or ""), a usable device,
        % backend in use ("none" on the CPU path)
        function val = compiled_backends()
            val = mLibKriging("Gpu::compiled_backends");
        end

        function val = available()
            val = mLibKriging("Gpu::available");
        end

        function val = active_backend()
            val = mLibKriging("Gpu::active_backend");
        end

        % Use the GPU when available (default true; false when LK_ITERATIVE_GPU=0)
        function val = get_enabled()
            val = mLibKriging("Gpu::get_enabled");
        end

        function set_enabled(val)
            mLibKriging("Gpu::set_enabled", logical(val));
        end

        % Device-memory budget (MiB) of the dense correlation matrix, 0: matrix-free
        function val = get_dense_max_mb()
            val = mLibKriging("Gpu::get_dense_max_mb");
        end

        function set_dense_max_mb(val)
            mLibKriging("Gpu::set_dense_max_mb", int32(val));
        end

        % TF32 / fp32 products of the dense CG matvec (CUDA)
        function val = get_mixed_precision()
            val = mLibKriging("Gpu::get_mixed_precision");
        end

        function set_mixed_precision(val)
            mLibKriging("Gpu::set_mixed_precision", logical(val));
        end

        % Diagnostic of each GPU CG call
        function val = get_verbose()
            val = mLibKriging("Gpu::get_verbose");
        end

        function set_verbose(val)
            mLibKriging("Gpu::set_verbose", logical(val));
        end
    end
end
