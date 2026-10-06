classdef Gpu
    % Gpu - GPU acceleration of the iterative path (LLIterative fits and
    % predictIterative). Every other method always runs on the CPU.
    %
    % A GPU backend is only present if libKriging was built with one
    % (-DENABLE_CUDA_ITERATIVE=AUTO, the default, enables CUDA when a CUDA
    % toolkit is found at build time). When present, it is enabled by default
    % as soon as a usable device is found, unless the environment variable
    % LK_ITERATIVE_GPU is set to 0 before the first GPU query or iterative call.
    %
    %   Gpu.compiled_backends()  % 'cuda', 'hip', 'sycl', 'metal' (comma-separated) or ''
    %   Gpu.available()          % true iff a compiled-in backend found a device
    %   Gpu.backend()            % backend in use, or 'none' (CPU)
    %   Gpu.enabled()            % Gpu.backend() differs from 'none'
    %   Gpu.set_enabled(false)   % force the CPU path (true is ignored without a device)

    methods (Static)
        function val = compiled_backends()
            val = mLibKriging("Gpu::compiled_backends");
        end

        function val = available()
            val = mLibKriging("Gpu::available");
        end

        function val = backend()
            val = mLibKriging("Gpu::backend");
        end

        function val = enabled()
            val = mLibKriging("Gpu::enabled");
        end

        function set_enabled(val)
            mLibKriging("Gpu::set_enabled", logical(val));
        end
    end
end
