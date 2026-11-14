classdef Parallel
    % Parallel - thread budget of libKriging and how it is distributed between
    % its levels: multistart workers, OpenMP loops, BLAS threads
    % (docs/dev/parallel.md). Static methods; they modify global settings, to
    % set before the calls. Optim's thread settings are the same settings.

    methods (Static)
        function val = get_num_threads_percent()
            val = mLibKriging("Parallel::get_num_threads_percent");
        end

        % Thread budget: x <= 1 ratio, x > 1 percentage of the physical cores,
        % 0 automatic (100 %)
        function set_num_threads_percent(val)
            mLibKriging("Parallel::set_num_threads_percent", val);
        end

        % Resolved thread budget (number of threads), read only
        function val = get_num_threads()
            val = mLibKriging("Parallel::get_num_threads");
        end

        function val = get_thread_pool_size()
            val = mLibKriging("Parallel::get_thread_pool_size");
        end

        % Multistart workers (0: automatic, the budget)
        function set_thread_pool_size(val)
            mLibKriging("Parallel::set_thread_pool_size", int32(val));
        end

        function val = get_thread_start_delay_ms()
            val = mLibKriging("Parallel::get_thread_start_delay_ms");
        end

        function set_thread_start_delay_ms(val)
            mLibKriging("Parallel::set_thread_start_delay_ms", int32(val));
        end

        function val = get_blas_threads_min_n()
            val = mLibKriging("Parallel::get_blas_threads_min_n");
        end

        % Small-problem BLAS rule: -1 automatic, 0 off, n > 0 threshold
        function set_blas_threads_min_n(val)
            mLibKriging("Parallel::set_blas_threads_min_n", int32(val));
        end

        function val = get_omp_min_rows()
            val = mLibKriging("Parallel::get_omp_min_rows");
        end

        % Rows from which the covariance / gradient loops run on OpenMP threads
        function set_omp_min_rows(val)
            mLibKriging("Parallel::set_omp_min_rows", int32(val));
        end

        % Read only: 'mkl', 'openblas', 'accelerate' or 'unknown'; its thread count; OpenMP loops compiled in
        function val = blas_backend()
            val = mLibKriging("Parallel::blas_backend");
        end

        function val = blas_num_threads()
            val = mLibKriging("Parallel::blas_num_threads");
        end

        function val = omp_enabled()
            val = mLibKriging("Parallel::omp_enabled");
        end
    end
end
