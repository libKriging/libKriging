#ifndef LIBKRIGING_BINDINGS_OCTAVE_OPTIM_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_OPTIM_BINDING_HPP

#include <mex.h>

namespace OptimBinding {
void is_reparametrized(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void use_reparametrize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_theta_lower_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_theta_lower_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_theta_upper_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_theta_upper_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void variogram_bounds_heuristic_used(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void use_variogram_bounds_heuristic(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_log_level(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_log_level(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_max_iteration(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_max_iteration(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_gradient_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_gradient_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_objective_rel_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_objective_rel_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_thread_start_delay_ms(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_thread_start_delay_ms(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_thread_pool_size(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_thread_pool_size(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_num_threads_percent(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_num_threads_percent(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_num_threads(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace OptimBinding

namespace LinearAlgebraBinding {
void get_chol_jitter(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_chol_jitter(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_num_nugget(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_num_nugget(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_chol_warning(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void check_chol_rcond(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void chol_rcond_checked(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void check(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_cg_warning(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_cg_auto_precond(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_iterative_dense_max_mb(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_iterative_dense_max_mb(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace LinearAlgebraBinding

namespace ParallelBinding {
void get_num_threads_percent(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_num_threads_percent(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_num_threads(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_thread_pool_size(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_thread_pool_size(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_thread_start_delay_ms(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_thread_start_delay_ms(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_blas_threads_min_n(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_blas_threads_min_n(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_omp_min_rows(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_omp_min_rows(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void blas_backend(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void blas_num_threads(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void omp_enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace ParallelBinding

namespace GpuBinding {
void compiled_backends(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void available(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void active_backend(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_dense_max_mb(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_dense_max_mb(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_mixed_precision(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_mixed_precision(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void get_verbose(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_verbose(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace GpuBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_OPTIM_BINDING_HPP
