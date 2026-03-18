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
}  // namespace OptimBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_OPTIM_BINDING_HPP
