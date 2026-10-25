#include "Optim_binding.hpp"

#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"

#include "tools/MxMapper.hpp"

namespace OptimBinding {

void is_reparametrized(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::is_reparametrized(), "is_reparametrized");
}

void use_reparametrize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::use_reparametrize(input.get<bool>(0, "reparametrize"));
}

void get_theta_lower_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_theta_lower_factor(), "theta_lower_factor");
}

void set_theta_lower_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_theta_lower_factor(input.get<double>(0, "theta_lower_factor"));
}

void get_theta_upper_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_theta_upper_factor(), "theta_upper_factor");
}

void set_theta_upper_factor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_theta_upper_factor(input.get<double>(0, "theta_upper_factor"));
}

void variogram_bounds_heuristic_used(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::variogram_bounds_heuristic_used(), "variogram_bounds_heuristic");
}

void use_variogram_bounds_heuristic(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::use_variogram_bounds_heuristic(input.get<bool>(0, "variogram_bounds_heuristic"));
}

void get_log_level(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_log_level(), "log_level");
}

void set_log_level(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_log_level(input.get<int>(0, "log_level"));
}

void get_max_iteration(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_max_iteration(), "max_iteration");
}

void set_max_iteration(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_max_iteration(input.get<int>(0, "max_iteration"));
}

void get_gradient_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_gradient_tolerance(), "gradient_tolerance");
}

void set_gradient_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_gradient_tolerance(input.get<double>(0, "gradient_tolerance"));
}

void get_objective_rel_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_objective_rel_tolerance(), "objective_rel_tolerance");
}

void set_objective_rel_tolerance(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_objective_rel_tolerance(input.get<double>(0, "objective_rel_tolerance"));
}

void get_thread_start_delay_ms(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_thread_start_delay_ms(), "thread_start_delay_ms");
}

void set_thread_start_delay_ms(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_thread_start_delay_ms(input.get<int>(0, "thread_start_delay_ms"));
}

void get_thread_pool_size(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_thread_pool_size(), "thread_pool_size");
}

void set_thread_pool_size(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_thread_pool_size(input.get<int>(0, "thread_pool_size"));
}

void get_num_threads_percent(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_num_threads_percent(), "num_threads_percent");
}

void set_num_threads_percent(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  Optim::set_num_threads_percent(input.get<double>(0, "num_threads_percent"));
}

void get_num_threads(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, Optim::get_num_threads(), "num_threads");
}

}  // namespace OptimBinding

namespace LinearAlgebraBinding {

void get_chol_jitter(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, LinearAlgebra::get_chol_jitter(), "chol_jitter");
}

void set_chol_jitter(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  LinearAlgebra::set_chol_jitter(input.get<double>(0, "chol_jitter"));
}

void get_num_nugget(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, LinearAlgebra::get_num_nugget(), "num_nugget");
}

void set_num_nugget(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  LinearAlgebra::set_num_nugget(input.get<double>(0, "num_nugget"));
}

// BLAS / LAPACK self-test (#429): [ok, library, report] = check_blas(int32 n_threads)
void check_blas(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{0, 1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 3}};
  const auto c = LinearAlgebra::check_blas(input.getOptional<int>(0, "n_threads").value_or(0));
  output.set(0, c.ok, "ok");
  output.setOptional(1, c.library, "library");
  output.setOptional(2, c.report, "report");
}

}  // namespace LinearAlgebraBinding
