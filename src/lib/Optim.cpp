// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Optim.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

bool Optim::reparametrize = true;

LIBKRIGING_EXPORT void Optim::set_reparametrize(bool do_reparametrize) {
  Optim::reparametrize = do_reparametrize;
};

LIBKRIGING_EXPORT bool Optim::get_reparametrize() {
  return Optim::reparametrize;
};

double Optim::theta_lower_factor = 0.05;

LIBKRIGING_EXPORT void Optim::set_theta_lower_factor_heuristic(double _theta_lower_factor) {
  Optim::theta_lower_factor = _theta_lower_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_lower_factor_heuristic() {
  return Optim::theta_lower_factor;
};

int Optim::log_level = 0;

LIBKRIGING_EXPORT void Optim::log(int l) {
  Optim::log_level = l;
};

int Optim::max_iteration = 20;

LIBKRIGING_EXPORT void Optim::set_max_iteration(int max_iteration_val) {
  Optim::max_iteration = max_iteration_val;
};

LIBKRIGING_EXPORT int Optim::get_max_iteration() {
  return Optim::max_iteration;
};

double Optim::gradient_tolerance = 0.01;

LIBKRIGING_EXPORT void Optim::set_gradient_tolerance(double gradient_tolerance_val) {
  Optim::gradient_tolerance = gradient_tolerance_val;
};

LIBKRIGING_EXPORT double Optim::get_gradient_tolerance() {
  return Optim::gradient_tolerance;
};

double Optim::objective_rel_tolerance = 0.001;

LIBKRIGING_EXPORT void Optim::set_objective_rel_tolerance(double objective_rel_tolerance_val) {
  Optim::objective_rel_tolerance = objective_rel_tolerance_val;
};

LIBKRIGING_EXPORT double Optim::get_objective_rel_tolerance() {
  return Optim::objective_rel_tolerance;
};