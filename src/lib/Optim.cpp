// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Optim.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

bool Optim::reparametrize = true;

LIBKRIGING_EXPORT void Optim::use_reparametrize(bool do_reparametrize) {
  Optim::reparametrize = do_reparametrize;
};

LIBKRIGING_EXPORT bool Optim::is_reparametrized() {
  return Optim::reparametrize;
};

std::function<float(const float&)> Optim::reparam_to_ = [](const float& _theta) { return -std::log(_theta); };
std::function<arma::fvec(const arma::fvec&)> Optim::reparam_to
    = [](const arma::fvec& _theta) { return arma::conv_to<arma::fcolvec>::from(-arma::log(_theta)); };
std::function<float(const float&)> Optim::reparam_from_ = [](const float& _gamma) { return std::exp(-_gamma); };
std::function<arma::fvec(const arma::fvec&)> Optim::reparam_from
    = [](const arma::fvec& _gamma) { return arma::conv_to<arma::fcolvec>::from(arma::exp(-_gamma)); };
std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> Optim::reparam_from_deriv =
    [](const arma::fvec& _theta, const arma::fvec& _grad) { return arma::conv_to<arma::fcolvec>::from(-_grad % _theta); };
std::function<arma::fmat(const arma::fvec&, const arma::fvec&, const arma::fmat&)> Optim::reparam_from_deriv2
    = [](const arma::fvec& _theta, const arma::fvec& _grad, const arma::fmat& _hess) {
        return arma::conv_to<arma::fmat>::from(_grad - _hess % _theta);
      };

float Optim::theta_lower_factor = 0.02;

LIBKRIGING_EXPORT void Optim::set_theta_lower_factor(float _theta_lower_factor) {
  Optim::theta_lower_factor = _theta_lower_factor;
};

LIBKRIGING_EXPORT float Optim::get_theta_lower_factor() {
  return Optim::theta_lower_factor;
};

float Optim::theta_upper_factor = 10.0;

LIBKRIGING_EXPORT void Optim::set_theta_upper_factor(float _theta_upper_factor) {
  Optim::theta_upper_factor = _theta_upper_factor;
};

LIBKRIGING_EXPORT float Optim::get_theta_upper_factor() {
  return Optim::theta_upper_factor;
};

bool Optim::variogram_bounds_heuristic = true;

LIBKRIGING_EXPORT void Optim::use_variogram_bounds_heuristic(bool _variogram_bounds_heuristic) {
  Optim::variogram_bounds_heuristic = _variogram_bounds_heuristic;
};

LIBKRIGING_EXPORT bool Optim::variogram_bounds_heuristic_used() {
  return Optim::variogram_bounds_heuristic;
};

int Optim::log_level = 0;

LIBKRIGING_EXPORT void Optim::log(int l) {
  Optim::log_level = l;
};

int Optim::max_restart = 10;

int Optim::max_iteration = 20;

LIBKRIGING_EXPORT void Optim::set_max_iteration(int max_iteration_val) {
  Optim::max_iteration = max_iteration_val;
};

LIBKRIGING_EXPORT int Optim::get_max_iteration() {
  return Optim::max_iteration;
};

float Optim::gradient_tolerance = 0.001;

LIBKRIGING_EXPORT void Optim::set_gradient_tolerance(float gradient_tolerance_val) {
  Optim::gradient_tolerance = gradient_tolerance_val;
};

LIBKRIGING_EXPORT float Optim::get_gradient_tolerance() {
  return Optim::gradient_tolerance;
};

float Optim::objective_rel_tolerance = 0.001;

LIBKRIGING_EXPORT void Optim::set_objective_rel_tolerance(float objective_rel_tolerance_val) {
  Optim::objective_rel_tolerance = objective_rel_tolerance_val;
};

LIBKRIGING_EXPORT float Optim::get_objective_rel_tolerance() {
  return Optim::objective_rel_tolerance;
};
