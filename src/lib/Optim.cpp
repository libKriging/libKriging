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

std::function<double(const double&)> Optim::reparam_to_ = [](const double& _theta) {
  return -std::log(_theta); 
};
std::function<arma::vec(const arma::vec&)> Optim::reparam_to = [](const arma::vec& _theta) {
  return -arma::log(_theta); 
};
std::function<double(const double&)> Optim::reparam_from_ = [](const double& _gamma) {
  return std::exp(-_gamma); 
};
std::function<arma::vec(const arma::vec&)> Optim::reparam_from = [](const arma::vec& _gamma) {
  return arma::exp(-_gamma); 
};
std::function<arma::vec(const arma::vec&, const arma::vec&)> Optim::reparam_from_deriv = [](const arma::vec& _theta, const arma::vec& _grad) { 
  return -_grad % _theta; 
};
std::function<arma::mat(const arma::vec&, const arma::vec&, const arma::mat&)> Optim::reparam_from_deriv2 = [](const arma::vec& _theta, const arma::vec& _grad, const arma::mat& _hess) {
  return _grad - _hess % _theta;
};

double Optim::theta_lower_factor = 0.02;

LIBKRIGING_EXPORT void Optim::set_theta_lower_factor(double _theta_lower_factor) {
  Optim::theta_lower_factor = _theta_lower_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_lower_factor() {
  return Optim::theta_lower_factor;
};

double Optim::theta_upper_factor = 10.0;

LIBKRIGING_EXPORT void Optim::set_theta_upper_factor(double _theta_upper_factor) {
  Optim::theta_upper_factor = _theta_upper_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_upper_factor() {
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
