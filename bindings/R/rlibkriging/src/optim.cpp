// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Optim.hpp"

bool optim_is_reparametrized() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->is_reparametrized();
}

void optim_use_reparametrize(bool reparametrize) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->use_reparametrize(reparametrize);
}

double optim_get_theta_lower_factor() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_theta_lower_factor();
}

void optim_set_theta_lower_factor(double theta_lower_factor) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_theta_lower_factor(theta_lower_factor);
}

double optim_get_theta_upper_factor() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_theta_upper_factor();
}

void optim_set_theta_upper_factor(double theta_upper_factor) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_theta_upper_factor(theta_upper_factor);
}

bool optim_variogram_bounds_heuristic_used() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->variogram_bounds_heuristic_used();
}

void optim_use_variogram_bounds_heuristic(bool variogram_bounds_heuristic) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->use_variogram_bounds_heuristic(variogram_bounds_heuristic);
}

void optim_log(int l) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->log(l);
}

int optim_get_max_iteration() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_max_iteration();
}

void optim_set_max_iteration(int max_iteration) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_max_iteration(max_iteration);
}

double optim_get_gradient_tolerance() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_gradient_tolerance();
}

void optim_set_gradient_tolerance(double gradient_tolerance) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_gradient_tolerance(gradient_tolerance);
}

double optim_get_objective_rel_tolerance() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_objective_rel_tolerance();
}

void optim_set_objective_rel_tolerance(double objective_rel_tolerance) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_objective_rel_tolerance(objective_rel_tolerance);
}
