// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Optim.hpp"

// [[Rcpp::export]]
bool optim_is_reparametrized() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->is_reparametrized();
}

// [[Rcpp::export]]
void optim_use_reparametrize(bool reparametrize) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->use_reparametrize(reparametrize);
}

// [[Rcpp::export]]
double optim_get_theta_lower_factor() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_theta_lower_factor();
}

// [[Rcpp::export]]
void optim_set_theta_lower_factor(double theta_lower_factor) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_theta_lower_factor(theta_lower_factor);
}

// [[Rcpp::export]]
double optim_get_theta_upper_factor() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_theta_upper_factor();
}

// [[Rcpp::export]]
void optim_set_theta_upper_factor(double theta_upper_factor) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_theta_upper_factor(theta_upper_factor);
}

// [[Rcpp::export]]
bool optim_variogram_bounds_heuristic_used() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->variogram_bounds_heuristic_used();
}

// [[Rcpp::export]]
void optim_use_variogram_bounds_heuristic(bool variogram_bounds_heuristic) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->use_variogram_bounds_heuristic(variogram_bounds_heuristic);
}

// [[Rcpp::export]]
void optim_set_log_level(int l) {
  Optim::set_log_level(l);
}

// [[Rcpp::export]]
int optim_get_log_level() {
  return Optim::get_log_level();
}

// [[Rcpp::export]]
int optim_get_max_iteration() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_max_iteration();
}

// [[Rcpp::export]]
void optim_set_max_iteration(int max_iteration) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_max_iteration(max_iteration);
}

// [[Rcpp::export]]
double optim_get_gradient_tolerance() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_gradient_tolerance();
}

// [[Rcpp::export]]
void optim_set_gradient_tolerance(double gradient_tolerance) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_gradient_tolerance(gradient_tolerance);
}

// [[Rcpp::export]]
double optim_get_objective_rel_tolerance() {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  return impl_ptr->get_objective_rel_tolerance();
}

// [[Rcpp::export]]
void optim_set_objective_rel_tolerance(double objective_rel_tolerance) {
  Optim* la = new Optim();
  Rcpp::XPtr<Optim> impl_ptr(la);
  impl_ptr->set_objective_rel_tolerance(objective_rel_tolerance);
}

// [[Rcpp::export]]
int optim_get_thread_start_delay_ms() {
  return Optim::get_thread_start_delay_ms();
}

// [[Rcpp::export]]
void optim_set_thread_start_delay_ms(int delay_ms) {
  Optim::set_thread_start_delay_ms(delay_ms);
}

// [[Rcpp::export]]
int optim_get_thread_pool_size() {
  return Optim::get_thread_pool_size();
}

// [[Rcpp::export]]
void optim_set_thread_pool_size(int pool_size) {
  Optim::set_thread_pool_size(pool_size);
}
