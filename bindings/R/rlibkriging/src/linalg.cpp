// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/LinearAlgebra.hpp"

// [[Rcpp::export]]
double linalg_get_num_nugget() {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->get_num_nugget();
}

// [[Rcpp::export]]
void linalg_set_num_nugget(double nugget) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  impl_ptr->set_num_nugget(nugget);
}

// Fixed Cholesky jitter of noise-free fits: < 0 automatic per kernel (default),
// 0 disabled (former behaviour), > 0 this value for every kernel
// [[Rcpp::export]]
double linalg_get_chol_jitter() {
  return LinearAlgebra::get_chol_jitter();
}

// [[Rcpp::export]]
void linalg_set_chol_jitter(double jitter) {
  LinearAlgebra::set_chol_jitter(jitter);
}

// [[Rcpp::export]]
void linalg_check_chol_rcond(bool cr) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  impl_ptr->check_chol_rcond(cr);
}

// [[Rcpp::export]]
bool linalg_chol_rcond_checked() {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->chol_rcond_checked();
}

// [[Rcpp::export]]
arma::mat linalg_chol_safe(arma::mat X) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->safe_chol_lower(X);
}

// [[Rcpp::export]]
void linalg_set_chol_warning(bool warn) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  impl_ptr->set_chol_warning(warn);
}

// [[Rcpp::export]]
double linalg_rcond_approx_chol(arma::mat X) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->rcond_approx_chol(X);
}

// [[Rcpp::export]]
double linalg_rcond_chol(arma::mat X) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->rcond_chol(X);
}

// [[Rcpp::export]]
arma::mat linalg_chol_block(arma::mat C, arma::mat Loo) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->chol_block(C, Loo);
}

// Self-test of the detected BLAS / LAPACK backend (single thread, then
// n_threads concurrent threads; 0: hardware concurrency), see
// LinearAlgebra::check and #429; exported as linearalgebra_check()
// [[Rcpp::export]]
Rcpp::List linalg_check(int n_threads = 0) {
  const LinearAlgebra::CheckResult c = LinearAlgebra::check(n_threads);
  return Rcpp::List::create(
      Rcpp::Named("ok") = c.ok, Rcpp::Named("backend") = c.backend, Rcpp::Named("report") = c.report);
}
