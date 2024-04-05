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

// [[Rcpp::export]]
void linalg_check_chol_rcond(bool cr) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  impl_ptr->check_chol_rcond(cr);
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
arma::mat linalg_chol_block(arma::mat C, arma::mat Loo, arma::mat Coo) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->chol_block(C, Loo, Coo);
}