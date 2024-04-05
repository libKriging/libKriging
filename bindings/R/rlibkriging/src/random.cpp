// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Random.hpp"

// [[Rcpp::export]]
void random_reset_seed(unsigned int seed) {
  Random* r = new Random();
  Rcpp::XPtr<Random> impl_ptr(r);
  impl_ptr->reset_seed(seed);
}

// [[Rcpp::export]]
double random_randu() {
  Random* r = new Random();
  Rcpp::XPtr<Random> impl_ptr(r);
  return impl_ptr->randu();
}

// [[Rcpp::export]]
arma::fvec random_randu_vec(unsigned int n) {
  Random* r = new Random();
  Rcpp::XPtr<Random> impl_ptr(r);
  return impl_ptr->randu_vec(n);
}

// [[Rcpp::export]]
arma::fmat random_randu_mat(unsigned int n, unsigned int d) {
  Random* r = new Random();
  Rcpp::XPtr<Random> impl_ptr(r);
  return impl_ptr->randu_mat(n, d);
}

// [[Rcpp::export]]
arma::fmat random_randn_mat(unsigned int n, unsigned int d) {
  Random* r = new Random();
  Rcpp::XPtr<Random> impl_ptr(r);
  return impl_ptr->randn_mat(n, d);
}
