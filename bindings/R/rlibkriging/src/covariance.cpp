// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Covariance.hpp"

// [[Rcpp::export]]
bool covariance_approx_singular_used() {
  Covariance* c = new Covariance();
  Rcpp::XPtr<Covariance> impl_ptr(c);
  return impl_ptr->approx_singular_used();
}

// [[Rcpp::export]]
void covariance_use_approx_singular(bool approx_singular) {
  Covariance* c = new Covariance();
  Rcpp::XPtr<Covariance> impl_ptr(c);
  impl_ptr->use_approx_singular(approx_singular);
}
