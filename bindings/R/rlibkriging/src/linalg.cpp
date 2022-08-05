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
arma::mat linalg_chol_safe(arma::mat X) {
  LinearAlgebra* la = new LinearAlgebra();
  Rcpp::XPtr<LinearAlgebra> impl_ptr(la);
  return impl_ptr->safe_chol_lower(X);
}
