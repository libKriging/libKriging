#include <RcppArmadillo.h>
#include "libKriging/demo/DemoArmadilloClass.hpp"

// [[Rcpp::export]]
arma::vec getEigenValues(arma::mat M) {
  return DemoArmadilloClass().getEigenValues(M);
}