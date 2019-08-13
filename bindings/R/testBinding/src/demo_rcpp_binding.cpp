#include <RcppArmadillo.h>
#include "libKriging/DemoArmadilloClass.hpp"

// [[Rcpp::export]]
arma::vec getEigenValues(arma::mat M) {
  return DemoArmadilloClass::getEigenValues(M);
}