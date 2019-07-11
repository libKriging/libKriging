#include <RcppArmadillo.h>
#include "libKriging/ArmadilloTestClass.hpp"

// [[Rcpp::export]]
arma::vec getEigenValues(arma::mat M) {
  return ArmadilloTestClass::getEigenValues(M);
}