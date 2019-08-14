#include <RcppArmadillo.h>
#include "libKriging/LinearRegression.hpp"

// [[Rcpp::export]]
Rcpp::List linear_regression(arma::vec y, arma::mat X) {
  LinearRegression rl;
  auto ans = rl.apply(y, X);
  return Rcpp::List::create(Rcpp::Named("coefficients") = std::get<0>(ans),
                            Rcpp::Named("stderr") = std::get<1>(ans));
}