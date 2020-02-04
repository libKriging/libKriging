#include <RcppArmadillo.h>
#include "libKriging/LinearRegression.hpp"

// [[Rcpp::export]]
Rcpp::List linear_regression(arma::vec y, arma::mat X) {
  LinearRegression rl;
  auto ans = rl.apply(y, X);
  return Rcpp::List::create(Rcpp::Named("coefficients") = std::get<0>(ans),
                            Rcpp::Named("stderr") = std::get<1>(ans));
}

#include "libKriging/OrdinaryKriging.hpp"

// [[Rcpp::export]]
Rcpp::List ordinary_kriging(arma::vec y, arma::mat X,arma::vec theta) {
    OrdinaryKriging ok;
    auto ans = ok.fit(y, X, theta);
    return Rcpp::List::create(Rcpp::Named("gamma") = std::get<0>(ans),
                              Rcpp::Named("theta") = std::get<1>(ans));
}
