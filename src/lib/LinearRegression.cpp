//
// Created by Pascal Havé on 2019-08-13.
//

#include "libKriging/LinearRegression.hpp"

LIBKRIGING_EXPORT
LinearRegression::LinearRegression() {}

LIBKRIGING_EXPORT
std::tuple<arma::colvec, arma::colvec> LinearRegression::apply(const arma::vec& y, const arma::mat& X) {
  int n = X.n_rows, k = X.n_cols;

  arma::colvec coef = arma::solve(X, y);
  arma::colvec resid = y - X * coef;

  double sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
  arma::colvec stderrest = arma::sqrt(sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));

  return std::make_tuple(std::move(coef), std::move(stderrest));
}