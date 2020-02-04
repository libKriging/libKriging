//
// Created by Pascal Havé on 2019-08-13.
//

#include "libKriging/LinearRegression.hpp"

LIBKRIGING_EXPORT
LinearRegression::LinearRegression() = default;

LIBKRIGING_EXPORT arma::colvec coef;
LIBKRIGING_EXPORT double sig2;
LIBKRIGING_EXPORT arma::colvec stderrest;


LIBKRIGING_EXPORT
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
// returned object should hold error state instead of void
void LinearRegression::fit(const arma::vec y, const arma::mat X) {
  int n = X.n_rows;
  int k = X.n_cols;

  coef = arma::solve(X, y);
  arma::colvec resid = y - X * coef;

  sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
  stderrest = arma::sqrt(sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));
}

std::tuple<arma::colvec, arma::colvec> LinearRegression::predict(const arma::mat X) {
  // should test that X.n_cols == fit.X.n_cols
  int n = X.n_rows;
  // int k = X.n_cols;

  arma::colvec y = X * coef;
  arma::colvec stderr = arma::sqrt(sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));

  return std::make_tuple(std::move(y), std::move(stderr));
}