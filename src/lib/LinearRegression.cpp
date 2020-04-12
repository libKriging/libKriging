//
// Created by Pascal Havé on 2019-08-13.
//

#include "libKriging/LinearRegression.hpp"

LIBKRIGING_EXPORT
LinearRegression::LinearRegression() = default;

LIBKRIGING_EXPORT
// returned object should hold error state instead of void
void LinearRegression::fit(const arma::vec& y, const arma::mat& X) {
  int n = X.n_rows;
  int k = X.n_cols;

  m_coef = arma::solve(X, y);
  // arma::cout << "Coef: " << m_coef << arma::endl;
  arma::colvec resid = y - X * m_coef;

  m_sig2 = arma::as_scalar(arma::trans(resid) * resid / (n - k));
  m_stderrest = arma::sqrt(m_sig2 * arma::diagvec(arma::inv(arma::trans(X) * X)));
}

std::tuple<arma::colvec, arma::colvec> LinearRegression::predict(const arma::mat& X) {
  // should test that X.n_cols == fit.X.n_cols
  // int n = X.n_rows;
  // int k = X.n_cols;

  arma::colvec y = X * m_coef;
  arma::colvec stderr_v = arma::sqrt(arma::diagvec(X * arma::diagmat(m_stderrest) * arma::trans(X)));

  return std::make_tuple(std::move(y), std::move(stderr_v));
}