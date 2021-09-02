//
// Created by Pascal Havé on 2019-08-13.
//

#ifndef LIBKRIGING_LINEARREGRESSION_HPP
#define LIBKRIGING_LINEARREGRESSION_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

/** Basic linear regression
 * @ingroup Regression
 */
class LinearRegression {
 public:
  /** Trivial constructor */
  LIBKRIGING_EXPORT LinearRegression();

  [[nodiscard]] const arma::colvec& coef() const { return m_coef; };
  [[nodiscard]] const double& sig2() const { return m_sig2; };
  [[nodiscard]] const arma::colvec& stderrest() const { return m_stderrest; };

  /** True linear regression computation
   * has to find s such that y ~= X * s
   * The accuracy may be evaluated using the returned standard error
   *
   * @param y : rhs vector of size n
   * @param X : matrix of size n * m
   */
  LIBKRIGING_EXPORT void fit(const arma::vec& y, const arma::mat& X);

  LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec> predict(const arma::mat& X);

 private:
  arma::colvec m_coef;
  double m_sig2{};
  arma::colvec m_stderrest;
};

#endif  // LIBKRIGING_LINEARREGRESSION_HPP
