#ifndef LIBKRIGING_LINEARREGRESSIONOPTIM_HPP
#define LIBKRIGING_LINEARREGRESSIONOPTIM_HPP

#include <armadillo>
#include "libKriging/libKriging_exports.h"

/** Basic linear regression
 * @ingroup Regression
 */
class LinearRegressionOptim {
 public:
  /** Trivial constructor */
  LIBKRIGING_EXPORT LinearRegressionOptim();

    // should be not exported ?
    LIBKRIGING_EXPORT arma::colvec coef;
    LIBKRIGING_EXPORT double sig2;
    LIBKRIGING_EXPORT arma::colvec stderrest;

  /** True linear regression computation
   * has to find s such that y ~= X * s
   * The accuracy may be evaluated using the returned standard error
   *
   * @param y : rhs vector of size n
   * @param X : matrix of size n * m
   */
  LIBKRIGING_EXPORT void fit(const arma::vec y, const arma::mat X);

  LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec> predict(const arma::mat X);

};

#endif  // LIBKRIGING_LINEARREGRESSIONOPTIM_HPP
