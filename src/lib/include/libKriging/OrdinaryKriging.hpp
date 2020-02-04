#ifndef LIBKRIGING_ORDINARYKRIGING_HPP
#define LIBKRIGING_ORDINARYKRIGING_HPP

#include <armadillo>
#include "libKriging_exports.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class OrdinaryKriging {
 public:
  /** Trivial constructor */
  LIBKRIGING_EXPORT OrdinaryKriging();
  LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec> fit(const arma::vec y, const arma::mat X, const arma::vec theta);
};

#endif  // LIBKRIGING_ORDINARYKRIGING_HPP
