#ifndef LIBKRIGING_BENCH_HPP
#define LIBKRIGING_BENCH_HPP

#include <armadillo>

#include "libKriging/libKriging_exports.h"
// #include "covariance.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class Bench {
 public:
  int n;

  LIBKRIGING_EXPORT Bench(int _n);  // const std::string & covType);

  LIBKRIGING_EXPORT arma::mat SolveTri(const arma::mat& Xtri, const arma::vec& y);

  LIBKRIGING_EXPORT arma::mat CholSym(const arma::mat& Rsym);

  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat> QR(const arma::mat& M);

  LIBKRIGING_EXPORT arma::mat InvSymPD(const arma::mat& Rsympd);
};

#endif  // LIBKRIGING_BENCH_HPP