#ifndef LIBKRIGING_BENCH_HPP
#define LIBKRIGING_BENCH_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Kriging.hpp"
#include "libKriging/libKriging_exports.h"
// #include "covariance.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class Bench {
 public:
  static std::chrono::high_resolution_clock::time_point tic();
  static std::chrono::high_resolution_clock::time_point toc(std::string what,
                                                            std::chrono::high_resolution_clock::time_point t0);

  int n;

  LIBKRIGING_EXPORT Bench(int _n);  // const std::string & covType);

  LIBKRIGING_EXPORT arma::mat SolveTri(const arma::mat& Xtri, const arma::vec& y);

  LIBKRIGING_EXPORT arma::mat CholSym(const arma::mat& Rsym);

  LIBKRIGING_EXPORT std::tuple<arma::mat, arma::mat> QR(const arma::mat& M);

  LIBKRIGING_EXPORT arma::mat InvSymPD(const arma::mat& Rsympd);

  LIBKRIGING_EXPORT double LogLik(Kriging& ok, const arma::vec& theta);

  LIBKRIGING_EXPORT arma::vec LogLikGrad(Kriging& ok, const arma::vec& theta);
};

#endif  // LIBKRIGING_BENCH_HPP
