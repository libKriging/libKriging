#ifndef LIBKRIGING_BENCH_HPP
#define LIBKRIGING_BENCH_HPP

#include <armadillo>

#include "libKriging/OrdinaryKriging.hpp"
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

  LIBKRIGING_EXPORT double LogLik(OrdinaryKriging& ok, const arma::vec& theta);

  LIBKRIGING_EXPORT arma::vec LogLikGrad(OrdinaryKriging& ok, const arma::vec& theta);
  
  LIBKRIGING_EXPORT double Rosenbrock(arma::vec& x) ;
  LIBKRIGING_EXPORT arma::vec RosenbrockGrad(arma::vec& x) ;
  LIBKRIGING_EXPORT arma::mat OptimRosenbrock(arma::vec& x0) ;
};

#endif  // LIBKRIGING_BENCH_HPP
