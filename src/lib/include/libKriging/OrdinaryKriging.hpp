#ifndef LIBKRIGING_ORDINARYKRIGING_HPP
#define LIBKRIGING_ORDINARYKRIGING_HPP

#include <armadillo>
#include "libKriging_exports.h"
#include "covariance.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class OrdinaryKriging {

   CorrelationFunction k;

   arma::mat X;
   arma::colvec y;

  /* root distance function. Example for "gauss" kernel:
   double temp = 1;
   for (int k = 0; k < d; k++) {
   double dij = (X(i,k)-X(j,k))/parameters(k);
   temp *= exp(-0.5*std::pow(dij,2));
   }
   CovMatrix(i,j) = temp;
   */
  double dist(const arma::rowvec & xi, const arma::rowvec & xj);

  // returns distance matrix form Xp to X
  LIBKRIGING_EXPORT arma::mat distX(const arma::mat & Xp);
  // same for one point
  LIBKRIGING_EXPORT arma::colvec distX(const arma::rowvec & x);

  // This will create the dist(xi,xj) function above. Need to parse "kernel".
  void make_dist(const string & kernel);

public:
  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT OrdinaryKriging(string kernel);

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param parameters is starting value for hyper-parameters
   * @param optim_method is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::colvec y, const arma::mat X, const arma::vec parameters, const string optim_method, const string optim_objective);

  /** Compute the prediction for given points X'
   * @param Xp is m*d matrix of points where to predict output
   * @param std is true if return also stdev column vector
   * @param cov is true if return also cov matrix between Xp
   * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
   */
  LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat> predict(const arma::mat & Xp, const bool withStd, const bool withCov);

  /** Draw sample trajectories of kriging at given points X'
   * @param Xp is m*d matrix of points where to simulate output
   * @param nsim is number of simulations to draw
   * @return output is m*nsim matrix of simulations at Xp
   */
  LIBKRIGING_EXPORT arma::mat simulate(const int nsim, const arma::mat & Xp);

  /** Add new conditional data points to previous (X,y)
   * @param newy is m length column vector of new output
   * @param newX is m*d matrix of new input
   * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
   * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
   */
  LIBKRIGING_EXPORT void update(const arma::vec & newy, const arma::mat & newX, const string & optim_method, const string & optim_objective);

};

#endif  // LIBKRIGING_ORDINARYKRIGING_HPP
