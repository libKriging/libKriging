#ifndef LIBKRIGING_ORDINARYKRIGING_HPP
#define LIBKRIGING_ORDINARYKRIGING_HPP

#include <armadillo>
#include "libKriging/libKriging_exports.h"
// #include "covariance.h"

/** Ordinary kriging regression
 * @ingroup Regression
 */
class OrdinaryKriging {

   arma::mat X;
   arma::colvec y;
   arma::mat T;
   arma::colvec z;

   std::function<double(arma::rowvec,arma::rowvec,arma::rowvec)> Cov; // Covariance function
   std::function<double(arma::rowvec,arma::rowvec,arma::rowvec)> DCov; // Covaraince function derivative vs. theta
   arma::vec theta;
   double sigma2;

   // returns distance matrix form Xp to X
   LIBKRIGING_EXPORT arma::mat Cov(const arma::mat & X, const arma::mat & Xp, const arma::rowvec & theta);
   // same for one point
   LIBKRIGING_EXPORT arma::colvec Cov(const arma::mat & X, const arma::rowvec & x, const arma::rowvec & theta);

   // This will create the dist(xi,xj) function above. Need to parse "kernel".
   void make_Cov(const string & kernel);

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
   LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat> predict(const arma::mat & Xp, bool withStd, bool withCov);

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
   LIBKRIGING_EXPORT void update(const arma::vec & newy, const arma::mat & newX, const std::string & optim_method, const std::string & optim_objective);

};

#endif  // LIBKRIGING_ORDINARYKRIGING_HPP
