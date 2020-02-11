#include <tuple>

#include "libKriging/OrdinaryKriging.hpp"
#include "libKriging/covariance.h"

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf (same variables names)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)

/* root distance function. Example for "gauss" kernel:
 double temp = 1;
 for (int k = 0; k < d; k++) {
 double dij = (X(i,k)-X(j,k))/parameters(k);
 temp *= exp(-0.5*std::pow(dij,2));
 }
 CovMatrix(i,j) = temp;
 */
double dist(const arma::rowvec & xi, const arma::rowvec & xj) {
  return distX(arma::mat(x)).col(1); // TODO to be optimized: return k.corr(new Point(xi), new Point(xj),...);
}

// returns distance matrix form Xp to X
LIBKRIGING_EXPORT arma::mat distX(const arma::mat & Xp) {
  return getCrossCorrMatrix(X,Xp,parameters,covType);
}
// same for one point
LIBKRIGING_EXPORT arma::colvec distX(const arma::rowvec & x) {
  return distX(arma::mat(x)).col(1); // TODO to be optimized...
}

// This will create the dist(xi,xj) function above. Need to parse "covType".
void make_dist(const string & covType) {
  k = CovarianceParameters::getCorrelationFunction(covType);
}

  // at least, just call make_dist(kernel)
  LIBKRIGING_EXPORT OrdinaryKriging(std::string covType) {
    make_dist(covType);
  }

  /** Fit the kriging object on (X,y):
   * @param y is n length column vector of output
   * @param X is n*d matrix of input
   * @param parameters is starting value for hyper-parameters
   * @param optim_method is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
   * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
   */
  LIBKRIGING_EXPORT void fit(const arma::colvec y, const arma::mat X, const arma::vec parameters, const string optim_objective, const string optim_method) {
    this.X = X; // enfin qlq chose comme ça, hein...
    this.y = y;

    int n = X.n_rows, d = X.n_cols;

    // Define regression matrix
    int nreg = 1;
    arma::mat F(n,nreg); F.ones();

    // Allocate the matrix we will return (Gauss or Matern 5/2)
    arma::mat CovMatrix = distX(X); // Should be stored at instance level for further usage...
    // Pevious version: version
    // arma::mat CovMatrix(n,n); CovMatrix.zeros();
    // for (int i = 0; i < n; i++) {
    //   for (int j = 0; j < i; j++) {
    //     double temp = 1;
    //     for (int k = 0; k < m; k++) {
    //       // double d = -std::sqrt(5)*std::abs(X(i,k)-X(j,k))/parameters(k);
    //       // temp *= (1+d+std::pow(d,2)/3)*std::exp(d);
    //       double d = (X(i,k)-X(j,k))/parameters(k);
    //       temp *= exp(-0.5*std::pow(d,2));
    //     }
    //     CovMatrix(i,j) = temp;
    //   }
    // }
    // CovMatrix = CovMatrix + trans(CovMatrix);
    // CovMatrix.diag().ones();

    getCrossCorrMatrix(X,Xp,parameters,covType)
    // Cholesky decompostion of covariance matrix
    arma::mat C = trans(chol(CovMatrix));
    // Compute intermediate useful matrices
    arma::mat Ft = arma::solve(C, F);
    arma::mat Q, G;
    qr_econ(Q,G,Ft);
    arma::colvec Yt = arma::solve(C, y);
    arma::colvec beta = arma::solve(G, trans(Q)*Yt);
    arma::colvec rho = Yt - Ft*beta;

    if (optim_method.compare('none')!=0) {
      // ... optim loop here
    }

    double  sigma2 = arma::as_scalar(arma::sum(pow(rho,2))/n);
    arma::mat gamma = trans(arma::solve(trans(C), rho));

    parameters = arma::join_cols(sigma,theta);

  }

  /** Compute the prediction for given points X'
   * @param Xp is m*d matrix of points where to predict output
   * @param std is true if return also stdev column vector
   * @param cov is true if return also cov matrix between Xp
   * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
   */
  LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat> predict(const arma::mat & Xp, const bool withStd, const bool withCov) {
    int m = Xp.n_rows;
    arma::colvec mean(m);
    arma::colvec stdev(m);
    arma::mat cor(m,m);

    // ...

    if (withStd)
      if (withCov)
        return std::make_tuple(std::move(mean), std::move(stdev), std::move(cov));
      else
        return std::make_tuple(std::move(mean), std::move(stdev));
    else
      if (withCov)
        return std::make_tuple(std::move(mean), std::move(cov));
      else
        return std::make_tuple(std::move(mean));
  }

  /** Draw sample trajectories of kriging at given points X'
   * @param Xp is m*d matrix of points where to simulate output
   * @param nsim is number of simulations to draw
   * @return output is m*nsim matrix of simulations at Xp
   */
  LIBKRIGING_EXPORT arma::mat simulate(const int nsim, const arma::mat & Xp) {
    arma::mat y(n,nsim);

    // ...

    return std::move(yp);
  }

  /** Add new conditional data points to previous (X,y)
   * @param newy is m length column vector of new output
   * @param newX is m*d matrix of new input
   * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
   * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
   */
  LIBKRIGING_EXPORT void update(const arma::vec & newy, const arma::mat & newX, const string & optim_method, const string & optim_objective) {
    // ...
  }
