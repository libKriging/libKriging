#include "libKriging/OrdinaryKriging.hpp"

#include <armadillo>
#include <optim.hpp>
#include <tuple>
// #include "libKriging/covariance.h"

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf
//'  (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (same variables names)

// Utilities
// TODO: will be moved in utility file when this file will be ok
//template<typename... Args>
//std::string asString(Args&& ...args) {
//  std::ostringstream oss;
//  (void)(int[]){0, (void(oss << std::forward<Args>(args)), 0)...};
//  return oss.str();
//};
//
//class ParameterReader {
// private:
//  using Parameters = std::map<std::string, arma::vec>;
//
// public:
//  explicit ParameterReader(const Parameters& parameters) : m_parameters(parameters) {}
//
//  const arma::vec& operator[](const std::string& name) const {
//    auto finder = m_parameters.find(name);
//    if (finder != m_parameters.end()) {
//      return finder->second;
//    }
//    else {
//      throw std::invalid_argument(asString("Parameter ",name," not found"));
//    }
//  }
//
//  bool has(const std::string& name) const {
//    auto finder = m_parameters.find(name);
//    return (finder != m_parameters.end());
//  }
//
// private:
//  const Parameters& m_parameters;
//};

// returns distance matrix form Xp to X
LIBKRIGING_EXPORT
arma::mat OrdinaryKriging::Cov(const arma::mat& X, const arma::mat& Xp, const arma::colvec& theta) {
  // Should be tyaken from covariance.h from nestedKriging ?
  // return getCrossCorrMatrix(X,Xp,parameters,covType);

  arma::uword n = X.n_rows;

  // Should bre replaced by for_each
  arma::mat R(n, n);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      // FIXME WARNING : theta parameter shadows theta attribute
      R(i, j) = Cov_fun(X.row(i), Xp.row(j), theta);
    }
  }
  R = arma::symmatu(R);  // R + trans(R);
  R.diag().ones();
  return R;
}

//// same for one point
//LIBKRIGING_EXPORT arma::colvec OrdinaryKriging::Cov(const arma::mat& X,
//                                                    const arma::rowvec& x,
//                                                    const arma::colvec& theta) {
//  // FIXME mat(x) : an arma::mat from a arma::rowvec ?
//  return OrdinaryKriging::Cov(&X, arma::mat(&x), &theta).col(1);  // TODO to be optimized...
//}

// This will create the dist(xi,xj) function above. Need to parse "covType".
void OrdinaryKriging::make_Cov(const std::string& covType) {
  // if (covType.compareTo("gauss")==0)
  //' @ref https://github.com/cran/DiceKriging/blob/master/src/CovFuns.c
  Cov_fun = [](const arma::rowvec &xi, const arma::rowvec &xj, const arma::rowvec &theta) {
    double temp = 1;
    for (arma::uword k = 0; k < theta.n_elem; k++) {
      double d = (xi(k) - xj(k)) / theta(k);
      temp *= exp(-0.5 * std::pow(d, 2));
    }
    return temp;
  };
  Cov_deriv = [](const arma::rowvec &xi, const arma::rowvec &xj, const arma::rowvec &theta, int dim) {
    double temp = 1;
    for (arma::uword k = 0; k < theta.n_elem; k++) {
      double d = (xi(k) - xj(k)) / theta(k);
      temp *= exp(-0.5 * std::pow(d, 2));
      if (k == dim) {  //-0.5*(xi(k)-xj(k))*(-2)/(theta(k)^3);
        temp *= d / theta(k);
      }
    }
    return temp;
  };
}

// at least, just call make_dist(kernel)
LIBKRIGING_EXPORT OrdinaryKriging::OrdinaryKriging(const std::string & covType) {
  make_Cov(covType);
  // FIXME: sigma2 attribute not initialized
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param parameters is starting value for hyper-parameters
 * @param optim_method is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
 */
LIBKRIGING_EXPORT void OrdinaryKriging::fit(const arma::colvec& y,
                                            const arma::mat& X,
                                            const Parameters& parameters,
                                            const std::string& optim_objective, // FIXME never used
                                            const std::string& optim_method) {
  this->X = X;
  this->y = y;

  if (optim_method == "none") {  // just keep given theta, no optimisation of ll
    theta = parameters.theta;
  } else if (optim_method.rfind("bfgs", 0) == 0) {
    arma::mat theta0;
    // FIXME parameters.has needs to implemtented (no use case in current code)
    if (not parameters.has_theta) {  // no theta given, so draw 10 random uniform starting values
      int multistart = 10;  // TODO? stoi(substr(optim_method,)) to hold 'bfgs10' as a 10 multistart bfgs
      theta0 = arma::randu(multistart, X.n_cols);
    } else {  // just use given theta(s) as starting values for multi-bfgs
      theta0 = arma::mat(parameters.theta);
    }

    optim::algo_settings_t algo_settings;
    algo_settings.iter_max = 10;  // TODO change by default?

    double ll = -std::numeric_limits<double>::infinity(); // FIXME prefer use of C++ value without narrowing
    for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
      arma::vec theta_tmp = theta0.row(i);     // FIXME arma::mat replaced by arma::vec
      arma::mat T;
      arma::mat z;
      OKModel okm_data{y, X, T, z};
      bool bfgs_ok = optim::bfgs(
          theta_tmp,
          [this, &okm_data](const arma::vec& vals_inp, arma::vec* grad_out, void*) -> double {
            return this->fit_ofn(vals_inp, grad_out, &okm_data);
          },
          nullptr,
          algo_settings);
      if (bfgs_ok) {
        double ll_tmp
            = fit_ofn(theta_tmp,
                      nullptr,
                      &okm_data);  // this last call also ensure that T and z are up-to-date with solution found.
        if (ll_tmp > ll) {
          theta = theta_tmp;
          ll = ll_tmp;
          T = okm_data.T;
          z = okm_data.z;
        }
      }
    }
  } else
    throw std::runtime_error("Not a suitable optim_method: " + optim_method);

  if (not parameters.has_sigma2) {
    sigma2 = arma::as_scalar(sum(pow(z, 2)) / X.n_rows);
  } else {
    sigma2 = parameters.sigma2;
  }
}

double OrdinaryKriging::fit_ofn(const arma::vec& theta, arma::vec* grad_out, OKModel* okm_data) {
  OKModel* fd = okm_data;

  //' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikFun.R
  //  model@covariance <- vect2covparam(model@covariance, param)
  //  model@covariance@sd2 <- 1		# to get the correlation matrix
  //
  //  aux <- covMatrix(model@covariance, model@X)
  //
  //  R <- aux[[1]]
  //  T <- chol(R)
  //
  //  x <- backsolve(t(T), model@y, upper.tri = FALSE)
  //  M <- backsolve(t(T), model@F, upper.tri = FALSE)
  //  z <- compute.z(x=x, M=M, beta=beta)
  //  sigma2.hat <- compute.sigma2.hat(z)
  //  logLik <- -0.5*(model@n * log(2*pi*sigma2.hat) + 2*sum(log(diag(T))) + model@n)

  int n = fd->X.n_rows;

  // Define regression matrix
  int nreg = 1;
  arma::mat F(n, nreg);
  F.ones();

  // Allocate the matrix
  arma::mat R = Cov(fd->X, fd->X, theta);

  // Cholesky decompostion of covariance matrix
  fd->T = trans(chol(R));

  // Compute intermediate useful matrices
  arma::mat M = solve(fd->T, F);
  arma::mat Q, G;
  qr_econ(Q, G, M);
  arma::colvec Yt = solve(fd->T, fd->y);
  arma::colvec beta = solve(G, trans(Q) * Yt);
  fd->z = Yt - M * beta;

  //' @ref https://github.com/cran/DiceKriging/blob/master/R/computeAuxVariables.R
  double sigma2_hat = as_scalar(sum(pow(fd->z, 2)) / n);

  double ll = -0.5 * (n * log(2 * M_PI * sigma2_hat) + 2 * sum(log(fd->T.diag())) + n);

  if (grad_out != nullptr) {
    //' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikGrad.R
    //  logLik.derivative <- matrix(0,nparam,1)
    //  x <- backsolve(T,z)			# compute x := T^(-1)*z
    //  Rinv <- chol2inv(T)			# compute inv(R) by inverting T
    //
    //  Rinv.upper <- Rinv[upper.tri(Rinv)]
    //  xx <- x%*%t(x)
    //  xx.upper <- xx[upper.tri(xx)]
    //
    //  for (k in 1:nparam) {
    //    gradR.k <- CovMatrixDerivative(model@covariance, X=model@X, C0=R, k=k)
    //    gradR.k.upper <- gradR.k[upper.tri(gradR.k)]
    //
    //    terme1 <- sum(xx.upper*gradR.k.upper)   / sigma2.hat
    //    # quick computation of t(x)%*%gradR.k%*%x /  ...
    //    terme2 <- - sum(Rinv.upper*gradR.k.upper)
    //    # quick computation of trace(Rinv%*%gradR.k)
    //    logLik.derivative[k] <- terme1 + terme2
    //  }

    arma::mat Rinv = inv_sympd(R);
    // arma::mat Rinv_upper = trimatu(Rinv);

    arma::mat x = trans(solve(trans(fd->T), fd->z));
    arma::mat xx = x * trans(x);
    // arma::mat xx_upper = trimatu(xx);

    for (int k = 0; k < fd->X.n_cols; k++) {
      arma::mat gradR_k_upper(n, n);
      gradR_k_upper.zeros();
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          gradR_k_upper(i, j) = OrdinaryKriging::Cov_deriv(fd->X.row(i), fd->X.row(j), theta, k);
        }
      }
      // gradR_k = symmatu(gradR_k);
      // gradR_k.diag().zeros();

      (*grad_out)(k) = arma::accu(dot(xx / sigma2_hat + Rinv, gradR_k_upper));
    }
    // cout << "Grad: " << *grad_out << endl;
  }
  // cout<<"Y = X * s :\n"<<y<<endl;

  return ll;
}

/** Compute the prediction for given points X'
 * @param Xp is m*d matrix of points where to predict output
 * @param std is true if return also stdev column vector
 * @param cov is true if return also cov matrix between Xp
 * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat> OrdinaryKriging::predict(const arma::mat& Xp,
                                                                                             bool withStd,
                                                                                             bool withCov) {
  int m = Xp.n_rows;
  arma::colvec mean(m);
  arma::colvec stdev(m);
  arma::mat cov(m, m);

  // ...

  // FIXME what size ? (output tuple has 3 fields)
  if (withStd)
    if (withCov)
      return std::make_tuple(std::move(mean), std::move(stdev), std::move(cov));
    else
      return std::make_tuple(std::move(mean), std::move(stdev), nullptr);
  else if (withCov)
    return std::make_tuple(std::move(mean), std::move(cov), nullptr);
  else
    return std::make_tuple(std::move(mean), nullptr, nullptr);
}

/** Draw sample trajectories of kriging at given points X'
 * @param Xp is m*d matrix of points where to simulate output
 * @param nsim is number of simulations to draw
 * @return output is m*nsim matrix of simulations at Xp
 */
LIBKRIGING_EXPORT arma::mat OrdinaryKriging::simulate(const int nsim, const arma::mat& Xp) {
  arma::mat yp(X.n_rows, nsim);

  // ...

  return yp; // NB: move not required due to copy ellision mechanism
}

/** Add new conditional data points to previous (X,y)
 * @param newy is m length column vector of new output
 * @param newX is m*d matrix of new input
 * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
 * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
 */
LIBKRIGING_EXPORT void OrdinaryKriging::update(const arma::vec& newy,
                                               const arma::mat& newX,
                                               const std::string& optim_objective,
                                               const std::string& optim_method) {
  // rebuild data
  X = join_rows(X, newX);
  y = join_rows(y, newy);

  // rebuild starting parameters
  Parameters parameters{this->sigma2, true, this->theta, true};
  // re-fit
  this->fit(y, X, parameters, optim_objective, optim_method);
}
