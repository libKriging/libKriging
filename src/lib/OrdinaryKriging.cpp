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
  arma::uword np = Xp.n_rows;
  
  // Should bre replaced by for_each
  arma::mat R(n, np);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < np; j++) {
      // FIXME WARNING : theta parameter shadows theta attribute
      R(i, j) = Cov_fun(X.row(i), Xp.row(j), theta);
    }
  }
  return R;
}

// Optimized version when Xp=X
LIBKRIGING_EXPORT
  arma::mat OrdinaryKriging::Cov(const arma::mat& X, const arma::colvec& theta) {
    // Should be tyaken from covariance.h from nestedKriging ?
    // return getCrossCorrMatrix(X,Xp,parameters,covType);
    
    arma::uword n = X.n_rows;
    
    // Should bre replaced by for_each
    arma::mat R(n, n);
    R.zeros();
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        // FIXME WARNING : theta parameter shadows theta attribute
        R(i, j) = Cov_fun(X.row(i), X.row(j), theta);
      }
    }
    R = arma::symmatl(R);  // R + trans(R);
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
  Cov_fun = [](const arma::rowvec &xi, const arma::rowvec &xj, const arma::vec & _theta) {
    double temp = 1;
    for (arma::uword k = 0; k < _theta.n_elem; k++) {
      double d = (xi(k) - xj(k)) / _theta(k);
      temp *= exp(-0.5 * std::pow(d, 2));
    }
    return temp;
  };
  Cov_deriv = [](const arma::rowvec &xi, const arma::rowvec &xj, const arma::vec & _theta, int dim) {
    double temp = 1;
    for (arma::uword k = 0; k < _theta.n_elem; k++) {
      double d = (xi(k) - xj(k)) / _theta(k);
      temp *= exp(-0.5 * std::pow(d, 2));
      if (k == dim) {  //-0.5*(xi(k)-xj(k))*(-2)/(theta(k)^3);
        temp *= std::pow(d, 2) / _theta(k);
      }
    }
    return temp;
  };
  
  // arma::cout << "make_Cov done." << arma::endl;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT OrdinaryKriging::OrdinaryKriging(){//const std::string & covType) {
  make_Cov("gauss");//covType);
  // FIXME: sigma2 attribute not initialized
}

// Objective function for fit : -logLikelihood
double fit_ofn(const arma::vec& _theta, arma::vec* grad_out, OrdinaryKriging::OKModel* okm_data) {//void* okm_data){//
  // OrdinaryKriging::OKModel* fd = reinterpret_cast<OrdinaryKriging::OKModel*> (okm_data);
  OrdinaryKriging::OKModel* fd = okm_data;
  
  // arma::cout << "_theta:" << _theta << arma::endl;
  
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
  
  // Allocate the matrix // arma::mat R = Cov(fd->X, _theta);
  // Should be replaced by for_each
  arma::mat R(n, n);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      // FIXME WARNING : theta parameter shadows theta attribute
      R(i, j) = fd->cov_fun(fd->X.row(i), fd->X.row(j), _theta);
    }
  }
  R = arma::symmatl(R);  // R + trans(R);
  R.diag().ones();
  // arma::cout << "R:" << R << arma::endl;
  
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
  // arma::cout << "sigma2_hat:" << sigma2_hat << arma::endl;
  
  double minus_ll = /*-*/ 0.5 * (n * log(2 * M_PI * sigma2_hat) + 2 * sum(log(fd->T.diag())) + n);
  // arma::cout << "ll:" << -minus_ll << arma::endl;
  
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
    
    arma::mat x = solve(trans(fd->T), fd->z);
    arma::mat xx = x * trans(x);
     // arma::mat xx_upper = trimatu(xx);
    
    for (int k = 0; k < fd->X.n_cols; k++) {
      arma::mat gradR_k_upper(n, n);
      gradR_k_upper.zeros();
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          gradR_k_upper(i,j) = fd->cov_deriv(fd->X.row(i), fd->X.row(j), _theta, k);
        }
      }
      gradR_k_upper = trans(gradR_k_upper);
      // arma::mat gradR_k = symmatu(gradR_k_upper);
      // gradR_k.diag().zeros();
      
      double terme1 = arma::accu(xx/*_upper*/ % gradR_k_upper) / sigma2_hat;//as_scalar((trans(x) * gradR_k) * x)/ sigma2_hat;
      double terme2 = -arma::accu(Rinv/*_upper*/ % gradR_k_upper);//-arma::trace(Rinv * gradR_k);
      (*grad_out)(k) = - (terme1 + terme2);
      // (*grad_out)(k) = - arma::accu(dot(xx / sigma2_hat - Rinv, gradR_k_upper));
    }
     // arma::cout << "Grad: " << *grad_out <<  arma::endl;
  }
  
  return minus_ll;
}

LIBKRIGING_EXPORT double OrdinaryKriging::logLikelihood(const arma::vec& _theta) {
  arma::mat T;
  arma::mat z;
  OrdinaryKriging::OKModel okm_data{y, X, T, z, Cov_fun, Cov_deriv};
  
  return -fit_ofn(_theta, nullptr, &okm_data); 
}

LIBKRIGING_EXPORT arma::vec OrdinaryKriging::logLikelihoodGrad(const arma::vec& _theta) {
  arma::mat T;
  arma::mat z;
  OrdinaryKriging::OKModel okm_data{y, X, T, z, Cov_fun, Cov_deriv};
  
  arma::vec grad(_theta.n_elem);
  
  double ll = fit_ofn(_theta, &grad, &okm_data); 
  
  return -grad;
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param parameters is starting value for hyper-parameters
 * @param optim_method is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
 */
LIBKRIGING_EXPORT void OrdinaryKriging::fit(const arma::colvec& y,
                                            const arma::mat& X){//,
                                            // const Parameters& parameters,
                                            // const std::string& optim_objective, // will support "logLik" or "leaveOneOut"
                                            // const std::string& optim_method) {
                                            
  std::string optim_objective="ll";
  std::string optim_method="bfgs";
  Parameters parameters{0, false, arma::vec(1), false};
  
  this->X = X;
  this->y = y;
  
  // arma::cout << "optim_method:" << optim_method << arma::endl;
  
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
    
    // arma::cout << "theta0:" << theta0 << arma::endl;
    
    optim::algo_settings_t algo_settings;
    algo_settings.iter_max = 100;  // TODO change by default?
    algo_settings.vals_bound = true;
    algo_settings.lower_bounds = arma::zeros<arma::vec>(X.n_cols);
    algo_settings.upper_bounds = 2*arma::ones<arma::vec>(X.n_cols);
    double minus_ll = std::numeric_limits<double>::infinity(); // FIXME prefer use of C++ value without narrowing
    for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
      arma::vec theta_tmp = trans(theta0.row(i));     // FIXME arma::mat replaced by arma::vec
      arma::mat T;
      arma::mat z;
      OrdinaryKriging::OKModel okm_data{y, X, T, z, Cov_fun, Cov_deriv};
      bool bfgs_ok = optim::bfgs(
          theta_tmp,
          [&okm_data](const arma::vec& vals_inp, arma::vec* grad_out, void*) -> double {
            return fit_ofn(vals_inp, grad_out, &okm_data);
          },
          nullptr,
          // fit_ofn,
          // reinterpret_cast<OrdinaryKriging::OKModel*> (&okm_data),
          algo_settings);
      
      // if (bfgs_ok) {    
        double minus_ll_tmp
            = fit_ofn(theta_tmp,
                      nullptr,
                      &okm_data);  // this last call also ensure that T and z are up-to-date with solution found.
        if (minus_ll_tmp < minus_ll) {
          theta = std::move(theta_tmp);
          minus_ll = minus_ll_tmp;
          T = std::move(okm_data.T);
          z = std::move(okm_data.z);
        }
      // }
    }
  } else
    throw std::runtime_error("Not a suitable optim_method: " + optim_method);

  arma::cout << "theta:" << theta << arma::endl;
  
  
  if (not parameters.has_sigma2) {
    sigma2 = arma::as_scalar(sum(pow(z, 2)) / X.n_rows);
  } else {
    sigma2 = parameters.sigma2;
  }
  
  arma::cout << "sigma2:" << sigma2 << arma::endl;
  
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
  this->fit(y, X);//, parameters, optim_objective, optim_method);
}
