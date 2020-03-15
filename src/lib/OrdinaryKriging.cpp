// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/OrdinaryKriging.hpp"

#include <armadillo>
#include <optim.hpp>
#include <tuple>

// #include "libKriging/covariance.h"

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf
//'  (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (same variables names)

// returns distance matrix form Xp to X
LIBKRIGING_EXPORT
arma::mat OrdinaryKriging::Cov(const arma::mat& X, const arma::mat& Xp) {
  // Should be tyaken from covariance.h from nestedKriging ?
  // return getCrossCorrMatrix(X,Xp,parameters,covType);
  arma::mat Xtnorm = trans(X); Xtnorm.each_col() /= m_theta;
  arma::mat Xptnorm = trans(Xp); Xptnorm.each_col() /= m_theta;
  
  arma::uword n = X.n_rows;
  arma::uword np = Xp.n_rows;

  // Should bre replaced by for_each
  arma::mat R(n, np);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < np; j++) {
      // FIXME WARNING : theta parameter shadows theta attribute
      R.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xptnorm.col(j));
    }
  }
  return R;
}

// Optimized version when Xp=X
LIBKRIGING_EXPORT
  arma::mat OrdinaryKriging::Cov(const arma::mat& X) {
    // Should be tyaken from covariance.h from nestedKriging ?
    // return getCrossCorrMatrix(X,Xp,parameters,covType);
    
    arma::mat Xtnorm = trans(X); Xtnorm.each_col() /= m_theta;
    arma::uword n = X.n_rows;
    
    // Should bre replaced by for_each
    arma::mat R(n, n);
    R.zeros();
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        // FIXME WARNING : theta parameter shadows theta attribute
        R.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xtnorm.col(j));
      }
    }
    R = arma::symmatl(R);  // R + trans(R);
    R.diag().ones();
    return R;
  }
//// same for one point
// LIBKRIGING_EXPORT arma::colvec OrdinaryKriging::Cov(const arma::mat& X,
//                                                    const arma::rowvec& x,
//                                                    const arma::colvec& theta) {
//  // FIXME mat(x) : an arma::mat from a arma::rowvec ?
//  return OrdinaryKriging::Cov(&X, arma::mat(&x), &theta).col(1);  // TODO to be optimized...
//}

// This will create the dist(xi,xj) function above. Need to parse "covType".
void OrdinaryKriging::make_Cov(const std::string& covType) {
  // if (covType.compareTo("gauss")==0)
  //' @ref https://github.com/cran/DiceKriging/blob/master/src/CovFuns.c
  CovNorm_fun = [](const arma::vec &xi, const arma::vec &xj) {
    double temp = 0;
    for (arma::uword k = 0; k < xi.n_elem; k++) {
      double d = (xi(k) - xj(k));
      temp += d*d;
    }
    return exp(-0.5*temp);
  };
  CovNorm_deriv = [](const arma::vec &xi, const arma::vec &xj, int dim) {
    double temp = 0;
    for (arma::uword k = 0; k < xi.n_elem; k++) {
      double d = (xi(k) - xj(k));
      temp += d*d;
    }
    return exp(-.5*temp) * (xi(dim) - xj(dim))*(xi(dim) - xj(dim));
  };
  
  // arma::cout << "make_Cov done." << arma::endl;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT OrdinaryKriging::OrdinaryKriging() {  // const std::string & covType) {
  make_Cov("gauss");                                    // covType);
  // FIXME: sigma2 attribute not initialized
}

// Objective function for fit : -logLikelihood
double fit_ofn(const arma::vec& _theta, arma::vec* grad_out, OrdinaryKriging::OKModel* okm_data) {
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
  
  arma::mat Xtnorm = trans(fd->X); Xtnorm.each_col() /= _theta;
  
  arma::uword n = fd->X.n_rows;
  
  // Define regression matrix
  arma::uword nreg = 1;
  arma::mat F = arma::ones(n, nreg);

  // Allocate the matrix // arma::mat R = Cov(fd->X, _theta);
  // Should be replaced by for_each
  arma::mat R = arma::zeros(n,n);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = fd->covnorm_fun(Xtnorm.col(i), Xtnorm.col(j));
    }
  }
  R = arma::symmatl(R);  // R + trans(R);
  R.diag().ones();
  // arma::cout << "R:" << R << arma::endl;

  // Cholesky decompostion of covariance matrix
  fd->T = trans(chol(R));

  // Compute intermediate useful matrices
  fd->M = solve(trimatl(fd->T), F,arma::solve_opts::fast);
  arma::mat Q;
  arma::mat G;
  qr_econ(Q, G, fd->M);
  arma::colvec Yt = solve(trimatl(fd->T), fd->y,arma::solve_opts::fast);
  fd->beta = solve(trimatu(G), trans(Q) * Yt,arma::solve_opts::fast);
  fd->z = Yt - fd->M * fd->beta;

  //' @ref https://github.com/cran/DiceKriging/blob/master/R/computeAuxVariables.R
  double sigma2_hat = arma::accu(fd->z % fd->z) / n;
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

    arma::mat Linv = solve(trimatl(fd->T), arma::eye(n,n),arma::solve_opts::fast);
    arma::mat Rinv = trans(Linv) * Linv;  //inv_sympd(R);

    arma::mat x = solve(trimatu(trans(fd->T)), fd->z,arma::solve_opts::fast);
    arma::mat xx = x * trans(x);
    // arma::mat xx_upper = trimatu(xx);

    for (arma::uword k = 0; k < fd->X.n_cols; k++) {
      arma::mat gradR_k_upper = arma::zeros(n, n);
      for (arma::uword i = 0; i < n; i++) {
        for (arma::uword j = 0; j < i; j++) {
          gradR_k_upper.at(j,i) = fd->covnorm_deriv(Xtnorm.col(i), Xtnorm.col(j), k);
        }
      }
      gradR_k_upper /= _theta(k);
      gradR_k_upper = trans(gradR_k_upper);
      // arma::mat gradR_k = symmatu(gradR_k_upper);
      // gradR_k.diag().zeros();

      double terme1 = arma::accu(xx/*_upper*/ % gradR_k_upper) / sigma2_hat;//as_scalar((trans(x) * gradR_k) * x)/ sigma2_hat;
      double terme2 = -arma::accu(Rinv/*_upper*/ % gradR_k_upper);//-arma::trace(Rinv * gradR_k);
      (*grad_out).at(k) = - (terme1 + terme2);
      // (*grad_out)(k) = - arma::accu(dot(xx / sigma2_hat - Rinv, gradR_k_upper));
    }
    // arma::cout << "Grad: " << *grad_out <<  arma::endl;
  }

  return minus_ll;
}

// Utility function for LOO
arma::colvec DiagABA(const arma::mat& A,const arma::mat& B) {
  arma::mat D=trimatu(2*B);
  D.diag() = B.diag();
  D = (A*D)%A;
  arma::colvec c = sum(D,1);

  return c;
}

// Objective function for fit : -LOO
  double fit_ofn2(const arma::vec& _theta, arma::vec* grad_out, OrdinaryKriging::OKModel* okm_data) {
  OrdinaryKriging::OKModel* fd = okm_data;
  
  arma::mat Xtnorm = trans(fd->X); Xtnorm.each_col() /= _theta;
  
  arma::uword n = fd->X.n_rows;
  
  // Define regression matrix
  arma::uword nreg = 1;
  arma::mat F(n, nreg);
  F.ones();

  // Allocate the matrix // arma::mat R = Cov(fd->X, _theta);
  // Should be replaced by for_each
  arma::mat R(n, n);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      R(i, j) = fd->covnorm_fun(Xtnorm.col(i), Xtnorm.col(j));
    }
  }
  R = arma::symmatl(R);  // R + trans(R);
  R.diag().ones();
  // arma::cout << "R:" << R << arma::endl;

  // Cholesky decompostion of covariance matrix
  fd->T = trans(chol(R));

  // Compute intermediate useful matrices
  // arma::mat M = solve(fd->T, F);
  fd->M = solve(trimatl(fd->T), F,arma::solve_opts::fast);
  // arma::mat Rinv = inv_sympd(R); // didn't find chol2inv equivalent in armadillo
  arma::mat Rinv = inv(trimatl(fd->T));
  Rinv = trimatl(Rinv) * trimatl(Rinv);
  arma::mat RinvF = Rinv * F;
  arma::mat TM = chol(trans(fd->M)*fd->M); // Can be optimized with a crossprod equivalent in armadillo ? 
  // arma::mat aux = solve(trans(TM), trans(RinvF));
  arma::mat aux = solve(trimatl(trans(TM)), trans(RinvF),arma::solve_opts::fast);
  arma::mat Q = Rinv - trans(aux)*aux; // Can be optimized with a crossprod equivalent in armadillo ?
  arma::mat Qy = Q * fd->y;
  arma::colvec sigma2LOO = 1/Q.diag();
  arma::colvec errorsLOO = sigma2LOO % Qy;
  double minus_loo = - arma::accu(errorsLOO % errorsLOO) / n;

  if (grad_out != nullptr) {
    //' @ref hhttps://github.com/cran/DiceKriging/blob/master/R/leaveOneOutGrad.R
    // LOOfunDer <- matrix(0, nparam, 1)							
		//for (k in 1:nparam) {
		//	gradR.k <- covMatrixDerivative(model@covariance, X=model@X, C0=R, k=k)
		//	diagdQ <- - diagABA(A=Q, B=gradR.k)
		//	dsigma2LOO <- - (sigma2LOO^2) * diagdQ
		//	derrorsLOO <- dsigma2LOO * Q.y - sigma2LOO * (Q%*%(gradR.k%*%Q.y))
		//	LOOfunDer[k] <- 2*crossprod(errorsLOO, derrorsLOO)/model@n
		//}

    for (arma::uword k = 0; k < fd->X.n_cols; k++) {
      arma::mat gradR_k(n, n);
      gradR_k.zeros();
      for (arma::uword i = 0; i < n; i++) {
        for (arma::uword j = 0; j < i; j++) {
          gradR_k(i,j) = fd->covnorm_deriv(Xtnorm.col(i), Xtnorm.col(j), k);
        }
      }
      gradR_k /= _theta(k);
      gradR_k = arma::symmatl(gradR_k);  // gradR_k + trans(gradR_k);
      gradR_k.diag().zeros();

      arma::colvec diagdQ = -DiagABA(Q,gradR_k);
      arma::colvec dsigma2LOO = - sigma2LOO%sigma2LOO%diagdQ;
      arma::colvec derrorsLOO = dsigma2LOO%Qy - sigma2LOO%(Q*(gradR_k*Qy));
      (*grad_out)(k) = - 2*dot(errorsLOO, derrorsLOO)/n;
    }
    // arma::cout << "Grad: " << *grad_out <<  arma::endl;
  }

  return minus_loo;
}

LIBKRIGING_EXPORT double OrdinaryKriging::logLikelihood(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  OrdinaryKriging::OKModel okm_data{m_y, m_X, T, M, z, beta, CovNorm_fun, CovNorm_deriv};
  
  return -fit_ofn(_theta, nullptr, &okm_data); 
}

LIBKRIGING_EXPORT arma::vec OrdinaryKriging::logLikelihoodGrad(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  OrdinaryKriging::OKModel okm_data{m_y, m_X, T, M, z, beta, CovNorm_fun, CovNorm_deriv};
  
  arma::vec grad(_theta.n_elem);

  double ll = fit_ofn(_theta, &grad, &okm_data);

  return -grad;
}

LIBKRIGING_EXPORT double OrdinaryKriging::loofun(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  OrdinaryKriging::OKModel okm_data{m_y, m_X, T, M, z, beta, CovNorm_fun, CovNorm_deriv};
  
  return -fit_ofn2(_theta, nullptr, &okm_data); 
}

LIBKRIGING_EXPORT arma::vec OrdinaryKriging::loofungrad(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  OrdinaryKriging::OKModel okm_data{m_y, m_X, T, M, z, beta, CovNorm_fun, CovNorm_deriv};
  
  arma::vec grad(_theta.n_elem);

  double ll = fit_ofn2(_theta, &grad, &okm_data);

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

  // Normalization of inputs and output
  arma::rowvec centerX = min(X,0);
  arma::rowvec scaleX = max(X,0) - min(X,0);
  double centerY = min(y);
  double scaleY = max(y) - min(y);
  m_centerX = centerX;
  m_scaleX = scaleX;
  m_centerY = centerY;
  m_scaleY = scaleY;
  arma::mat newX = X;
  newX.each_row() -= centerX;
  newX.each_row() /= scaleX;
  arma::colvec newy = (y-centerY)/scaleY;

  this->m_X = newX;
  this->m_y = newy;

  // arma::cout << "optim_method:" << optim_method << arma::endl;

  if (optim_method == "none") {  // just keep given theta, no optimisation of ll
    m_theta = parameters.theta;
  } else if (optim_method.rfind("bfgs", 0) == 0) {
    arma::mat theta0;
    // FIXME parameters.has needs to implemtented (no use case in current code)
    if (!parameters.has_theta) {  // no theta given, so draw 10 random uniform starting values
      int multistart = 10;        // TODO? stoi(substr(optim_method,)) to hold 'bfgs10' as a 10 multistart bfgs
      theta0 = arma::randu(multistart, X.n_cols);
    } else {  // just use given theta(s) as starting values for multi-bfgs
      theta0 = arma::mat(parameters.theta);
    }

    // arma::cout << "theta0:" << theta0 << arma::endl;

    optim::algo_settings_t algo_settings;
    algo_settings.iter_max = 10;  // TODO change by default?
    algo_settings.err_tol = 1e-5; 
    algo_settings.vals_bound = true;
    algo_settings.lower_bounds = 0.001*arma::ones<arma::vec>(X.n_cols);
    algo_settings.upper_bounds = 2*sqrt(X.n_cols)*arma::ones<arma::vec>(X.n_cols);
    double minus_ll = std::numeric_limits<double>::infinity(); // FIXME prefer use of C++ value without narrowing
    for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
      arma::vec theta_tmp = trans(theta0.row(i));     // FIXME arma::mat replaced by arma::vec
      arma::mat T;
      arma::mat M;
      arma::mat z;
      arma::colvec beta;
      OrdinaryKriging::OKModel okm_data{newy, newX, T, M, z, beta, CovNorm_fun, CovNorm_deriv};
      bool bfgs_ok = optim::lbfgs(
          theta_tmp,
          [&okm_data](const arma::vec& vals_inp, arma::vec* grad_out, void*) -> double {
            return fit_ofn(vals_inp, grad_out, &okm_data);
          },
          nullptr,
          algo_settings);

      // if (bfgs_ok) { // FIXME always succeeds ?
      double minus_ll_tmp
          = fit_ofn(theta_tmp,
                    nullptr,
                    &okm_data);  // this last call also ensure that T and z are up-to-date with solution found.
      if (minus_ll_tmp < minus_ll) {
        m_theta = std::move(theta_tmp);
        minus_ll = minus_ll_tmp;
        m_T = std::move(okm_data.T);
        m_M = std::move(okm_data.M);
        m_z = std::move(okm_data.z);
        m_beta = std::move(okm_data.beta);
      }
      // }
    }
  } else
    throw std::runtime_error("Not a suitable optim_method: " + optim_method);

  // arma::cout << "theta:" << m_theta << arma::endl;

  if (!parameters.has_sigma2) {
    m_sigma2 = arma::as_scalar(sum(pow(m_z, 2)) / X.n_rows);
    m_sigma2 = arma::as_scalar(accu(m_z%m_z) / X.n_rows);
    // Un-normalize
    m_sigma2 *= scaleY*scaleY;
  } else {
    m_sigma2 = parameters.sigma2;
  }
  
  // arma::cout << "sigma2:" << m_sigma2 << arma::endl;
  
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
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::colvec pred_mean(m);
  arma::colvec pred_stdev(m);
  arma::mat pred_cov(m, m);
  pred_stdev.zeros();
  pred_cov.zeros();

  // Define regression matrix
  arma::uword nreg = 1;
  arma::mat Ftest(m, nreg);
  Ftest.ones();

  // Compute covariance between training data and new data to predict
  arma::mat R(n, m);
  arma::mat Xtnorm = trans(m_X);
  Xtnorm.each_col() /= m_theta;
  arma::mat Xpnorm = trans(Xp);
  // Normalize Xp
  Xpnorm.each_col() -= m_centerX;
  Xpnorm.each_col() /= m_scaleX;
  Xpnorm.each_col() /= m_theta;
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      R.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xpnorm.col(j));
    }
  }
  arma::mat Tinv_newdata = solve(trimatl(m_T), R,arma::solve_opts::fast);
  pred_mean = Ftest*m_beta + trans(Tinv_newdata)*m_z;
  // Un-normalize predictor
  pred_mean = m_centerY + m_scaleY*pred_mean;
  
  if (withStd){
    double total_sd2 = m_sigma2;
    // s2.predict.1 <- apply(Tinv.c.newdata, 2, crossprod)
    arma::colvec s2_predict_1 = m_sigma2 * trans(sum(Tinv_newdata % Tinv_newdata,0));
    // Type = "UK"
    // T.M <- chol(t(M)%*%M)
    arma::mat TM = trans(chol(trans(m_M)*m_M));
    // s2.predict.mat <- backsolve(t(T.M), t(F.newdata - t(Tinv.c.newdata)%*%M) , upper.tri = FALSE)
    arma::mat s2_predict_mat = solve(trimatl(TM), trans(Ftest-trans(Tinv_newdata)*m_M),arma::solve_opts::fast);
    // s2.predict.2 <- apply(s2.predict.mat, 2, crossprod)
    arma::colvec s2_predict_2 = m_sigma2 * trans(sum(s2_predict_mat% s2_predict_mat,0));
    // s2.predict <- pmax(total.sd2 - s2.predict.1 + s2.predict.2, 0)
    arma::mat s2_predict = total_sd2 - s2_predict_1 + s2_predict_2;
    s2_predict.elem( find(pred_stdev < 0) ).zeros();
    pred_stdev = sqrt(s2_predict);
    if (withCov) {
      // C.newdata <- covMatrix(object@covariance, newdata)[[1]]
      arma::mat C_newdata(m,m);
      for (arma::uword i = 0; i < m; i++) {
        for (arma::uword j = 0; j < m; j++) {
          C_newdata.at(i, j) = CovNorm_fun(Xpnorm.col(i), Xpnorm.col(j));
        }
      }
      // cond.cov <- C.newdata - crossprod(Tinv.c.newdata)
      // cond.cov <- cond.cov + crossprod(s2.predict.mat)
      pred_cov = m_sigma2 * (C_newdata - trans(Tinv_newdata) * Tinv_newdata + trans(s2_predict_mat) * s2_predict_mat);
    } 
  }else if (withCov){
    arma::mat C_newdata(m,m);
      for (arma::uword i = 0; i < m; i++) {
        for (arma::uword j = 0; j < m; j++) {
          C_newdata.at(i, j) = CovNorm_fun(Xpnorm.col(i), Xpnorm.col(j));
        }
      }
      // Need to compute matrices computed in withStd case
      arma::mat TM = trans(chol(trans(m_M)*m_M));
      arma::mat s2_predict_mat = solve(trimatl(TM), trans(Ftest-trans(Tinv_newdata)*m_M),arma::solve_opts::fast);
      pred_cov = m_sigma2 * (C_newdata - trans(Tinv_newdata) * Tinv_newdata + trans(s2_predict_mat) * s2_predict_mat);
  }

  return std::make_tuple(std::move(pred_mean), std::move(pred_stdev), std::move(pred_cov));
  /*if (withStd)
    if (withCov)
      return std::make_tuple(std::move(pred_mean), std::move(pred_stdev), std::move(pred_cov));
    else
      return std::make_tuple(std::move(pred_mean), std::move(pred_stdev), nullptr);
  else if (withCov)
    return std::make_tuple(std::move(pred_mean), std::move(pred_cov), nullptr);
  else
    return std::make_tuple(std::move(pred_mean), nullptr, nullptr);*/
}

/** Draw sample trajectories of kriging at given points X'
 * @param Xp is m*d matrix of points where to simulate output
 * @param nsim is number of simulations to draw
 * @return output is m*nsim matrix of simulations at Xp
 */
LIBKRIGING_EXPORT arma::mat OrdinaryKriging::simulate(const int nsim, const arma::mat& Xp) {
  // Here nugget.sim = 1e-10 to avoid chol failures of Sigma_cond)
  double nugget_sim = 1e-10;
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::mat yp(m, nsim);

  // Define regression matrix
  arma::uword nreg = 1;
  arma::mat F_newdata(m, nreg);
  F_newdata.ones();
  arma::colvec y_trend = F_newdata * m_beta;

  // Compute covariance between new data
  arma::mat Sigma(m, m);
  arma::mat Xpnorm = trans(Xp);
  // Normalize Xp
  Xpnorm.each_col() -= m_centerX;
  Xpnorm.each_col() /= m_scaleX;
  Xpnorm.each_col() /= m_theta;
  for (arma::uword i = 0; i < m; i++) {
    for (arma::uword j = 0; j < i; j++) {
      Sigma.at(i, j) = CovNorm_fun(Xpnorm.col(i), Xpnorm.col(j));
    }
  }
  Sigma = arma::symmatl(Sigma);  // R + trans(R);
  Sigma.diag().ones();
  // arma::mat T_newdata = chol(Sigma);
  // Compute covariance between training data and new data to predict
  // Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)
  arma::mat Sigma21(n, m);
  arma::mat Xtnorm = trans(m_X);
  Xtnorm.each_col() /= m_theta;
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      Sigma21.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xpnorm.col(j));
    }
  }
  // Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE
  arma::mat Tinv_Sigma21 = solve(trimatl(m_T), Sigma21,arma::solve_opts::fast);
  // y.trend.cond <- y.trend + t(Tinv.Sigma21) %*% object@z
  y_trend += trans(Tinv_Sigma21) * m_z;
  // Sigma.cond <- Sigma11 - t(Tinv.Sigma21) %*% Tinv.Sigma21 
  arma::mat Sigma_cond = Sigma - trans(Tinv_Sigma21) * Tinv_Sigma21;
  // T.cond <- chol(Sigma.cond + diag(nugget.sim, m, m))	
  Sigma_cond.diag() += nugget_sim;
  arma::mat T_cond = chol(m_sigma2*Sigma_cond);
  // white.noise <- matrix(rnorm(m*nsim), m, nsim)
  // y.rand.cond <- t(T.cond) %*% white.noise
  // y <- matrix(y.trend.cond, m, nsim) + y.rand.cond	
  yp.each_col() = y_trend;
  yp += trans(T_cond) * arma::randn(m,nsim);
  // Un-normalize simulations
  yp = m_centerY + m_scaleY*yp;

  return yp;  // NB: move not required due to copy ellision mechanism
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
  m_X = join_rows(m_X, newX);
  m_y = join_rows(m_y, newy);

  // rebuild starting parameters
  Parameters parameters{this->m_sigma2, true, this->m_theta, true};
  // re-fit
  this->fit(m_y, m_X);  //, parameters, optim_objective, optim_method);
}
