// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/CacheFunction.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/KrigingException.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NuggetKriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/custom_hash_function.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"

#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <tuple>
#include <vector>

/************************************************/
/**      NuggetKriging implementation        **/
/************************************************/

// This will create the dist(xi,xj) function above. Need to parse "covType".
void NuggetKriging::make_Cov(const std::string& covType) {
  m_covType = covType;
  if (covType.compare("gauss") == 0) {
    Cov = Covariance::Cov_gauss;
    DlnCovDtheta = Covariance::DlnCovDtheta_gauss;
    DlnCovDx = Covariance::DlnCovDx_gauss;
    Cov_pow = 2;
  } else if (covType.compare("exp") == 0) {
    Cov = Covariance::Cov_exp;
    DlnCovDtheta = Covariance::DlnCovDtheta_exp;
    DlnCovDx = Covariance::DlnCovDx_exp;
    Cov_pow = 1;
  } else if (covType.compare("matern3_2") == 0) {
    Cov = Covariance::Cov_matern32;
    DlnCovDtheta = Covariance::DlnCovDtheta_matern32;
    DlnCovDx = Covariance::DlnCovDx_matern32;
    Cov_pow = 1.5;
  } else if (covType.compare("matern5_2") == 0) {
    Cov = Covariance::Cov_matern52;
    DlnCovDtheta = Covariance::DlnCovDtheta_matern52;
    DlnCovDx = Covariance::DlnCovDx_matern52;
    Cov_pow = 2.5;
  } else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);

  // arma::cout << "make_Cov done." << arma::endl;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const arma::colvec& y,
                                               const arma::mat& X,
                                               const std::string& covType,
                                               const Trend::RegressionModel& regmodel,
                                               bool normalize,
                                               const std::string& optim,
                                               const std::string& objective,
                                               const Parameters& parameters) {
  if (y.n_elem != X.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X.n_rows) + "x"
                             + std::to_string(X.n_cols) + "), y: (" + std::to_string(y.n_elem) + ")");

  make_Cov(covType);
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

// arma::mat XtX(arma::mat &X) {
//   arma::mat XtX = arma::zeros(X.n_cols,X.n_cols);
//   for (arma::uword i = 0; i < X.n_cols; i++) {
//     for (arma::uword j = 0; j <= i; j++) {
//       for (arma::uword k = 0; k < X.n_rows; k++) {
//         XtX.at(i,j) += X.at(k,i) * X.at(k,j);
//       }
//     }
//   }
//   return(symmatl(XtX));
// }

// Objective function for fit : -logLikelihood

double NuggetKriging::_logLikelihood(const arma::vec& _theta_alpha,
                                     arma::vec* grad_out,
                                     NuggetKriging::OKModel* okm_data) const {
  // arma::cout << " theta, alpha: " << _theta_alpha.t() << arma::endl;
  //' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikFun.R
  //   model@covariance <- vect2covparam(model@covariance, param[1:(nparam - 1)])
  //   model@covariance@sd2 <- 1
  //   model@covariance@nugget <- 0
  //   alpha <- param[nparam]
  //
  //   aux <- covMatrix(model@covariance, model@X)
  //   R0 <- aux[[1]] - diag(aux[[2]])
  //   R <- alpha * R0 + (1 - alpha) * diag(model@n)
  //
  //   T <- chol(R)
  //   x <- backsolve(t(T), model@y, upper.tri = FALSE)
  //   M <- backsolve(t(T), model@F, upper.tri = FALSE)
  //   z <- compute.z(x=x, M=M, beta=beta)
  //   v <- compute.sigma2.hat(z)
  //   logLik <- -0.5 * (model@n * log(2 * pi * v) + 2*sum(log(diag(T))) + model@n)

  NuggetKriging::OKModel* fd = okm_data;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;

  double _alpha = _theta_alpha.at(d);
  arma::vec _theta = _theta_alpha.head(d);

  arma::mat R = arma::mat(n, n);
  for (arma::uword i = 0; i < n; i++) {
    R.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = R.at(j, i) = Cov(m_dX.col(i * n + j), _theta) * _alpha;
    }
  }

  // Cholesky decompostion of covariance matrix
  fd->T = LinearAlgebra::safe_chol_lower(R);  // Do NOT trimatl T (slower because copy): trimatl(chol(R, "lower"));

  // Compute intermediate useful matrices
  fd->M = solve(fd->T, m_F, LinearAlgebra::default_solve_opts);
  arma::mat Q;
  arma::mat G;
  qr_econ(Q, G, fd->M);

  arma::colvec Yt = solve(fd->T, m_y, LinearAlgebra::default_solve_opts);
  if (fd->is_beta_estim)
    fd->beta = solve(G, Q.t() * Yt, LinearAlgebra::default_solve_opts);
  fd->z = Yt - fd->M * fd->beta;

  fd->var = arma::accu(fd->z % fd->z) / n;
  if (fd->is_nugget_estim) {
    fd->nugget = (1 - _alpha) * fd->var;
    if (fd->is_sigma2_estim)
      fd->sigma2 = _alpha * fd->var;
  } else {
    if (fd->is_sigma2_estim)
      fd->sigma2 = fd->var - fd->nugget;
  }

  double ll = -0.5 * (n * log(2 * M_PI * fd->var) + 2 * sum(log(fd->T.diag())) + n);
  // arma::cout << " ll:" << ll << arma::endl;

  if (grad_out != nullptr) {
    //' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikGrad.R
    //  logLik.derivative <- matrix(0,nparam,1)
    //  x <- backsolve(T, z) # x := T^(-1)*z
    //  Cinv <- chol2inv(T) # Invert R from given T

    //  Cinv.upper <- Cinv[upper.tri(Cinv)]
    //  xx <- x %*% t(x)
    //  xx.upper <- xx[upper.tri(xx)]
    //
    //  # partial derivative with respect to parameters except sigma^2
    //  for (k in 1:(nparam - 1)) {
    //      gradC.k <- covMatrixDerivative(model@covariance, X = model@X, C0 = R0, k = k)
    //      gradC.k <- alpha * gradC.k
    //      gradC.k.upper <- gradC.k[upper.tri(gradC.k)]
    //
    //      term1 <- sum(xx.upper * gradC.k.upper) / v
    //      # economic computation of - t(x)%*%gradC.k%*%x / v
    //      term2 <- -sum(Cinv.upper * gradC.k.upper)
    //      # economic computation of trace(Cinv%*%gradC.k)
    //      logLik.derivative[k] <- term1 + term2
    //  }

    std::vector<arma::mat> gradsC(d);  // if (hess_out != nullptr)
    arma::vec term1 = arma::vec(d);    // if (hess_out != nullptr)

    arma::mat Linv = solve(fd->T, arma::eye(n, n), LinearAlgebra::default_solve_opts);
    arma::mat Cinv = (Linv.t() * Linv);  // Do NOT inv_sympd (slower): inv_sympd(R);

    arma::mat tT = fd->T.t();  // trimatu(trans(fd->T));
    arma::mat x = solve(tT, fd->z, LinearAlgebra::default_solve_opts);
    arma::mat xx = x * x.t();

    arma::cube gradC = arma::cube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradC.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }

    for (arma::uword k = 0; k < d; k++) {
      arma::mat gradC_k = arma::mat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradC_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradC_k.at(i, j) = gradC_k.at(j, i) = gradC.slice(i).col(j)[k];
        }
      }

      // should make a fast function trace_prod(A,B) -> sum_i(sum_j(Ai,j*Bj,i))
      term1.at(k)
          = as_scalar((trans(x) * gradC_k) * x) / fd->var;  //; //as_scalar((trans(x) * gradR_k) * x)/ sigma2_hat;
      double term2 = -arma::trace(Cinv * gradC_k);          //-arma::accu(Rinv % gradR_k_upper)
      (*grad_out).at(k) = (term1.at(k) + term2) / 2;
    }  // for (arma::uword k = 0; k < m_X.n_cols; k++)

    //  # partial derivative with respect to v = sigma^2 + delta^2
    //  dCdv <- R0 - diag(model@n)
    //  term1 <- -t(x) %*% dCdv %*% x / v
    //  term2 <- sum(Cinv * dCdv) # economic computation of trace(Cinv%*%C0)
    //  logLik.derivative[nparam] <- -0.5 * (term1 + term2) # /sigma2

    arma::mat dCdv = R / _alpha;
    dCdv.diag().zeros();
    double _term1 = -as_scalar((trans(x) * dCdv) * x) / fd->var;
    double _term2 = arma::accu(arma::dot(Cinv, dCdv));
    (*grad_out).at(d) = -0.5 * (_term1 + _term2);
    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }  // if (grad_out != nullptr)
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> NuggetKriging::logLikelihoodFun(const arma::vec& _theta_alpha,
                                                                                const bool _grad) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  double nugget{};
  double var{};
  NuggetKriging::OKModel okm_data{T, M, z, beta, true, sigma2, true, nugget, true, var};

  double ll = -1;
  arma::vec grad;
  if (_grad) {
    grad = arma::vec(_theta_alpha.n_elem);
    ll = _logLikelihood(_theta_alpha, &grad, &okm_data);
  } else
    ll = _logLikelihood(_theta_alpha, nullptr, &okm_data);

  return std::make_tuple(ll, std::move(grad));
}
// Objective function for fit: bayesian-like approach fromm RobustGaSP

double NuggetKriging::_logMargPost(const arma::vec& _theta_alpha,
                                   arma::vec* grad_out,
                                   NuggetKriging::OKModel* okm_data) const {
  // arma::cout << " theta: " << _theta << arma::endl;

  // In RobustGaSP:
  // neg_log_marginal_post_approx_ref <- function(param,nugget,
  // nugget.est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha) {
  //  lml=log_marginal_lik(param,nugget,nugget.est,R0,X,zero_mean,output,kernel_type,alpha);
  //  lp=log_approx_ref_prior(param,nugget,nugget.est,CL,a,b);
  //  -(lml+lp)
  //}
  // double log_marginal_lik(const Vec param,double nugget, const bool nugget_est, const List R0, const
  // Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output, Eigen::VectorXi
  // kernel_type,const Eigen::VectorXd alpha ){
  //  double nu=nugget;
  //  int param_size=param.size();
  //  Eigen::VectorXd beta= param.array().exp().matrix();
  //  ...beta
  //  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure
  //
  //  LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  //  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition
  //
  //  if(zero_mean=="Yes"){...}else{
  //
  //  int q=X.cols();
  //
  //  MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward
  //  and one backward to compute R.inv%*%X MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X; //Xt%*%R.inv%*%X
  //
  //  LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X); // cholesky decomposition of Xt_R_inv_X called lltOfXRinvX
  //  MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition
  //  MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv=
  //  R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
  //  //compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward and one backward solve MatrixXd yt_R_inv=
  //  (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); MatrixXd S_2=
  //  (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output); double log_S_2=log(S_2(0,0)); return
  //  (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2);
  //  }
  //}
  // double log_approx_ref_prior(const Vec param,double nugget, bool nugget_est, const Eigen::VectorXd CL,const double
  // a,const double b ){
  //  double nu=nugget;
  //  int param_size=param.size();beta
  //  Eigen::VectorX beta= param.array().exp().matrix();
  //  ...
  //  double t=CL.cwiseProduct(beta).sum()+nu;
  //  return -b*t + a*log(t);
  //}

  NuggetKriging::OKModel* fd = okm_data;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;

  double _alpha = _theta_alpha.at(d);
  arma::vec _theta = _theta_alpha.head(d);

  arma::mat R = arma::mat(n, n);
  for (arma::uword i = 0; i < n; i++) {
    R.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = R.at(j, i) = Cov(m_dX.col(i * n + j), _theta) * _alpha;
    }
  }

  // Cholesky decompostion of covariance matrix
  fd->T = LinearAlgebra::safe_chol_lower(R);

  //  // Compute intermediate useful matrices
  //  fd->M = solve(fd->T, m_F, LinearAlgebra::solve_opts);
  //  arma::mat Q;
  //  arma::mat G;
  //  qr_econ(Q, G, fd->M);
  //
  //  arma::mat H = Q * Q.t();  // if (hess_out != nullptr)
  //  arma::colvec Yt = solve(fd->T, m_y, LinearAlgebra::solve_opts);
  //  if (fd->is_beta_estim)
  // fd->beta = solve(trimatu(G), Q.t() * Yt, LinearAlgebra::solve_opts);
  //  fd->z = Yt - fd->M * fd->beta;

  // Keep RobustGaSP naming from now...
  arma::mat X = m_F;
  arma::mat L = fd->T;

  fd->M = solve(L, X, LinearAlgebra::default_solve_opts);
  arma::mat R_inv_X = solve(trans(L), fd->M, LinearAlgebra::default_solve_opts);
  arma::mat Xt_R_inv_X = trans(X) * R_inv_X;  // Xt%*%R.inv%*%X

  arma::mat LX = chol(Xt_R_inv_X, "lower");  //  retrieve factor LX  in the decomposition
  arma::mat R_inv_X_Xt_R_inv_X_inv_Xt_R_inv
      = R_inv_X
        * (solve(trans(LX),
                 solve(LX, trans(R_inv_X), LinearAlgebra::default_solve_opts),
                 LinearAlgebra::default_solve_opts));  // compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward
                                                       // and one backward solve
  arma::colvec Yt = solve(L, m_y, LinearAlgebra::default_solve_opts);
  if (fd->is_beta_estim) {
    arma::mat Q;
    arma::mat G;
    qr_econ(Q, G, fd->M);
    fd->beta = solve(G, Q.t() * Yt, LinearAlgebra::default_solve_opts);
    fd->z = Yt - fd->M * fd->beta;
  }

  arma::mat yt_R_inv = trans(solve(trans(L), Yt, LinearAlgebra::default_solve_opts));
  arma::mat S_2 = (yt_R_inv * m_y - trans(m_y) * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv * m_y);

  fd->var = S_2(0, 0) / (n - d);
  if (fd->is_nugget_estim) {
    fd->nugget = (1 - _alpha) * fd->var;
    if (fd->is_sigma2_estim)
      fd->sigma2 = _alpha * fd->var;
  } else {
    if (fd->is_sigma2_estim)
      fd->sigma2 = fd->var - fd->nugget;
  }

  double log_S_2 = log(S_2(0, 0));

  double log_marginal_lik = -sum(log(L.diag())) - sum(log(LX.diag())) - (m_X.n_rows - m_F.n_cols) / 2.0 * log_S_2;
  // arma::cout << " log_marginal_lik:" << log_marginal_lik << arma::endl;

  // Default prior params
  double a = 0.2;
  double b = 1.0 / pow(m_X.n_rows, 1.0 / m_X.n_cols) * (a + 1.0);

  arma::vec CL = trans(max(m_X, 0) - min(m_X, 0)) / pow(m_X.n_rows, 1.0 / m_X.n_cols);
  double t = arma::accu(CL % pow(_theta, -1.0)) + fd->nugget;
  double log_approx_ref_prior = -b * t + a * log(t);
  // arma::cout << " log_approx_ref_prior:" << log_approx_ref_prior << arma::endl;

  if (grad_out != nullptr) {
    // Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param,double nugget,  bool nugget_est, const List
    // R0, const Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output,
    // Eigen::VectorXi kernel_type,const Eigen::VectorXd alpha){
    // ...
    // VectorXd ans=VectorXd::Ones(param_size);
    // ...
    // MatrixXd Q_output= yt_R_inv.transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output;
    // MatrixXd dev_R_i;
    // MatrixXd Wb_ti;
    // //allow different choices of kernels
    //
    // for(int ti=0;ti<p;ti++){
    //   //kernel_type_ti=kernel_type[ti];
    //   if(kernel_type[ti]==3){
    //     dev_R_i=matern_5_2_deriv( R0[ti],R_ori,beta[ti]);  //now here I have R_ori instead of R
    //   }else {...}
    //   Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
    //   ans[ti]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0);
    // }
    arma::vec ans = arma::ones(m_X.n_cols);
    arma::mat Q_output = trans(yt_R_inv) - R_inv_X_Xt_R_inv_X_inv_Xt_R_inv * m_y;

    arma::cube gradR = arma::cube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradR.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }

    arma::mat Wb_k;
    for (arma::uword k = 0; k < m_X.n_cols; k++) {
      arma::mat gradR_k = arma::mat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradR_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradR_k.at(i, j) = gradR_k.at(j, i) = gradR.slice(i).col(j)[k];
        }
      }

      Wb_k = trans(solve(
                 trans(L), solve(L, gradR_k, LinearAlgebra::default_solve_opts), LinearAlgebra::default_solve_opts))
             - gradR_k * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      ans[k] = -0.5 * sum(Wb_k.diag())
               + (m_X.n_rows - m_F.n_cols) / 2.0 * (trans(m_y) * trans(Wb_k) * Q_output / S_2(0, 0))[0];
    }
    // arma::cout << " log_marginal_lik_deriv:" << -ans * pow(_theta,2) << arma::endl;
    // arma::cout << " log_approx_ref_prior_deriv:" <<  a*CL/t - b*CL << arma::endl;

    (*grad_out).head(d) = ans - (a * CL / t - b * CL) / pow(_theta, 2.0);

    arma::mat gradR_d = arma::ones(n, n);
    Wb_k = trans(
               solve(trans(L), solve(L, gradR_d, LinearAlgebra::default_solve_opts), LinearAlgebra::default_solve_opts))
           - gradR_d * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
    double ans_d = -0.5 * sum(Wb_k.diag())
                   + (m_X.n_rows - m_F.n_cols) / 2.0 * (trans(m_y) * trans(Wb_k) * Q_output / S_2(0, 0))[0];

    (*grad_out).at(d) = ans_d - (a * 1.0 / t - b * 1.0);

    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }

  // arma::cout << " lmp:" << (log_marginal_lik+log_approx_ref_prior) << arma::endl;
  return (log_marginal_lik + log_approx_ref_prior);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> NuggetKriging::logMargPostFun(const arma::vec& _theta,
                                                                              const bool _grad) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  double nugget{};
  double var{};
  NuggetKriging::OKModel okm_data{T, M, z, beta, true, sigma2, true, nugget, false, var};

  double lmp = -1;
  arma::vec grad;
  if (_grad) {
    grad = arma::vec(_theta.n_elem);
    lmp = _logMargPost(_theta, &grad, &okm_data);
  } else
    lmp = _logMargPost(_theta, nullptr, &okm_data);

  return std::make_tuple(lmp, std::move(grad));
}

LIBKRIGING_EXPORT double NuggetKriging::logLikelihood() {
  int d = m_theta.n_elem;
  arma::vec _theta_alpha = arma::vec(d + 1);
  _theta_alpha.head(d) = m_theta;
  _theta_alpha.at(d) = m_sigma2 / (m_sigma2 + m_nugget);
  return std::get<0>(NuggetKriging::logLikelihoodFun(_theta_alpha, false));
}

LIBKRIGING_EXPORT double NuggetKriging::logMargPost() {
  int d = m_theta.n_elem;
  arma::vec _theta_alpha = arma::vec(d + 1);
  _theta_alpha.head(d) = m_theta;
  _theta_alpha.at(d) = m_sigma2 / (m_sigma2 + m_nugget);
  return std::get<0>(NuggetKriging::logMargPostFun(_theta_alpha, false));
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 */
LIBKRIGING_EXPORT void NuggetKriging::fit(const arma::colvec& y,
                                          const arma::mat& X,
                                          const Trend::RegressionModel& regmodel,
                                          bool normalize,
                                          const std::string& optim,
                                          const std::string& objective,
                                          const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<double(const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::OKModel* okm_data)> fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = CacheFunction{[this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::OKModel* okm_data) {
        // Change variable for opt: . -> 1/exp(.)
        // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
        const arma::vec _theta_alpha = Optim::reparam_from(_gamma);
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        double ll = this->_logLikelihood(_theta_alpha, grad_out, okm_data);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > ll: " << ll << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
          *grad_out = -Optim::reparam_from_deriv(_theta_alpha, *grad_out);
        }
        return -ll;
      }};
    } else {
      fit_ofn = CacheFunction{[this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::OKModel* okm_data) {
        const arma::vec _theta_alpha = _gamma;
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        double ll = this->_logLikelihood(_theta_alpha, grad_out, okm_data);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > ll: " << ll << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
          *grad_out = -*grad_out;
        }
        return -ll;
      }};
    }
  } else if (objective.compare("LMP") == 0) {
    // Our impl. of https://github.com/cran/RobustGaSP/blob/5cf21658e6a6e327be6779482b93dfee25d24592/R/rgasp.R#L303
    //@see Mengyang Gu, Xiao-jing Wang and Jim Berger, 2018, Annals of Statistics.
    if (Optim::reparametrize) {
      fit_ofn = CacheFunction{[this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::OKModel* okm_data) {
        // Change variable for opt: . -> 1/exp(.)
        // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
        const arma::vec _theta_alpha = Optim::reparam_from(_gamma);
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        double lmp = this->_logMargPost(_theta_alpha, grad_out, okm_data);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
          *grad_out = -Optim::reparam_from_deriv(_theta_alpha, *grad_out);
        }
        return -lmp;
      }};
    } else {
      fit_ofn = CacheFunction{[this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::OKModel* okm_data) {
        const arma::vec _theta_alpha = _gamma;
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        double lmp = this->_logMargPost(_theta_alpha, grad_out, okm_data);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
          *grad_out = -*grad_out;
        }
        return -lmp;
      }};
    }
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LMP)");

  arma::rowvec centerX(d);
  arma::rowvec scaleX(d);
  double centerY;
  double scaleY;
  // Normalization of inputs and output
  if (normalize) {
    centerX = min(X, 0);
    scaleX = max(X, 0) - min(X, 0);
    centerY = min(y);
    scaleY = max(y) - min(y);
  } else {
    centerX.zeros();
    scaleX.ones();
    centerY = 0;
    scaleY = 1;
  }
  m_centerX = centerX;
  m_scaleX = scaleX;
  m_centerY = centerY;
  m_scaleY = scaleY;
  {  // FIXME why copies of newX and newy
    arma::mat newX = X;
    newX.each_row() -= centerX;
    newX.each_row() /= scaleX;
    arma::colvec newy = (y - centerY) / scaleY;
    this->m_X = newX;
    this->m_y = newy;
  }

  // Now we compute the distance matrix between points. Will be used to compute R(theta) later (e.g. when fitting)
  m_dX = arma::zeros(d, n * n);
  for (arma::uword ij = 0; ij < m_dX.n_cols; ij++) {
    int i = (int)ij / n;
    int j = ij % n;  // i,j <-> i*n+j
    if (i < j) {
      m_dX.col(ij) = trans(m_X.row(i) - m_X.row(j));
      m_dX.col(j * n + i) = m_dX.col(ij);
    }
  }

  // Define regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);

  arma::mat theta0;
  if (parameters.theta.has_value()) {
    theta0 = parameters.theta.value();
    if (parameters.theta.value().n_cols != d && parameters.theta.value().n_rows == d)
      theta0 = parameters.theta.value().t();
    if (theta0.n_cols != d)
      throw std::runtime_error("Dimension of theta should be nx" + std::to_string(d) + " instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
  }

  if (optim == "none") {  // just keep given theta, no optimisation of ll
    if (!parameters.theta.has_value())
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");
    if (!parameters.nugget.has_value())
      throw std::runtime_error("Nugget should be given, when optim=none");
    if (!parameters.sigma2.has_value())
      throw std::runtime_error("Sigma2 should be given, when optim=none");

    m_theta = trans(theta0.row(0));
    m_est_theta = false;
    arma::mat T;
    arma::mat M;
    arma::colvec z;
    arma::colvec beta;
    if (parameters.beta.has_value())
      beta = parameters.beta.value();
    double sigma2 = -1;
    if (parameters.sigma2.has_value())
      sigma2 = parameters.sigma2.value()[0];  // otherwise sigma2 will be re-calculated using given theta
    double nugget = -1;
    if (parameters.nugget.has_value())
      nugget = parameters.nugget.value()[0];

    NuggetKriging::OKModel okm_data{T,
                                    M,
                                    z,
                                    beta,
                                    parameters.is_beta_estim,
                                    sigma2,
                                    parameters.is_sigma2_estim,
                                    nugget,
                                    parameters.is_nugget_estim,
                                    nugget + sigma2};

    arma::vec gamma_tmp = arma::vec(d + 1);
    gamma_tmp.head(d) = m_theta;
    gamma_tmp.at(d) = sigma2 / (nugget + sigma2);
    if (Optim::reparametrize) {
      gamma_tmp.head(d) = Optim::reparam_to(m_theta);
      gamma_tmp.at(d) = Optim::reparam_to_(sigma2 / (nugget + sigma2));
    }

    /* double min_ofn_tmp = */ fit_ofn(gamma_tmp, nullptr, &okm_data);

    m_T = std::move(okm_data.T);
    m_M = std::move(okm_data.M);
    m_z = std::move(okm_data.z);
    m_beta = std::move(okm_data.beta);
    m_est_beta = parameters.is_beta_estim;
    m_sigma2 = okm_data.sigma2;
    m_est_sigma2 = parameters.is_sigma2_estim;
    m_nugget = okm_data.nugget;
    m_est_nugget = parameters.is_nugget_estim;

  } else if (optim.rfind("BFGS", 0) == 0) {
    Random::init();

    arma::vec theta_lower = Optim::theta_lower_factor * trans(max(X, 0) - min(X, 0));
    arma::vec theta_upper = Optim::theta_upper_factor * trans(max(X, 0) - min(X, 0));

    if (Optim::variogram_bounds_heuristic) {
      arma::vec dy2 = arma::zeros(n * n);
      for (arma::uword ij = 0; ij < dy2.n_elem; ij++) {
        int i = (int)ij / n;
        int j = ij % n;  // i,j <-> i*n+j
        if (i < j) {
          dy2[ij] = m_y.at(i) - m_y.at(j);
          dy2[ij] *= dy2[ij];
          dy2[j * n + i] = dy2[ij];
        }
      }
      // dy2 /= arma::var(m_y);
      arma::vec dy2dX2_slope = dy2 / arma::sum(m_dX % m_dX, 0).t();
      // arma::cout << "dy2dX_slope:" << dy2dX_slope << arma::endl;
      dy2dX2_slope.replace(arma::datum::nan, 0.0);  // we are not interested in same points where dX=0, and dy=0
      arma::vec w = dy2dX2_slope / sum(dy2dX2_slope);
      arma::mat steepest_dX_mean = arma::abs(m_dX) * w;

      theta_lower = arma::max(theta_lower, Optim::theta_lower_factor / steepest_dX_mean);
      theta_upper = arma::min(theta_upper, Optim::theta_upper_factor / steepest_dX_mean);
      theta_lower = arma::min(theta_lower, theta_upper);
      theta_upper = arma::max(theta_lower, theta_upper);
    }
    // arma::cout << "theta_lower:" << theta_lower << arma::endl;
    // arma::cout << "theta_upper:" << theta_upper << arma::endl;

    // FIXME parameters.has needs to implemtented (no use case in current code)
    if (!parameters.theta.has_value()) {  // no theta given, so draw 10 random uniform starting values
      int multistart = 1;
      try {
        multistart = std::stoi(optim.substr(4));
      } catch (std::invalid_argument&) {
        // let multistart = 1
      }
      theta0 = arma::repmat(trans(theta_lower), multistart, 1)
               + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
      // theta0 = arma::abs(0.5 + Random::randn_mat(multistart, d) / 6.0)
      //          % arma::repmat(max(m_X, 0) - min(m_X, 0), multistart, 1);
    } else {  // just use given theta(s) as starting values for multi-bfgs
      theta0 = arma::mat(parameters.theta.value());
    }
    // arma::cout << "theta0:" << theta0 << arma::endl;

    double alpha_upper = 1.0;
    double alpha_lower = 1E-3;

    arma::vec alpha0;
    if (parameters.sigma2.has_value() && parameters.nugget.has_value()) {
      alpha0 = arma::vec(parameters.sigma2.value().n_elem * parameters.nugget.value().n_elem);
      for (size_t i = 0; i < parameters.sigma2.value().n_elem; i++) {
        for (size_t j = 0; j < parameters.nugget.value().n_elem; j++) {
          if ((parameters.sigma2.value()[i] < 0) || (parameters.nugget.value()[j] < 0)
              || (parameters.sigma2.value()[i] + parameters.nugget.value()[j] < 0))
            alpha0[i + j * parameters.sigma2.value().n_elem]
                = alpha_lower + (alpha_upper - alpha_lower) * Random::randu();
          else
            alpha0[i + j * parameters.sigma2.value().n_elem]
                = parameters.sigma2.value()[i] / (parameters.sigma2.value()[i] + parameters.nugget.value()[j]);
        }
      }
    } else {
      alpha0 = alpha_lower + (alpha_upper - alpha_lower) * Random::randu_vec(theta0.n_rows);
    }
    // arma::cout << "alpha0:" << alpha0 << arma::endl;

    arma::vec gamma_lower = arma::vec(d + 1);
    gamma_lower.head(d) = theta_lower;
    gamma_lower.at(d) = alpha_lower;
    arma::vec gamma_upper = arma::vec(d + 1);
    gamma_upper.head(d) = theta_upper;
    gamma_upper.at(d) = alpha_upper;
    if (Optim::reparametrize) {
      arma::vec gamma_lower_tmp = gamma_lower;
      gamma_lower = Optim::reparam_to(gamma_upper);
      gamma_upper = Optim::reparam_to(gamma_lower_tmp);
    }

    double min_ofn = std::numeric_limits<double>::infinity();

    for (arma::uword i = 0; i < theta0.n_rows; i++) {
      arma::vec gamma_tmp = arma::vec(d + 1);
      gamma_tmp.head(d) = theta0.row(i).t();
      gamma_tmp.at(d) = alpha0[i % alpha0.n_elem];
      if (Optim::reparametrize) {
        gamma_tmp = Optim::reparam_to(gamma_tmp);
      }

      gamma_lower = arma::min(gamma_tmp, gamma_lower);
      gamma_upper = arma::max(gamma_tmp, gamma_upper);

      if (Optim::log_level > 0) {
        arma::cout << "BFGS:" << arma::endl;
        arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
        arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
        arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
        arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
        arma::cout << "  lower_bounds: " << theta_lower.t() << "";
        arma::cout << "                " << alpha_lower << arma::endl;
        arma::cout << "  upper_bounds: " << theta_upper.t() << "";
        arma::cout << "                " << alpha_upper << arma::endl;
        arma::cout << "  start_point: " << theta0.row(i).t() << "";
        arma::cout << "               " << alpha0[i % alpha0.n_elem] << arma::endl;
      }

      arma::mat T;
      arma::mat M;
      arma::colvec z;
      arma::colvec beta;
      if (parameters.beta.has_value())
        beta = parameters.beta.value();
      double sigma2 = -1;
      if (parameters.sigma2.has_value())
        sigma2 = parameters.sigma2.value()[0];  // pass the initial given value (useful if not to be estimated)
      double nugget = -1;
      if (parameters.nugget.has_value())
        nugget = parameters.nugget.value()[0];  // pass the initial given value (useful if not to be estimated)

      NuggetKriging::OKModel okm_data{T,
                                      M,
                                      z,
                                      beta,
                                      parameters.is_beta_estim,
                                      sigma2,
                                      parameters.is_sigma2_estim,
                                      nugget,
                                      parameters.is_nugget_estim,
                                      nugget + sigma2};

      lbfgsb::Optimizer optimizer{d + 1};
      optimizer.iprint = Optim::log_level - 2;
      optimizer.max_iter = Optim::max_iteration;
      optimizer.pgtol = Optim::gradient_tolerance;
      optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
      arma::ivec bounds_type{d + 1, arma::fill::value(2)};  // means both upper & lower bounds
      int retry = 0;
      while (retry <= Optim::max_restart) {
        auto result = optimizer.minimize(
            [&okm_data, &fit_ofn](const arma::vec& vals_inp, arma::vec& grad_out) -> double {
              return fit_ofn(vals_inp, &grad_out, &okm_data);
            },
            gamma_tmp,
            gamma_lower.memptr(),
            gamma_upper.memptr(),
            bounds_type.memptr());
        arma::vec sol_to_lb = gamma_tmp.head(d) - theta_lower;
        if (Optim::reparametrize) {
          sol_to_lb = gamma_tmp - gamma_upper;
          sol_to_lb = sol_to_lb.head(d);
        }
        if ((retry < Optim::max_restart) && (result.num_iters <= 2 * d)
            && (any(abs(sol_to_lb) < arma::datum::eps))) {  // we fastly converged to one bound
          gamma_tmp.head(d)
              = (theta0.row(i).t() + theta_lower)
                / pow(2.0, retry + 1);  // so, re-use previous starting point and change it to middle-point
          gamma_tmp.at(d) = alpha0[i % alpha0.n_elem];
          if (Optim::log_level > 0)
            arma::cout << "    start_point: " << gamma_tmp.t() << " ";
          if (Optim::reparametrize)
            gamma_tmp = Optim::reparam_to(gamma_tmp);
          if (Optim::log_level > 0) {
            arma::cout << "    iterations: " << result.num_iters << arma::endl;
          }
          retry++;
        } else {
          if (Optim::log_level > 1)
            result.print();
          break;
        }
      }

      // this last call also ensure that T and z are up-to-date with solution found.
      double min_ofn_tmp = fit_ofn(gamma_tmp, nullptr, &okm_data);

      if (Optim::log_level > 0) {
        arma::cout << "  best objective: " << min_ofn_tmp << arma::endl;
        if (Optim::reparametrize)
          arma::cout << "  best solution: " << Optim::reparam_from(gamma_tmp.t()) << " ";
        else
          arma::cout << "  best solution: " << gamma_tmp.t() << "";
      }

      if (min_ofn_tmp < min_ofn) {
        m_theta = gamma_tmp.head(d);
        if (Optim::reparametrize)
          m_theta = Optim::reparam_from(m_theta);
        m_est_theta = true;
        min_ofn = min_ofn_tmp;
        m_T = std::move(okm_data.T);
        m_M = std::move(okm_data.M);
        m_z = std::move(okm_data.z);
        m_beta = std::move(okm_data.beta);
        m_est_beta = parameters.is_beta_estim;
        m_sigma2 = okm_data.sigma2;
        m_est_sigma2 = parameters.is_sigma2_estim;
        m_nugget = okm_data.nugget;
        m_est_nugget = parameters.is_nugget_estim;
      }
    }
  } else
    throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS**)");

  if (!parameters.sigma2.has_value())
    m_sigma2 *= scaleY * scaleY;
  if (!parameters.nugget.has_value())
    m_nugget *= scaleY * scaleY;

  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param Xp is m*d matrix of points where to predict output
 * @param std is true if return also stdev column vector
 * @param cov is true if return also cov matrix between Xp
 * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat, arma::mat, arma::mat>
NuggetKriging::predict(const arma::mat& Xp, bool withStd, bool withCov, bool withDeriv) {
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (Xp.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(Xp.n_cols) + " instead of "
                             + std::to_string(d));

  arma::colvec pred_mean(m);
  arma::colvec pred_stdev = arma::zeros(m);
  arma::mat pred_cov = arma::zeros(m, m);
  arma::mat pred_mean_deriv = arma::zeros(m, d);
  arma::mat pred_stdev_deriv = arma::zeros(m, d);

  arma::mat Xtnorm = trans(m_X);  // already normalized if needed
  arma::mat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_p = Trend::regressionModelMatrix(m_regmodel, Xpnorm);
  Xpnorm = trans(Xpnorm);

  // Compute covariance between training data and new data to predict
  arma::mat R_pred = arma::ones(n, m);
  // double total_sd2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) + m_nugget;
  double total_sd2 = (m_sigma2 + m_nugget) * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  // R *= m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;
  R_pred *= m_sigma2 / total_sd2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      arma::vec dij = Xtnorm.col(i) - Xpnorm.col(j);
      if (arma::all(dij == 0))
        R_pred.at(i, j) = 1.0;  // m_sigma2 + m_nugget;
      else
        R_pred.at(i, j) *= Cov(dij, m_theta);
    }
  }
  arma::mat Tinv_pred = solve(m_T, R_pred, arma::solve_opts::fast);
  pred_mean = F_p * m_beta + trans(Tinv_pred) * m_z;
  // Un-normalize predictor
  pred_mean = m_centerY + m_scaleY * pred_mean;

  arma::mat s2_predict_mat;
  arma::mat FinvMtM;
  if (withStd || withCov) {
    arma::mat TM = trans(chol(trans(m_M) * m_M));
    s2_predict_mat = solve(TM, trans(F_p - trans(Tinv_pred) * m_M), arma::solve_opts::fast);

    if (withDeriv) {
      arma::mat m = trans(F_p - trans(Tinv_pred) * m_M);
      arma::mat invMtM = inv_sympd(m_M.t() * m_M);
      FinvMtM = (F_p - trans(Tinv_pred) * m_M) * inv_sympd(m_M.t() * m_M);
    }
  }
  if (withStd) {
    // s2.predict.1 <- apply(Tinv.c.newdata, 2, crossprod)
    arma::colvec s2_predict_1 = trans(sum(Tinv_pred % Tinv_pred, 0));
    s2_predict_1.transform([](double val) {
      return (val > 1.0 ? 1.0 : val);
    });  // constrain this first part to not be negative (rationale: it is the whole stdev for simple kriging)

    // s2.predict.2 <- apply(s2.predict.mat, 2, crossprod)
    arma::colvec s2_predict_2 = trans(sum(s2_predict_mat % s2_predict_mat, 0));
    // s2.predict <- pmax(total.sd2 - s2.predict.1 + s2.predict.2, 0)

    arma::mat s2_predict = total_sd2 * (1.0 - s2_predict_1 + s2_predict_2);
    s2_predict.transform([](double val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
    pred_stdev = sqrt(s2_predict);
  }

  if (withCov) {
    // C.newdata <- covMatrix(object@covariance, newdata)[[1]]
    arma::mat R_predpred = arma::ones(m, m);
    // C_newdata *= m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;
    R_predpred *= m_sigma2 / total_sd2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
    for (arma::uword i = 0; i < m; i++) {
      R_predpred.at(i, i) = 1;
      for (arma::uword j = 0; j < i; j++) {
        R_predpred.at(i, j) = R_predpred.at(j, i) *= Cov((Xpnorm.col(i) - Xpnorm.col(j)), m_theta);
      }
    }
    // cond.cov <- C.newdata - crossprod(Tinv.c.newdata)
    // cond.cov <- cond.cov + crossprod(s2.predict.mat)

    pred_cov = total_sd2 * (R_predpred - trans(Tinv_pred) * Tinv_pred + trans(s2_predict_mat) * s2_predict_mat);
  }

  if (withDeriv) {
    // # Compute derivatives of the covariance and trend functions
    for (arma::uword i = 0; i < m; i++) {  // for each Xp predict point... should be parallel ?

      arma::mat dc = arma::mat(n, d);
      for (arma::uword j = 0; j < n; j++) {
        dc.row(j) = R_pred.at(j, i) * trans(DlnCovDx(Xpnorm.col(i) - Xtnorm.col(j), m_theta));
      }

      const double h = 1.0E-5;  // Value is sensitive only for non linear trends. Otherwise, it gives exact results.
      arma::mat tXpn_i_repd = arma::trans(Xpnorm.col(i) * arma::ones(1, d));  // just duplicate Xp.row(i) d times

      arma::mat F_dx = (Trend::regressionModelMatrix(m_regmodel, tXpn_i_repd + h * arma::eye(d, d))
                        - Trend::regressionModelMatrix(m_regmodel, tXpn_i_repd - h * arma::eye(d, d)))
                       / (2 * h);

      // # Compute gradients of the kriging mean and variance
      arma::mat W = solve(m_T, dc, LinearAlgebra::default_solve_opts);

      pred_mean_deriv.row(i) = trans(F_dx * m_beta + trans(W) * m_z);

      if (withStd) {
        arma::mat pred_stdev_deriv_noTrend = Tinv_pred.t() * W;
        pred_stdev_deriv.row(i) = (-pred_stdev_deriv_noTrend.row(i) + FinvMtM.row(i) * (F_dx.t() - trans(m_M) * W))
                                  * total_sd2 / pred_stdev.at(i);
      }
    }
    pred_mean_deriv *= m_scaleY;
    pred_stdev_deriv *= m_scaleY;
  }

  return std::make_tuple(std::move(pred_mean),
                         std::move(pred_stdev),
                         std::move(pred_cov),
                         std::move(pred_mean_deriv),
                         std::move(pred_stdev_deriv));
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
LIBKRIGING_EXPORT arma::mat NuggetKriging::simulate(const int nsim, const int seed, const arma::mat& Xp) {
  // Here nugget.sim = 1e-10 to avoid chol failures of Sigma_cond)
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (Xp.n_cols != d)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(Xp.n_cols) + " instead of "
                             + std::to_string(d));

  arma::mat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_p = Trend::regressionModelMatrix(m_regmodel, Xpnorm);
  Xpnorm = trans(Xpnorm);
  // t0 = Bench::toc("Xpnorm         ", t0);

  // auto t0 = Bench::tic();
  arma::colvec y_trend = F_p * m_beta;  // / std::sqrt(m_sigma2);
  // t0 = Bench::toc("y_trend        ", t0);

  // Compute covariance between new data
  arma::mat Sigma = arma::ones(m, m);
  // double total_sd2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) + m_nugget;
  double total_sd2 = (m_sigma2 + m_nugget) * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  // Sigma *=  m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;
  Sigma *= m_sigma2 / total_sd2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  for (arma::uword i = 0; i < m; i++) {
    Sigma.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      Sigma.at(i, j) = Sigma.at(j, i) *= Cov((Xpnorm.col(i) - Xpnorm.col(j)), m_theta);
    }
  }

  // arma::mat T_newdata = chol(Sigma);
  // Compute covariance between training data and new data to predict
  // Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)
  arma::mat Xtnorm = trans(m_X);
  arma::mat Sigma21(n, m);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      // if (arma::all(dij==0))
      //  Sigma21.at(i, j) = 1.0;//m_sigma2 + m_nugget;
      // else
      Sigma21.at(i, j) = Cov((Xtnorm.col(i) - Xpnorm.col(j)), m_theta);
    }
  }
  Sigma21 *= m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;

  // Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE
  arma::mat Tinv_Sigma21 = solve(m_T, Sigma21, LinearAlgebra::default_solve_opts);

  // y.trend.cond <- y.trend + t(Tinv.Sigma21) %*% object@z
  y_trend += trans(Tinv_Sigma21) * m_z;

  // Sigma.cond <- Sigma11 - t(Tinv.Sigma21) %*% Tinv.Sigma21
  // arma::mat Sigma_cond = Sigma - XtX(Tinv_Sigma21);
  // arma::mat Sigma_cond = Sigma - trans(Tinv_Sigma21) * Tinv_Sigma21;
  arma::mat Sigma_cond = trimatl(Sigma);
  Sigma_cond.diag() += m_nugget;
  for (arma::uword i = 0; i < Tinv_Sigma21.n_cols; i++) {
    for (arma::uword j = 0; j <= i; j++) {
      Sigma_cond.at(i, j) -= cdot(Tinv_Sigma21.col(i), Tinv_Sigma21.col(j));
      Sigma_cond.at(j, i) = Sigma_cond.at(i, j);
    }
  }

  // T.cond <- chol(Sigma.cond + diag(nugget.sim, m, m))
  arma::mat tT_cond = LinearAlgebra::safe_chol_lower(Sigma_cond);

  // white.noise <- matrix(rnorm(m*nsim), m, nsim)
  // y.rand.cond <- t(T.cond) %*% white.noise
  // y <- matrix(y.trend.cond, m, nsim) + y.rand.cond
  arma::mat yp(m, nsim);
  yp.each_col() = y_trend;

  Random::reset_seed(seed);
  yp += tT_cond * Random::randn_mat(m, nsim) * std::sqrt(total_sd2);

  // Un-normalize simulations
  yp = m_centerY + m_scaleY * yp;  // * std::sqrt(m_sigma2);

  return yp;
}

/** Add new conditional data points to previous (X,y)
 * @param newy is m length column vector of new output
 * @param newX is m*d matrix of new input
 * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
 * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
 */
LIBKRIGING_EXPORT void NuggetKriging::update(const arma::vec& newy, const arma::mat& newX) {
  if (newy.n_elem != newX.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(newX.n_rows) + "x"
                             + std::to_string(newX.n_cols) + "), y: (" + std::to_string(newy.n_elem) + ")");

  // rebuild starting parameters
  Parameters parameters{std::make_optional(arma::vec(1, arma::fill::value(this->m_nugget))),
                        this->m_est_nugget,
                        std::make_optional(arma::vec(1, arma::fill::value(this->m_sigma2))),
                        this->m_est_sigma2,
                        std::make_optional(trans(this->m_theta)),
                        this->m_est_theta,
                        std::make_optional(trans(this->m_beta)),
                        this->m_est_beta};
  // re-fit
  // TODO refit() method which will use Shurr forms to fast update matrix (R, ...)
  bool normalize = (m_centerY == 0.0 && m_scaleY == 1) && (m_centerX.is_zero() && m_scaleX.is_zero());
  this->fit(
      arma::join_cols(m_y, newy), arma::join_cols(m_X, newX), m_regmodel, normalize, m_optim, m_objective, parameters);
}

LIBKRIGING_EXPORT std::string NuggetKriging::summary() const {
  std::ostringstream oss;
  auto colvec_printer = [&oss](const arma::colvec& v) {
    v.for_each([&oss, i = 0](const arma::colvec::elem_type& val) mutable {
      if (i++ > 0)
        oss << ", ";
      oss << val;
    });
  };

  oss << "* data: " << m_X.n_rows << " x " << m_X.n_cols << " -> " << m_y.n_rows << " x " << m_y.n_cols << "\n";
  oss << "* trend " << Trend::toString(m_regmodel);
  oss << ((m_est_beta) ? " (est.): " : ": ");
  colvec_printer(m_beta);
  oss << "\n";
  oss << "* variance";
  oss << ((m_est_sigma2) ? " (est.): " : ": ");
  oss << m_sigma2;
  oss << "\n";
  oss << "* covariance:\n";
  oss << "  * kernel: " << m_covType << "\n";
  oss << "  * range";
  oss << ((m_est_theta) ? " (est.): " : ": ");
  colvec_printer(m_theta);
  oss << "\n";
  oss << "  * nugget";
  oss << ((m_est_nugget) ? " (est.): " : ": ");
  oss << m_nugget;
  oss << "\n";
  oss << "  * fit:\n";
  oss << "    * objective: " << m_objective << "\n";
  oss << "    * optim: " << m_optim << "\n";
  return oss.str();
}
