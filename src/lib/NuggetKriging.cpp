// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio

#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Bench.hpp"
//#include "libKriging//*CacheFunction*/.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/KrigingException.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NuggetKriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/custom_hash_function.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <map>
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

LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const arma::fvec& y,
                                               const arma::fmat& X,
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

LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const NuggetKriging& other, ExplicitCopySpecifier)
    : NuggetKriging{other} {}

// arma::fmat XtX(arma::fmat &X) {
//   arma::fmat XtX = arma::zeros(X.n_cols,X.n_cols);
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

float NuggetKriging::_logLikelihood(const arma::fvec& _theta_alpha,
                                     arma::fvec* grad_out,
                                     NuggetKriging::OKModel* okm_data,
                                     std::map<std::string, double>* bench) const {
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

  float _alpha = _theta_alpha.at(d);
  arma::fvec _theta = _theta_alpha.head(d);

  auto t0 = Bench::tic();
  arma::fmat R = arma::fmat(n, n);
  if ((m_theta.size() == _theta.size()) && (_theta - m_theta).is_zero() && (this->m_T.memptr() != nullptr) && (n > this->m_T.n_rows) ) { // means that we want to recompute LL for same theta, for augmented Xy (using cholesky fast update).
    fd->T = LinearAlgebra::update_cholCov(&R, m_dX, _theta, Cov, _alpha, arma::fvec(n,arma::fill::ones), m_T);
  } else 
    fd->T = LinearAlgebra::cholCov(&R, m_dX, _theta, Cov, _alpha, arma::fvec(n,arma::fill::ones));
  t0 = Bench::toc(bench, "R = Cov(dX) & T = Chol(R)", t0);

  // Compute intermediate useful matrices
  fd->M = solve(fd->T, m_F, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "M = F \\ T", t0);

  arma::fvec Yt = solve(fd->T, m_y, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "Yt = y \\ T", t0);

  if (fd->is_beta_estim) {
    arma::fmat Q;
    arma::fmat G;
    qr_econ(Q, G, fd->M);
    t0 = Bench::toc(bench, "Q,G = QR(M)", t0);
    fd->beta = solve(G, Q.t() * Yt, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "B = Qt * Yt \\ G", t0);
  }

  fd->z = Yt - fd->M * fd->beta;
  t0 = Bench::toc(bench, "z = Yt - M * B", t0);

  fd->var = arma::accu(fd->z % fd->z) / n;
  t0 = Bench::toc(bench, "S2 = Acc(z * z) / n", t0);

  if (fd->is_nugget_estim)
    fd->nugget = (1 - _alpha) * fd->var;

  if (fd->is_sigma2_estim)
    fd->sigma2 = _alpha * fd->var;

  float ll = -0.5 * (n * log(2 * M_PI * fd->var) + 2 * sum(log(fd->T.diag())) + n);
  t0 = Bench::toc(bench, "ll = ...log(S2) + Sum(log(Td))...", t0);
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

    t0 = Bench::tic();
    std::vector<arma::fmat> gradsC(d);  // if (hess_out != nullptr)
    arma::fvec term1 = arma::fvec(d);    // if (hess_out != nullptr)

    arma::fmat Linv = solve(fd->T, arma::fmat(n, n,arma::fill::eye), LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "Li = I \\ T", t0);
    arma::fmat Cinv = (Linv.t() * Linv);  // Do NOT inv_sympd (slower): inv_sympd(R);
    t0 = Bench::toc(bench, "Ri = Lit * Li", t0);

    arma::fmat tT = fd->T.t();  // trimatu(trans(fd->T));
    t0 = Bench::toc(bench, "tT = Tt", t0);

    arma::fmat x = solve(tT, fd->z, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "x = z \\ tT", t0);

    arma::fcube gradC = arma::fcube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradC.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();
      arma::fmat gradC_k = arma::fmat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradC_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradC_k.at(i, j) = gradC_k.at(j, i) = gradC.slice(i).col(j)[k];
        }
      }
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      // should make a fast function trace_prod(A,B) -> sum_i(sum_j(Ai,j*Bj,i))
      term1.at(k)
          = as_scalar((trans(x) * gradC_k) * x) / fd->var;  //; //as_scalar((trans(x) * gradR_k) * x)/ sigma2_hat;
      float term2 = -arma::trace(Cinv * gradC_k);          //-arma::accu(Rinv % gradR_k_upper)
      (*grad_out).at(k) = (term1.at(k) + term2) / 2;
      t0 = Bench::toc(bench, "grad_ll[k] = xt * gradR_k / S2 + tr(Ri * gradR_k)", t0);
    }  // for (arma::uword k = 0; k < m_X.n_cols; k++)

    //  # partial derivative with respect to v = sigma^2 + delta^2
    //  dCdv <- R0 - diag(model@n)
    //  term1 <- -t(x) %*% dCdv %*% x / v
    //  term2 <- sum(Cinv * dCdv) # economic computation of trace(Cinv%*%C0)
    //  logLik.derivative[nparam] <- -0.5 * (term1 + term2) # /sigma2

    arma::fmat dCdv = R / _alpha;
    dCdv.diag().zeros();
    float _term1 = -as_scalar((trans(x) * dCdv) * x) / fd->var;
    float _term2 = arma::accu(arma::dot(Cinv, dCdv));
    (*grad_out).at(d) = -0.5 * (_term1 + _term2);
    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }  // if (grad_out != nullptr)
  return ll;
}

LIBKRIGING_EXPORT std::tuple<float, arma::fvec> NuggetKriging::logLikelihoodFun(const arma::fvec& _theta_alpha,
                                                                                const bool _grad,
                                                                                const bool _bench) {
  arma::fmat T;
  arma::fmat M;
  arma::fvec z;
  arma::fvec beta;
  float sigma2{};
  float nugget{};
  float var{};
  NuggetKriging::OKModel okm_data{T, M, z, beta, true, sigma2, true, nugget, true, var};

  float ll = -1;
  arma::fvec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::fvec(_theta_alpha.n_elem);
      ll = _logLikelihood(_theta_alpha, &grad, &okm_data, &bench);
    } else
      ll = _logLikelihood(_theta_alpha, nullptr, &okm_data, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad) {
      grad = arma::fvec(_theta_alpha.n_elem);
      ll = _logLikelihood(_theta_alpha, &grad, &okm_data, nullptr);
    } else
      ll = _logLikelihood(_theta_alpha, nullptr, &okm_data, nullptr);
  }

  return std::make_tuple(ll, std::move(grad));
}
// Objective function for fit: bayesian-like approach fromm RobustGaSP

float NuggetKriging::_logMargPost(const arma::fvec& _theta_alpha,
                                   arma::fvec* grad_out,
                                   NuggetKriging::OKModel* okm_data,
                                   std::map<std::string, double>* bench) const {
  // arma::cout << " theta: " << _theta << arma::endl;

  // In RobustGaSP:
  // neg_log_marginal_post_approx_ref <- function(param,nugget,
  // nugget.est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha) {
  //  lml=log_marginal_lik(param,nugget,nugget.est,R0,X,zero_mean,output,kernel_type,alpha);
  //  lp=log_approx_ref_prior(param,nugget,nugget.est,CL,a,b);
  //  -(lml+lp)
  //}
  // float log_marginal_lik(const Vec param,float nugget, const bool nugget_est, const List R0, const
  // Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output, Eigen::VectorXi
  // kernel_type,const Eigen::VectorXd alpha ){
  //  float nu=nugget;
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
  //  (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output); float log_S_2=log(S_2(0,0)); return
  //  (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2);
  //  }
  //}
  // float log_approx_ref_prior(const Vec param,float nugget, bool nugget_est, const Eigen::VectorXd CL,const float
  // a,const float b ){
  //  float nu=nugget;
  //  int param_size=param.size();beta
  //  Eigen::VectorX beta= param.array().exp().matrix();
  //  ...
  //  float t=CL.cwiseProduct(beta).sum()+nu;
  //  return -b*t + a*log(t);
  //}

  NuggetKriging::OKModel* fd = okm_data;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;

  float _alpha = _theta_alpha.at(d);
  arma::fvec _theta = _theta_alpha.head(d);

  auto t0 = Bench::tic();
  arma::fmat R = arma::fmat(n, n);
  if ((m_theta.size() == _theta.size()) && (_theta - m_theta).is_zero() && (this->m_T.memptr() != nullptr) && (n > this->m_T.n_rows) ) { // means that we want to recompute LL for same theta, for augmented Xy (using cholesky fast update).
    fd->T = LinearAlgebra::update_cholCov(&R, m_dX, _theta, Cov, _alpha, arma::fvec(n,arma::fill::ones), m_T);
  } else 
    fd->T = LinearAlgebra::cholCov(&R, m_dX, _theta, Cov, _alpha, arma::fvec(n,arma::fill::ones));
  t0 = Bench::toc(bench, "R = Cov(dX) & T = Chol(R)", t0);

  //  // Compute intermediate useful matrices
  //  fd->M = solve(fd->T, m_F, LinearAlgebra::solve_opts);
  //  arma::fmat Q;
  //  arma::fmat G;
  //  qr_econ(Q, G, fd->M);
  //
  //  arma::fmat H = Q * Q.t();  // if (hess_out != nullptr)
  //  arma::fvec Yt = solve(fd->T, m_y, LinearAlgebra::solve_opts);
  //  if (fd->is_beta_estim)
  // fd->beta = solve(trimatu(G), Q.t() * Yt, LinearAlgebra::solve_opts);
  //  fd->z = Yt - fd->M * fd->beta;

  // Keep RobustGaSP naming from now...
  arma::fmat X = m_F;
  arma::fmat L = fd->T;

  fd->M = solve(L, X, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "M = F \\ T", t0);

  arma::fmat R_inv_X = solve(trans(L), fd->M, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "RiF = Ri * F", t0);

  arma::fmat Xt_R_inv_X = trans(X) * R_inv_X;  // Xt%*%R.inv%*%X
  t0 = Bench::toc(bench, "FtRiF = Ft * RiF", t0);

  arma::fmat LX = chol(Xt_R_inv_X, "lower");  //  retrieve factor LX  in the decomposition
  t0 = Bench::toc(bench, "TF = Chol(FtRiF)", t0);

  arma::fmat R_inv_X_Xt_R_inv_X_inv_Xt_R_inv
      = R_inv_X
        * (solve(trans(LX),
                 solve(LX, trans(R_inv_X), LinearAlgebra::default_solve_opts),
                 LinearAlgebra::default_solve_opts));  // compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward
  // and one backward solve
  t0 = Bench::toc(bench, "RiFFtRiFiFtRi = RiF * RiFt \\ M \\ Mt", t0);

  arma::fvec Yt = solve(L, m_y, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "Yt = y \\ T", t0);

  if (fd->is_beta_estim) {
    arma::fmat Q;
    arma::fmat G;
    qr_econ(Q, G, fd->M);
    t0 = Bench::toc(bench, "Q,G = QR(M)", t0);
    fd->beta = solve(G, Q.t() * Yt, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "B = Qt * Yt \\ G", t0);
  }

  fd->z = Yt - fd->M * fd->beta;  // required for later predict
  t0 = Bench::toc(bench, "z = Yt - M * B", t0);

  arma::fmat yt_R_inv = trans(solve(trans(L), Yt, LinearAlgebra::default_solve_opts));
  t0 = Bench::toc(bench, "YtRi = Yt \\ Tt", t0);

  arma::fmat S_2 = (yt_R_inv * m_y - trans(m_y) * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv * m_y);
  t0 = Bench::toc(bench, "S2 = YtRi * y - yt * RiFFtRiFiFtRi * y", t0);
  // arma::cout << " S_2:" << S_2 << arma::endl;

  if (fd->is_sigma2_estim)
    fd->sigma2 = S_2(0, 0) / (n - m_F.n_cols);

  if (fd->is_nugget_estim)
    fd->nugget = S_2(0, 0) / (n - m_F.n_cols) * (1 / _alpha - 1);

  fd->var = S_2(0, 0) / (n - m_F.n_cols) / _alpha;

  float log_S_2 = log(S_2(0, 0));
  float log_marginal_lik = -sum(log(L.diag())) - sum(log(LX.diag())) - (n - m_F.n_cols) / 2.0 * log_S_2;
  t0 = Bench::toc(bench, "lml = -Sum(log(diag(T))) - Sum(log(diag(TF)))...", t0);
  // arma::cout << " log_marginal_lik:" << log_marginal_lik << arma::endl;

  // Default prior params
  float a = 0.2;
  float b = 1.0 / pow(n, 1.0 / d) * (a + d);

  arma::fvec CL = trans(max(m_X, 0) - min(m_X, 0)) / pow(n, 1.0 / d);
  t0 = Bench::toc(bench, "CL = (max(X) - min(X)) / n^1/d", t0);
  // arma::cout << " CL:" << CL << arma::endl;

  float t = arma::accu(CL % pow(_theta, -1.0)) + fd->nugget / fd->sigma2;
  // arma::cout << " t:" << t << arma::endl;
  float log_approx_ref_prior = -b * t + a * log(t);
  // arma::cout << " log_approx_ref_prior:" << log_approx_ref_prior << arma::endl;

  if (grad_out != nullptr) {
    // Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param,float nugget,  bool nugget_est, const List
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
    t0 = Bench::tic();
    arma::fvec ans = arma::fvec(m_X.n_cols,arma::fill::ones);
    arma::fmat Q_output = trans(yt_R_inv) - R_inv_X_Xt_R_inv_X_inv_Xt_R_inv * m_y;
    t0 = Bench::toc(bench, "Qo = YtRi - RiFFtRiFiFtRi * y", t0);

    arma::fcube gradR = arma::fcube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradR.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    arma::fmat Wb_k;
    for (arma::uword k = 0; k < m_X.n_cols; k++) {
      t0 = Bench::tic();
      arma::fmat gradR_k = arma::fmat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradR_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradR_k.at(i, j) = gradR_k.at(j, i) = gradR.slice(i).col(j)[k];
        }
      }
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      Wb_k = trans(solve(
                 trans(L), solve(L, gradR_k, LinearAlgebra::default_solve_opts), LinearAlgebra::default_solve_opts))
             - gradR_k * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      t0 = Bench::toc(bench, "Wb_k = gradR_k \\ T \\ Tt - gradR_k * RiFFtRiFiFtRi", t0);

      ans[k] = -0.5 * sum(Wb_k.diag()) + (n - m_F.n_cols) / 2.0 * (trans(m_y) * trans(Wb_k) * Q_output / S_2(0, 0))[0];
      t0 = Bench::toc(bench, "ans[k] = Sum(diag(Wb_k)) + yt * Wb_kt * Qo / S2...", t0);
    }
    // arma::cout << " log_marginal_lik_deriv:" << -ans * pow(_theta,2) << arma::endl;
    // arma::cout << " log_approx_ref_prior_deriv:" <<  a*CL/t - b*CL << arma::endl;
    (*grad_out).head(d) = ans - (a * CL / t - b * CL) / pow(_theta, 2.0);

    arma::fmat gradR_d = R / _alpha;
    gradR_d.diag().zeros();
    Wb_k = trans(
               solve(trans(L), solve(L, gradR_d, LinearAlgebra::default_solve_opts), LinearAlgebra::default_solve_opts))
           - gradR_d * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
    float ans_d
        = -0.5 * sum(Wb_k.diag()) + (n - m_F.n_cols) / 2.0 * (trans(m_y) * trans(Wb_k) * Q_output / S_2(0, 0))[0];

    (*grad_out).at(d) = ans_d - (a / t - b) / pow(_alpha, 2.0);

    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }

  // arma::cout << " lmp:" << (log_marginal_lik+log_approx_ref_prior) << arma::endl;
  return (log_marginal_lik + log_approx_ref_prior);
}

LIBKRIGING_EXPORT std::tuple<float, arma::fvec> NuggetKriging::logMargPostFun(const arma::fvec& _theta_alpha,
                                                                              const bool _grad,
                                                                              const bool _bench) {
  arma::fmat T;
  arma::fmat M;
  arma::fvec z;
  arma::fvec beta;
  float sigma2{};
  float nugget{};
  float var{};
  NuggetKriging::OKModel okm_data{T, M, z, beta, true, sigma2, true, nugget, true, var};

  float lmp = -1;
  arma::fvec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::fvec(_theta_alpha.n_elem);
      lmp = _logMargPost(_theta_alpha, &grad, &okm_data, &bench);
    } else
      lmp = _logMargPost(_theta_alpha, nullptr, &okm_data, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad) {
      grad = arma::fvec(_theta_alpha.n_elem);
      lmp = _logMargPost(_theta_alpha, &grad, &okm_data, nullptr);
    } else
      lmp = _logMargPost(_theta_alpha, nullptr, &okm_data, nullptr);
  }

  return std::make_tuple(lmp, std::move(grad));
}

LIBKRIGING_EXPORT float NuggetKriging::logLikelihood() {
  int d = m_theta.n_elem;
  arma::fvec _theta_alpha = arma::fvec(d + 1);
  _theta_alpha.head(d) = m_theta;
  _theta_alpha.at(d) = m_sigma2 / (m_sigma2 + m_nugget);
  return std::get<0>(NuggetKriging::logLikelihoodFun(_theta_alpha, false, false));
}

LIBKRIGING_EXPORT float NuggetKriging::logMargPost() {
  int d = m_theta.n_elem;
  arma::fvec _theta_alpha = arma::fvec(d + 1);
  _theta_alpha.head(d) = m_theta;
  _theta_alpha.at(d) = m_sigma2 / (m_sigma2 + m_nugget);
  return std::get<0>(NuggetKriging::logMargPostFun(_theta_alpha, false, false));
}

std::function<arma::fvec(const arma::fvec&)> NuggetKriging::reparam_to = [](const arma::fvec& _theta_alpha) {
  arma::fvec _theta_malpha = _theta_alpha;
  const arma::uword d = _theta_alpha.n_elem - 1;
  _theta_malpha.at(d) = 1 + alpha_lower - _theta_malpha.at(d);
  return Optim::reparam_to(_theta_malpha);
};

std::function<arma::fvec(const arma::fvec&)> NuggetKriging::reparam_from = [](const arma::fvec& _gamma) {
  arma::fvec _theta_alpha = Optim::reparam_from(_gamma);
  const arma::uword d = _theta_alpha.n_elem - 1;
  _theta_alpha.at(d) = 1 + alpha_lower - _theta_alpha.at(d);
  return _theta_alpha;
};

std::function<arma::fvec(const arma::fvec&, const arma::fvec&)> NuggetKriging::reparam_from_deriv
    = [](const arma::fvec& _theta_alpha, const arma::fvec& _grad) {
        arma::fvec D_theta_alpha = arma::conv_to<arma::fvec>::from(-_grad % _theta_alpha);
        const arma::uword d = D_theta_alpha.n_elem - 1;
        D_theta_alpha.at(d) = (1 + alpha_lower - _theta_alpha.at(d)) * _grad.at(d);
        return D_theta_alpha;
      };

float NuggetKriging::alpha_upper = 1.0;
float NuggetKriging::alpha_lower = 1E-3;

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 */
LIBKRIGING_EXPORT void NuggetKriging::fit(const arma::fvec& y,
                                          const arma::fmat& X,
                                          const Trend::RegressionModel& regmodel,
                                          bool normalize,
                                          const std::string& optim,
                                          const std::string& objective,
                                          const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<float(const arma::fvec& _gamma, arma::fvec* grad_out, NuggetKriging::OKModel* okm_data)> fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = /*CacheFunction*/([this](const arma::fvec& _gamma, arma::fvec* grad_out, NuggetKriging::OKModel* okm_data) {
        // Change variable for opt: . -> 1/exp(.)
        // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
        const arma::fvec _theta_alpha = NuggetKriging::reparam_from(_gamma);
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        float ll = this->_logLikelihood(_theta_alpha, grad_out, okm_data, nullptr);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > -ll: " << -ll << arma::endl;
        if (grad_out != nullptr) {
          *grad_out = -NuggetKriging::reparam_from_deriv(_theta_alpha, *grad_out);
          // DEBUG:
          // if (Optim::log_level>3) {
          //  arma::cout << "  > grad -ll: " << *grad_out << arma::endl;
          //  //// Check with numerical gradient:
          //  //for (size_t i = 0; i <_gamma.n_elem; i++) {
          //  //  arma::fvec eps = arma::zeros(_gamma.n_elem);
          //  //  eps(i) = 0.000001;
          //  //  const arma::fvec _theta_alpha_eps = reparam_from(_gamma + eps);
          //  //  float ll_eps = this->_logLikelihood(_theta_alpha_eps, nullptr, okm_data);
          //  //  arma::cout << "  > num_grad -ll: " << -(ll_eps - ll)/0.000001 << arma::endl;
          //  //}
          //}
        }
        return -ll;
      });
    } else {
      fit_ofn = /*CacheFunction*/([this](const arma::fvec& _gamma, arma::fvec* grad_out, NuggetKriging::OKModel* okm_data) {
        const arma::fvec _theta_alpha = _gamma;
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        float ll = this->_logLikelihood(_theta_alpha, grad_out, okm_data, nullptr);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > ll: " << ll << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
          *grad_out = -*grad_out;
        }
        return -ll;
      });
    }
  } else if (objective.compare("LMP") == 0) {
    // Our impl. of https://github.com/cran/RobustGaSP/blob/5cf21658e6a6e327be6779482b93dfee25d24592/R/rgasp.R#L303
    //@see Mengyang Gu, Xiao-jing Wang and Jim Berger, 2018, Annals of Statistics.
    if (Optim::reparametrize) {
      fit_ofn = /*CacheFunction*/([this](const arma::fvec& _gamma, arma::fvec* grad_out, NuggetKriging::OKModel* okm_data) {
        // Change variable for opt: . -> 1/exp(.)
        // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
        const arma::fvec _theta_alpha = NuggetKriging::reparam_from(_gamma);
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        float lmp = this->_logMargPost(_theta_alpha, grad_out, okm_data, nullptr);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
          *grad_out = -NuggetKriging::reparam_from_deriv(_theta_alpha, *grad_out);
        }
        return -lmp;
      });
    } else {
      fit_ofn = /*CacheFunction*/([this](const arma::fvec& _gamma, arma::fvec* grad_out, NuggetKriging::OKModel* okm_data) {
        const arma::fvec _theta_alpha = _gamma;
        // DEBUG: if (Optim::log_level>3) arma::cout << "> theta_alpha: " << _theta_alpha << arma::endl;
        float lmp = this->_logMargPost(_theta_alpha, grad_out, okm_data, nullptr);
        // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
        if (grad_out != nullptr) {
          // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
          *grad_out = -*grad_out;
        }
        return -lmp;
      });
    }
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LMP)");

  arma::frowvec centerX(d);
  arma::frowvec scaleX(d);
  float centerY;
  float scaleY;
  // Normalization of inputs and output
  m_normalize = normalize;
  if (m_normalize) {
    centerX = min(X, 0);
    scaleX = max(X, 0) - min(X, 0);
    centerY = min(y);
    scaleY = max(y) - min(y);
  } else {
    centerX = arma::frowvec(d, arma::fill::zeros);
    scaleX = arma::frowvec(d, arma::fill::ones);
    centerY = 0;
    scaleY = 1;
  }
  m_centerX = centerX;
  m_scaleX = scaleX;
  m_centerY = centerY;
  m_scaleY = scaleY;
  {  // FIXME why copies of newX and newy
    arma::fmat newX = X;
    newX.each_row() -= centerX;
    newX.each_row() /= scaleX;
    arma::fvec newy = (y - centerY) / scaleY;
    this->m_X = newX;
    this->m_y = newy;
  }

  // Now we compute the distance matrix between points. Will be used to compute R(theta) later (e.g. when fitting)
  m_dX = arma::fmat(d, n * n, arma::fill::none);
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

  arma::fmat theta0;
  if (parameters.theta.has_value()) {
    theta0 = parameters.theta.value();
    if (parameters.theta.value().n_cols != d && parameters.theta.value().n_rows == d)
      theta0 = parameters.theta.value().t();
    if (m_normalize)
      theta0.each_row() /= scaleX;
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
    arma::fmat T;
    arma::fmat M;
    arma::fvec z;
    arma::fvec beta;
    bool is_beta_estim = parameters.is_beta_estim;
    if (parameters.beta.has_value()) {
      beta = parameters.beta.value();
      if (m_normalize)
        beta /= scaleY;
    } else {
      is_beta_estim = true;  // force estim if no value given
    }
    float sigma2 = -1;
    bool is_sigma2_estim = parameters.is_sigma2_estim;
    if (parameters.sigma2.has_value()) {
      sigma2 = parameters.sigma2.value()[0];  // otherwise sigma2 will be re-calculated using given theta
      if (m_normalize)
        sigma2 /= (scaleY * scaleY);
    } else {
      is_sigma2_estim = true;  // force estim if no value given
    }
    float nugget = -1;
    bool is_nugget_estim = parameters.is_nugget_estim;
    if (parameters.nugget.has_value()) {
      nugget = parameters.nugget.value()[0];
      if (m_normalize)
        nugget /= (scaleY * scaleY);
    } else {
      is_nugget_estim = true;  // force estim if no value given
    }

    NuggetKriging::OKModel okm_data{
        T, M, z, beta, is_beta_estim, sigma2, is_sigma2_estim, nugget, is_nugget_estim, nugget + sigma2};

    arma::fvec gamma_tmp = arma::fvec(d + 1);
    gamma_tmp.head(d) = m_theta;
    gamma_tmp.at(d) = sigma2 / (nugget + sigma2);
    if (Optim::reparametrize) {
      gamma_tmp.head(d) = Optim::reparam_to(m_theta);
      gamma_tmp.at(d) = Optim::reparam_to_(1 + alpha_lower - sigma2 / (nugget + sigma2));
    }

    /* float min_ofn_tmp = */ fit_ofn(gamma_tmp, nullptr, &okm_data);

    m_T = std::move(okm_data.T);
    m_M = std::move(okm_data.M);
    m_z = std::move(okm_data.z);
    m_est_beta = is_beta_estim;
    if (m_est_beta) {
      m_beta = std::move(okm_data.beta);
    } else {
      m_beta = beta;
    }
    m_est_sigma2 = is_sigma2_estim;
    if (m_est_sigma2) {
      m_sigma2 = okm_data.sigma2;
    } else {
      m_sigma2 = sigma2;
    }
    m_est_nugget = is_nugget_estim;
    if (m_est_nugget) {
      m_nugget = okm_data.nugget;
    } else {
      m_nugget = nugget;
    }

  } else if (optim.rfind("BFGS", 0) == 0) {
    Random::init();

    arma::fvec theta_lower = Optim::theta_lower_factor * trans(max(m_X, 0) - min(m_X, 0));
    arma::fvec theta_upper = Optim::theta_upper_factor * trans(max(m_X, 0) - min(m_X, 0));

    if (Optim::variogram_bounds_heuristic) {
      arma::fvec dy2 = arma::fvec(n * n,arma::fill::zeros);
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
      arma::fvec dy2dX2_slope = dy2 / arma::sum(m_dX % m_dX, 0).t();
      // arma::cout << "dy2dX_slope:" << dy2dX_slope << arma::endl;
      dy2dX2_slope.replace(arma::datum::nan, 0.0);  // we are not interested in same points where dX=0, and dy=0
      arma::fvec w = dy2dX2_slope / sum(dy2dX2_slope);
      arma::fmat steepest_dX_mean = arma::abs(m_dX) * w;

      theta_lower = arma::max(theta_lower, Optim::theta_lower_factor * steepest_dX_mean);
      // no, only relevant for inf bound: theta_upper = arma::min(theta_upper, Optim::theta_upper_factor *
      // steepest_dX_mean);
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
      theta0 = arma::fmat(parameters.theta.value());
      if (m_normalize)
        theta0.each_row() /= scaleX;
    }
    // arma::cout << "theta0:" << theta0 << arma::endl;

    arma::fvec alpha0;
    if (parameters.sigma2.has_value() && parameters.nugget.has_value()) {
      alpha0 = arma::fvec(parameters.sigma2.value().n_elem * parameters.nugget.value().n_elem);
      for (size_t i = 0; i < parameters.sigma2.value().n_elem; i++) {
        for (size_t j = 0; j < parameters.nugget.value().n_elem; j++) {
          if ((parameters.sigma2.value()[i] < 0) || (parameters.nugget.value()[j] < 0)
              || (parameters.sigma2.value()[i] + parameters.nugget.value()[j] < 0))
            alpha0[i + j * parameters.sigma2.value().n_elem]
                = alpha_lower + (alpha_upper - alpha_lower) * (1 - std::pow(Random::randu(), 3.0));
          else
            alpha0[i + j * parameters.sigma2.value().n_elem]
                = parameters.sigma2.value()[i] / (parameters.sigma2.value()[i] + parameters.nugget.value()[j]);
        }
      }
    } else {
      alpha0 = alpha_lower + (alpha_upper - alpha_lower) * (1 - arma::pow(Random::randu_vec(theta0.n_rows), 3.0));
    }
    // arma::cout << "alpha0:" << alpha0 << arma::endl;

    arma::fvec gamma_lower = arma::fvec(d + 1);
    gamma_lower.head(d) = theta_lower;
    gamma_lower.at(d) = alpha_lower;
    arma::fvec gamma_upper = arma::fvec(d + 1);
    gamma_upper.head(d) = theta_upper;
    gamma_upper.at(d) = alpha_upper;
    if (Optim::reparametrize) {
      arma::fvec gamma_lower_tmp = gamma_lower;
      gamma_lower = NuggetKriging::reparam_to(gamma_upper);
      gamma_upper = NuggetKriging::reparam_to(gamma_lower_tmp);
      float gamma_lower_at_d = gamma_lower.at(d);
      gamma_lower.at(d) = gamma_upper.at(d);
      gamma_upper.at(d) = gamma_lower_at_d;
    }

    float min_ofn = std::numeric_limits<float>::infinity();

    for (arma::uword i = 0; i < theta0.n_rows; i++) {
      arma::fvec gamma_tmp = arma::fvec(d + 1);
      gamma_tmp.head(d) = theta0.row(i).t();
      gamma_tmp.at(d) = alpha0[i % alpha0.n_elem];
      if (Optim::reparametrize) {
        gamma_tmp = NuggetKriging::reparam_to(gamma_tmp);
      }

      gamma_lower = arma::min(gamma_tmp, gamma_lower);
      gamma_upper = arma::max(gamma_tmp, gamma_upper);

      if (Optim::log_level > 0) {
        arma::cout << "BFGS:" << arma::endl;
        arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
        arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
        arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
        arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
        arma::cout << "  normalize: " << m_normalize << arma::endl;
        arma::cout << "  lower_bounds: " << theta_lower << "";
        arma::cout << "                " << alpha_lower << arma::endl;
        arma::cout << "  upper_bounds: " << theta_upper << "";
        arma::cout << "                " << alpha_upper << arma::endl;
        arma::cout << "  start_point: " << theta0.row(i) << "";
        arma::cout << "               " << alpha0[i % alpha0.n_elem] << arma::endl;
      }

      arma::fmat T;
      arma::fmat M;
      arma::fvec z;
      arma::fvec beta;
      if (parameters.beta.has_value()) {
        beta = parameters.beta.value();
        if (m_normalize)
          beta /= scaleY;
      }
      float sigma2 = -1;
      if (parameters.sigma2.has_value()) {
        sigma2 = parameters.sigma2.value()[0];  // otherwise sigma2 will be re-calculated using given theta
        if (m_normalize)
          sigma2 /= (scaleY * scaleY);
      }
      float nugget = -1;
      if (parameters.nugget.has_value()) {
        nugget = parameters.nugget.value()[0];
        if (m_normalize)
          nugget /= (scaleY * scaleY);
      }

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
      optimizer.factr = Optim::objective_rel_tolerance / 1E-11;
      arma::ivec bounds_type{d + 1, arma::fill::value(2)};  // means both upper & lower bounds

      int retry = 0;
      while (retry <= Optim::max_restart) {
        arma::fvec gamma_0 = gamma_tmp;
        /*auto result = optimizer.minimize(
            [&okm_data, &fit_ofn](const arma::fvec& vals_inp, arma::fvec& grad_out) -> float {
                return fit_ofn(vals_inp, &grad_out, &okm_data);
              }, 
              arma::conv_to<arma::vec>::from(gamma_tmp),
              arma::conv_to<arma::vec>::from(gamma_lower).memptr(),
              arma::conv_to<arma::vec>::from(gamma_upper).memptr(),
            bounds_type.memptr());

        if (Optim::log_level > 0) {
          arma::cout << "     iterations: " << result.num_iters << arma::endl;
          if (Optim::reparametrize) {
            arma::cout << "     start_point: " << NuggetKriging::reparam_from(gamma_0).t() << " ";
            arma::cout << "     solution: " << NuggetKriging::reparam_from(gamma_tmp).t() << " ";
          } else {
            arma::cout << "     start_point: " << gamma_0.t() << " ";
            arma::cout << "     solution: " << gamma_tmp.t() << " ";
          }
        }

        float sol_to_lb_theta = arma::min(arma::abs(gamma_tmp.head(d) - gamma_lower.head(d)));
        float sol_to_ub_theta = arma::min(arma::abs(gamma_tmp.head(d) - gamma_upper.head(d)));
        float sol_to_b_theta
            = Optim::reparametrize ? sol_to_ub_theta : sol_to_lb_theta;         // just consider theta lower bound
        float sol_to_b_alpha = std::abs(gamma_tmp.at(d) - gamma_lower.at(d));  // just consider alpha lower bound
        // Optim::reparametrize
        //    ? std::abs(gamma_tmp.at(d) - gamma_upper.at(d))
        //    : std::abs(gamma_tmp.at(d) - gamma_lower.at(d));
        float sol_to_b = sol_to_b_theta < sol_to_b_alpha ? sol_to_b_theta : sol_to_b_alpha;
        if ((retry < Optim::max_restart)       //&& (result.num_iters <= 2 * d)
            && ((sol_to_b < arma::datum::eps)  // we fastly converged to one bound
                || (result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0))) {
          gamma_tmp.head(d)
              = (theta0.row(i).t() + theta_lower)
                / pow(2.0, retry + 1);  // so, re-use previous starting point and change it to middle-point
          gamma_tmp.at(d) = alpha_upper - (alpha0[i % alpha0.n_elem] + alpha_upper) / pow(2.0, retry + 1);

          if (Optim::reparametrize)
            gamma_tmp = NuggetKriging::reparam_to(gamma_tmp);

          gamma_lower = arma::min(gamma_tmp, gamma_lower);
          gamma_upper = arma::max(gamma_tmp, gamma_upper);

          retry++;
        } else {
          if (Optim::log_level > 1)
            result.print();
          break;
        }*/
      }

      // this last call also ensure that T and z are up-to-date with solution found.
      float min_ofn_tmp = fit_ofn(gamma_tmp, nullptr, &okm_data);

      if (Optim::log_level > 0) {
        arma::cout << "  best objective: " << min_ofn_tmp << arma::endl;
        if (Optim::reparametrize)
          arma::cout << "  best solution: " << NuggetKriging::reparam_from(gamma_tmp) << " ";
        else
          arma::cout << "  best solution: " << gamma_tmp << "";
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
    throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS[#])");

  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param Xp is m*d matrix of points where to predict output
 * @param std is true if return also stdev column vector
 * @param cov is true if return also cov matrix between Xp
 * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::fvec, arma::fvec, arma::fmat, arma::fmat, arma::fmat>
NuggetKriging::predict(const arma::fmat& Xp, bool withStd, bool withCov, bool withDeriv) {
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (Xp.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(Xp.n_cols) + " instead of "
                             + std::to_string(d));

  arma::fvec pred_mean(m);
  arma::fvec pred_stdev = arma::fvec(m,arma::fill::zeros);
  arma::fmat pred_cov = arma::fmat(m, m,arma::fill::zeros);
  arma::fmat pred_mean_deriv = arma::fmat(m, d,arma::fill::zeros);
  arma::fmat pred_stdev_deriv = arma::fmat(m, d,arma::fill::zeros);

  arma::fmat Xtnorm = trans(m_X);  // already normalized if needed
  arma::fmat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::fmat F_p = Trend::regressionModelMatrix(m_regmodel, Xpnorm);
  Xpnorm = trans(Xpnorm);

  // Compute covariance between training data and new data to predict
  arma::fmat R_pred = arma::fmat(n, m, arma::fill::ones);
  // float total_sd2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) + m_nugget;
  float total_sd2 = (m_sigma2 + m_nugget) * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  // R *= m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;
  R_pred *= m_sigma2 / total_sd2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      arma::fvec dij = Xtnorm.col(i) - Xpnorm.col(j);
      if (arma::any(dij != 0))
        R_pred.at(i, j) *= Cov(dij, m_theta);
      else
        R_pred.at(i, j) = 1.0;  // nugget kriging interpolates
    }
  }
  arma::fmat Tinv_pred = solve(m_T, R_pred, arma::solve_opts::fast);
  pred_mean = F_p * m_beta + trans(Tinv_pred) * m_z;
  // Un-normalize predictor
  pred_mean = m_centerY + m_scaleY * pred_mean;

  arma::fmat s2_predict_mat;
  arma::fmat FinvMtM;
  if (withStd || withCov) {
    arma::fmat TM = trans(chol(trans(m_M) * m_M));
    s2_predict_mat = solve(TM, trans(F_p - trans(Tinv_pred) * m_M), arma::solve_opts::fast);

    if (withDeriv) {
      arma::fmat m = trans(F_p - trans(Tinv_pred) * m_M);
      arma::fmat invMtM = inv_sympd(m_M.t() * m_M);
      FinvMtM = (F_p - trans(Tinv_pred) * m_M) * inv_sympd(m_M.t() * m_M);
    }
  }
  if (withStd) {
    // s2.predict.1 <- apply(Tinv.c.newdata, 2, crossprod)
    arma::fvec s2_predict_1 = trans(sum(Tinv_pred % Tinv_pred, 0));
    s2_predict_1.transform([](float val) {
      return (val > 1.0 ? 1.0 : val);
    });  // constrain this first part to not be negative (rationale: it is the whole stdev for simple kriging)

    // s2.predict.2 <- apply(s2.predict.mat, 2, crossprod)
    arma::fvec s2_predict_2 = trans(sum(s2_predict_mat % s2_predict_mat, 0));
    // s2.predict <- pmax(total.sd2 - s2.predict.1 + s2.predict.2, 0)

    arma::fmat s2_predict = total_sd2 * (1.0 - s2_predict_1 + s2_predict_2);
    s2_predict.transform([](float val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
    pred_stdev = sqrt(s2_predict);

    pred_stdev *= m_scaleY;
  }

  if (withCov) {
    // C.newdata <- covMatrix(object@covariance, newdata)[[1]]
    arma::fmat R_predpred = arma::fmat(m, m, arma::fill::ones);
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

    pred_cov *= m_scaleY;
  }

  if (withDeriv) {
    // # Compute derivatives of the covariance and trend functions
    for (arma::uword i = 0; i < m; i++) {  // for each Xp predict point... should be parallel ?

      arma::fmat dc = arma::fmat(n, d);
      for (arma::uword j = 0; j < n; j++) {
        dc.row(j) = R_pred.at(j, i) * trans(DlnCovDx(Xpnorm.col(i) - Xtnorm.col(j), m_theta));
      }

      const float h = 1.0E-5;  // Value is sensitive only for non linear trends. Otherwise, it gives exact results.
      arma::fmat tXpn_i_repd = arma::trans(Xpnorm.col(i) * arma::fmat(1, d, arma::fill::ones));  // just duplicate Xp.row(i) d times

      arma::fmat F_dx = (Trend::regressionModelMatrix(m_regmodel, tXpn_i_repd + h * arma::fmat(d, d,arma::fill::eye))
                        - Trend::regressionModelMatrix(m_regmodel, tXpn_i_repd - h * arma::fmat(d, d,arma::fill::eye)))
                       / (2 * h);

      // # Compute gradients of the kriging mean and variance
      arma::fmat W = solve(m_T, dc, LinearAlgebra::default_solve_opts);

      pred_mean_deriv.row(i) = trans(F_dx * m_beta + trans(W) * m_z);

      if (withStd) {
        arma::fmat pred_stdev_deriv_noTrend = Tinv_pred.t() * W;
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
LIBKRIGING_EXPORT arma::fmat NuggetKriging::simulate(const int nsim, const int seed, const arma::fmat& Xp) {
  // Here nugget.sim = 1e-10 to avoid chol failures of Sigma_cond)
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (Xp.n_cols != d)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(Xp.n_cols) + " instead of "
                             + std::to_string(d));

  arma::fmat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::fmat F_p = Trend::regressionModelMatrix(m_regmodel, Xpnorm);
  Xpnorm = trans(Xpnorm);
  // t0 = Bench::toc("Xpnorm         ", t0);

  // auto t0 = Bench::tic();
  arma::fvec y_trend = F_p * m_beta;  // / std::sqrt(m_sigma2);
  // t0 = Bench::toc("y_trend        ", t0);

  // Compute covariance between new data
  arma::fmat Sigma = arma::fmat(m, m, arma::fill::ones);
  // float total_sd2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) + m_nugget;
  float total_sd2 = (m_sigma2 + m_nugget) * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  // Sigma *=  m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;
  Sigma *= m_sigma2 / total_sd2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  for (arma::uword i = 0; i < m; i++) {
    Sigma.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      Sigma.at(i, j) = Sigma.at(j, i) *= Cov((Xpnorm.col(i) - Xpnorm.col(j)), m_theta);
    }
  }

  // arma::fmat T_newdata = chol(Sigma);
  // Compute covariance between training data and new data to predict
  // Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)
  arma::fmat Xtnorm = trans(m_X);
  arma::fmat Sigma21(n, m);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      arma::fvec dij = Xtnorm.col(i) - Xpnorm.col(j);
      if (arma::any(dij != 0))
        Sigma21.at(i, j) = Cov(dij, m_theta);
      else
        Sigma21.at(i, j) = 1.0;  // to interpolate design points
    }
  }
  Sigma21 *= m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0) / total_sd2;

  // Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE
  arma::fmat Tinv_Sigma21 = solve(m_T, Sigma21, LinearAlgebra::default_solve_opts);

  // y.trend.cond <- y.trend + t(Tinv.Sigma21) %*% object@z
  y_trend += trans(Tinv_Sigma21) * m_z;

  // Sigma.cond <- Sigma11 - t(Tinv.Sigma21) %*% Tinv.Sigma21
  // arma::fmat Sigma_cond = Sigma - XtX(Tinv_Sigma21);
  // arma::fmat Sigma_cond = Sigma - trans(Tinv_Sigma21) * Tinv_Sigma21;
  arma::fmat Sigma_cond = trimatl(Sigma);
  Sigma_cond.diag() += m_nugget;
  for (arma::uword i = 0; i < Tinv_Sigma21.n_cols; i++) {
    for (arma::uword j = 0; j <= i; j++) {
      Sigma_cond.at(i, j) -= cdot(Tinv_Sigma21.col(i), Tinv_Sigma21.col(j));
      Sigma_cond.at(j, i) = Sigma_cond.at(i, j);
    }
  }

  // T.cond <- chol(Sigma.cond + diag(nugget.sim, m, m))
  arma::fmat tT_cond = LinearAlgebra::safe_chol_lower(Sigma_cond);

  // white.noise <- matrix(rnorm(m*nsim), m, nsim)
  // y.rand.cond <- t(T.cond) %*% white.noise
  // y <- matrix(y.trend.cond, m, nsim) + y.rand.cond
  arma::fmat yp(m, nsim);
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
LIBKRIGING_EXPORT void NuggetKriging::update(const arma::fvec& newy, const arma::fmat& newX) {
  if (newy.n_elem != newX.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(newX.n_rows) + "x"
                             + std::to_string(newX.n_cols) + "), y: (" + std::to_string(newy.n_elem) + ")");

  // rebuild starting parameters
  Parameters parameters{
      std::make_optional(arma::fvec(1, arma::fill::value(this->m_nugget * this->m_scaleY * this->m_scaleY))),
      this->m_est_nugget,
      std::make_optional(arma::fvec(1, arma::fill::value(this->m_sigma2 * this->m_scaleY * this->m_scaleY))),
      this->m_est_sigma2,
      std::make_optional(trans(this->m_theta) % this->m_scaleX),
      this->m_est_theta,
      std::make_optional(trans(this->m_beta) * this->m_scaleY),
      this->m_est_beta};
  // re-fit
  // TODO refit() method which will use Shurr forms to fast update matrix (R, ...)
  this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                            newy),  // de-normalize previous data according to suite unnormed new data
            arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, newX),
            m_regmodel,
            m_normalize,
            m_optim,
            m_objective,
            parameters);
}

LIBKRIGING_EXPORT std::string NuggetKriging::summary() const {
  std::ostringstream oss;
  auto vec_printer = [&oss](const arma::fvec& v) {
    v.for_each([&oss, i = 0](const arma::fvec::elem_type& val) mutable {
      if (i++ > 0)
        oss << ", ";
      oss << val;
    });
  };

  if (m_X.is_empty() || m_X.n_rows == 0) {  // not yet fitted
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
  } else {
    oss << "* data";
    oss << ((m_normalize) ? " (normalized): " : ": ") << m_X.n_rows << "x";
    arma::frowvec Xmins = arma::min(m_X, 0);
    arma::frowvec Xmaxs = arma::max(m_X, 0);
    for (arma::uword i = 0; i < m_X.n_cols; i++) {
      oss << "[" << Xmins[i] << "," << Xmaxs[i] << "]";
      if (i < m_X.n_cols - 1)
        oss << ",";
    }
    oss << " -> " << m_y.n_elem << "x[" << arma::min(m_y) << "," << arma::max(m_y) << "]\n";
    oss << "* trend " << Trend::toString(m_regmodel);
    oss << ((m_est_beta) ? " (est.): " : ": ");
    vec_printer(m_beta);
    oss << "\n";
    oss << "* variance";
    oss << ((m_est_sigma2) ? " (est.): " : ": ");
    oss << m_sigma2;
    oss << "\n";
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
    oss << "  * range";
    oss << ((m_est_theta) ? " (est.): " : ": ");
    vec_printer(m_theta);
    oss << "\n";
    oss << "  * nugget";
    oss << ((m_est_nugget) ? " (est.): " : ": ");
    oss << m_nugget;
    oss << "\n";
    oss << "  * fit:\n";
    oss << "    * objective: " << m_objective << "\n";
    oss << "    * optim: " << m_optim << "\n";
  }
  return oss.str();
}

void NuggetKriging::save(const std::string filename) const {
  nlohmann::json j;

  j["version"] = 2;
  j["content"] = "NuggetKriging";

  // Cov_pow & std::function embedded by make_Cov
  j["covType"] = m_covType;
  j["X"] = to_json(arma::conv_to<arma::mat>::from(m_X));
  j["centerX"] = to_json(arma::conv_to<arma::rowvec>::from(m_centerX));
  j["scaleX"] = to_json(arma::conv_to<arma::rowvec>::from(m_scaleX));
  j["y"] = to_json(arma::conv_to<arma::colvec>::from(m_y));
  j["centerY"] = m_centerY;
  j["scaleY"] = m_scaleY;
  j["normalize"] = m_normalize;

  j["regmodel"] = Trend::toString(m_regmodel);
  j["optim"] = m_optim;
  j["objective"] = m_objective;
  j["dX"] = to_json(arma::conv_to<arma::mat>::from(m_dX));
  j["F"] = to_json(arma::conv_to<arma::mat>::from(m_F));
  j["T"] = to_json(arma::conv_to<arma::mat>::from(m_T));
  j["M"] = to_json(arma::conv_to<arma::mat>::from(m_M));
  j["z"] = to_json(arma::conv_to<arma::mat>::from(m_z));
  j["beta"] = to_json(arma::conv_to<arma::colvec>::from(m_beta));
  j["est_beta"] = m_est_beta;
  j["theta"] = to_json(arma::conv_to<arma::colvec>::from(m_theta));
  j["est_theta"] = m_est_theta;
  j["sigma2"] = m_sigma2;
  j["est_sigma2"] = m_est_sigma2;
  j["nugget"] = m_nugget;
  j["est_nugget"] = m_est_nugget;

  std::ofstream f(filename);
  f << std::setw(4) << j;
}

NuggetKriging NuggetKriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version != 2) {
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
  }
  std::string content = j["content"].template get<std::string>();
  if (content != "NuggetKriging") {
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'NuggetKriging'"));
  }

  std::string covType = j["covType"].template get<std::string>();
  NuggetKriging kr(covType);  // Cov_pow & std::function embedded by make_Cov

  kr.m_X = arma::conv_to<arma::fmat>::from(mat_from_json(j["X"]));
  kr.m_centerX = arma::conv_to<arma::frowvec>::from(rowvec_from_json(j["centerX"]));
  kr.m_scaleX = arma::conv_to<arma::frowvec>::from(rowvec_from_json(j["scaleX"]));
  kr.m_y = arma::conv_to<arma::fvec>::from(colvec_from_json(j["y"]));
  kr.m_centerY = j["centerY"].template get<decltype(kr.m_centerY)>();
  kr.m_scaleY = j["scaleY"].template get<decltype(kr.m_scaleY)>();
  kr.m_normalize = j["normalize"].template get<decltype(kr.m_normalize)>();

  std::string model = j["regmodel"].template get<std::string>();
  kr.m_regmodel = Trend::fromString(model);

  kr.m_optim = j["optim"].template get<decltype(kr.m_optim)>();
  kr.m_objective = j["objective"].template get<decltype(kr.m_objective)>();
  kr.m_dX = arma::conv_to<arma::fmat>::from(mat_from_json(j["dX"]));
  kr.m_F = arma::conv_to<arma::fmat>::from(mat_from_json(j["F"]));
  kr.m_T = arma::conv_to<arma::fmat>::from(mat_from_json(j["T"]));
  kr.m_M = arma::conv_to<arma::fmat>::from(mat_from_json(j["M"]));
  kr.m_z = arma::conv_to<arma::fvec>::from(colvec_from_json(j["z"]));
  kr.m_beta = arma::conv_to<arma::fvec>::from(colvec_from_json(j["beta"]));
  kr.m_est_beta = j["est_beta"].template get<decltype(kr.m_est_beta)>();
  kr.m_theta = arma::conv_to<arma::fvec>::from(colvec_from_json(j["theta"]));
  kr.m_est_theta = j["est_theta"].template get<decltype(kr.m_est_theta)>();
  kr.m_sigma2 = j["sigma2"].template get<decltype(kr.m_sigma2)>();
  kr.m_est_sigma2 = j["est_sigma2"].template get<decltype(kr.m_est_sigma2)>();
  kr.m_nugget = j["nugget"].template get<decltype(kr.m_nugget)>();
  kr.m_est_nugget = j["est_nugget"].template get<decltype(kr.m_est_nugget)>();

  return kr;
}
