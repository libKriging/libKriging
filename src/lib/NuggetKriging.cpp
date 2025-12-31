// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio

#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Bench.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/KrigingException.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NuggetKriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <map>
#include <tuple>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

#ifdef _OPENMP
#include <omp.h>

// Helper function to safely get optimal thread count
// Windows MSVC OpenMP can sometimes return unexpected values
inline int get_optimal_threads(int max_default = 2) {
  int max_threads = omp_get_max_threads();
  if (max_threads <= 0) {
    return 1;
  }
  return (max_threads > max_default) ? max_default : max_threads;
}
#endif

// Helper to get OpenBLAS thread control function (if available)
// Note: On macOS ARM64, use Accelerate framework instead of OpenBLAS
#if !defined(__APPLE__) || !defined(__arm64__)
#if defined(_MSC_VER)
  // MSVC doesn't support weak symbols; use runtime dynamic loading
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  namespace {
    typedef void (*openblas_set_num_threads_t)(int);
    openblas_set_num_threads_t get_openblas_set_num_threads() {
      static openblas_set_num_threads_t func = nullptr;
      static bool initialized = false;
      if (!initialized) {
        initialized = true;
        // Try to load from OpenBLAS DLL (used by numpy/scipy)
        HMODULE hModule = GetModuleHandleA("libopenblas.dll");
        if (!hModule) hModule = GetModuleHandleA("openblas.dll");
        if (hModule) {
          func = (openblas_set_num_threads_t)GetProcAddress(hModule, "openblas_set_num_threads");
        }
      }
      return func;
    }
  }
#else
  // GCC/Clang support weak symbols
  extern "C" {
    void openblas_set_num_threads(int num_threads) __attribute__((weak));
  }
  namespace {
    typedef void (*openblas_set_num_threads_t)(int);
    openblas_set_num_threads_t get_openblas_set_num_threads() {
      return openblas_set_num_threads;
    }
  }
#endif
#endif

/************************************************/
/**      NuggetKriging implementation        **/
/************************************************/

// This will create the dist(xi,xj) function above. Need to parse "covType".
void NuggetKriging::make_Cov(const std::string& covType) {
  m_covType = covType;
  if (covType.compare("gauss") == 0) {
    _Cov = Covariance::Cov_gauss;
    _DlnCovDtheta = Covariance::DlnCovDtheta_gauss;
    _DlnCovDx = Covariance::DlnCovDx_gauss;
    _Cov_pow = 2;
  } else if (covType.compare("exp") == 0) {
    _Cov = Covariance::Cov_exp;
    _DlnCovDtheta = Covariance::DlnCovDtheta_exp;
    _DlnCovDx = Covariance::DlnCovDx_exp;
    _Cov_pow = 1;
  } else if (covType.compare("matern3_2") == 0) {
    _Cov = Covariance::Cov_matern32;
    _DlnCovDtheta = Covariance::DlnCovDtheta_matern32;
    _DlnCovDx = Covariance::DlnCovDx_matern32;
    _Cov_pow = 1.5;
  } else if (covType.compare("matern5_2") == 0) {
    _Cov = Covariance::Cov_matern52;
    _DlnCovDtheta = Covariance::DlnCovDtheta_matern52;
    _DlnCovDx = Covariance::DlnCovDx_matern52;
    _Cov_pow = 2.5;
  } else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);

  // arma::cout << "make_Cov done." << arma::endl;
}

LIBKRIGING_EXPORT arma::mat NuggetKriging::covMat(const arma::mat& X1, const arma::mat& X2) {
  arma::mat Xn1 = X1;
  arma::mat Xn2 = X2;
  Xn1.each_row() -= m_centerX;
  Xn1.each_row() /= m_scaleX;
  Xn2.each_row() -= m_centerX;
  Xn2.each_row() /= m_scaleX;

  arma::mat R = arma::mat(X1.n_rows, X2.n_rows, arma::fill::none);
  LinearAlgebra::covMat_rect(&R, Xn1.t(), Xn2.t(), m_theta, _Cov, m_sigma2);
  return R;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const arma::vec& y,
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

LIBKRIGING_EXPORT NuggetKriging::NuggetKriging(const NuggetKriging& other, ExplicitCopySpecifier)
    : NuggetKriging{other} {}

arma::vec NuggetKriging::ones = arma::ones<arma::vec>(0);

void NuggetKriging::populate_Model(KModel& m,
                                   const arma::vec& theta,
                                   const double alpha,
                                   std::map<std::string, double>* bench) const {
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::uword p = m_F.n_cols;

  auto t0 = Bench::tic();
  // Reuse existing m.R allocation
  // check if we want to recompute model for same theta, for augmented Xy (using cholesky fast update).
  bool update = false;
  if (!m_is_empty)
    update = (m_sigma2 / (m_sigma2 + m_nugget) == alpha) && (m_theta.size() == theta.size())
             && (theta - m_theta).is_zero() && (this->m_T.memptr() != nullptr) && (n > this->m_T.n_rows);
  if (update) {
    m.L = LinearAlgebra::update_cholCov(&(m.R), m_dX, theta, _Cov, alpha, NuggetKriging::ones, m_T, m_R);
  } else
    m.L = LinearAlgebra::cholCov(&(m.R), m_dX, theta, _Cov, alpha, NuggetKriging::ones);
  t0 = Bench::toc(bench, "R = _Cov(dX) & L = Chol(R)", t0);

  // Compute intermediate useful matrices
  arma::mat Fystar = LinearAlgebra::solve(m.L, arma::join_rows(m_F, m_y));
  t0 = Bench::toc(bench, "Fy* = L \\ [F,y]", t0);
  m.Fstar = Fystar.head_cols(p);
  m.ystar = Fystar.tail_cols(1);

  arma::mat Q_qr;
  arma::mat R_qr;
  arma::qr_econ(Q_qr, R_qr, Fystar);
  t0 = Bench::toc(bench, "Q_qr,R_qr = QR(Fy*)", t0);

  m.Rstar = R_qr.head_cols(p);
  m.Qstar = Q_qr.head_cols(p);
  m.Estar = Q_qr.tail_cols(1) * R_qr.at(p, p);
  m.SSEstar = R_qr.at(p, p) * R_qr.at(p, p);

  if (m_est_beta) {
    m.betahat = LinearAlgebra::solve(m.Rstar, R_qr.tail_cols(1));
    t0 = Bench::toc(bench, "^b = R* \\ R_qr[1:p, p+1]", t0);
  } else {
    m.betahat = arma::vec(p, arma::fill::zeros);  // whatever: not used
  }
}

NuggetKriging::KModel NuggetKriging::make_Model(const arma::vec& theta,
                                                const double alpha,
                                                std::map<std::string, double>* bench) const {
  arma::mat R;
  arma::mat L;
  arma::mat Linv;
  arma::mat Fstar;
  arma::vec ystar;
  arma::mat Rstar;
  arma::mat Qstar;
  arma::vec Estar;
  double SSEstar{};
  arma::vec betahat;
  NuggetKriging::KModel m{R, L, Linv, Fstar, ystar, Rstar, Qstar, Estar, SSEstar, betahat};

  arma::uword n = m_X.n_rows;
  arma::uword p = m_F.n_cols;

  // Allocate matrices
  m.R = arma::mat(n, n, arma::fill::none);
  m.L = arma::mat(n, n, arma::fill::none);
  m.Linv = arma::mat();  // Empty matrix, will be filled on demand in gradient computation
  m.Fstar = arma::mat(n, p, arma::fill::none);
  m.ystar = arma::vec(n, arma::fill::none);
  m.Rstar = arma::mat(p, p, arma::fill::none);
  m.Qstar = arma::mat(n, p, arma::fill::none);
  m.Estar = arma::vec(n, arma::fill::none);
  m.betahat = arma::vec(p, arma::fill::none);

  // Populate the model
  populate_Model(m, theta, alpha, bench);

  return m;
}

// Objective function for fit : -logLikelihood

double NuggetKriging::_logLikelihood(const arma::vec& _theta_alpha,
                                     arma::vec* grad_out,
                                     NuggetKriging::KModel* model,
                                     std::map<std::string, double>* bench) const {
  // arma::cout << " theta, alpha: " << _theta_alpha.t() << arma::endl;

  arma::uword d = m_X.n_cols;
  double _alpha = _theta_alpha.at(d);
  // closure: (alpha,sigma2,nugget)
  // * alpha = sigma2 / (nugget + sigma2)
  // * sigma2 = nugget * alpha / (1-alpha)
  // * nugget = sigma2 * (1-alpha) / alpha
  double _sigma2 = m_sigma2;
  double _nugget = m_nugget;
  if (m_est_sigma2) {
    if (m_est_nugget) {
      // nothing to do now: use alpha as input, compute var = sigma2 + nugget later and then sigma2 and nugget
    } else {
      _sigma2 = m_nugget * _alpha / (1 - _alpha);
    }
  } else {
    if (m_est_nugget) {
      _nugget = m_sigma2 * (1 - _alpha) / _alpha;
    } else {
      _alpha
          = m_sigma2
            / (m_nugget + m_sigma2);  // force alpha value, and its derivative will be 0 (to force gradient convergence)
    }
  }
  arma::vec _theta = _theta_alpha.head(d);

  NuggetKriging::KModel m = make_Model(_theta, _alpha, bench);
  if (model != nullptr)
    *model = m;

  arma::uword n = m_X.n_rows;

  if (m_est_sigma2 && m_est_nugget) {  // DiceKriging: model@case == "LLconcentration_beta_alpha"
    // back to case 1: sigma2 + nugget = var (=z*z/n)
    double var = as_scalar(LinearAlgebra::crossprod(m.Estar)) / n;
    _sigma2 = _alpha * var;
    _nugget = (1 - _alpha) * var;
  }

  double ll = -0.5
              * (n * log(2 * M_PI * _sigma2 / _alpha) + 2 * sum(log(m.L.diag()))
                 + as_scalar(LinearAlgebra::crossprod(m.Estar)) * (_alpha / _sigma2));

  if (grad_out != nullptr) {
    auto t0 = Bench::tic();
    arma::vec terme1 = arma::vec(d);

    if ((m.Linv.memptr() == nullptr) || (arma::size(m.Linv) != arma::size(m.L))) {
      m.Linv = LinearAlgebra::solve(m.L, arma::mat(n, n, arma::fill::eye));
      t0 = Bench::toc(bench, "L ^-1", t0);
    }

    arma::mat Rinv = LinearAlgebra::crossprod(m.Linv);
    t0 = Bench::toc(bench, "R^-1 = t(L^-1) * L^-1", t0);

    arma::mat x = LinearAlgebra::solve(m.L.t(), m.Estar);
    t0 = Bench::toc(bench, "x = tL \\ z", t0);

    arma::cube gradR = arma::cube(n, n, d, arma::fill::none);
    const arma::vec zeros = arma::vec(d, arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++) {
      gradR.tube(i, i) = zeros;
      for (arma::uword j = 0; j < i; j++) {
        gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
        gradR.tube(j, i) = gradR.tube(i, j);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = gradR.slice(k);
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      // should make a fast function trace_prod(A,B) -> sum_i(sum_j(Ai,j*Bj,i))
      terme1.at(k) = as_scalar(x.t() * gradR_k * x) / (_sigma2 + _nugget);
      double terme2 = -arma::trace(Rinv * gradR_k);  //-arma::accu(Rinv % gradR_k_upper)
      (*grad_out).at(k) = (terme1.at(k) + terme2) / 2;
      t0 = Bench::toc(bench, "grad_ll[k] = xt * gradR_k / S2 + tr(Ri * gradR_k)", t0);
    }

    //  # partial derivative with respect to (alpha=) v = sigma^2 + delta^2
    //  dCdv <- R0 - diag(model@n)
    //  term1 <- -t(x) %*% dCdv %*% x / v
    //  term2 <- sum(Cinv * dCdv) # economic computation of trace(Cinv%*%C0)
    //  logLik.derivative[nparam] <- -0.5 * (term1 + term2) # /sigma2

    if (m_est_sigma2 && m_est_nugget) {
      arma::mat dRdv = m.R / _alpha;
      dRdv.diag().zeros();
      double _terme1 = -as_scalar((trans(x) * dRdv) * x) / (_sigma2 + _nugget);
      double _terme2 = arma::accu(arma::dot(Rinv, dRdv));
      (*grad_out).at(d) = -0.5 * (_terme1 + _terme2);
    } else if (m_est_sigma2 && !m_est_nugget) {
      arma::mat dRdv = m.R / _alpha;
      dRdv.diag().ones();
      double _terme1 = -as_scalar((trans(x) * dRdv) * x) / ((_sigma2 + _nugget) * (_sigma2 + _nugget));
      double _terme2 = arma::accu(arma::dot(Rinv / (_sigma2 + _nugget), dRdv));
      (*grad_out).at(d) = -0.5 * (_terme1 + _terme2) * _nugget / (1 - _alpha) / (1 - _alpha);
    } else                    // we do not support explicitely the case where nugget is estimated and sigma2 is fixed
      (*grad_out).at(d) = 0;  // if sigma2 and nugget are defined & fixed by user

    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> NuggetKriging::logLikelihoodFun(const arma::vec& _theta_alpha,
                                                                                const bool _grad,
                                                                                const bool _bench) {
  double ll = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta_alpha.n_elem);
      ll = _logLikelihood(_theta_alpha, &grad, nullptr, &bench);
    } else
      ll = _logLikelihood(_theta_alpha, nullptr, nullptr, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench) {
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;
    }

  } else {
    if (_grad) {
      grad = arma::vec(_theta_alpha.n_elem);
      ll = _logLikelihood(_theta_alpha, &grad, nullptr, nullptr);
    } else
      ll = _logLikelihood(_theta_alpha, nullptr, nullptr, nullptr);
  }

  return std::make_tuple(ll, std::move(grad));
}

// Objective function for fit: bayesian-like approach fromm RobustGaSP

double NuggetKriging::_logMargPost(const arma::vec& _theta_alpha,
                                   arma::vec* grad_out,
                                   NuggetKriging::KModel* model,
                                   std::map<std::string, double>* bench) const {
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
  //  MatrixXd Rinv_X_Xt_Rinv_X_inv_Xt_Rinv=
  //  R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
  //  //compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward and one backward solve MatrixXd yt_R_inv=
  //  (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); MatrixXd S_2=
  //  (yt_R_inv*output-output.transpose()*Rinv_X_Xt_Rinv_X_inv_Xt_Rinv*output); double log_S_2=log(S_2(0,0)); return
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

  arma::uword d = m_X.n_cols;
  double _alpha = _theta_alpha.at(d);
  // closure: (alpha,sigma2,nugget)
  // * alpha = sigma2 / (nugget + sigma2)
  // * sigma2 = nugget * alpha / (1-alpha)
  // * nugget = sigma2 * (1-alpha) / alpha
  double _sigma2;
  double _nugget;
  if (m_est_sigma2) {
    if (m_est_nugget) {
      // nothing to do now: use alpha as input, compute var = sigma2 + nugget later and then sigma2 and nugget
    } else {
      _sigma2 = m_nugget * _alpha / (1 - _alpha);
    }
  } else {
    if (m_est_nugget) {
      _nugget = m_sigma2 * (1 - _alpha) / _alpha;
    } else {
      _alpha
          = m_sigma2
            / (m_nugget + m_sigma2);  // force alpha value, and its derivative will be 0 (to force gradient convergence)
    }
  }
  arma::vec _theta = _theta_alpha.head(d);

  NuggetKriging::KModel m = make_Model(_theta, _alpha, bench);
  if (model != nullptr)
    *model = m;

  arma::uword n = m_X.n_rows;
  arma::uword p = m_F.n_cols;

  // RobustGaSP naming...
  // arma::mat X = m_F;
  // arma::mat L = fd->T;

  auto t0 = Bench::tic();
  // m.Fstar : fd->M = solve(L, X, LinearAlgebra::default_solve_opts);

  // arma::mat Rinv_X = solve(trans(L), fd->M, LinearAlgebra::default_solve_opts);
  arma::mat Rinv_X = LinearAlgebra::solve(m.L.t(), m.Fstar);

  // arma::mat Xt_Rinv_X = trans(X) * Rinv_X;  // Xt%*%R.inv%*%X
  arma::mat Xt_Rinv_X = m_F.t() * Rinv_X;

  // arma::mat LX = chol(Xt_Rinv_X, "lower");  //  retrieve factor LX  in the decomposition
  arma::mat LX = LinearAlgebra::safe_chol_lower(Xt_Rinv_X);

  // arma::mat Rinv_X_Xt_Rinv_X_inv_Xt_Rinv
  //     = Rinv_X
  //       * (solve(trans(LX),
  //                solve(LX, trans(Rinv_X), LinearAlgebra::default_solve_opts),
  //                LinearAlgebra::default_solve_opts));  // compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward
  arma::mat Rinv_X_Xt_Rinv_X_inv_Xt_Rinv
      = Rinv_X
        * (LinearAlgebra::solve(
            trans(LX),
            LinearAlgebra::solve(LX, trans(Rinv_X))));  // compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward

  arma::mat yt_Rinv = trans(solve(trans(m.L), m.ystar));
  t0 = Bench::toc(bench, "YtRi = Yt \\ Tt", t0);

  arma::mat S_2 = (yt_Rinv * m_y - trans(m_y) * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv * m_y);
  t0 = Bench::toc(bench, "S2 = YtRi * y - yt * RiFFtRiFiFtRi * y", t0);

  if (m_est_sigma2 && m_est_nugget) {
    _sigma2 = S_2(0, 0) / (n - p);  //? - 2);
    _nugget = _sigma2 * (1 - _alpha) / _alpha;
  }
  double log_S_2 = log(_sigma2 * (n - p));

  double log_marginal_lik = -sum(log(m.L.diag())) - sum(log(LX.diag())) - (n - p) / 2.0 * log_S_2;
  t0 = Bench::toc(bench, "lml = -Sum(log(diag(T))) - Sum(log(diag(TF)))...", t0);
  // arma::cout << " log_marginal_lik:" << log_marginal_lik << arma::endl;

  // Default prior params
  double a = 0.2;
  double b = 1.0 / pow(n, 1.0 / d) * (a + d);
  // t0 = Bench::toc(bench,"b             ", t0);

  arma::vec CL = trans(max(m_X, 0) - min(m_X, 0)) / pow(n, 1.0 / d);
  t0 = Bench::toc(bench, "CL = (max(X) - min(X)) / n^1/d", t0);

  double t = arma::accu(CL / _theta) + _nugget / _sigma2;
  // arma::cout << " a:" << a << arma::endl;
  // arma::cout << " b:" << b << arma::endl;
  // arma::cout << " t:" << t << arma::endl;
  double log_approx_ref_prior = -b * t + a * log(t);
  // arma::cout << " log_approx_ref_prior:" << log_approx_ref_prior << arma::endl;

  if (grad_out != nullptr) {
    // Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param,double nugget,  bool nugget_est, const List
    // R0, const Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output,
    // Eigen::VectorXi kernel_type,const Eigen::VectorXd alpha){
    // ...
    // VectorXd ans=VectorXd::Ones(param_size);
    // ...
    // MatrixXd Q_output= yt_R_inv.transpose()-Rinv_X_Xt_Rinv_X_inv_Xt_Rinv*output;
    // MatrixXd dev_R_i;
    // MatrixXd Wb_ti;
    // //allow different choices of kernels
    //
    // for(int ti=0;ti<p;ti++){
    //   //kernel_type_ti=kernel_type[ti];
    //   if(kernel_type[ti]==3){
    //     dev_R_i=matern_5_2_deriv( R0[ti],R_ori,beta[ti]);  //now here I have R_ori instead of R
    //   }else {...}
    //   Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
    //   ans[ti]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0);
    // }

    t0 = Bench::tic();
    arma::vec ans = arma::vec(d, arma::fill::ones);
    arma::mat Q_output = trans(yt_Rinv) - Rinv_X_Xt_Rinv_X_inv_Xt_Rinv * m_y;
    t0 = Bench::toc(bench, "Qo = YtRi - RiFFtRiFiFtRi * y", t0);

    arma::cube gradR = arma::cube(n, n, d, arma::fill::zeros);
    // const arma::vec zeros = arma::vec(d,arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++) {
      // gradR.tube(i, i) = zeros;
      for (arma::uword j = 0; j < i; j++) {
        gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
        gradR.tube(j, i) = gradR.tube(i, j);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    arma::mat Wb_k;
    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = gradR.slice(k);
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      Wb_k = trans(LinearAlgebra::solve(trans(m.L), LinearAlgebra::solve(m.L, gradR_k)))
             - gradR_k * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
      t0 = Bench::toc(bench, "Wb_k = gradR_k \\ L \\ Tt - gradR_k * RiFFtRiFiFtRi", t0);

      ans[k] = -sum(Wb_k.diag()) / 2.0 + as_scalar(trans(m_y) * trans(Wb_k) * Q_output) / (2.0 * _sigma2);
      t0 = Bench::toc(bench, "ans[k] = Sum(diag(Wb_k)) + yt * Wb_kt * Qo / S2...", t0);
    }
    // arma::cout << " log_marginal_lik_deriv:" << -ans * pow(_theta,2) << arma::endl;
    // arma::cout << " log_approx_ref_prior_deriv:" <<  a*CL/t - b*CL << arma::endl;

    (*grad_out).head(d) = ans - (a * CL / t - b * CL) / square(_theta);

    if (m_est_sigma2 || m_est_nugget) {
      arma::mat gradR_d = m.R / _alpha;
      gradR_d.diag().zeros();
      Wb_k = trans(LinearAlgebra::solve(trans(m.L), LinearAlgebra::solve(m.L, gradR_d)))
             - gradR_d * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
      double ans_d = -sum(Wb_k.diag()) / 2.0 + as_scalar(trans(m_y) * trans(Wb_k) * Q_output) / (2.0 * _sigma2);
      (*grad_out).at(d) = ans_d - (a / t - b) / pow(_alpha, 2.0);
    } else
      (*grad_out).at(d) = 0;  // if sigma2 and nugget are defined & fixed by user
    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }

  // arma::cout << " lmp:" << (log_marginal_lik+log_approx_ref_prior) << arma::endl;
  return (log_marginal_lik + log_approx_ref_prior);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> NuggetKriging::logMargPostFun(const arma::vec& _theta_alpha,
                                                                              const bool _grad,
                                                                              const bool _bench) {
  double lmp = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta_alpha.n_elem);
      lmp = _logMargPost(_theta_alpha, &grad, nullptr, &bench);
    } else
      lmp = _logMargPost(_theta_alpha, nullptr, nullptr, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench) {
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;
    }

  } else {
    if (_grad) {
      grad = arma::vec(_theta_alpha.n_elem);
      lmp = _logMargPost(_theta_alpha, &grad, nullptr, nullptr);
    } else
      lmp = _logMargPost(_theta_alpha, nullptr, nullptr, nullptr);
  }

  return std::make_tuple(lmp, std::move(grad));
}

LIBKRIGING_EXPORT double NuggetKriging::logLikelihood() {
  int d = m_theta.n_elem;
  arma::vec _theta_alpha = arma::vec(d + 1);
  _theta_alpha.head(d) = m_theta;
  _theta_alpha.at(d) = m_sigma2 / (m_sigma2 + m_nugget);
  return std::get<0>(NuggetKriging::logLikelihoodFun(_theta_alpha, false, false));
}

LIBKRIGING_EXPORT double NuggetKriging::logMargPost() {
  int d = m_theta.n_elem;
  arma::vec _theta_alpha = arma::vec(d + 1);
  _theta_alpha.head(d) = m_theta;
  _theta_alpha.at(d) = m_sigma2 / (m_sigma2 + m_nugget);
  return std::get<0>(NuggetKriging::logMargPostFun(_theta_alpha, false, false));
}

std::function<arma::vec(const arma::vec&)> NuggetKriging::reparam_to = [](const arma::vec& _theta_alpha) {
  arma::vec _theta_malpha = _theta_alpha;
  const arma::uword d = _theta_alpha.n_elem - 1;
  _theta_malpha.at(d) = 1 + alpha_lower - _theta_malpha.at(d);
  return Optim::reparam_to(_theta_malpha);
};

std::function<arma::vec(const arma::vec&)> NuggetKriging::reparam_from = [](const arma::vec& _gamma) {
  arma::vec _theta_alpha = Optim::reparam_from(_gamma);
  const arma::uword d = _theta_alpha.n_elem - 1;
  _theta_alpha.at(d) = 1 + alpha_lower - _theta_alpha.at(d);
  return _theta_alpha;
};

std::function<arma::vec(const arma::vec&, const arma::vec&)> NuggetKriging::reparam_from_deriv
    = [](const arma::vec& _theta_alpha, const arma::vec& _grad) {
        arma::vec D_theta_alpha = arma::conv_to<arma::vec>::from(-_grad % _theta_alpha);
        const arma::uword d = D_theta_alpha.n_elem - 1;
        D_theta_alpha.at(d) = (1 + alpha_lower - _theta_alpha.at(d)) * _grad.at(d);
        return D_theta_alpha;
      };

double NuggetKriging::alpha_upper = 1.0;
double NuggetKriging::alpha_lower = 1E-3;

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 */
LIBKRIGING_EXPORT void NuggetKriging::fit(const arma::vec& y,
                                          const arma::mat& X,
                                          const Trend::RegressionModel& regmodel,
                                          bool normalize,
                                          const std::string& optim,
                                          const std::string& objective,
                                          const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<double(const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::KModel* km_data)> fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::KModel* km_data) {
        // Change variable for opt: . -> 1/exp(.)
        const arma::vec _theta_alpha = NuggetKriging::reparam_from(_gamma);
        double ll = this->_logLikelihood(_theta_alpha, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -NuggetKriging::reparam_from_deriv(_theta_alpha, *grad_out);
        }
        return -ll;
      };
    } else {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::KModel* km_data) {
        const arma::vec _theta_alpha = _gamma;
        double ll = this->_logLikelihood(_theta_alpha, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -*grad_out;
        }
        return -ll;
      };
    }
  } else if (objective.compare("LMP") == 0) {
    // Our impl. of https://github.com/cran/RobustGaSP/blob/5cf21658e6a6e327be6779482b93dfee25d24592/R/rgasp.R#L303
    //@see Mengyang Gu, Xiao-jing Wang and Jim Berger, 2018, Annals of Statistics.
    if (Optim::reparametrize) {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::KModel* km_data) {
        // Change variable for opt: . -> 1/exp(.)
        const arma::vec _theta_alpha = NuggetKriging::reparam_from(_gamma);
        double lmp = this->_logMargPost(_theta_alpha, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -NuggetKriging::reparam_from_deriv(_theta_alpha, *grad_out);
        }
        return -lmp;
      };
    } else {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NuggetKriging::KModel* km_data) {
        const arma::vec _theta_alpha = _gamma;
        double lmp = this->_logMargPost(_theta_alpha, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -*grad_out;
        }
        return -lmp;
      };
    }
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LMP)");

  arma::rowvec centerX;
  arma::rowvec scaleX;
  double centerY;
  double scaleY;
  // Normalization of inputs and output
  m_normalize = normalize;
  if (m_normalize) {
    centerX = min(X, 0);
    scaleX = max(X, 0) - min(X, 0);
    centerY = min(y);
    scaleY = max(y) - min(y);
  } else {
    centerX = arma::rowvec(d, arma::fill::zeros);
    scaleX = arma::rowvec(d, arma::fill::ones);
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
    arma::vec newy = (y - centerY) / scaleY;
    this->m_X = newX;
    this->m_y = newy;
  }

  // Now we compute the distance matrix between points. Will be used to compute R(theta) later (e.g. when fitting)
  // Note: m_dX is transposed compared to m_X
  m_dX = LinearAlgebra::compute_dX(m_X);
  m_maxdX = arma::max(arma::abs(m_dX), 1);

  // Define regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);
  m_est_beta = parameters.is_beta_estim && (m_regmodel != Trend::RegressionModel::None);
  if (!m_est_beta && parameters.beta.has_value()
      && parameters.beta.value().n_elem > 0) {  // Then force beta to be fixed (not estimated, no variance)
    m_est_beta = false;
    m_beta = parameters.beta.value();
    if (m_normalize)
      m_beta /= scaleY;
  } else
    m_est_beta = true;

  arma::mat theta0;
  if (parameters.theta.has_value()) {
    theta0 = parameters.theta.value();
    if ((theta0.n_cols != d) && (theta0.n_rows == d))
      theta0 = theta0.t();
    if (m_normalize)
      theta0.each_row() /= scaleX;
    if (theta0.n_cols != d)
      throw std::runtime_error("Dimension of theta should be nx" + std::to_string(d) + " instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
  }

  if (optim == "none") {  // just keep given theta, no optimisation of ll (but estim beta still possible)
    if (!parameters.theta.has_value())
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");
    if (!parameters.nugget.has_value())
      throw std::runtime_error("Nugget should be given, when optim=none");
    if (!parameters.sigma2.has_value())
      throw std::runtime_error("Sigma2 should be given, when optim=none");

    m_theta = trans(theta0.row(0));
    m_est_theta = false;

    double sigma2 = -1;
    m_est_sigma2 = parameters.is_sigma2_estim;
    if (parameters.sigma2.has_value()) {
      sigma2 = parameters.sigma2.value()[0];  // otherwise sigma2 will be re-calculated using given theta
      if (m_normalize)
        sigma2 /= (scaleY * scaleY);
    } else
      m_est_sigma2 = true;
    double nugget = 0;
    m_est_nugget = parameters.is_nugget_estim;
    if (parameters.nugget.has_value()) {
      nugget = parameters.nugget.value()[0];
      if (m_normalize)
        nugget /= (scaleY * scaleY);
    } else
      m_est_nugget = true;

    NuggetKriging::KModel m = make_Model(m_theta, sigma2 / (nugget + sigma2), nullptr);
    m_is_empty = false;
    m_T = std::move(m.L);
    m_R = std::move(m.R);
    m_M = std::move(m.Fstar);
    m_circ = std::move(m.Rstar);
    m_star = std::move(m.Qstar);
    if (m_est_beta) {
      m_beta = std::move(m.betahat);
      m_z = std::move(m.Estar);
    } else {
      // m_beta = parameters.beta.value(); already done above
      m_z = std::move(m.ystar) - m_M * m_beta;
    }
    if (m_est_sigma2) {
      m_sigma2 = m.SSEstar / n;
    } else {
      m_sigma2 = sigma2;
    }
    if (m_est_nugget) {
      m_nugget = 0;
    } else {
      m_nugget = nugget;
    }

  } else if (optim.rfind("BFGS", 0) == 0) {
    Random::init();

    arma::vec theta_lower = Optim::theta_lower_factor * m_maxdX;
    arma::vec theta_upper = Optim::theta_upper_factor * m_maxdX;

    if (Optim::variogram_bounds_heuristic) {
      arma::vec dy2 = arma::vec(n * n, arma::fill::zeros);
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
      // arma::cout << "steepest_dX_mean:" << steepest_dX_mean << arma::endl;

      theta_lower = arma::max(theta_lower, Optim::theta_lower_factor * steepest_dX_mean);
      // no, only relevant for inf bound: theta_upper = arma::min(theta_upper, Optim::theta_upper_factor *
      // steepest_dX_mean);
      theta_lower = arma::min(theta_lower, theta_upper);
      theta_upper = arma::max(theta_lower, theta_upper);
    }
    // arma::cout << "theta_lower:" << theta_lower << arma::endl;
    // arma::cout << "theta_upper:" << theta_upper << arma::endl;

    int multistart = 1;
    try {
      multistart = std::stoi(optim.substr(4));
    } catch (std::invalid_argument&) {
      // let multistart = 1
    }

    // Configure threads for Armadillo/BLAS to balance nested parallelism
    // Each of the 'multistart' threads will use internal parallelism
    unsigned int n_cpu = std::thread::hardware_concurrency();
    if (n_cpu > 0 && multistart > 1) {
      unsigned int threads_per_worker = std::max(1u, n_cpu / multistart);
      
      // Set OpenBLAS threads (if available)
      // Note: Skip on macOS ARM64 where Accelerate framework is used
#if !defined(__APPLE__) || !defined(__arm64__)
      auto openblas_fn = get_openblas_set_num_threads();
      if (openblas_fn != nullptr) {
        openblas_fn(threads_per_worker);
      }
#endif
      
      // Set OpenMP threads (for Armadillo operations that use OpenMP)
      #ifdef _OPENMP
      omp_set_num_threads(threads_per_worker);
      #endif
      
      if (Optim::log_level > Optim::log_none) {
        arma::cout << "Threads per worker: " << threads_per_worker 
                   << " (total CPUs: " << n_cpu << ", multistart: " << multistart << ")" << arma::endl;
      }
    }

    arma::mat theta0_rand
        = arma::repmat(trans(theta_lower), multistart, 1)
          + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
    // theta0 = arma::abs(0.5 + Random::randn_mat(multistart, d) / 6.0)
    //          % arma::repmat(max(m_X, 0) - min(m_X, 0), multistart, 1);

    if (parameters.theta.has_value()) {  // just use given theta(s) as starting values for multi-bfgs
      multistart = std::max(multistart, (int)theta0.n_rows);
      theta0 = arma::join_cols(theta0, theta0_rand);  // append random starting points to given ones
      theta0.resize(multistart, theta0.n_cols);       // keep only multistart first rows
    } else {
      theta0 = theta0_rand;
    }
    // arma::cout << "theta0:" << theta0 << arma::endl;

    arma::vec alpha0;
    if (parameters.sigma2.has_value() && parameters.nugget.has_value()) {
      alpha0 = arma::vec(parameters.sigma2.value().n_elem * parameters.nugget.value().n_elem);
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

    arma::vec gamma_lower = arma::vec(d + 1);
    gamma_lower.head(d) = theta_lower;
    gamma_lower.at(d) = alpha_lower;
    arma::vec gamma_upper = arma::vec(d + 1);
    gamma_upper.head(d) = theta_upper;
    gamma_upper.at(d) = alpha_upper;
    if (Optim::reparametrize) {
      arma::vec gamma_lower_tmp = gamma_lower;
      gamma_lower = NuggetKriging::reparam_to(gamma_upper);
      gamma_upper = NuggetKriging::reparam_to(gamma_lower_tmp);
      double gamma_lower_at_d = gamma_lower.at(d);
      gamma_lower.at(d) = gamma_upper.at(d);
      gamma_upper.at(d) = gamma_lower_at_d;
    }

    double min_ofn = std::numeric_limits<double>::infinity();

    // Set estimation flags before parallel execution (these are read by fit_ofn in all threads)
    m_est_sigma2 = parameters.is_sigma2_estim;
    if ((!m_est_sigma2) && (parameters.sigma2.has_value())) {
      m_sigma2 = parameters.sigma2.value()[0];
      if (m_normalize)
        m_sigma2 /= (scaleY * scaleY);
    } else {
      m_est_sigma2 = true;  // force estim if no value given
    }
    m_est_nugget = parameters.is_nugget_estim;
    if ((!m_est_nugget) && (parameters.nugget.has_value())) {
      m_nugget = parameters.nugget.value()[0];
      if (m_normalize)
        m_nugget /= (scaleY * scaleY);
    } else {
      m_est_nugget = true;  // force estim if no value given
    }

      // Preallocate KModels for each thread to avoid race conditions
    arma::uword n_data = n;
    arma::uword p_data = m_F.n_cols;
    std::vector<NuggetKriging::KModel> preallocated_models(multistart);
    
    if (Optim::log_level > Optim::log_none) {
      arma::cout << "Preallocating " << multistart << " KModel structures (n=" 
                 << n_data << ", p=" << p_data << ")..." << arma::endl;
    }
    
    for (int i = 0; i < multistart; i++) {
      auto& m = preallocated_models[i];
      m.R = arma::mat(n_data, n_data, arma::fill::none);
      m.L = arma::mat(n_data, n_data, arma::fill::none);
      m.Linv = arma::mat();  // Empty matrix
      m.Fstar = arma::mat(n_data, p_data, arma::fill::none);
      m.ystar = arma::vec(n_data, arma::fill::none);
      m.Rstar = arma::mat(p_data, p_data, arma::fill::none);
      m.Qstar = arma::mat(n_data, p_data, arma::fill::none);
      m.Estar = arma::vec(n_data, arma::fill::none);
      m.betahat = arma::vec(p_data, arma::fill::none);
      m.SSEstar = 0.0;
    }

    // Multi-threading implementation for BFGS multistart
    // Each thread uses its own preallocated KModel, so no mutex needed
    
    // Structure to hold optimization results from each worker thread
    struct OptimizationResult {
      arma::uword start_index;
      double objective_value;
      arma::vec gamma;
      arma::vec theta_alpha;
      arma::mat L;
      arma::mat R;
      arma::mat Fstar;
      arma::mat Rstar;
      arma::mat Qstar;
      arma::vec Estar;
      arma::vec ystar;
      double SSEstar;
      arma::vec betahat;
      bool success;
      std::string error_message;
      
      OptimizationResult() : start_index(0), objective_value(std::numeric_limits<double>::infinity()), success(false) {}
    };

    // Worker function for each optimization start point
    auto optimize_worker = [&](arma::uword start_idx) -> OptimizationResult {
      OptimizationResult result;
      result.start_index = start_idx;
      
      try {
        arma::vec gamma_tmp = arma::vec(d + 1);
        gamma_tmp.head(d) = theta0.row(start_idx % multistart).t();
        gamma_tmp.at(d) = alpha0[start_idx % alpha0.n_elem];
        if (Optim::reparametrize) {
          gamma_tmp = NuggetKriging::reparam_to(gamma_tmp);
        }

        arma::vec gamma_lower_local = gamma_lower;
        arma::vec gamma_upper_local = gamma_upper;
        gamma_lower_local = arma::min(gamma_tmp, gamma_lower_local);
        gamma_upper_local = arma::max(gamma_tmp, gamma_upper_local);

        // Use pre-allocated KModel for this thread (thread-safe)
        if (start_idx >= preallocated_models.size()) {
          throw std::runtime_error("Preallocated model index out of bounds");
        }
        
        NuggetKriging::KModel& m = preallocated_models[start_idx];
        populate_Model(m, theta0.row(start_idx % multistart).t(), alpha0[start_idx % alpha0.n_elem], nullptr);

        lbfgsb::Optimizer optimizer{d + 1};
        optimizer.iprint = -1;  // Disable output in parallel mode
        optimizer.max_iter = Optim::max_iteration;
        optimizer.pgtol = Optim::gradient_tolerance;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
        arma::ivec bounds_type{d + 1, arma::fill::value(2)};

        if (Optim::log_level > Optim::log_none) {
          arma::cout << "BFGS (start " << (start_idx+1) << "/" << multistart << "):" << arma::endl;
          arma::cout << "  objective: " << m_objective << arma::endl;
          arma::cout << "  max iterations: " << optimizer.max_iter << arma::endl;
          arma::cout << "  null gradient tolerance: " << optimizer.pgtol << arma::endl;
          arma::cout << "  constant objective tolerance: " << optimizer.factr * 1E-13 << arma::endl;          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower.t() << "";
          arma::cout << "                " << alpha_lower << arma::endl;
          arma::cout << "  upper_bounds: " << theta_upper.t() << "";
          arma::cout << "                " << alpha_upper << arma::endl;
          arma::cout << "  start_point: " << theta0.row(start_idx % multistart) << "";
          arma::cout << "               " << alpha0[start_idx % alpha0.n_elem] << arma::endl;
        }

        int retry = 0;
        double best_f_opt = std::numeric_limits<double>::infinity();
        arma::vec best_gamma = gamma_tmp;

        while (retry <= Optim::max_restart) {
          arma::vec gamma_0 = gamma_tmp;
          
          auto opt_result = optimizer.minimize(
              [&m, &fit_ofn](const arma::vec& vals_inp, arma::vec& grad_out) -> double {
                return fit_ofn(vals_inp, &grad_out, &m);
              },
              gamma_tmp,
              gamma_lower_local.memptr(),
              gamma_upper_local.memptr(),
              bounds_type.memptr());

          if (Optim::log_level > Optim::log_info) {
            arma::cout << "  Start " << (start_idx + 1) << ", Retry " << (retry)
                       << ": f_opt=" << opt_result.f_opt << ", num_iters=" << opt_result.num_iters
                       << ", task=" << opt_result.task << arma::endl;
          }
          
          if (opt_result.f_opt < best_f_opt) {
            best_f_opt = opt_result.f_opt;
            best_gamma = gamma_tmp;
          }

          double sol_to_lb_theta = Optim::reparametrize
                                        ? arma::min(arma::abs(NuggetKriging::reparam_from(gamma_tmp.head(d)) - theta_lower))
                                        : arma::min(arma::abs(gamma_tmp.head(d) - theta_upper));
          double sol_to_ub_theta = Optim::reparametrize
                                        ? arma::min(arma::abs(NuggetKriging::reparam_from(gamma_tmp.head(d)) - theta_upper))
                                        : arma::min(arma::abs(gamma_tmp.head(d) - theta_lower));
          double sol_to_lb_alpha = Optim::reparametrize
                                        ? std::abs(NuggetKriging::reparam_from(gamma_tmp).at(d) - alpha_lower)
                                        : std::abs(gamma_tmp.at(d) - alpha_lower);
          double sol_to_ub_alpha = Optim::reparametrize
                                        ? std::abs(NuggetKriging::reparam_from(gamma_tmp).at(d) - alpha_upper)
                                        : std::abs(gamma_tmp.at(d) - alpha_upper);

          if ((retry < Optim::max_restart)
              && ((opt_result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0) // Check for abnormal termination
                  || (opt_result.num_iters <= 2) // Start point is strangely quite optimal...
                  || (sol_to_lb_theta < arma::datum::eps) // Stuck at theta lower bound
                  || (sol_to_ub_alpha < arma::datum::eps) // Stuck at sigma2 upper bound, i.e. nugget lower bound
                  || (opt_result.f_opt > best_f_opt))) { // No improvement

            if (Optim::log_level > Optim::log_none) {
              arma::cout << "  Restarting BFGS (start " << (start_idx+1) << ", retry " << (retry+1)
                         << "): f_opt=" << opt_result.f_opt
                         << ", sol_to_lb=" << sol_to_lb_theta
                         << ", sol_to_ub=" << sol_to_ub_theta
                         << ", sol_to_lb_alpha=" << sol_to_lb_alpha
                         << ", sol_to_ub_alpha=" << sol_to_ub_alpha << arma::endl;
            }

            gamma_tmp.head(d) = (theta0.row(start_idx % multistart).t() + theta_lower) / pow(2.0, retry + 1);
            gamma_tmp.at(d) = alpha_upper - (alpha0[start_idx % alpha0.n_elem] + alpha_upper) / pow(2.0, retry + 1);
            
            if (Optim::reparametrize)
              gamma_tmp = NuggetKriging::reparam_to(gamma_tmp);

            gamma_lower_local = arma::min(gamma_tmp, gamma_lower_local);
            gamma_upper_local = arma::max(gamma_tmp, gamma_upper_local);
            retry++;
          } else {
            break;
          }
        }

        // Final evaluation to update model
        double min_ofn_tmp = fit_ofn(best_gamma, nullptr, &m);

        result.objective_value = min_ofn_tmp;
        result.gamma = best_gamma;
        if (Optim::reparametrize)
          result.theta_alpha = NuggetKriging::reparam_from(best_gamma);
        else
          result.theta_alpha = best_gamma;
        
        // Copy (not move) since m is a reference to preallocated memory
        // Force DEEP copy to avoid any shared memory issues
        result.L = arma::mat(m.L);  // Force copy constructor
        result.R = arma::mat(m.R);  // Force copy constructor
        result.Fstar = arma::mat(m.Fstar);  // Force copy constructor
        result.Rstar = arma::mat(m.Rstar);  // Force copy constructor
        result.Qstar = arma::mat(m.Qstar);  // Force copy constructor
        result.Estar = arma::vec(m.Estar);  // Force copy constructor
        result.ystar = arma::vec(m.ystar);  // Force copy constructor
        result.SSEstar = m.SSEstar;
        result.betahat = arma::vec(m.betahat);  // Force copy constructor
        result.success = true;

      } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
        if (Optim::log_level > Optim::log_none) {
          arma::cout << "Warning: start point " << (start_idx + 1) << " failed: " << e.what() << arma::endl;
        }
      }

      return result;
    };

    // Execute optimizations (parallel if multistart > 1)
    std::vector<OptimizationResult> results(multistart);
    std::mutex results_mutex;  // Protect results vector writes

    if (multistart == 1) {
      results[0] = optimize_worker(0);
    } else {
      // Determine thread pool size
      unsigned int n_cpu = std::thread::hardware_concurrency();
      int pool_size = Optim::thread_pool_size;
      if (pool_size <= 0) {
        pool_size = std::max(1u, n_cpu);
      }
      pool_size = std::min(pool_size, (int)multistart);  // Don't exceed number of tasks
      
      if (Optim::log_level > Optim::log_none) {
        arma::cout << "Thread pool: " << pool_size << " workers (ncpu=" << n_cpu 
                   << ", multistart=" << multistart << ")" << arma::endl;
      }
      
      // Thread pool implementation: use semaphore-like counter
      std::atomic<int> next_task(0);
      std::vector<std::thread> threads;
      threads.reserve(pool_size);

      // RAII guard to ensure threads are always joined, even on exception
      struct ThreadJoiner {
        std::vector<std::thread>& threads_ref;
        explicit ThreadJoiner(std::vector<std::thread>& t) : threads_ref(t) {}
        ~ThreadJoiner() {
          for (auto& t : threads_ref) {
            if (t.joinable()) {
              t.join();
            }
          }
        }
      };

      for (int worker_id = 0; worker_id < pool_size; worker_id++) {
        threads.emplace_back([&, worker_id]() {
          while (true) {
            int task_id = next_task.fetch_add(1);
            if (task_id >= multistart) break;

            // Add staggered startup delay to avoid thread initialization race conditions
            int delay_ms = task_id * Optim::thread_start_delay_ms;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

            // FIX: Eliminate intermediate stack allocation by writing directly to results
            // This reduces C stack pressure for R bindings (issue with BFGS10)
            {
              std::lock_guard<std::mutex> lock(results_mutex);
              results[task_id] = optimize_worker(task_id);
            }
          }
        });
      }

      // Ensure threads are joined when leaving this scope
      ThreadJoiner joiner(threads);
    }

    // Find best result
    int best_idx = -1;
    int successful_optimizations = 0;

    for (size_t i = 0; i < results.size(); i++) {
      const auto& r = results[i];
      if (r.success) {
        successful_optimizations++;
        if (r.objective_value < min_ofn) {
          min_ofn = r.objective_value;
          best_idx = static_cast<int>(i);
        }
      }
    }

    if (successful_optimizations == 0) {
      throw std::runtime_error("All " + std::to_string(multistart)
                               + " optimization attempts failed");
    }

    if (Optim::log_level > Optim::log_none && successful_optimizations < multistart) {
      arma::cout << "\nOptimization summary: " << successful_optimizations << "/" << multistart
                << " succeeded" << arma::endl;
    }

    // Update member variables with best result
    if (best_idx >= 0) {
      const auto& best = results[best_idx];
      m_theta = best.theta_alpha.head(d);  // copy
      m_est_theta = true;
      m_is_empty = false;
      m_T = best.L;          // copy instead of move to avoid issues
      m_R = best.R;
      m_M = best.Fstar;
      m_circ = best.Rstar;
      m_star = best.Qstar;

      if (m_est_beta) {
        m_beta = best.betahat;
        m_z = best.Estar;
      } else {
        m_z = best.ystar - m_M * m_beta;
      }

      double m_alpha = best.theta_alpha.at(d);
      if (m_est_sigma2) {
        if (m_est_nugget) {
          m_sigma2 = m_alpha * as_scalar(LinearAlgebra::crossprod(m_z)) / n;
          if (m_objective.compare("LMP") == 0) {
            m_sigma2 = m_sigma2 * n / (n - m_F.n_cols - 2);
          }
          m_nugget = m_sigma2 / m_alpha - m_sigma2;
        } else {
          m_sigma2 = m_nugget * m_alpha / (1 - m_alpha);
        }
      } else {
        if (m_est_nugget) {
          m_nugget = m_sigma2 * (1 - m_alpha) / m_alpha;
        }
      }

      if (Optim::log_level > Optim::log_none) {
        arma::cout << "\nBest solution from start point " << (best_idx + 1) << " with objective: " << min_ofn
                  << arma::endl;
      }
    }
  } else
    throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS[#])");

  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param X_n is n_n*d matrix of points where to predict output
 * @param return_stdev is true if return also stdev column vector
 * @param return_cov is true if return also cov matrix between X_n
 * @param return_deriv is true if return also derivative of prediction wrt x
 * @return output prediction: n_n means, [n_n standard deviations], [n_n*n_n full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat>
NuggetKriging::predict(const arma::mat& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  arma::vec yhat_n(n_n);
  arma::vec ysd2_n = arma::vec(n_n, arma::fill::zeros);
  arma::mat Sigma_n = arma::mat(n_n, n_n, arma::fill::zeros);
  arma::mat Dyhat_n = arma::mat(n_n, d, arma::fill::zeros);
  arma::mat Dysd2_n = arma::mat(n_n, d, arma::fill::zeros);

  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = X_n;
  // Normalize X_n
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  double sigma2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n_o - m_F.n_cols) / (n_o - m_F.n_cols - 2) : 1.0);

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  Xn_n = trans(Xn_n);

  double m_alpha = m_sigma2 / (m_sigma2 + m_nugget);

  auto t0 = Bench::tic();
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  #ifdef _OPENMP
  arma::uword total_work = n_o * n_n;
  if (total_work >= 40000) {  // Only use OpenMP for sufficient work (avoid overhead for small matrices)
    int optimal_threads = get_optimal_threads(2);
    #pragma omp parallel for schedule(static) collapse(2) num_threads(optimal_threads) if(total_work >= 40000)
    for (arma::uword i = 0; i < n_o; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
        if (dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * m_alpha;
      }
    }
  } else {
  #endif
    for (arma::uword i = 0; i < n_o; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
        if (dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * m_alpha;
      }
    }
  #ifdef _OPENMP
  }
  #endif
  t0 = Bench::toc(nullptr, "R_on       ", t0);

  arma::mat Rstar_on = LinearAlgebra::solve(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n     ", t0);

  // Un-normalize predictor
  yhat_n = m_centerY + m_scaleY * yhat_n;

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n    ", t0);

  if (return_stdev) {
    ysd2_n = 1.0 - sum(Rstar_on % Rstar_on, 0).as_col() + sum(Ecirc_n % Ecirc_n, 1).as_col();
    ysd2_n.transform([](double val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
    ysd2_n *= sigma2 / m_alpha * m_scaleY * m_scaleY;
    t0 = Bench::toc(nullptr, "ysd2_n     ", t0);
  }

  if (return_cov) {
    // Compute the covariance matrix between new data points
    arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
    arma::vec diag_nn = arma::vec(n_n, arma::fill::ones);
    LinearAlgebra::covMat_sym_X(&R_nn, Xn_n, m_theta, _Cov, m_alpha, diag_nn);
    t0 = Bench::toc(nullptr, "R_nn       ", t0);

    Sigma_n = R_nn - trans(Rstar_on) * Rstar_on + Ecirc_n * trans(Ecirc_n);
    Sigma_n *= sigma2 / m_alpha * m_scaleY * m_scaleY;
    t0 = Bench::toc(nullptr, "Sigma_n    ", t0);
  }

  if (return_deriv) {
    //// https://github.com/libKriging/dolka/blob/bb1dbf0656117756165bdcff0bf5e0a1f963fbef/R/kmStuff.R#L322C1-L363C10
    // for (i in 1:n_n) {
    //
    //   ## =================================================================
    //   ## 'DF_n_i' and 'DR_on_i' are matrices with
    //   ## dimension c(n_n, d)
    //   ## =================================================================
    //
    //   DF_n_i <- trend.deltax(x = XNew[i, ], model = object)
    //   KOldNewi <- as.vector(KOldNew[ , i])
    //   DR_on_i <- covVector.dx(x = as.vector(XNew[i, ]),
    //                               X = X,
    //                               object = object@covariance,
    //                               c = KOldNewi)
    //
    //   KOldNewStarDer[ , i, i, ] <-
    //       backsolve(L, DR_on_i, upper.tri = FALSE)
    //
    //   ## Gradient of the kriging trend and mean
    //   muNewHatDer[i, i, ] <- crossprod(DF_n_i, betaHat)
    //   ## dim in product c(d, n) and NULL(length d)
    //   mean_nHatDer[i, i, ] <- muNewHatDer[i, i, ] +
    //       crossprod(KOldNewStarDer[ , i, i,  ], zStar)
    //   ## dim in product c(d, n) and NULL(length n)
    //   s2Der[i, i,  ] <-
    //       - 2 * crossprod(KOldNewStarDer[ , i, i, ],
    //                       drop(KOldNewStar[ , i]))
    //
    //   ## dim in product c(d, n) and c(n, p)
    //
    //   if (type == "UK") {
    //       ENewDagDer[i, i, , ] <-
    //           backsolve(t(RStar),
    //                     DF_n_i -
    //                     t(crossprod(KOldNewStarDer[ , i, i, ], FStar)),
    //                     upper.tri = FALSE)
    //       ## dim in product NULL (length p) and c(p, d) because of 'drop'
    //       s2Der[i, i, ] <- s2Der[i, i, ] + 2 * drop(ENewDagT[ , i]) %*%
    //           drop(ENewDagDer[i, i, , ])
    //   }
    //  numerical derivative step : value is sensitive only for non linear trends. Otherwise, it gives exact results.
    const double h = 1.0E-5;
    arma::mat h_eye_d = h * arma::mat(d, d, arma::fill::eye);

    // Compute the derivatives of the covariance and trend functions
    for (arma::uword i = 0; i < n_n; i++) {  // for each predict point... should be parallel ?
      arma::mat DR_on_i = arma::mat(n_o, d, arma::fill::none);
      for (arma::uword j = 0; j < n_o; j++) {
        DR_on_i.row(j) = R_on.at(j, i) * trans(_DlnCovDx(Xn_n.col(i) - Xn_o.col(j), m_theta));
      }
      t0 = Bench::toc(nullptr, "DR_on_i    ", t0);

      arma::mat tXn_n_repd_i
          = arma::trans(Xn_n.col(i) * arma::mat(1, d, arma::fill::ones));  // just duplicate X_n.row(i) d times
      arma::mat DF_n_i = (Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i + h_eye_d)
                          - Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i - h_eye_d))
                         / (2 * h);
      t0 = Bench::toc(nullptr, "DF_n_i     ", t0);

      arma::mat W_i = LinearAlgebra::solve(m_T, DR_on_i);
      t0 = Bench::toc(nullptr, "W_i        ", t0);
      Dyhat_n.row(i) = trans(DF_n_i * m_beta + trans(W_i) * m_z);
      t0 = Bench::toc(nullptr, "Dyhat_n    ", t0);

      if (return_stdev) {
        arma::mat DEcirc_n_i = LinearAlgebra::solve(m_circ.t(), trans(DF_n_i - W_i.t() * m_M));
        Dysd2_n.row(i) = -2 * Rstar_on.col(i).t() * W_i + 2 * Ecirc_n.row(i) * DEcirc_n_i;
        t0 = Bench::toc(nullptr, "Dysd2_n    ", t0);
      }
    }
    Dyhat_n *= m_scaleY;
    Dysd2_n *= sigma2 / m_alpha * m_scaleY * m_scaleY;
  }

  return std::make_tuple(std::move(yhat_n),
                         std::move(arma::sqrt(ysd2_n)),
                         std::move(Sigma_n),
                         std::move(Dyhat_n),
                         std::move(Dysd2_n / (2 * arma::sqrt(ysd2_n) * arma::mat(1, d, arma::fill::ones))));
  /*if (return_stdev)
    if (return_cov)
      return std::make_tuple(std::move(yhat_n), std::move(pred_stdev), std::move(pred_cov));
    else
      return std::make_tuple(std::move(yhat_n), std::move(pred_stdev), nullptr);
  else if (return_cov)
    return std::make_tuple(std::move(yhat_n), std::move(pred_cov), nullptr);
  else
    return std::make_tuple(std::move(yhat_n), nullptr, nullptr);*/
}

/** Draw sample trajectories of kriging at given points X'
 * @param X_n is n_n*d matrix of points where to simulate output
 * @param seed is seed for random number generator
 * @param nsim is number of simulations to draw
 * @param with_nugget is true if we want to include nugget effect in simulations
 * @param will_update is true if we want to keep simulations data for future update
 * @return output is n_n*nsim matrix of simulations at X_n
 */
LIBKRIGING_EXPORT arma::mat NuggetKriging::simulate(const int nsim,
                                                    const int seed,
                                                    const arma::mat& X_n,
                                                    const bool with_nugget,
                                                    const bool will_update) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = X_n;
  // Normalize X_n
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);

  Xn_n = trans(Xn_n);

  double m_alpha = m_sigma2 / (m_sigma2 + m_nugget);

  auto t0 = Bench::tic();
  // Compute covariance between new data
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  arma::vec diag_nn = (with_nugget) ? arma::vec(n_n, arma::fill::ones) : arma::vec();
  LinearAlgebra::covMat_sym_X(&R_nn, Xn_n, m_theta, _Cov, m_alpha, diag_nn);
  t0 = Bench::toc(nullptr, "R_nn          ", t0);

  // Compute covariance between training data and new data to predict
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  #ifdef _OPENMP
  arma::uword total_work = n_o * n_n;
  if (total_work >= 40000) {  // Only use OpenMP for sufficient work (avoid overhead for small matrices)
    int optimal_threads = get_optimal_threads(2);
    #pragma omp parallel for schedule(static) collapse(2) num_threads(optimal_threads) if(total_work >= 40000)
    for (arma::uword i = 0; i < n_o; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::mat dij = Xn_o.col(i) - Xn_n.col(j);
        if (with_nugget && dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * m_alpha;  // force m_alpha here, to be consistent with R_oo (=T*t(T))
        // R_on.at(i, j) = _Cov(Xn_o.col(i) - Xn_n.col(j), m_theta);
      }
    }
  } else {
  #endif
    for (arma::uword i = 0; i < n_o; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::mat dij = Xn_o.col(i) - Xn_n.col(j);
        if (with_nugget && dij.is_zero(arma::datum::eps))
          R_on.at(i, j) = 1.0;
        else
          R_on.at(i, j) = _Cov(dij, m_theta) * m_alpha;  // force m_alpha here, to be consistent with R_oo (=T*t(T))
        // R_on.at(i, j) = _Cov(Xn_o.col(i) - Xn_n.col(j), m_theta);
      }
    }
  #ifdef _OPENMP
  }
  #endif
  // R_on *= alpha;
  t0 = Bench::toc(nullptr, "R_on       ", t0);

  arma::mat Rstar_on = LinearAlgebra::solve(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  arma::vec yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n        ", t0);

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n       ", t0);

  arma::mat SigmaNoTrend_nKo = R_nn - trans(Rstar_on) * Rstar_on;

  arma::mat Sigma_nKo = SigmaNoTrend_nKo + Ecirc_n * trans(Ecirc_n);
  t0 = Bench::toc(nullptr, "Sigma_nKo     ", t0);

  arma::mat LSigma_nKo = LinearAlgebra::safe_chol_lower(Sigma_nKo / m_alpha);
  t0 = Bench::toc(nullptr, "LSigma_nKo     ", t0);

  arma::mat y_n = arma::mat(n_n, nsim, arma::fill::none);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += LSigma_nKo * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  // Un-normalize simulations
  y_n = m_centerY + m_scaleY * y_n;

  if (will_update) {
    lastsimup_Xn_u.clear();  // force reset to force update_simulate consider new data
    lastsim_y_n = y_n;

    lastsim_Xn_n = Xn_n;
    lastsim_seed = seed;
    lastsim_nsim = nsim;
    lastsim_with_nugget = with_nugget;

    lastsim_R_nn = R_nn;
    lastsim_F_n = F_n;

    lastsim_L_oCn = Rstar_on;
    lastsim_L_nCn = LinearAlgebra::safe_chol_lower(SigmaNoTrend_nKo);
    t0 = Bench::toc(nullptr, "L_nCn     ", t0);

    lastsim_L_on = arma::join_rows(arma::join_cols(m_T, lastsim_L_oCn.t()),
                                   arma::join_cols(arma::zeros(n_o, n_n), lastsim_L_nCn));

    arma::mat Linv_on = LinearAlgebra::solve(lastsim_L_on, arma::mat(n_o + n_n, n_o + n_n, arma::fill::eye));
    t0 = Bench::toc(nullptr, "Linv_on     ", t0);
    lastsim_Rinv_on = Linv_on.t() * Linv_on;
    t0 = Bench::toc(nullptr, "Rinv_on     ", t0);

    lastsim_F_on = arma::join_cols(m_F, lastsim_F_n);
    lastsim_Fstar_on = LinearAlgebra::solve(lastsim_L_on, lastsim_F_on);
    t0 = Bench::toc(nullptr, "Fstar_on     ", t0);
    arma::mat Q_Fstar_on;
    arma::qr(Q_Fstar_on, lastsim_circ_on, lastsim_Fstar_on);
    lastsim_Fcirc_on = LinearAlgebra::rsolve(lastsim_circ_on, lastsim_F_on);
    t0 = Bench::toc(nullptr, "Fcirc_on     ", t0);

    lastsim_Fhat_nKo = lastsim_L_oCn.t() * m_M;
    t0 = Bench::toc(nullptr, "Fhat_nKo     ", t0);
    lastsim_Ecirc_nKo = LinearAlgebra::rsolve(m_circ, F_n - lastsim_Fhat_nKo);
    t0 = Bench::toc(nullptr, "Ecirc_nKo     ", t0);
  }

  // Un-normalize simulations
  return y_n;
}

LIBKRIGING_EXPORT arma::mat NuggetKriging::update_simulate(const arma::vec& y_u, const arma::mat& X_u) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  if (lastsim_y_n.is_empty() || lastsim_y_n.n_rows == 0)
    throw std::runtime_error("No previous simulation data available");

  if (lastsim_Xn_n.n_rows != X_u.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(X_u.n_cols)
                             + "), last sim X: (...x" + std::to_string(lastsim_Xn_n.n_rows) + ")");

  arma::uword n_n = lastsim_Xn_n.n_cols;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::mat Xn_o = trans(m_X);    // already normalized if needed
  arma::mat Xn_n = lastsim_Xn_n;  // already normalized

  arma::uword n_on = n_o + n_n;
  // arma::mat Xn_on = arma::join_cols(Xn_o, Xn_n);
  arma::mat F_on = arma::join_cols(m_F, lastsim_F_n);

  arma::uword n_u = X_u.n_rows;
  // Normalize X_u
  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_u = Trend::regressionModelMatrix(m_regmodel, Xn_u);

  auto t0 = Bench::tic();
  Xn_u = trans(Xn_u);
  t0 = Bench::toc(nullptr, "Xn_u.t()      ", t0);

  double m_alpha = m_sigma2 / (m_sigma2 + m_nugget);

  bool use_lastsimup = (!lastsimup_Xn_u.is_empty()) && (lastsimup_Xn_u - Xn_u).is_zero(arma::datum::eps);
  if (!use_lastsimup) {
    lastsimup_Xn_u = Xn_u;

    // Compute covariance between updated data
    lastsimup_R_uu = arma::mat(n_u, n_u, arma::fill::none);
    arma::vec diag_uu = (lastsim_with_nugget) ? arma::vec(n_u, arma::fill::ones) : arma::vec();
    LinearAlgebra::covMat_sym_X(&lastsimup_R_uu, Xn_u, m_theta, _Cov, m_alpha, diag_uu);
    t0 = Bench::toc(nullptr, "R_uu          ", t0);

    // Compute covariance between updated/old data
    lastsimup_R_uo = arma::mat(n_u, n_o, arma::fill::none);
    LinearAlgebra::covMat_rect(&lastsimup_R_uo, Xn_u, Xn_o, m_theta, _Cov, m_alpha);
    t0 = Bench::toc(nullptr, "R_uo          ", t0);

    // Compute covariance between updated/new data
    lastsimup_R_un = arma::mat(n_u, n_n, arma::fill::none);
    for (arma::uword i = 0; i < n_u; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        arma::vec dij = Xn_u.col(i) - Xn_n.col(j);
        if (lastsim_with_nugget && dij.is_zero(arma::datum::eps))
          lastsimup_R_un.at(i, j) = 1.0;
        else
          lastsimup_R_un.at(i, j) = _Cov(dij, m_theta) * m_alpha;
        // lastsimup_R_un.at(i, j) = _Cov(Xn_u.col(i) - Xn_n.col(j), m_theta) * m_alpha;
      }
    }
    t0 = Bench::toc(nullptr, "R_un          ", t0);
  }

  // ======================================================================
  // FOXY step #1 Extend the simulation to the design 'X_u' IF
  // NECESSARY. Remind that the simulation number j is
  // conditional on the given 'y_o' and on 'y_n = Y_sim[ , j]'
  //
  // CAUTION. To avoid unnecessary re-computations we here use
  // auxiliary variables that where computed in the creation of
  // the KM0 step AND later in the simulation. The first ones are
  // in 'theKM0$Extra' and the second ones are in 'Ex'
  //
  // In indices 'C' means comma and 'K' means pipe '|'
  //
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat R_onCu = arma::join_rows(lastsimup_R_uo, lastsimup_R_un).t();
    arma::mat Rstar_onCu = LinearAlgebra::solve(lastsim_L_on, R_onCu);
    t0 = Bench::toc(nullptr, "Rstar_onCu          ", t0);

    arma::mat Ecirc_uKon = LinearAlgebra::rsolve(lastsim_circ_on, F_u - Rstar_onCu.t() * lastsim_Fstar_on);
    t0 = Bench::toc(nullptr, "Ecirc_uKon          ", t0);

    arma::mat Sigma_uKon = lastsimup_R_uu - Rstar_onCu.t() * Rstar_onCu + Ecirc_uKon * Ecirc_uKon.t();
    t0 = Bench::toc(nullptr, "Sigma_uKon          ", t0);

    arma::mat LSigma_uKon = LinearAlgebra::safe_chol_lower(Sigma_uKon / m_alpha);
    t0 = Bench::toc(nullptr, "LSigma_uKon          ", t0);

    arma::mat W_uCon = (R_onCu.t() + Ecirc_uKon * lastsim_Fcirc_on.t()) * lastsim_Rinv_on;
    t0 = Bench::toc(nullptr, "W_uCon          ", t0);

    arma::mat m_u = W_uCon.head_cols(n_o) * m_y;
    arma::mat M_u = arma::repmat(m_u, 1, lastsim_nsim) + W_uCon.tail_cols(n_n) * lastsim_y_n;

    Random::reset_seed(lastsim_seed);
    lastsimup_y_u = M_u + LSigma_uKon * Random::randn_mat(n_u, lastsim_nsim) * std::sqrt(m_sigma2);
    t0 = Bench::toc(nullptr, "y_u          ", t0);
  }

  // ======================================================================
  // FOXY step #2   Update the simulated paths on 'X_n'
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat Rstar_ou = LinearAlgebra::solve(m_T, lastsimup_R_uo.t());
    t0 = Bench::toc(nullptr, "Rstar_ou          ", t0);

    arma::mat Fhat_uKo = Rstar_ou.t() * m_M;
    arma::mat Ecirc_uKo = LinearAlgebra::rsolve(m_circ, F_u - Fhat_uKo);
    t0 = Bench::toc(nullptr, "Ecirc_uKo          ", t0);

    arma::mat Rtild_uCu = lastsimup_R_uu - Rstar_ou.t() * Rstar_ou + Ecirc_uKo * Ecirc_uKo.t();
    t0 = Bench::toc(nullptr, "Rtild_uCu          ", t0);

    arma::mat Rtild_nCu = lastsimup_R_un - Rstar_ou.t() * lastsim_L_oCn + Ecirc_uKo * lastsim_Ecirc_nKo.t();
    t0 = Bench::toc(nullptr, "Rtild_nCu          ", t0);

    lastsimup_Wtild_nKu = LinearAlgebra::solve(Rtild_uCu, Rtild_nCu).t();
    t0 = Bench::toc(nullptr, "Wtild_nKu          ", t0);
  }

  return lastsim_y_n + lastsimup_Wtild_nKu * (arma::repmat(y_u, 1, lastsim_nsim) - lastsimup_y_u);
}

/** Add new conditional data points to previous (X,y), then perform new fit.
 * @param y_u is n_u length column vector of new output
 * @param X_u is n_u*d matrix of new input
 * @param refit is true if we want to re-fit the model
 */
LIBKRIGING_EXPORT void NuggetKriging::update(const arma::vec& y_u, const arma::mat& X_u, const bool refit) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");
  // rebuild starting parameters
  Parameters parameters;
  if (refit) {  // re-fit
    if (m_est_beta) {
      if (m_est_nugget && m_est_sigma2 && m_est_theta) {
        // All parameters are being estimated - use default initialization for better convergence
        parameters = Parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
      } else {
        parameters = Parameters{
            std::make_optional(arma::vec(1, arma::fill::value(this->m_nugget * this->m_scaleY * this->m_scaleY))),
            this->m_est_nugget,
            std::make_optional(arma::vec(1, arma::fill::value(this->m_sigma2 * this->m_scaleY * this->m_scaleY))),
            this->m_est_sigma2,
            std::make_optional(trans(this->m_theta) % this->m_scaleX),
            this->m_est_theta,
            std::make_optional(arma::ones<arma::vec>(0)),
            true};
      }
    } else {
      if (m_est_nugget && m_est_sigma2 && m_est_theta) {
        // All parameters except beta are being estimated - use default initialization
        parameters = Parameters{std::nullopt,
                                true,
                                std::nullopt,
                                true,
                                std::nullopt,
                                true,
                                std::make_optional(trans(this->m_beta) * this->m_scaleY),
                                false};
      } else {
        parameters = Parameters{
            std::make_optional(arma::vec(1, arma::fill::value(this->m_nugget * this->m_scaleY * this->m_scaleY))),
            this->m_est_nugget,
            std::make_optional(arma::vec(1, arma::fill::value(this->m_sigma2 * this->m_scaleY * this->m_scaleY))),
            this->m_est_sigma2,
            std::make_optional(trans(this->m_theta) % this->m_scaleX),
            this->m_est_theta,
            std::make_optional(trans(this->m_beta) * this->m_scaleY),
            false};
      }
    }
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                              y_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
              m_regmodel,
              m_normalize,
              m_optim,
              m_objective,
              parameters);
  } else {  // just update
    parameters = Parameters{
        std::make_optional(arma::vec(1, arma::fill::value(this->m_nugget * this->m_scaleY * this->m_scaleY))),
        false,
        std::make_optional(arma::vec(1, arma::fill::value(this->m_sigma2 * this->m_scaleY * this->m_scaleY))),
        false,
        std::make_optional(trans(this->m_theta) % this->m_scaleX),
        false,
        std::make_optional(arma::vec(this->m_beta) * this->m_scaleY),
        false};
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                              y_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
              m_regmodel,
              m_normalize,
              "none",
              m_objective,
              parameters);
  }
}

LIBKRIGING_EXPORT std::string NuggetKriging::summary() const {
  std::ostringstream oss;
  auto vec_printer = [&oss](const arma::vec& v) {
    v.for_each([&oss, i = 0](const arma::vec::elem_type& val) mutable {
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
    arma::rowvec Xmins = arma::min(m_X, 0);
    arma::rowvec Xmaxs = arma::max(m_X, 0);
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

  // _Cov_pow & std::function embedded by make_Cov
  j["covType"] = m_covType;
  j["X"] = to_json(m_X);
  j["centerX"] = to_json(m_centerX);
  j["scaleX"] = to_json(m_scaleX);
  j["y"] = to_json(m_y);
  j["centerY"] = m_centerY;
  j["scaleY"] = m_scaleY;
  j["normalize"] = m_normalize;
  j["regmodel"] = Trend::toString(m_regmodel);
  j["optim"] = m_optim;
  j["objective"] = m_objective;
  // Auxiliary data
  j["dX"] = to_json(m_dX);
  j["maxdX"] = to_json(m_maxdX);
  j["F"] = to_json(m_F);
  j["T"] = to_json(m_T);
  j["R"] = to_json(m_R);
  j["M"] = to_json(m_M);
  j["star"] = to_json(m_star);
  j["circ"] = to_json(m_circ);
  j["z"] = to_json(m_z);
  j["beta"] = to_json(m_beta);
  j["est_beta"] = m_est_beta;
  j["theta"] = to_json(m_theta);
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
  NuggetKriging kr(covType);  // _Cov_pow & std::function embedded by make_Cov

  kr.m_X = mat_from_json(j["X"]);
  kr.m_centerX = rowvec_from_json(j["centerX"]);
  kr.m_scaleX = rowvec_from_json(j["scaleX"]);
  kr.m_y = colvec_from_json(j["y"]);
  kr.m_centerY = j["centerY"].template get<decltype(kr.m_centerY)>();
  kr.m_scaleY = j["scaleY"].template get<decltype(kr.m_scaleY)>();
  kr.m_normalize = j["normalize"].template get<decltype(kr.m_normalize)>();

  std::string model = j["regmodel"].template get<std::string>();
  kr.m_regmodel = Trend::fromString(model);

  kr.m_optim = j["optim"].template get<decltype(kr.m_optim)>();
  kr.m_objective = j["objective"].template get<decltype(kr.m_objective)>();
  // Auxiliary data
  kr.m_dX = mat_from_json(j["dX"]);
  kr.m_maxdX = colvec_from_json(j["maxdX"]);
  kr.m_F = mat_from_json(j["F"]);
  kr.m_T = mat_from_json(j["T"]);
  kr.m_R = mat_from_json(j["R"]);
  kr.m_M = mat_from_json(j["M"]);
  kr.m_star = mat_from_json(j["star"]);
  kr.m_circ = mat_from_json(j["circ"]);
  kr.m_z = colvec_from_json(j["z"]);
  kr.m_beta = colvec_from_json(j["beta"]);
  kr.m_est_beta = j["est_beta"].template get<decltype(kr.m_est_beta)>();
  kr.m_theta = colvec_from_json(j["theta"]);
  kr.m_est_theta = j["est_theta"].template get<decltype(kr.m_est_theta)>();
  kr.m_sigma2 = j["sigma2"].template get<decltype(kr.m_sigma2)>();
  kr.m_est_sigma2 = j["est_sigma2"].template get<decltype(kr.m_est_sigma2)>();
  kr.m_nugget = j["nugget"].template get<decltype(kr.m_nugget)>();
  kr.m_est_nugget = j["est_nugget"].template get<decltype(kr.m_est_nugget)>();
  kr.m_is_empty = false;

  return kr;
}
