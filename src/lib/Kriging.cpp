// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio

#include <algorithm>
#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Bench.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/KrigingException.hpp"
#include "libKriging/LinearAlgebra.hpp"
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
#include <thread>
#include <tuple>
#include <vector>

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
    if (!hModule)
      hModule = GetModuleHandleA("openblas.dll");
    if (hModule) {
      func = (openblas_set_num_threads_t)GetProcAddress(hModule, "openblas_set_num_threads");
    }
  }
  return func;
}
}  // namespace
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
}  // namespace
#endif
#endif

// Register pthread_atfork handlers so that forked child processes (e.g. R's
// parallel::mclapply) inherit a clean, single-threaded BLAS/OMP state instead
// of a locked idle thread pool that causes deadlocks.
//
// The prepare handler runs in the parent just before fork(): it sets BLAS and
// OMP to 1 thread so no new parallel work can start during the fork window.
// The child handler re-asserts single-threading in case the runtime inherited
// stale state.  The parent handler is omitted (nullptr): the next fit() call
// will reset threads_per_worker as needed.
#if !defined(_WIN32) && defined(_POSIX_VERSION)
#include <pthread.h>
namespace {

void libkriging_atfork_quiesce() {
#if !defined(__APPLE__) || !defined(__arm64__)
  auto fn = get_openblas_set_num_threads();
  if (fn)
    fn(1);
#endif
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
}

struct ForkSafeRegistrar {
  ForkSafeRegistrar() {
    pthread_atfork(libkriging_atfork_quiesce,   // prepare: quiesce before fork
                   nullptr,                     // parent:  restored by next fit()
                   libkriging_atfork_quiesce);  // child:   ensure clean state
  }
};
static ForkSafeRegistrar fork_safe_registrar;

}  // namespace
#endif  // !_WIN32 && _POSIX_VERSION

// =============================================================================
// Subset-of-data pre-fit reduction (see Kriging.hpp for the full doc) -- a
// pure pre-processing layer ahead of an ordinary fit, no interaction with
// the rest of the class (static, no member access).
// =============================================================================

LIBKRIGING_EXPORT arma::uvec Kriging::subsetOfData(const arma::mat& X,
                                                   arma::uword n_max,
                                                   const std::string& method,
                                                   int seed) {
  const arma::uword n = X.n_rows;
  if (n_max >= n)
    return arma::regspace<arma::uvec>(0, n - 1);
  if (n_max == 0)
    return arma::uvec();

  arma::arma_rng::set_seed(static_cast<arma::arma_rng::seed_type>(seed));

  if (method == "kmeans") {
    arma::mat centroids;
    // arma::kmeans expects d x n data (rows = dimensions, cols = observations).
    const bool kmeans_ok = arma::kmeans(centroids, X.t(), n_max, arma::random_subset, 10, false);
    if (kmeans_ok) {
      arma::uvec idx(n_max, arma::fill::none);
      std::vector<bool> taken(n, false);
      for (arma::uword c = 0; c < n_max; ++c) {
        arma::vec dist2(n, arma::fill::none);
        for (arma::uword i = 0; i < n; ++i)
          dist2(i) = taken[i] ? arma::datum::inf : arma::accu(arma::square(X.row(i).t() - centroids.col(c)));
        const arma::uword nearest = dist2.index_min();
        idx(c) = nearest;
        taken[nearest] = true;
      }
      return arma::sort(idx);
    }
    // else fall through to the random method below
  } else if (method != "random") {
    throw std::invalid_argument("subsetOfData: unknown method '" + method + "' (expected \"kmeans\" or \"random\")");
  }

  return arma::sort(arma::uvec(arma::randperm(n, n_max)));
}

/************************************************/
/**      Kriging implementation        **/
/************************************************/

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT Kriging::Kriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT Kriging::Kriging(const std::string& covType, NoiseModel noise_model) : m_noise_model(noise_model) {
  make_Cov(covType);
}

arma::uword Kriging::gamma_dim() const {
  return m_X.n_cols + (m_noise_model == NoiseModel::None ? 0 : 1);
}

arma::vec Kriging::current_gamma() const {
  arma::uword d = m_theta.n_elem;
  if (m_noise_model == NoiseModel::None)
    return m_theta;
  arma::vec g(d + 1);
  g.head(d) = m_theta;
  g.at(d) = (m_noise_model == NoiseModel::Nugget) ? m_alpha : m_sigma2;
  return g;
}

LIBKRIGING_EXPORT Kriging::Kriging(const arma::vec& y,
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

LIBKRIGING_EXPORT Kriging::Kriging(const Kriging& other, ExplicitCopySpecifier) : Kriging{other} {}

void Kriging::populate_Model(KModel& m,
                             const arma::vec& theta,
                             double extra_param,
                             std::map<std::string, double>* bench) const {
  double alpha = 1.0;
  arma::vec diag = KrigingImpl::ones;
  bool update_eligible = false;

  if (m_noise_model == NoiseModel::Nugget) {
    alpha = extra_param;
    update_eligible = !m_is_empty && (extra_param == m_alpha) && (m_theta.size() == theta.size())
                      && (theta - m_theta).is_zero() && (m_T.memptr() != nullptr) && (m_X.n_rows > m_T.n_rows);
  } else if (m_noise_model == NoiseModel::Heterogeneous) {
    diag = 1.0 + m_noise / extra_param;
    update_eligible = !m_is_empty && (extra_param == m_sigma2) && (m_theta.size() == theta.size())
                      && (theta - m_theta).is_zero() && (m_T.memptr() != nullptr) && (m_X.n_rows > m_T.n_rows);
  } else {
    update_eligible = !m_is_empty && (m_theta.size() == theta.size()) && (theta - m_theta).is_zero()
                      && (m_T.memptr() != nullptr) && (m_X.n_rows > m_T.n_rows);
  }

  KrigingImpl::populate_Model(m, theta, alpha, diag, update_eligible, bench);
  if (!m_est_beta)
    m.betahat = arma::vec(m_F.n_cols, arma::fill::zeros);
}

void Kriging::populate_Model(KModel& m, const arma::vec& theta, std::map<std::string, double>* bench) const {
  double extra = (m_noise_model == NoiseModel::Nugget) ? m_alpha : m_sigma2;
  populate_Model(m, theta, extra, bench);
}

Kriging::KModel Kriging::make_Model(const arma::vec& theta,
                                    double extra_param,
                                    std::map<std::string, double>* bench) const {
  KModel m = allocate_KModel();
  populate_Model(m, theta, extra_param, bench);
  return m;
}

Kriging::KModel Kriging::make_Model(const arma::vec& theta, std::map<std::string, double>* bench) const {
  KModel m = allocate_KModel();
  populate_Model(m, theta, bench);
  return m;
}

// Objective function for fit : -logLikelihood
// gamma = [theta] for None, [theta, alpha] for Nugget, [theta, sigma2] for Heterogeneous.

double Kriging::_logLikelihood(const arma::vec& _gamma,
                               arma::vec* grad_out,
                               Kriging::KModel* model,
                               std::map<std::string, double>* bench) const {
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  const arma::vec _theta = _gamma.head(d);

  // Extract or default extra_param (alpha or sigma2)
  double extra_param;
  if (_gamma.n_elem > d)
    extra_param = _gamma.at(d);
  else
    extra_param = (m_noise_model == NoiseModel::Nugget) ? m_alpha : m_sigma2;

  // For Heterogeneous with fixed sigma2, or Nugget with both fixed,
  // force extra_param so the model is consistent with the LL formula
  if (m_noise_model == NoiseModel::Heterogeneous && !m_est_sigma2)
    extra_param = m_sigma2;
  else if (m_noise_model == NoiseModel::Nugget && !m_est_sigma2 && !m_est_nugget)
    extra_param = m_sigma2 / (m_sigma2 + m_nugget);

  Kriging::KModel m_local;
  if (model != nullptr)
    populate_Model(*model, _theta, extra_param, bench);
  else
    m_local = make_Model(_theta, extra_param, bench);
  Kriging::KModel& m = (model != nullptr) ? *model : m_local;

  // LL value — differs by noise model
  double sigma2_grad;  // normalizer for the theta-gradient
  double ll;

  if (m_noise_model == NoiseModel::Nugget) {
    double _alpha = extra_param;
    double _sigma2_loc = m_sigma2;
    double _nugget_loc = m_nugget;
    if (m_est_sigma2) {
      if (m_est_nugget) {
        double var = m.SSEstar / n;
        _sigma2_loc = _alpha * var;
        _nugget_loc = (1.0 - _alpha) * var;
      } else {
        _sigma2_loc = m_nugget * _alpha / (1.0 - _alpha);
      }
    } else {
      if (m_est_nugget) {
        _nugget_loc = m_sigma2 * (1.0 - _alpha) / _alpha;
      } else {
        _alpha = m_sigma2 / (m_sigma2 + m_nugget);
      }
    }
    double total_var = _sigma2_loc + _nugget_loc;
    ll = -0.5 * (n * log(2 * M_PI * total_var) + 2 * arma::sum(log(m.L.diag())) + m.SSEstar / total_var);
    sigma2_grad = total_var;

  } else if (m_noise_model == NoiseModel::Heterogeneous) {
    // Unconcentrated form: sigma2 is a free parameter
    double _sigma2 = extra_param;
    if (!m_est_sigma2)
      _sigma2 = m_sigma2;
    ll = -0.5 * (n * log(2 * M_PI * _sigma2) + 2 * arma::sum(log(m.L.diag())) + m.SSEstar / _sigma2);
    sigma2_grad = _sigma2;

  } else {
    // None: concentrated form (original Kriging behavior)
    if (m_est_sigma2) {
      sigma2_grad = m.SSEstar / n;
      ll = -0.5 * (n * log(2 * M_PI * sigma2_grad) + 2 * arma::sum(log(m.L.diag())) + n);
    } else {
      sigma2_grad = m_sigma2;
      ll = -0.5
           * (n * log(2 * M_PI * sigma2_grad) + 2 * arma::sum(log(m.L.diag()))
              + as_scalar(LinearAlgebra::crossprod(m.Estar)) / sigma2_grad);
    }
  }

  if (grad_out != nullptr) {
    auto t0 = Bench::tic();
    const arma::mat& Rinv = m.Rinv;
    arma::mat x = LinearAlgebra::solve_upper(m.L.t(), m.Estar);
    t0 = Bench::toc(bench, "x = tL \\ z", t0);

    arma::vec term1_vec(d, arma::fill::zeros);
    arma::vec term2_vec(d, arma::fill::zeros);
    t0 = Bench::tic();
    compute_ll_grad_theta_vecs(m.R, Rinv, x, _theta, term1_vec, term2_vec);
    t0 = Bench::toc(bench, "gradR computation [optimized]", t0);

    for (arma::uword k = 0; k < d; k++)
      (*grad_out).at(k) = (term1_vec.at(k) / sigma2_grad + term2_vec.at(k)) / 2.0;

    // Extra gradient dimension for Nugget / Heterogeneous
    if (grad_out->n_elem > d) {
      if (m_noise_model == NoiseModel::Nugget) {
        double _alpha = extra_param;
        if (m_est_sigma2 && m_est_nugget) {
          double var = sigma2_grad;
          arma::mat dRdv = m.R / _alpha;
          dRdv.diag().zeros();
          double term1_a = -as_scalar((trans(x) * dRdv) * x) / var;
          double term2_a = arma::dot(Rinv, dRdv);
          (*grad_out).at(d) = -0.5 * (term1_a + term2_a);
        } else if (m_est_sigma2 && !m_est_nugget) {
          double total_var = sigma2_grad;
          arma::mat dRdv = m.R / _alpha;
          dRdv.diag().ones();
          double term1_a = -as_scalar((trans(x) * dRdv) * x) / (total_var * total_var);
          double term2_a = arma::dot(Rinv / total_var, dRdv);
          (*grad_out).at(d) = -0.5 * (term1_a + term2_a) * m_nugget / (1.0 - _alpha) / (1.0 - _alpha);
        } else {
          (*grad_out).at(d) = 0.0;
        }
      } else if (m_noise_model == NoiseModel::Heterogeneous) {
        double _sigma2 = extra_param;
        if (!m_est_sigma2) {
          (*grad_out).at(d) = 0.0;
        } else {
          double s2sq = _sigma2 * _sigma2;
          double noise_Rinv = arma::dot(m_noise, Rinv.diag());
          double noise_x2 = arma::dot(m_noise, x % x);
          (*grad_out).at(d) = -0.5 * (n / _sigma2 - noise_Rinv / s2sq + noise_x2 / (s2sq * _sigma2) - m.SSEstar / s2sq);
        }
      }
    }
  }
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logLikelihoodFun(const arma::vec& _theta,
                                                                          const bool _grad,
                                                                          const bool _bench) {
  return eval_objective(_theta.n_elem, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _logLikelihood(_theta, g, nullptr, b);
  });
}

// Objective function for fit : -LOO

double Kriging::_leaveOneOut(const arma::vec& _theta,
                             arma::vec* grad_out,
                             arma::mat* yhat_out,
                             Kriging::KModel* model,
                             std::map<std::string, double>* bench) const {
  // arma::cout << " theta: " << _theta << arma::endl;
  //' @ref https://github.com/DiceKrigingClub/DiceKriging/blob/master/R/leaveOneOutFun.R
  // model@covariance <- vect2covparam(model@covariance, param)
  // model@covariance@sd2 <- 1		# to get the correlation matrix
  //
  // R <- covMatrix(model@covariance, model@X)[[1]]
  // T <- chol(R)
  //
  // M <- backsolve(t(T), model@F, upper.tri = FALSE)
  //
  // Rinv <- chol2inv(T)             # cost : n*n*n/3
  //...
  //  Rinv.F <- Rinv %*% (model@F)    # cost : 2*n*n*p
  //  T.M <- chol(crossprod(M))       # cost : p*p*p/3, neglected
  //  aux <- backsolve(t(T.M), t(Rinv.F), upper.tri=FALSE)   # cost : p*p*n, neglected
  //  Q <- Rinv - crossprod(aux)      # cost : 2*n*n*(p-1/2)
  //  Q.y <- Q %*% (model@y)          # cost : 2*n*n
  //  ## Remark:   Q <- Cinv - Cinv.F %*% solve(t(M)%*%M) %*% t(Cinv.F)
  //...
  // sigma2LOO <- 1/diag(Q)
  // errorsLOO <- sigma2LOO * (Q.y)       # cost : n, neglected
  //
  // LOOfun <- as.numeric(crossprod(errorsLOO)/model@n)

  // arma::cout << " theta: " << _theta << arma::endl;
  Kriging::KModel m_local;
  if (model != nullptr) {
    populate_Model(*model, _theta, bench);
  } else {
    m_local = make_Model(_theta, bench);
  }
  Kriging::KModel& m = (model != nullptr) ? *model : m_local;

  arma::uword n = m_X.n_rows;

  auto t0 = Bench::tic();
  if ((m.Linv.memptr() == nullptr) || (arma::size(m.Linv) != arma::size(m.L))) {
    m.Linv = LinearAlgebra::solve_lower(m.L, arma::mat(n, n, arma::fill::eye));
    t0 = Bench::toc(bench, "L ^-1", t0);
  }
  arma::mat By = m.Linv.t() * m.Estar;
  t0 = Bench::toc(bench, "By = L^-1 * E*", t0);
  // Compute Qstar on demand (not stored in populate_Model)
  arma::mat Qstar_loo;
  arma::mat Rstar_tmp;
  LinearAlgebra::qr_econ(Qstar_loo, Rstar_tmp, m.Fstar);
  arma::mat A = Qstar_loo.t() * m.Linv;
  t0 = Bench::toc(bench, "A = Q* * L^-1", t0);
  arma::mat B = LinearAlgebra::crossprod(m.Linv) - LinearAlgebra::crossprod(A);
  t0 = Bench::toc(bench, "B = t(L^-1) * L^-1 - t(A) * A", t0);

  arma::vec sigma2LOO = 1 / B.diag();
  t0 = Bench::toc(bench, "S2l = 1 / diag(Q)", t0);

  arma::vec errorsLOO = sigma2LOO % By;
  t0 = Bench::toc(bench, "E = S2l * Qy", t0);

  double loo = arma::accu(errorsLOO % errorsLOO) / n;
  t0 = Bench::toc(bench, "loo = Acc(E * E) / n", t0);

  if (yhat_out != nullptr) {
    (*yhat_out).col(0) = m_y - errorsLOO;
    (*yhat_out).col(1) = arma::sqrt(sigma2LOO);
  }

  if (grad_out != nullptr) {
    //' @ref https://github.com/cran/DiceKriging/blob/master/R/leaveOneOutGrad.R
    // leaveOneOutDer <- matrix(0, nparam, 1)
    // for (k in 1:nparam) {
    //	gradR.k <- covMatrixDerivative(model@covariance, X=model@X, C0=R, k=k)
    //	diagdQ <- - diagABA(A=Q, B=gradR.k)
    //	dsigma2LOO <- - (sigma2LOO^2) * diagdQ
    //	derrorsLOO <- dsigma2LOO * Q.y - sigma2LOO * (Q%*%(gradR.k%*%Q.y))
    //	leaveOneOutDer[k] <- 2*crossprod(errorsLOO, derrorsLOO)/model@n
    //}

    arma::uword d = m_X.n_cols;

    // Optimized gradient computation: compute gradR_k on-the-fly without storing full gradR cube
    // This eliminates expensive tube() operations and reduces memory usage

    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();

      // Build gradR_k matrix on-the-fly for this dimension only
      arma::mat gradR_k(n, n, arma::fill::zeros);
      for (arma::uword i = 0; i < n; i++) {
        for (arma::uword j = 0; j < i; j++) {
          arma::vec dlnCov = _DlnCovDtheta(m_dX.col(i * n + j), _theta);
          double gradR_k_ij = m.R.at(i, j) * dlnCov.at(k);
          gradR_k.at(i, j) = gradR_k_ij;
          gradR_k.at(j, i) = gradR_k_ij;
        }
      }
      t0 = Bench::toc(bench, "gradR_k [optimized]", t0);

      arma::vec diagdB = -LinearAlgebra::diagABA(B, gradR_k);
      t0 = Bench::toc(bench, "diagdB = DiagABA(B, gradR_k)", t0);

      arma::vec dsigma2LOO = -sigma2LOO % sigma2LOO % diagdB;
      t0 = Bench::toc(bench, "dS2l = -S2l % S2l % diagdB", t0);

      arma::vec derrorsLOO = dsigma2LOO % By - sigma2LOO % (B * (gradR_k * By));
      t0 = Bench::toc(bench, "dE = dS2l * By - S2l * (B * gradR_k * By)", t0);

      (*grad_out)(k) = 2 * dot(errorsLOO, derrorsLOO) / n;
      t0 = Bench::toc(bench, "grad_loo[k] = E * dE / n", t0);
    }
  }
  return loo;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::leaveOneOutFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  return eval_objective(_theta.n_elem, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _leaveOneOut(_theta, g, nullptr, nullptr, b);
  });
}

LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::leaveOneOutVec(const arma::vec& _theta) {
  double loo = -1;
  arma::mat yhat = arma::mat(m_y.n_elem, 2, arma::fill::none);
  loo = _leaveOneOut(_theta, nullptr, &yhat, nullptr, nullptr);

  return std::make_tuple(std::move(yhat.col(0)), std::move(yhat.col(1) * std::sqrt(m_sigma2)));
}

// Objective function for fit: bayesian-like approach fromm RobustGaSP

double Kriging::_logMargPost(const arma::vec& _gamma,
                             arma::vec* grad_out,
                             Kriging::KModel* model,
                             std::map<std::string, double>* bench) const {
  // arma::cout << " theta: " << _gamma << arma::endl;

  // In RobustGaSP:
  // neg_log_marginal_post_approx_ref <- function(param,nugget,
  // nugget.est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha) {
  //  lml=log_marginal_lik(param,nugget,nugget.est,R0,X,zero_mean,output,kernel_type,alpha);
  //  lp=log_approx_ref_prior(param,nugget,nugget.est,CL,a,b);
  //  -(lml+lp)
  //}
  // double log_marginal_lik(const Vec param,double nugget, const bool nugget_est, const List R0, const
  // Eigen::Map<Eigen::MatrixXd>  &X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd>  &output, Eigen::VectorXi
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
  //  MatrixXd Rinv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward
  //  and one backward to compute R.inv%*%X MatrixXd Xt_Rinv_X=X.transpose()*Rinv_X; //Xt%*%R.inv%*%X
  //
  //  LLT<MatrixXd> lltOfXRinvX(Xt_Rinv_X); // cholesky decomposition of Xt_Rinv_X called lltOfXRinvX
  //  MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition
  //  MatrixXd Rinv_X_Xt_Rinv_X_inv_Xt_Rinv=
  //  Rinv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(Rinv_X.transpose())));
  //  //compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward and one backward solve MatrixXd yt_Rinv=
  //  (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); MatrixXd S_2=
  //  (yt_Rinv*output-output.transpose()*Rinv_X_Xt_Rinv_X_inv_Xt_Rinv*output); double log_S_2=log(S_2(0,0)); return
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

  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::uword p = m_F.n_cols;

  arma::vec _theta = _gamma.head(d);

  // For Nugget mode, extract alpha from gamma[d]; compute sigma2/nugget after SSE
  double _alpha = (m_noise_model == NoiseModel::Nugget) ? _gamma.at(d) : m_alpha;

  Kriging::KModel m_local;
  if (model != nullptr) {
    populate_Model(*model, _theta, _alpha, bench);
  } else {
    m_local = make_Model(_theta, _alpha, bench);
  }
  Kriging::KModel& m = (model != nullptr) ? *model : m_local;

  // RobustGaSP naming...
  // arma::mat X = m_F;
  // arma::mat L = fd->T;

  auto t0 = Bench::tic();
  // m.Fstar : fd->M = solve(L, X, LinearAlgebra::default_solve_opts);

  // arma::mat Rinv_X = solve(trans(L), fd->M, LinearAlgebra::default_solve_opts);
  arma::mat Rinv_X = LinearAlgebra::solve_upper(m.L.t(), m.Fstar);

  // arma::mat Xt_Rinv_X = trans(X) * Rinv_X;  // Xt%*%R.inv%*%X
  arma::mat Xt_Rinv_X = m_F.t() * Rinv_X;

  // arma::mat LX = chol(Xt_Rinv_X, "lower");  //  retrieve factor LX  in the decomposition
  arma::mat LX = LinearAlgebra::safe_chol_lower(Xt_Rinv_X);

  arma::mat Rinv_X_Xt_Rinv_X_inv_Xt_Rinv
      = Rinv_X * (LinearAlgebra::solve_upper(LX.t(), LinearAlgebra::solve_lower(LX, Rinv_X.t())));

  arma::mat yt_Rinv = LinearAlgebra::solve_upper(m.L.t(), m.ystar).t();
  t0 = Bench::toc(bench, "YtRi = Yt \\ Tt", t0);

  arma::mat S_2 = (yt_Rinv * m_y - trans(m_y) * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv * m_y);
  t0 = Bench::toc(bench, "S2 = YtRi * y - yt * RiFFtRiFiFtRi * y", t0);

  double sigma2;
  if (m_noise_model == NoiseModel::Nugget) {
    if (m_est_sigma2 && m_est_nugget) {
      sigma2 = S_2(0, 0) / (n - p);
    } else if (m_est_sigma2 || m_est_nugget) {
      sigma2 = m_sigma2 / _alpha;
    } else {
      sigma2 = m_sigma2 + m_nugget;
    }
  } else if (m_est_sigma2) {
    sigma2 = S_2(0, 0) / (n - p);
  } else {
    sigma2 = m_sigma2;
  }
  double log_S_2 = log(sigma2 * (n - p));

  double log_marginal_lik = -sum(log(m.L.diag())) - sum(log(LX.diag())) - (n - p) / 2.0 * log_S_2;
  t0 = Bench::toc(bench, "lml = -Sum(log(diag(T))) - Sum(log(diag(TF)))...", t0);

  // Default prior params
  double a = 0.2;
  double b = 1.0 / pow(n, 1.0 / d) * (a + d);

  arma::vec CL = trans(max(m_X, 0) - min(m_X, 0)) / pow(n, 1.0 / d);
  t0 = Bench::toc(bench, "CL = (max(X) - min(X)) / n^1/d", t0);

  double nugget_ratio = (m_noise_model == NoiseModel::Nugget) ? (1.0 - _alpha) / _alpha : 0.0;
  double t = arma::accu(CL / _theta) + nugget_ratio;

  double log_approx_ref_prior = -b * t + a * log(t);

  if (grad_out != nullptr) {
    grad_out->set_size(_gamma.n_elem);
    if (m_est_sigma2) {
      t0 = Bench::tic();
      arma::mat Q_output = trans(yt_Rinv) - Rinv_X_Xt_Rinv_X_inv_Xt_Rinv * m_y;
      t0 = Bench::toc(bench, "Qo = YtRi - RiFFtRiFiFtRi * y", t0);
      arma::vec ans = compute_lmp_theta_ans(m, _theta, sigma2, Rinv_X_Xt_Rinv_X_inv_Xt_Rinv, Q_output, bench);
      grad_out->head(d) = ans - (a * CL / t - b * CL) / square(_theta);
      if (m_noise_model == NoiseModel::Nugget) {
        if (m_est_sigma2 || m_est_nugget) {
          arma::mat gradR_d = m.R / _alpha;
          gradR_d.diag().zeros();
          arma::mat Wb_k = trans(LinearAlgebra::solve_upper(m.L.t(), LinearAlgebra::solve_lower(m.L, gradR_d)))
                           - gradR_d * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
          double ans_d = -sum(Wb_k.diag()) / 2.0 + as_scalar(trans(m_y) * trans(Wb_k) * Q_output) / (2.0 * sigma2);
          grad_out->at(d) = ans_d - (a / t - b) / pow(_alpha, 2.0);
        } else {
          grad_out->at(d) = 0.0;
        }
      }
    } else {
      grad_out->head(d).zeros();
      double _eps = 1e-6;
      for (arma::uword k = 0; k < d; k++) {
        arma::vec gamma_eps = _gamma;
        gamma_eps[k] += _eps;
        (*grad_out)[k]
            = (_logMargPost(gamma_eps, nullptr, nullptr, nullptr) - (log_marginal_lik + log_approx_ref_prior)) / _eps;
      }
      if (m_noise_model == NoiseModel::Nugget)
        grad_out->at(d) = 0.0;
    }
  }

  return (log_marginal_lik + log_approx_ref_prior);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logMargPostFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  // For Nugget mode, _theta may be d-dim; augment with current alpha
  arma::vec _gamma = _theta;
  if (m_noise_model == NoiseModel::Nugget && _theta.n_elem == m_X.n_cols)
    _gamma = arma::join_cols(_theta, arma::vec{m_alpha});
  return eval_objective(_gamma.n_elem, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _logMargPost(_gamma, g, nullptr, b);
  });
}

LIBKRIGING_EXPORT double Kriging::logLikelihood() {
  return std::get<0>(Kriging::logLikelihoodFun(current_gamma(), false, false));
}

LIBKRIGING_EXPORT double Kriging::leaveOneOut() {
  return std::get<0>(Kriging::leaveOneOutFun(m_theta, false, false));
}

LIBKRIGING_EXPORT double Kriging::logMargPost() {
  return std::get<0>(Kriging::logMargPostFun(m_theta, false, false));
}

// alpha reparametrization for Nugget mode:
//   gamma_alpha = -log(1 + alpha_lower - alpha)   [alpha in [alpha_lower, 1] -> gamma_alpha in [0, inf)]
//   alpha       = 1 + alpha_lower - exp(-gamma_alpha)
//   d(alpha)/d(gamma_alpha) = 1 + alpha_lower - alpha
static constexpr double nugget_alpha_lower = 1e-3;

static arma::vec nugget_reparam_to(const arma::vec& _theta_alpha) {
  const arma::uword d = _theta_alpha.n_elem - 1;
  arma::vec gamma(_theta_alpha.n_elem);
  gamma.head(d) = Optim::reparam_to(_theta_alpha.head(d));
  gamma.at(d) = -std::log(1.0 + nugget_alpha_lower - _theta_alpha.at(d));
  return gamma;
}

static arma::vec nugget_reparam_from(const arma::vec& _gamma) {
  const arma::uword d = _gamma.n_elem - 1;
  arma::vec theta_alpha(_gamma.n_elem);
  theta_alpha.head(d) = Optim::reparam_from(_gamma.head(d));
  theta_alpha.at(d) = 1.0 + nugget_alpha_lower - std::exp(-_gamma.at(d));
  return theta_alpha;
}

static arma::vec nugget_reparam_from_deriv(const arma::vec& _theta_alpha, const arma::vec& _grad) {
  const arma::uword d = _theta_alpha.n_elem - 1;
  arma::vec D(_theta_alpha.n_elem);
  D.head(d) = Optim::reparam_from_deriv(_theta_alpha.head(d), _grad.head(d));
  D.at(d) = _grad.at(d) * (1.0 + nugget_alpha_lower - _theta_alpha.at(d));
  return D;
}

// =============================================================================
// Vecchia approximated log-likelihood (objective="LLVecchia(m)")
//
// Vecchia (1988): log p(y) = sum_i log p(y_i | y_{N(i)}) where N(i) is the set
// of (at most) m nearest previously-ordered neighbors in a maxmin ordering.
// Cost O(n m^3) per evaluation instead of O(n^3); valid Gaussian density
// (sparse inverse Cholesky); exact for m = n-1. See Guinness (2018),
// Katzfuss & Guinness (2021).
//
// Profiling matches the exact "LL" objective: sigma2 in closed form, beta by
// (per-conditional) GLS. Gradient in theta is analytic; the theta-dependence
// of beta_hat is ignored by the envelope theorem (d/dbeta = 0 at beta_hat).
// =============================================================================

arma::uword Kriging::parse_vll_m(const std::string& objective) {
  // "LLVecchia" -> default 30 ; "LLVecchia(m)" -> m
  if (objective == "LLVecchia")
    return 30;
  if (objective.rfind("LLVecchia(", 0) == 0 && objective.back() == ')') {
    const std::string inside = objective.substr(10, objective.size() - 11);
    try {
      const long m = std::stol(inside);
      if (m >= 1)
        return static_cast<arma::uword>(m);
    } catch (const std::exception&) {
      // fall through to the throw below
    }
  }
  throw std::invalid_argument("Invalid Vecchia objective '" + objective
                              + "': expected \"LLVecchia\" or \"LLVecchia(m)\" with m >= 1 (e.g. \"LLVecchia(30)\")");
}

void Kriging::make_vecchia_sets() {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword m = std::min<arma::uword>(m_vecchia_m, n - 1);

  // column-major layout (d x n) for cache-friendly point access in the O(n^2)
  // loops below; arma row access in tight loops allocates temporaries
  const arma::mat Xt = m_X.t();
  const double* xp = Xt.memptr();
  auto dist2 = [xp, d](arma::uword i, arma::uword j) {
    const double* a = xp + i * d;
    const double* b = xp + j * d;
    double s = 0;
    for (arma::uword k = 0; k < d; ++k) {
      const double dk = a[k] - b[k];
      s += dk * dk;
    }
    return s;
  };

  // --- greedy maxmin ordering (Guinness 2018) on normalized inputs ---------
  // start from the point closest to the centroid, then repeatedly add the
  // point maximizing its minimal distance to already-ordered points. O(n^2 d).
  m_vecchia_order.set_size(n);
  const arma::vec centroid = arma::mean(Xt, 1);
  arma::vec mind2(n);
  for (arma::uword i = 0; i < n; ++i) {
    double s = 0;
    for (arma::uword k = 0; k < d; ++k) {
      const double dk = xp[i * d + k] - centroid(k);
      s += dk * dk;
    }
    mind2(i) = s;
  }
  m_vecchia_order(0) = mind2.index_min();
  for (arma::uword i = 0; i < n; ++i)
    mind2(i) = dist2(i, m_vecchia_order(0));
  mind2(m_vecchia_order(0)) = -arma::datum::inf;
  for (arma::uword t = 1; t < n; ++t) {
    const arma::uword next = mind2.index_max();
    m_vecchia_order(t) = next;
    double* mp = mind2.memptr();
    for (arma::uword i = 0; i < n; ++i) {
      if (mp[i] == -arma::datum::inf)
        continue;
      const double d2 = dist2(i, next);
      if (d2 < mp[i])
        mp[i] = d2;
    }
    mind2(next) = -arma::datum::inf;
  }

  // --- m nearest previously-ordered neighbors (global row indices) ---------
  // partial selection (nth_element) instead of a full sort: O(n^2 d) total
  m_vecchia_neighbors.assign(n, arma::uvec());
  std::vector<std::pair<double, arma::uword>> cand;
  for (arma::uword t = 1; t < n; ++t) {
    const arma::uword k = std::min<arma::uword>(m, t);
    cand.resize(t);
    for (arma::uword j = 0; j < t; ++j)
      cand[j] = {dist2(m_vecchia_order(t), m_vecchia_order(j)), m_vecchia_order(j)};
    if (k < t)
      std::nth_element(cand.begin(), cand.begin() + k, cand.end());
    arma::uvec nb(k);
    for (arma::uword j = 0; j < k; ++j)
      nb(j) = cand[j].second;
    m_vecchia_neighbors[t] = nb;
  }
}

double Kriging::_logLikelihoodVecchia(const arma::vec& _theta,
                                      arma::vec* grad_out,
                                      arma::vec* beta_out,
                                      double* sigma2_out) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword p = m_F.n_cols;
  const bool with_grad = (grad_out != nullptr);
  constexpr double v_floor = 1e-15;

  arma::vec u(n);     // y_i - a' y_N  (trend-free residual numerator)
  arma::vec v(n);     // conditional correlation variance 1 - r'a
  arma::mat W(p, n);  // w_i = f_i - F_N' a  (per-conditional trend design)
  arma::mat du, dv;
  arma::cube dW;
  if (with_grad) {
    du.zeros(n, d);
    dv.zeros(n, d);
    dW.zeros(p, n, d);
  }

  for (arma::uword t = 0; t < n; ++t) {
    const arma::uword oi = m_vecchia_order(t);
    const arma::uvec& Ni = m_vecchia_neighbors[t];
    const arma::uword mi = Ni.n_elem;
    if (mi == 0) {
      u(t) = m_y(oi);
      v(t) = 1.0;
      W.col(t) = m_F.row(oi).t();
      continue;
    }

    // correlations (and their theta-derivatives) within {N(i), i}
    arma::mat RN(mi, mi, arma::fill::ones);
    arma::vec r(mi);
    arma::cube dRN;
    arma::mat dr;
    if (with_grad) {
      dRN.zeros(mi, mi, d);
      dr.zeros(mi, d);
    }
    for (arma::uword a = 0; a < mi; ++a) {
      for (arma::uword b = a + 1; b < mi; ++b) {
        const arma::vec dx = (m_X.row(Ni(a)) - m_X.row(Ni(b))).t();
        const double c = _Cov(dx, _theta);
        RN(a, b) = RN(b, a) = c;
        if (with_grad) {
          const arma::vec dln = _DlnCovDtheta(dx, _theta);
          for (arma::uword k = 0; k < d; ++k)
            dRN(a, b, k) = dRN(b, a, k) = c * dln(k);
        }
      }
      const arma::vec dx = (m_X.row(Ni(a)) - m_X.row(oi)).t();
      const double c = _Cov(dx, _theta);
      r(a) = c;
      if (with_grad) {
        const arma::vec dln = _DlnCovDtheta(dx, _theta);
        for (arma::uword k = 0; k < d; ++k)
          dr(a, k) = c * dln(k);
      }
    }
    RN.diag() += LinearAlgebra::num_nugget;

    const arma::mat LN = LinearAlgebra::safe_chol_lower(RN);
    const arma::vec a_vec = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), r));

    v(t) = std::max(1.0 - arma::dot(r, a_vec), v_floor);
    u(t) = m_y(oi) - arma::dot(a_vec, m_y(Ni));
    W.col(t) = m_F.row(oi).t() - m_F.rows(Ni).t() * a_vec;

    if (with_grad) {
      const arma::mat FN_t = m_F.rows(Ni).t();  // p x mi
      const arma::vec yN = m_y(Ni);
      for (arma::uword k = 0; k < d; ++k) {
        const arma::vec rhs = dr.col(k) - dRN.slice(k) * a_vec;
        const arma::vec da = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), rhs));
        dv(t, k) = -(arma::dot(dr.col(k), a_vec) + arma::dot(r, da));
        du(t, k) = -arma::dot(da, yN);
        dW.slice(k).col(t) = -FN_t * da;
      }
    }
  }

  // --- profiled beta (GLS over the per-conditional design) -------------------
  arma::vec beta;
  if (m_est_beta || m_beta.n_elem != p) {
    arma::mat A(p, p, arma::fill::zeros);
    arma::vec b(p, arma::fill::zeros);
    for (arma::uword t = 0; t < n; ++t) {
      A += W.col(t) * W.col(t).t() / v(t);
      b += W.col(t) * u(t) / v(t);
    }
    A.diag() += LinearAlgebra::num_nugget;
    beta = arma::solve(A, b, arma::solve_opts::likely_sympd);
  } else {
    beta = m_beta;
  }

  // --- profiled sigma2 & objective -------------------------------------------
  const arma::vec e = u - W.t() * beta;
  const double Q = arma::accu(e % e / v);
  const double sigma2 = Q / n;
  if (beta_out != nullptr)
    *beta_out = beta;
  if (sigma2_out != nullptr)
    *sigma2_out = sigma2;
  const double vll = -0.5 * n * std::log(2 * M_PI * sigma2) - 0.5 * arma::accu(arma::log(v)) - 0.5 * n;

  if (with_grad) {
    // envelope theorem: dQ/dbeta = 0 at beta_hat, so beta_hat's
    // theta-dependence does not contribute to the gradient
    for (arma::uword k = 0; k < d; ++k) {
      const arma::vec de = du.col(k) - dW.slice(k).t() * beta;
      const double dQ = arma::accu(2.0 * e % de / v - (e % e / (v % v)) % dv.col(k));
      (*grad_out)(k) = -0.5 * n * dQ / Q - 0.5 * arma::accu(dv.col(k) / v);
    }
  }
  return vll;
}

void Kriging::check_not_vecchia_light(const char* what) const {
  if (m_vecchia_light)
    throw std::runtime_error(std::string(what)
                             + ": not available on a light Vecchia fit "
                               "(refit with set_vecchia_exact_commit(true))");
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logLikelihoodVecchiaFun(const arma::vec& theta,
                                                                                 bool return_grad) {
  if (m_vecchia_m == 0 || m_vecchia_order.n_elem != m_X.n_rows)
    throw std::runtime_error("logLikelihoodVecchiaFun: model was not fitted with objective=\"LLVecchia(m)\"");
  arma::vec grad;
  if (return_grad) {
    grad.set_size(theta.n_elem);
    const double vll = _logLikelihoodVecchia(theta, &grad);
    return {vll, grad};
  }
  return {_logLikelihoodVecchia(theta, nullptr), grad};
}

/* Vecchia (local) prediction: condition each prediction point on its m
 * nearest observations only (Katzfuss & Guinness 2021, response-only
 * conditioning). Per point: O(m^3); no cross-covariances between prediction
 * points (use predict() for the exact joint version). */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::predictVecchia(const arma::mat& X_n,
                                                                           bool return_stdev,
                                                                           arma::uword m) {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword q = X_n.n_rows;
  if (X_n.n_cols != d)
    throw std::invalid_argument("X_n should have the same number of columns as X");
  if (m == 0)
    m = (m_vecchia_m > 0) ? m_vecchia_m : 30;
  m = std::min<arma::uword>(m, n);

  // normalize prediction inputs like predict() does
  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;
  const arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);

  arma::vec mean(q);
  arma::vec stdev(return_stdev ? q : 0);

  // column-major observed inputs for cache-friendly neighbor search
  const arma::mat Xt = m_X.t();
  const double* xp = Xt.memptr();

#if defined(_OPENMP) && !defined(LK_NESTED_NO_OMP)
#pragma omp parallel for schedule(dynamic, 16) if (q > 32 && n * m >= 40000)
#endif
  for (arma::sword t = 0; t < static_cast<arma::sword>(q); ++t) {
    // m nearest observations (normalized space), partial selection
    std::vector<std::pair<double, arma::uword>> cand(n);
    const arma::rowvec x_t = Xn_n.row(static_cast<arma::uword>(t));
    for (arma::uword j = 0; j < n; ++j) {
      double s = 0;
      for (arma::uword k = 0; k < d; ++k) {
        const double dk = x_t(k) - xp[j * d + k];
        s += dk * dk;
      }
      cand[j] = {s, j};
    }
    if (m < n)
      std::nth_element(cand.begin(), cand.begin() + m, cand.end());
    arma::uvec Ni(m);
    for (arma::uword j = 0; j < m; ++j)
      Ni(j) = cand[j].second;

    // local kriging on the neighborhood
    arma::mat RN(m, m, arma::fill::ones);
    arma::vec r(m);
    for (arma::uword a = 0; a < m; ++a) {
      for (arma::uword b = a + 1; b < m; ++b) {
        const arma::vec dx = (m_X.row(Ni(a)) - m_X.row(Ni(b))).t();
        RN(a, b) = RN(b, a) = _Cov(dx, m_theta);
      }
      const arma::vec dx = (m_X.row(Ni(a)) - x_t).t();
      r(a) = _Cov(dx, m_theta);
    }
    RN.diag() += LinearAlgebra::num_nugget;
    const arma::mat LN = LinearAlgebra::safe_chol_lower(RN);
    const arma::vec a_vec = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), r));

    const arma::vec resid = m_y(Ni) - m_F.rows(Ni) * m_beta;
    mean(t) = arma::dot(F_n.row(static_cast<arma::uword>(t)), m_beta) + arma::dot(a_vec, resid);
    if (return_stdev)
      stdev(t) = std::sqrt(m_sigma2 * std::max(0.0, 1.0 - arma::dot(r, a_vec)));
  }

  // de-normalize outputs
  mean = mean * m_scaleY + m_centerY;
  if (return_stdev)
    stdev = stdev * m_scaleY;
  return {mean, stdev};
}

// =============================================================================
// Nystrom approximated log-likelihood (objective="LLNystrom(k)")
//
// Global low-rank alternative to Vecchia: R ~= R_ns * R_ss^-1 * R_ns.t(),
// where S is a set of k landmark rows of X, chosen ONCE (make_nystrom_landmarks,
// called from fit() before optimization starts) and held FIXED across every
// theta evaluation. This is standard (fixed-landmark) Nystrom, as opposed to
// LinearAlgebra::nystromFactor's greedy pivoted-Cholesky landmark selection,
// which re-picks pivots from the current covariance values and is therefore
// only used here once, at a theta-neutral reference kernel, to seed the
// landmark set. Re-selecting greedily at every theta (i.e. calling
// nystromFactor directly inside this function) makes the pivot choice --
// and hence the objective value -- discontinuous in theta: an earlier version
// of this function did exactly that, and its finite-difference gradient was
// inconsistent between step sizes as a direct symptom (see
// KrigingNystromTest.cpp / git history).
//
// Once S is fixed, R_ss (k x k) and R_ns (n x k) are ordinary, smooth
// functions of theta; U := R_ns * L_ss^-T (L_ss = chol(R_ss)) satisfies
// U*U.t() = R_ns * R_ss^-1 * R_ns.t() exactly, and R^-1/log|R| go through
// LinearAlgebra::woodbury_solve/woodbury_logdet -- the n x n matrix R is
// never materialized. Cost O(n*k^2) per evaluation instead of O(n^3).
//
// Profiling matches the exact "LL" objective: sigma2 and beta in closed form
// (same GLS formulas as _logLikelihood, just solved via Woodbury instead of a
// dense Cholesky). Gradient in theta is analytic (envelope theorem, same
// principle as _logLikelihoodVecchia: beta_hat/sigma2_hat's own theta-dependence
// doesn't contribute since d(ll)/d(beta)=d(ll)/d(sigma2)=0 at their profiled
// values) -- see the derivation inside _logLikelihoodNystrom's grad_out block.
// An earlier version of this objective used a finite-difference gradient
// instead (see git history / KrigingNystromTest.cpp); it worked but was more
// sensitive to the likelihood surface's local curvature near small-theta /
// near-singular-R regions than the analytic form is.
// =============================================================================

arma::uword Kriging::parse_nystrom_k(const std::string& objective) {
  // "LLNystrom" -> default 50 ; "LLNystrom(k)" -> k
  if (objective == "LLNystrom")
    return 50;
  if (objective.rfind("LLNystrom(", 0) == 0 && objective.back() == ')') {
    const std::string inside = objective.substr(10, objective.size() - 11);
    try {
      const long k = std::stol(inside);
      if (k >= 1)
        return static_cast<arma::uword>(k);
    } catch (const std::exception&) {
      // fall through to the throw below
    }
  }
  throw std::invalid_argument("Invalid Nystrom objective '" + objective
                              + "': expected \"LLNystrom\" or \"LLNystrom(k)\" with k >= 1 (e.g. \"LLNystrom(50)\")");
}

void Kriging::make_nystrom_landmarks() {
  // Reference kernel used only to RANK points by spatial coverage via
  // nystromFactor's greedy residual criterion, never to evaluate the
  // likelihood itself. Scaled to the data's own extent (m_maxdX, per
  // dimension) rather than a fixed isotropic range: an unrelated scale (e.g.
  // range=1 on normalized [0,1] data) makes the reference correlation matrix
  // nearly rank-1 (all points look alike), so the greedy ranking picks a
  // poorly-spread/near-redundant landmark set -- which then makes R_ss at the
  // ACTUAL (possibly much shorter-range) theta ill-conditioned, in turn
  // making safe_chol_lower's adaptive nugget retry trigger inconsistently
  // across nearby theta and breaking the smoothness the fixed-landmark
  // scheme is meant to provide.
  const arma::vec ref_theta = m_maxdX;
  arma::vec diag_resid_unused;
  LinearAlgebra::nystromFactor(&diag_resid_unused,
                               m_X,
                               ref_theta,
                               _Cov,
                               /*factor=*/1.0,
                               KrigingImpl::ones,
                               m_nystrom_k,
                               1e-12,
                               &m_nystrom_landmarks);
}

double Kriging::_logLikelihoodNystrom(const arma::vec& _theta,
                                      arma::vec* grad_out,
                                      arma::vec* beta_out,
                                      double* sigma2_out,
                                      arma::mat* U_out,
                                      arma::vec* D_out) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword kL = m_nystrom_landmarks.n_elem;
  const arma::mat X_land = m_X.rows(m_nystrom_landmarks);  // k x d

  arma::mat R_ss(kL, kL, arma::fill::none);
  LinearAlgebra::covMat_sym_X(&R_ss, X_land.t(), _theta, _Cov, /*factor=*/1.0, KrigingImpl::ones);

  arma::mat R_ns(n, kL, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_ns, m_X.t(), X_land.t(), _theta, _Cov, /*factor=*/1.0);

  // A FIXED jitter (as opposed to safe_chol_lower's adaptive nugget-retry,
  // which decides whether/how much to add based on a per-call rcond check)
  // keeps this Cholesky deterministic and smooth in theta: an adaptive retry
  // can trigger on one side of a finite-difference stencil and not the other,
  // reintroducing the kind of discontinuity the fixed-landmark scheme is
  // meant to eliminate.
  R_ss.diag() += LinearAlgebra::num_nugget;
  const arma::mat L_ss = arma::chol(R_ss, "lower");
  const arma::mat U
      = arma::trans(LinearAlgebra::solve_lower(L_ss, R_ns.t()));  // n x k; U*U.t() = R_ns R_ss^-1 R_ns.t()

  const arma::vec captured = arma::sum(arma::square(U), 1);
  const arma::vec D = arma::clamp(1.0 - captured, LinearAlgebra::num_nugget, arma::datum::inf);

  const arma::mat RinvF = LinearAlgebra::woodbury_solve(U, D, m_F);
  const arma::vec Rinvy = LinearAlgebra::woodbury_solve(U, D, m_y);

  const arma::mat A = m_F.t() * RinvF;
  const arma::vec b = m_F.t() * Rinvy;
  const arma::vec beta = arma::solve(A, b, arma::solve_opts::likely_sympd);

  const arma::vec e = m_y - m_F * beta;
  const arma::vec Rinve = Rinvy - RinvF * beta;  // Rinv is linear: Rinv(y - F*beta) = Rinv*y - Rinv*F*beta
  const double SSE = arma::dot(e, Rinve);
  const double sigma2 = SSE / n;
  const double logdetR = LinearAlgebra::woodbury_logdet(U, D);

  if (beta_out != nullptr)
    *beta_out = beta;
  if (sigma2_out != nullptr)
    *sigma2_out = sigma2;
  if (U_out != nullptr)
    *U_out = U;
  if (D_out != nullptr)
    *D_out = D;

  if (grad_out != nullptr) {
    // Analytic gradient, envelope theorem (beta_hat and sigma2_hat held fixed
    // -- see the file-level comment above): for the standard concentrated GP
    // log-likelihood, d(ll)/d(theta_k) = 0.5*(x' dRhat/dtheta_k x / sigma2 -
    // trace(Rinv dRhat/dtheta_k)), x = Rinv*(y - F*beta_hat) = Rinve.
    //
    // Rhat_total = U*U.t() + diag(D) with D = 1 - diag(U*U.t()) (constant on
    // the diagonal, by construction): so dRhat_total/dtheta_k equals
    // d(U*U.t())/dtheta_k with its diagonal zeroed out. Writing U*U.t() =
    // M*K^-1*M.t() (M=R_ns, K=R_ss, both smooth in theta since the landmark
    // set is fixed), d(M K^-1 M.t())/dtheta_k = dM_k K^-1 M.t() + (same
    // transposed) - M K^-1 dK_k K^-1 M.t(); both the quadratic form x'(.)x
    // and the trace of Rinv*(.) reduce to O(n*k) / O(n*k^2) work per theta_k
    // via the identities below (derived directly in this function's body).
    const arma::uword d = _theta.n_elem;
    grad_out->set_size(d);

    const arma::vec& x = Rinve;

    const arma::mat Kinv = LinearAlgebra::inv_sympd(L_ss);           // R_ss^-1
    const arma::mat W = Kinv * R_ns.t();                             // k x n = R_ss^-1 * R_ns.t()
    const arma::mat RM = LinearAlgebra::woodbury_solve(U, D, R_ns);  // n x k = Rinv * R_ns

    const arma::vec Dinv = 1.0 / D;
    const arma::mat DinvU = U.each_col() % Dinv;
    const arma::mat Mcore = arma::eye<arma::mat>(kL, kL) + U.t() * DinvU;  // I_k + U.t() D^-1 U
    const arma::mat McoreInv = LinearAlgebra::inv_sympd(arma::chol(Mcore, "lower"));
    const arma::vec diagRinv = Dinv - LinearAlgebra::diagABA(DinvU, McoreInv);  // diag(Rinv)

    const arma::vec Mtx = R_ns.t() * x;    // k
    const arma::vec p = Kinv * Mtx;        // R_ss^-1 * R_ns.t() * x
    const arma::mat RMtM = RM.t() * R_ns;  // k x k = R_ns.t() * Rinv * R_ns

    for (arma::uword kk = 0; kk < d; ++kk) {
      arma::mat dM_k(n, kL, arma::fill::none);
      for (arma::uword i = 0; i < n; ++i)
        for (arma::uword j = 0; j < kL; ++j) {
          const arma::vec dx = (m_X.row(i) - X_land.row(j)).t();
          dM_k(i, j) = R_ns(i, j) * _DlnCovDtheta(dx, _theta)(kk);
        }
      arma::mat dK_k(kL, kL, arma::fill::zeros);
      for (arma::uword a = 0; a < kL; ++a)
        for (arma::uword bb = a + 1; bb < kL; ++bb) {
          const arma::vec dx = (X_land.row(a) - X_land.row(bb)).t();
          const double v = R_ss(a, bb) * _DlnCovDtheta(dx, _theta)(kk);
          dK_k(a, bb) = dK_k(bb, a) = v;
        }

      const arma::vec dMkx = dM_k.t() * x;
      const double raw1 = 2.0 * arma::dot(dMkx, p) - arma::as_scalar(p.t() * dK_k * p);
      const arma::vec rowdot = arma::sum(dM_k % W.t(), 1);         // n: dM_k(i,:) . w_i
      const arma::vec quad = LinearAlgebra::diagABA(W.t(), dK_k);  // n: w_i' dK_k w_i
      const double diagcorr1 = arma::dot(x % x, 2.0 * rowdot - quad);
      const double term1_k = raw1 - diagcorr1;

      const arma::mat RMt_dMk = RM.t() * dM_k;  // k x k
      const double raw2 = 2.0 * arma::trace(Kinv * RMt_dMk) - arma::trace(Kinv * dK_k * Kinv * RMtM);
      const double diagcorr2 = arma::dot(diagRinv, 2.0 * rowdot - quad);
      const double term2_k = raw2 - diagcorr2;

      (*grad_out)(kk) = 0.5 * (term1_k / sigma2 - term2_k);
    }
  }

  return -0.5 * (n * std::log(2 * M_PI * sigma2) + logdetR + n);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logLikelihoodNystromFun(const arma::vec& theta,
                                                                                 bool return_grad) {
  if (m_nystrom_k == 0)
    throw std::runtime_error("logLikelihoodNystromFun: model was not fitted with objective=\"LLNystrom(k)\"");
  arma::vec grad;
  if (return_grad) {
    grad.set_size(theta.n_elem);
    const double ll = _logLikelihoodNystrom(theta, &grad);
    return {ll, grad};
  }
  return {_logLikelihoodNystrom(theta), grad};
}

void Kriging::check_not_nystrom_light(const char* what) const {
  if (m_nystrom_light)
    throw std::runtime_error(std::string(what)
                             + ": not available on a Nystrom fit "
                               "(refit with objective=\"LL\", or use predictNystrom)");
}

/* Nystrom (global) prediction: uses the committed rank-k factors (U, D) via
 * Woodbury instead of the exact O(n^2) triangular solve. Mean is
 * universal-kriging-style with the committed beta; variance is the
 * simple-kriging one (beta treated as known, like predictVecchia). */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::predictNystrom(const arma::mat& X_n, bool return_stdev) {
  if (m_nystrom_U.is_empty())
    throw std::runtime_error("predictNystrom: model was not fitted with objective=\"LLNystrom(k)\"");
  const arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::invalid_argument("predictNystrom: X_n has wrong dimension: " + std::to_string(X_n.n_cols)
                                + " instead of " + std::to_string(d));

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);

  arma::mat R_on = arma::mat(m_X.n_rows, X_n.n_rows, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, m_X.t(), Xn_n.t(), m_theta, _Cov, 1.0);

  const arma::mat RinvR_on = LinearAlgebra::woodbury_solve(m_nystrom_U, m_nystrom_D, R_on);  // n_o x n_n

  arma::vec mean = F_n * m_beta + RinvR_on.t() * (m_y - m_F * m_beta);
  mean = mean * m_scaleY + m_centerY;

  arma::vec stdev;
  if (return_stdev) {
    const arma::vec quad = arma::sum(R_on % RinvR_on, 0).t();  // r_j' Rinv r_j per prediction point j
    stdev = arma::sqrt(arma::clamp(m_sigma2 * (1.0 - quad), 0.0, arma::datum::inf));
    stdev *= m_scaleY;
  }
  return {mean, stdev};
}

LIBKRIGING_EXPORT arma::mat Kriging::simulateNystrom(int nsim, int seed, const arma::mat& X_n) {
  if (m_nystrom_U.is_empty())
    throw std::runtime_error("simulateNystrom: model was not fitted with objective=\"LLNystrom(k)\"");
  const arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::invalid_argument("simulateNystrom: X_n has wrong dimension: " + std::to_string(X_n.n_cols)
                                + " instead of " + std::to_string(d));

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  const arma::uword n_n = X_n.n_rows;

  arma::mat R_on = arma::mat(m_X.n_rows, n_n, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, m_X.t(), Xn_n.t(), m_theta, _Cov, 1.0);

  const arma::mat RinvR_on = LinearAlgebra::woodbury_solve(m_nystrom_U, m_nystrom_D, R_on);  // n_o x n_n

  const arma::vec yhat_n = F_n * m_beta + RinvR_on.t() * (m_y - m_F * m_beta);

  // Joint (simple-kriging, beta treated as known -- like predictNystrom)
  // covariance among the n_n SIMULATION points: dense, but only n_n x n_n
  // (n_n is expected to be small; it is m_X, not X_n, that can be large).
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  LinearAlgebra::covMat_sym_X(&R_nn, Xn_n.t(), m_theta, _Cov, 1.0, arma::vec());
  arma::mat Sigma_n = arma::symmatu(R_nn - R_on.t() * RinvR_on);

  // Unlike a per-point marginal variance (predictNystrom's "1 - quad", simply
  // clamped to >= 0), the JOINT covariance among several simulation points
  // can have a genuinely (not just numerically-noise) negative eigenvalue: it
  // mixes an EXACT R_nn with cross-terms approximated through the same
  // rank-k landmark structure, and that mismatch does not have to be tiny for
  // a coarse k relative to n, or at an extreme theta (e.g. one an optimizer
  // drove very large on noise-free data -- a known, unrelated GP-MLE
  // degeneracy, not a Nystrom artifact, but one that makes R_ss ill
  // conditioned and hence this cross-term subtraction lose precision).
  //
  // safe_chol_lower's escalating (10x-per-retry) nugget, or a single uniform
  // diagonal shift calibrated to the WORST eigenvalue (tried first here; see
  // git history), are both the wrong tool: both inflate variance at EVERY
  // simulation point to fix what is usually one bad eigendirection shared by
  // only a few of them. Clip negative eigenvalues to a small floor instead
  // (nearest-PSD-matrix projection) -- this only removes the offending
  // eigendirection(s) and leaves well-conditioned points' variances alone.
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, Sigma_n);
  const double floor_val = std::max(LinearAlgebra::num_nugget, std::sqrt(arma::datum::eps) * eigval.max());
  eigval = arma::clamp(eigval, floor_val, arma::datum::inf);
  Sigma_n = arma::symmatu(eigvec * arma::diagmat(eigval) * eigvec.t());

  const arma::mat L_sigma = LinearAlgebra::safe_chol_lower(Sigma_n);

  arma::mat y_n = arma::mat(n_n, nsim, arma::fill::none);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += L_sigma * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  y_n = m_centerY + m_scaleY * y_n;
  return y_n;
}

void Kriging::update_nystrom(const arma::vec& y_u, const arma::mat& X_u, bool refit) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");
  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;
  const arma::vec yn_u = (y_u - m_centerY) / m_scaleY;

  // m_nystrom_landmarks holds row-indices into m_X; appending rows here
  // (never reordering/removing) keeps them valid without any adjustment.
  m_X = arma::join_cols(m_X, Xn_u);
  m_y = arma::join_cols(m_y, yn_u);
  m_F = Trend::regressionModelMatrix(m_regmodel, m_X);

  if (refit && m_optim != "none") {
    // Warm restart: single BFGS from the current theta, over the SAME
    // (fixed) landmark set -- landmarks are only re-picked by a full fit(),
    // which is what keeps this O((n_old+n_new)*k^2) rather than paying the
    // O(n*k) landmark-ranking pass (and losing the warm start) again.
    const FitOfn fit_ofn = make_fit_objective(m_objective);
    const arma::uword d = m_X.n_cols;

    // theta bounds from the per-dimension range of the extended data --
    // O(n*d), NOT Optim::theta_bounds's variogram-slope heuristic, which
    // needs the O(n^2) dX cube this update path is built to avoid.
    const arma::vec maxdX_local = arma::trans(arma::max(m_X, 0) - arma::min(m_X, 0));
    arma::vec theta_lower = arma::min(m_theta, Optim::theta_lower_factor * maxdX_local);
    arma::vec theta_upper = arma::max(m_theta, Optim::theta_upper_factor * maxdX_local);

    arma::vec gamma_start = m_theta;
    arma::vec gamma_lower = theta_lower;
    arma::vec gamma_upper = theta_upper;
    if (Optim::reparametrize) {
      gamma_start = Optim::reparam_to(gamma_start);
      gamma_lower = Optim::reparam_to(gamma_lower);
      gamma_upper = Optim::reparam_to(gamma_upper);
    }

    lbfgsb::Optimizer optimizer{static_cast<unsigned int>(d)};
    optimizer.iprint = Optim::log_level - 2;
    optimizer.max_iter = Optim::max_iteration;
    optimizer.pgtol = Optim::gradient_tolerance;
    optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
    const arma::ivec bounds_type{d, arma::fill::value(2)};

    optimizer.minimize([&fit_ofn](const arma::vec& vals_inp,
                                  arma::vec& grad_out) -> double { return fit_ofn(vals_inp, &grad_out, nullptr); },
                       gamma_start,
                       gamma_lower.memptr(),
                       gamma_upper.memptr(),
                       bounds_type.memptr());

    m_theta = Optim::reparametrize ? Optim::reparam_from(gamma_start) : gamma_start;
    m_est_theta = true;
  }

  arma::vec beta_v;
  double sigma2_v = -1;
  arma::mat U_v;
  arma::vec D_v;
  _logLikelihoodNystrom(m_theta, nullptr, &beta_v, &sigma2_v, &U_v, &D_v);
  if (m_est_beta)
    m_beta = beta_v;
  if (m_est_sigma2)
    m_sigma2 = sigma2_v;
  m_nystrom_U = std::move(U_v);
  m_nystrom_D = std::move(D_v);
}

// =============================================================================
// Iterative (matrix-free CG + stochastic log-det) approximated log-likelihood
// (objective="LLIterative(m)")
//
// Unlike LLVecchia/LLNystrom (which each replace R by a cheaper structured
// approximation -- local conditioning / global low rank), this keeps R
// itself EXACT: every term except log|R| is a CG-converged matrix-free
// solve, mathematically the same quantity a dense Cholesky factorization
// would give (up to CG's own convergence tolerance), just computed via
// O(n^2) matvecs instead of an O(n^3) factorization. Only the log-
// determinant -- the one term CG cannot produce directly -- is replaced by
// a Stochastic Lanczos Quadrature (SLQ) estimate
// (LinearAlgebra::stochasticLogDet), and correspondingly the gradient's
// trace(Rinv * dR/dtheta_k) term is a Hutchinson estimate sharing the SAME
// probe vectors as the log-det (both are computed from the same
// LinearAlgebra::conjugateGradient(Rmul, probes, ...) batch solve). This is
// the same overall strategy as GPyTorch's BBMM/Lanczos-based inference.
//
// Probes are drawn ONCE per fit (make_iterative_probes, fixed seed) and
// held fixed across every theta evaluation during optimization, for the
// same smoothness reason LLNystrom's landmarks are fixed: re-drawing fresh
// probes at every evaluation would make the objective (and its gradient)
// noisy/non-smooth between BFGS iterations.
// =============================================================================

arma::uword Kriging::parse_iterative_m(const std::string& objective) {
  // "LLIterative" -> default 30 ; "LLIterative(m)" -> m
  if (objective == "LLIterative")
    return 30;
  if (objective.rfind("LLIterative(", 0) == 0 && objective.back() == ')') {
    const std::string inside = objective.substr(12, objective.size() - 13);
    try {
      const long m = std::stol(inside);
      if (m >= 1)
        return static_cast<arma::uword>(m);
    } catch (const std::exception&) {
      // fall through to the throw below
    }
  }
  throw std::invalid_argument("Invalid Iterative objective '" + objective
                              + "': expected \"LLIterative\" or \"LLIterative(m)\" with m >= 1 (e.g. "
                                "\"LLIterative(30)\")");
}

void Kriging::make_iterative_probes() {
  const arma::uword n = m_X.n_rows;
  m_iterative_probes = LinearAlgebra::rademacherProbes(n, m_iterative_nprobe, /*seed=*/20260808u);
}

double Kriging::_logLikelihoodIterative(const arma::vec& _theta,
                                        arma::vec* grad_out,
                                        arma::vec* beta_out,
                                        double* sigma2_out) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword nprobe = m_iterative_nprobe;
  const arma::uword max_iter = (m_iterative_cg_max_iter == 0) ? 2 * n : m_iterative_cg_max_iter;

  const arma::mat Xt = m_X.t();  // d x n, cache-friendly columns
  const arma::vec& theta = _theta;
  const auto& cov = _Cov;
  auto Rmul = [&Xt, &theta, &cov, n](const arma::vec& v) -> arma::vec {
    arma::vec out(n, arma::fill::none);
#ifdef _OPENMP
    if (n >= 200) {
      int optimal_threads = get_optimal_threads(2);
#pragma omp parallel for schedule(static) num_threads(optimal_threads) if (n >= 200)
      for (arma::sword i = 0; i < static_cast<arma::sword>(n); ++i) {
        double acc = v(static_cast<arma::uword>(i));  // diag = 1
        for (arma::uword j = 0; j < n; ++j) {
          if (j == static_cast<arma::uword>(i))
            continue;
          acc += cov(Xt.col(i) - Xt.col(j), theta) * v(j);
        }
        out(static_cast<arma::uword>(i)) = acc;
      }
    } else {
#endif
      for (arma::uword i = 0; i < n; ++i) {
        double acc = v(i);  // diag = 1
        for (arma::uword j = 0; j < n; ++j) {
          if (j == i)
            continue;
          acc += cov(Xt.col(i) - Xt.col(j), theta) * v(j);
        }
        out(i) = acc;
      }
#ifdef _OPENMP
    }
#endif
    return out;
  };

  // One batched CG call solves R^-1 * [F | y] together (F has p <= a few
  // columns): p+1 right-hand sides sharing the same matvec, each an
  // independent Krylov solve (no block-CG subspace sharing, but far cheaper
  // than p+1 separate O(n^3) factorizations either way).
  arma::mat FY = arma::join_rows(m_F, m_y);
  const arma::mat RinvFY = LinearAlgebra::conjugateGradient(Rmul, FY, max_iter, m_iterative_cg_tol);
  const arma::mat RinvF = RinvFY.head_cols(m_F.n_cols);
  const arma::vec Rinvy = RinvFY.col(RinvFY.n_cols - 1);

  const arma::mat A = m_F.t() * RinvF;
  const arma::vec b = m_F.t() * Rinvy;
  const arma::vec beta = arma::solve(A, b, arma::solve_opts::likely_sympd);

  const arma::vec e = m_y - m_F * beta;
  const arma::vec x = Rinvy - RinvF * beta;  // Rinv is linear: Rinv(y - F*beta) = Rinv*y - Rinv*F*beta
  const double SSE = arma::dot(e, x);
  const double sigma2 = SSE / n;

  const double logdetR
      = LinearAlgebra::stochasticLogDet(Rmul, n, nprobe, m_iterative_lanczos_steps, m_iterative_probes);

  if (beta_out != nullptr)
    *beta_out = beta;
  if (sigma2_out != nullptr)
    *sigma2_out = sigma2;

  if (grad_out != nullptr) {
    // Envelope theorem (beta_hat/sigma2_hat's own theta-dependence doesn't
    // contribute at their profiled values -- same principle as
    // _logLikelihoodVecchia/_logLikelihoodNystrom):
    //   d(ll)/d(theta_k) = 0.5*(x' dR/dtheta_k x / sigma2 - trace(Rinv dR/dtheta_k))
    // trace(Rinv dR/dtheta_k) is a Hutchinson estimate sharing the SAME
    // probes as the log-determinant: w_p = Rinv*z_p (one batched CG call
    // for all probes at once), then trace_k ~= mean_p( w_p . (dR/dtheta_k * z_p) ).
    const arma::uword d = _theta.n_elem;
    grad_out->set_size(d);

    const arma::mat W = LinearAlgebra::conjugateGradient(Rmul, m_iterative_probes, max_iter, m_iterative_cg_tol);

    for (arma::uword kk = 0; kk < d; ++kk) {
      auto dRmul_k = [this, &Xt, &theta, &cov, n, kk](const arma::vec& v) -> arma::vec {
        arma::vec out(n, arma::fill::zeros);
        for (arma::uword i = 0; i < n; ++i) {
          double acc = 0.0;  // diagonal of dR/dtheta_k is 0 (diag(R) = 1 is theta-independent)
          for (arma::uword j = 0; j < n; ++j) {
            if (j == i)
              continue;
            const arma::vec dx = Xt.col(i) - Xt.col(j);
            acc += cov(dx, theta) * _DlnCovDtheta(dx, theta)(kk) * v(j);
          }
          out(i) = acc;
        }
        return out;
      };

      const double term1 = arma::dot(x, dRmul_k(x));

      double trace_k = 0.0;
      for (arma::uword p = 0; p < nprobe; ++p)
        trace_k += arma::dot(W.col(p), dRmul_k(m_iterative_probes.col(p)));
      trace_k /= static_cast<double>(nprobe);

      (*grad_out)(kk) = 0.5 * (term1 / sigma2 - trace_k);
    }
  }

  return -0.5 * (n * std::log(2 * M_PI * sigma2) + logdetR + n);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logLikelihoodIterativeFun(const arma::vec& theta,
                                                                                   bool return_grad) {
  if (m_iterative_nprobe == 0)
    throw std::runtime_error("logLikelihoodIterativeFun: model was not fitted with objective=\"LLIterative(m)\"");
  arma::vec grad;
  if (return_grad) {
    grad.set_size(theta.n_elem);
    const double ll = _logLikelihoodIterative(theta, &grad);
    return {ll, grad};
  }
  return {_logLikelihoodIterative(theta), grad};
}

void Kriging::check_not_iterative_light(const char* what) const {
  if (m_iterative_light)
    throw std::runtime_error(std::string(what)
                             + ": not available on an Iterative fit "
                               "(refit with objective=\"LL\", or use predictCG)");
}

Kriging::FitOfn Kriging::make_fit_objective(const std::string& objective) const {
  if (objective == "LL") {
    if (m_noise_model == NoiseModel::Nugget) {
      // d+1 gamma: composite reparam (theta part + alpha part)
      if (Optim::reparametrize) {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          const arma::vec _theta_alpha = nugget_reparam_from(_gamma);
          double ll = this->_logLikelihood(_theta_alpha, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -nugget_reparam_from_deriv(_theta_alpha, *grad_out);
          return -ll;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double ll = this->_logLikelihood(_gamma, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -*grad_out;
          return -ll;
        };
      }
    } else if (m_noise_model == NoiseModel::Heterogeneous) {
      // d+1 gamma: Optim::reparam_from on all components (sigma2 treated same as theta)
      if (Optim::reparametrize) {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          const arma::vec _theta_sigma2 = Optim::reparam_from(_gamma);
          double ll = this->_logLikelihood(_theta_sigma2, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -Optim::reparam_from_deriv(_theta_sigma2, *grad_out);
          return -ll;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double ll = this->_logLikelihood(_gamma, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -*grad_out;
          return -ll;
        };
      }
    } else {
      // None: d-dim theta only
      if (Optim::reparametrize) {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          const arma::vec _theta = Optim::reparam_from(_gamma);
          double ll = this->_logLikelihood(_theta, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
          return -ll;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double ll = this->_logLikelihood(_gamma, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -*grad_out;
          return -ll;
        };
      }
    }
  } else if (objective == "LOO") {
    if (m_noise_model != NoiseModel::None)
      throw std::invalid_argument("LOO objective not supported for Nugget/Heterogeneous noise modes");
    if (Optim::reparametrize) {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
        const arma::vec _theta = Optim::reparam_from(_gamma);
        double loo = this->_leaveOneOut(_theta, grad_out, nullptr, km_data, nullptr);
        if (grad_out != nullptr)
          *grad_out = Optim::reparam_from_deriv(_theta, *grad_out);
        return loo;
      };
    } else {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
        return this->_leaveOneOut(_gamma, grad_out, nullptr, km_data, nullptr);
      };
    }
  } else if (objective == "LMP") {
    if (m_noise_model == NoiseModel::Heterogeneous)
      throw std::invalid_argument("LMP objective not supported for Heterogeneous noise mode");
    if (m_noise_model == NoiseModel::Nugget) {
      if (Optim::reparametrize) {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          const arma::vec _theta_alpha = nugget_reparam_from(_gamma);
          double lmp = this->_logMargPost(_theta_alpha, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -nugget_reparam_from_deriv(_theta_alpha, *grad_out);
          return -lmp;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double lmp = this->_logMargPost(_gamma, grad_out, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -*grad_out;
          return -lmp;
        };
      }
    } else if (Optim::reparametrize) {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
        const arma::vec _theta = Optim::reparam_from(_gamma);
        double lmp = this->_logMargPost(_theta, grad_out, km_data, nullptr);
        if (grad_out != nullptr)
          *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
        return -lmp;
      };
    } else {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
        double lmp = this->_logMargPost(_gamma, grad_out, km_data, nullptr);
        if (grad_out != nullptr)
          *grad_out = -*grad_out;
        return -lmp;
      };
    }
  } else if (objective.rfind("LLVecchia", 0) == 0) {
    parse_vll_m(objective);  // validate the spec early (throws on malformed)
    if (m_noise_model != NoiseModel::None)
      throw std::invalid_argument("LLVecchia objective not supported for Nugget/Heterogeneous noise modes");
    // Protocol: during optimization the caller passes grad_out != nullptr and
    // we evaluate the O(n m^3) Vecchia likelihood without touching km_data.
    // The single final call (grad_out == nullptr, km_data != nullptr) performs
    // ONE exact O(n^3) factorization at theta* so that the committed model
    // (and thus predict/simulate/update) behaves exactly as after an "LL" fit.
    if (Optim::reparametrize) {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
        const arma::vec _theta = Optim::reparam_from(_gamma);
        if (grad_out == nullptr && km_data != nullptr && m_vecchia_exact_commit)
          return -this->_logLikelihood(_theta, nullptr, km_data, nullptr);
        double vll = this->_logLikelihoodVecchia(_theta, grad_out);
        if (grad_out != nullptr)
          *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
        return -vll;
      };
    } else {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
        if (grad_out == nullptr && km_data != nullptr && m_vecchia_exact_commit)
          return -this->_logLikelihood(_gamma, nullptr, km_data, nullptr);
        double vll = this->_logLikelihoodVecchia(_gamma, grad_out);
        if (grad_out != nullptr)
          *grad_out = -*grad_out;
        return -vll;
      };
    }
  } else if (objective.rfind("LLNystrom", 0) == 0) {
    parse_nystrom_k(objective);  // validate the spec early (throws on malformed)
    if (m_noise_model != NoiseModel::None)
      throw std::invalid_argument("LLNystrom objective not supported for Nugget/Heterogeneous noise modes");
    // Unlike LLVecchia, there is no exact-commit branch here: km_data (the dense
    // O(n^3) KModel) is never populated for this objective, by design.
    if (Optim::reparametrize) {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const arma::vec _theta = Optim::reparam_from(_gamma);
        arma::vec grad;
        const double ll = this->_logLikelihoodNystrom(_theta, grad_out != nullptr ? &grad : nullptr);
        if (grad_out != nullptr)
          *grad_out = -Optim::reparam_from_deriv(_theta, grad);
        return -ll;
      };
    } else {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const double ll = this->_logLikelihoodNystrom(_gamma, grad_out);
        if (grad_out != nullptr)
          *grad_out = -*grad_out;
        return -ll;
      };
    }
  } else if (objective.rfind("LLIterative", 0) == 0) {
    parse_iterative_m(objective);  // validate the spec early (throws on malformed)
    if (m_noise_model != NoiseModel::None)
      throw std::invalid_argument("LLIterative objective not supported for Nugget/Heterogeneous noise modes");
    // Like LLNystrom, there is no exact-commit branch here: km_data (the dense
    // O(n^3) KModel) is never populated for this objective, by design.
    if (Optim::reparametrize) {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const arma::vec _theta = Optim::reparam_from(_gamma);
        arma::vec grad;
        const double ll = this->_logLikelihoodIterative(_theta, grad_out != nullptr ? &grad : nullptr);
        if (grad_out != nullptr)
          *grad_out = -Optim::reparam_from_deriv(_theta, grad);
        return -ll;
      };
    } else {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const double ll = this->_logLikelihoodIterative(_gamma, grad_out);
        if (grad_out != nullptr)
          *grad_out = -*grad_out;
        return -ll;
      };
    }
  } else
    throw std::invalid_argument(
        "Unsupported fit objective: " + objective
        + " (supported are: LL, LOO, LMP, LLVecchia, LLVecchia(m), LLNystrom, LLNystrom(k), LLIterative, "
          "LLIterative(m))");
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LL' (default), 'LOO', 'LMP', or 'LLVecchia'/'LLVecchia(m)' for the
 *        Vecchia approximated log-likelihood with m conditioning neighbors
 *        (default m=30). Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 */
LIBKRIGING_EXPORT void Kriging::fit(const arma::vec& y,
                                    const arma::mat& X,
                                    const Trend::RegressionModel& regmodel,
                                    bool normalize,
                                    const std::string& optim,
                                    const std::string& objective,
                                    const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  m_optim = optim;
  m_objective = objective;
  FitOfn fit_ofn = make_fit_objective(objective);

  // Nystrom/Iterative never touch m_dX (they work straight from m_X via
  // covMat_sym_X/covMat_rect or a matrix-free matvec); a light Vecchia fit
  // (exact_commit=false) skips the exact factorization that would otherwise
  // need it. All are only true when optim != "none": that path always does
  // one exact make_Model call regardless of objective (see below), so it
  // always needs m_dX.
  const bool build_dX = (optim == "none")
                        || !((objective.rfind("LLNystrom", 0) == 0) || (objective.rfind("LLIterative", 0) == 0)
                             || (objective.rfind("LLVecchia", 0) == 0 && !m_vecchia_exact_commit));
  arma::mat theta0 = fit_setup_impl(
      y, X, regmodel, normalize, parameters.is_beta_estim, parameters.beta, parameters.theta, build_dX);

  m_vecchia_light = false;
  m_nystrom_light = false;
  m_nystrom_k = 0;
  m_nystrom_U.reset();
  m_nystrom_D.reset();
  m_iterative_light = false;
  m_iterative_nprobe = 0;
  m_iterative_probes.reset();
  if (objective.rfind("LLVecchia", 0) == 0) {
    m_vecchia_m = parse_vll_m(objective);
    make_vecchia_sets();
  } else {
    m_vecchia_m = 0;
  }
  if (objective.rfind("LLNystrom", 0) == 0) {
    m_nystrom_k = parse_nystrom_k(objective);
    make_nystrom_landmarks();
  }
  if (objective.rfind("LLIterative", 0) == 0) {
    m_iterative_nprobe = parse_iterative_m(objective);
    make_iterative_probes();
  }

  const double scaleY = m_scaleY;
  const arma::rowvec& scaleX = m_scaleX;

  if (optim == "none") {  // just keep given theta, no optimisation of ll (but estim sigma2  &beta still possible)
    if (!parameters.theta.has_value())
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");

    m_theta = trans(theta0.row(0));
    m_est_theta = false;

    double sigma2 = -1;
    m_est_sigma2 = parameters.is_sigma2_estim;
    if (parameters.sigma2.has_value()) {
      sigma2 = parameters.sigma2.value();  // otherwise sigma2 will be re-calculated using given theta
      if (m_normalize)
        sigma2 /= (scaleY * scaleY);
    } else
      m_est_sigma2 = true;

    if (m_nystrom_k > 0) {
      // LLNystrom objective with optim="none": commit a genuine Nystrom-light
      // fit at the given (fixed) theta -- beta/sigma2/U/D via the same
      // Woodbury likelihood the free-optimization path uses (mirrors
      // update_nystrom's refit=false path) -- instead of silently falling
      // through to an exact dense fit below, which would ignore the
      // requested approximate objective entirely and leave m_nystrom_light
      // false (so predictNystrom would then throw "model was not fitted
      // with objective=\"LLNystrom(k)\""). This is what makes a fully
      // deterministic (no BFGS, no platform-dependent convergence) Nystrom
      // fit possible -- see docs/math/Nystrom.md and issue #351's follow-up.
      arma::vec beta_v;
      double sigma2_v = -1;
      arma::mat U_v;
      arma::vec D_v;
      _logLikelihoodNystrom(m_theta, nullptr, &beta_v, &sigma2_v, &U_v, &D_v);
      if (m_est_beta)
        m_beta = beta_v;
      if (m_est_sigma2)
        m_sigma2 = sigma2_v;
      else
        m_sigma2 = sigma2;
      m_nystrom_U = std::move(U_v);
      m_nystrom_D = std::move(D_v);
      m_nystrom_light = true;
      m_is_empty = true;  // no committed factorization: predict routes to predictNystrom
      return;
    }

    double extra_param;         // alpha for Nugget, sigma2 for Heterogeneous, unused for None
    double nugget_param = 0.0;  // only used for Nugget mode
    if (m_noise_model == NoiseModel::Nugget) {
      m_est_nugget = parameters.is_nugget_estim;
      if (parameters.nugget.has_value()) {
        nugget_param = parameters.nugget.value();
        if (m_normalize)
          nugget_param /= (scaleY * scaleY);
      }
      if (sigma2 > 0 && (sigma2 + nugget_param) > 0)
        m_alpha = sigma2 / (sigma2 + nugget_param);
      else
        m_alpha = 1.0 - nugget_alpha_lower;
      extra_param = m_alpha;
    } else if (m_noise_model == NoiseModel::Heterogeneous) {
      extra_param = (sigma2 > 0) ? sigma2 : m_sigma2;
    } else {
      extra_param = m_sigma2;  // ignored by 2-param populate_Model for None
    }

    Kriging::KModel m = (m_noise_model != NoiseModel::None) ? make_Model(m_theta, extra_param, nullptr)
                                                            : make_Model(m_theta, nullptr);
    m_is_empty = false;
    m_T = std::move(m.L);
    m_R = std::move(m.R);
    m_M = std::move(m.Fstar);
    m_circ = std::move(m.Rstar);
    m_star = std::move(m.Qstar);
    m_Rinv = std::move(m.Rinv);
    if (m_est_beta) {
      m_beta = std::move(m.betahat);
      m_z = std::move(m.Estar);
    } else {
      // m_beta = parameters.beta.value(); already done above
      m_z = std::move(m.ystar) - m_M * m_beta;
    }
    if (m_noise_model == NoiseModel::Nugget) {
      if (m_est_sigma2) {
        double total_var = m.SSEstar / n;
        m_sigma2 = m_alpha * total_var;
        m_nugget = m_est_nugget ? (1.0 - m_alpha) * total_var : nugget_param;
      } else {
        m_sigma2 = sigma2;
        m_nugget = m_est_nugget ? 0.0 : nugget_param;
      }
    } else if (m_est_sigma2) {
      m_sigma2 = m.SSEstar / n;
    } else {
      m_sigma2 = sigma2;
    }

  } else {
    auto theta_bounds_pair = Optim::theta_bounds(m_maxdX, m_dX, m_y, n);
    arma::vec theta_lower = theta_bounds_pair.first;
    arma::vec theta_upper = theta_bounds_pair.second;

    if (optim.rfind("BFGS", 0) == 0) {
      Random::init();

      auto parsed_bfgs = Optim::parse_method(optim, "BFGS");
      int multistart = parsed_bfgs.second;

      // Configure threads for Armadillo/BLAS to balance nested parallelism.
      // Each of the 'multistart' workers uses internal BLAS/OMP parallelism.
      // A RAII guard resets to 1 thread when the scope exits — this collapses
      // the idle thread pool so a subsequent fork() (e.g. R's mclapply) does
      // not inherit locked mutexes and deadlock.
      unsigned int n_cpu = std::thread::hardware_concurrency();
      struct ThreadCountGuard {
        bool active = false;
        ThreadCountGuard() = default;
        void set(unsigned int n) {
          active = true;
#if !defined(__APPLE__) || !defined(__arm64__)
          auto fn = get_openblas_set_num_threads();
          if (fn)
            fn(static_cast<int>(n));
#endif
#ifdef _OPENMP
          omp_set_num_threads(static_cast<int>(n));
#endif
        }
        ~ThreadCountGuard() {
          if (!active)
            return;
#if !defined(__APPLE__) || !defined(__arm64__)
          auto fn = get_openblas_set_num_threads();
          if (fn)
            fn(1);
#endif
#ifdef _OPENMP
          omp_set_num_threads(1);
#endif
        }
      } thread_guard;

      if (n_cpu > 0 && multistart > 1) {
        unsigned int threads_per_worker = std::max(1u, n_cpu / multistart);
        thread_guard.set(threads_per_worker);
        if (Optim::log_level > Optim::log_none) {
          arma::cout << "Threads per worker: " << threads_per_worker << " (total CPUs: " << n_cpu
                     << ", multistart: " << multistart << ")" << arma::endl;
        }
      }

      arma::mat theta0_rand
          = arma::repmat(trans(theta_lower), multistart, 1)
            + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
      // theta0 = arma::abs(0.5 + Random::randn_mat(multistart, d) / 6.0)
      //          % arma::repmat(max(m_X, 0) - min(m_X, 0), multistart, 1);

      // Nystrom warm start: with no given theta, every multistart worker
      // otherwise begins from a fully random point in theta_bounds, so how
      // well/consistently BFGS converges depends on luck and platform RNG.
      // The landmarks were already chosen (make_nystrom_landmarks, greedy
      // pivoted-Cholesky) as a representative subsample of the correlation
      // structure, so a cheap EXACT LL fit restricted to just those k points
      // -- O(k^3), negligible next to the O(n*k^2) Nystrom cost -- tends to
      // land close to the full-data optimum. Seed one multistart worker with
      // it; keep the rest random so a single unlucky/degenerate landmark
      // subsample can't strand every worker in the same bad basin.
      arma::mat theta0_seed;
      if (m_nystrom_k > 0 && !parameters.theta.has_value()) {
        try {
          Kriging land_fit(m_y.elem(m_nystrom_landmarks),
                           m_X.rows(m_nystrom_landmarks),
                           m_covType,
                           m_regmodel,
                           /*normalize=*/false,
                           "BFGS",
                           "LL");
          theta0_seed = land_fit.theta().t();  // 1 x d, already in m_X's (normalized) scale
        } catch (const std::exception&) {
          // degenerate landmark subsample (e.g. too few distinct points): fall back to plain random multistart
        }
      }

      if (parameters.theta.has_value()) {  // just use given theta(s) as starting values for multi-bfgs
        multistart = std::max(multistart, (int)theta0.n_rows);
        theta0 = arma::join_cols(theta0, theta0_rand);  // append random starting points to given ones
        theta0.resize(multistart, theta0.n_cols);       // keep only multistart first rows
      } else if (!theta0_seed.is_empty()) {
        theta0 = arma::join_cols(theta0_seed, theta0_rand);  // landmark-fit theta first, then random
        theta0.resize(multistart, theta0.n_cols);
      } else {
        theta0 = theta0_rand;
      }
      // arma::cout << "theta0:" << theta0 << arma::endl;

      // extra0: starting values for the extra optimization parameter (alpha or sigma2)
      // extra_lower/upper: bounds for the extra parameter
      arma::vec extra0;
      double extra_lower_val = 0.0, extra_upper_val = 1.0;
      if (m_noise_model == NoiseModel::Nugget) {
        extra_lower_val = nugget_alpha_lower;
        extra_upper_val = 1.0;
        if (parameters.sigma2.has_value() && parameters.nugget.has_value()) {
          double s = parameters.sigma2.value(), nu = parameters.nugget.value();
          extra0 = arma::vec(1);
          extra0.at(0)
              = (s > 0 && (s + nu) > 0) ? s / (s + nu) : extra_lower_val + (extra_upper_val - extra_lower_val) * 0.5;
        } else {
          extra0 = extra_lower_val
                   + (extra_upper_val - extra_lower_val) * (1.0 - arma::pow(Random::randu_vec(theta0.n_rows), 3.0));
        }
      } else if (m_noise_model == NoiseModel::Heterogeneous) {
        // sigma2 bounds from variogram
        arma::vec dy2(n * n, arma::fill::zeros);
        for (arma::uword ij = 0; ij < dy2.n_elem; ij++) {
          arma::uword i = ij % n, j = ij / n;
          if (i != j) {
            dy2[ij] = m_y[i] - m_y[j];
            dy2[ij] *= dy2[ij];
          }
        }
        arma::vec dX2 = arma::sum(m_dX % m_dX, 0).t();
        double sigma2_variogram = 0.5 * arma::mean(dy2.elem(arma::find(dX2 >= arma::median(dX2))));
        extra_lower_val = 0.1 * (sigma2_variogram - arma::max(m_noise));
        extra_upper_val = 10.0 * (sigma2_variogram - arma::min(m_noise));
        if (parameters.sigma2.has_value()) {
          extra0 = arma::vec{parameters.sigma2.value()};
          if (m_normalize)
            extra0 /= scaleY;
        } else {
          extra0 = extra_lower_val + (extra_upper_val - extra_lower_val) * Random::randu_vec(theta0.n_rows);
        }
      }

      arma::vec gamma_lower(gamma_dim()), gamma_upper(gamma_dim());
      gamma_lower.head(d) = theta_lower;
      gamma_upper.head(d) = theta_upper;
      if (m_noise_model != NoiseModel::None) {
        gamma_lower.at(d) = extra_lower_val;
        gamma_upper.at(d) = extra_upper_val;
        if (Optim::reparametrize) {
          if (m_noise_model == NoiseModel::Nugget) {
            gamma_lower = nugget_reparam_to(gamma_lower);
            gamma_upper = nugget_reparam_to(gamma_upper);
          } else {
            gamma_lower = Optim::reparam_to(gamma_lower);
            gamma_upper = Optim::reparam_to(gamma_upper);
          }
        }
      } else {
        if (Optim::reparametrize) {
          gamma_lower.head(d) = Optim::reparam_to(theta_lower);
          gamma_upper.head(d) = Optim::reparam_to(theta_upper);
        }
      }

      double min_ofn = std::numeric_limits<double>::infinity();

      // Set estimation flags before threading
      m_est_sigma2 = parameters.is_sigma2_estim;
      if ((!m_est_sigma2) && (parameters.sigma2.has_value())) {
        m_sigma2 = parameters.sigma2.value();
        if (m_normalize)
          m_sigma2 /= (scaleY * scaleY);
      } else {
        m_est_sigma2 = true;  // force estim if no value given
      }
      if (m_noise_model == NoiseModel::Nugget) {
        m_est_nugget = parameters.is_nugget_estim;
        if ((!m_est_nugget) && parameters.nugget.has_value()) {
          m_nugget = parameters.nugget.value();
          if (m_normalize)
            m_nugget /= (scaleY * scaleY);
        } else {
          m_est_nugget = true;
        }
      }

      // Preallocate KModels for each thread to avoid race conditions
      arma::uword n_data = n;
      arma::uword p_data = m_F.n_cols;
      std::vector<Kriging::KModel> preallocated_models(multistart);

      if (Optim::log_level > Optim::log_none) {
        arma::cout << "Preallocating " << multistart << " KModel structures (n=" << n_data << ", p=" << p_data << ")..."
                   << arma::endl;
      }

      for (int i = 0; i < multistart; i++) {
        auto& m = preallocated_models[i];
        m.R = arma::mat(n_data, n_data, arma::fill::none);
        m.L = arma::mat(n_data, n_data, arma::fill::none);
        m.Linv = arma::mat();  // Empty matrix
        m.Rinv = arma::mat(n_data, n_data, arma::fill::none);
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

      // Structure to hold optimization results from each thread
      struct OptimizationResult {
        arma::uword start_index;
        double objective_value;
        arma::vec gamma;
        arma::vec theta;
        double extra_param = 0.0;  // alpha (Nugget) or sigma2 (Heterogeneous)
        arma::mat L;
        arma::mat R;
        arma::mat Fstar;
        arma::mat Rstar;
        arma::mat Qstar;
        arma::mat Rinv;
        arma::vec Estar;
        arma::vec ystar;
        double SSEstar;
        arma::vec betahat;
        bool success;
        std::string error_message;

        OptimizationResult()
            : start_index(0), objective_value(std::numeric_limits<double>::infinity()), success(false) {}
      };

      // Worker function for each thread
      auto optimize_worker = [&](arma::uword start_idx) -> OptimizationResult {
        OptimizationResult result;
        result.start_index = start_idx;

        try {
          const arma::uword gd = gamma_dim();
          arma::vec theta_start = theta0.row(start_idx % multistart).t();
          arma::vec gamma_tmp(gd);
          gamma_tmp.head(d) = theta_start;
          if (m_noise_model != NoiseModel::None)
            gamma_tmp.at(d) = extra0[start_idx % extra0.n_elem];
          if (Optim::reparametrize) {
            if (m_noise_model == NoiseModel::Nugget)
              gamma_tmp = nugget_reparam_to(gamma_tmp);
            else if (m_noise_model == NoiseModel::Heterogeneous)
              gamma_tmp = Optim::reparam_to(gamma_tmp);
            else
              gamma_tmp.head(d) = Optim::reparam_to(theta_start);
          }

          arma::vec gamma_lower_local = gamma_lower;
          arma::vec gamma_upper_local = gamma_upper;
          gamma_lower_local = arma::min(gamma_tmp, gamma_lower_local);
          gamma_upper_local = arma::max(gamma_tmp, gamma_upper_local);

          // Use pre-allocated KModel for this thread (thread-safe)
          if (start_idx >= preallocated_models.size()) {
            throw std::runtime_error("Preallocated model index out of bounds");
          }

          Kriging::KModel& m = preallocated_models[start_idx];
          // This "warm-up" populate_Model call primes `m` before the BFGS
          // loop; its result is unconditionally overwritten by the first
          // fit_ofn evaluation for objectives that touch km_data (LL/LOO/LMP,
          // and LLVecchia at its final exact-commit call). It always builds a dense
          // R via m_dX, so it must be skipped whenever m_dX was never built
          // (LLNystrom, and a light -- exact_commit=false -- LLVecchia fit): both
          // ignore km_data entirely, so skipping this call changes nothing
          // for them beyond avoiding the now-empty m_dX.
          if (build_dX) {
            if (m_noise_model != NoiseModel::None)
              populate_Model(m, theta_start, extra0[start_idx % extra0.n_elem], nullptr);
            else
              populate_Model(m, theta_start, nullptr);
          }

          lbfgsb::Optimizer optimizer{gd};
          optimizer.iprint = -1;  // Disable output in parallel mode. was Optim::log_level - 2;
          optimizer.max_iter = Optim::max_iteration;
          optimizer.pgtol = objective.compare("LOO") == 0
                                ? Optim::gradient_tolerance / (n * n)
                                : Optim::gradient_tolerance;  // scale by: n^2 for LOO vs. LL, and /10 because LOO is
                                                              // usually more smooth
          optimizer.factr = objective.compare("LOO") == 0 ? Optim::objective_rel_tolerance / 1E-13 / (n * n)
                                                          : Optim::objective_rel_tolerance / 1E-13;
          arma::ivec bounds_type{gd, arma::fill::value(2)};

          if (Optim::log_level > Optim::log_none) {
            arma::cout << "BFGS (start " << (start_idx + 1) << "/" << multistart << "):" << arma::endl;
            arma::cout << "  objective: " << m_objective << arma::endl;
            arma::cout << "  max iterations: " << optimizer.max_iter << arma::endl;
            arma::cout << "  null gradient tolerance: " << optimizer.pgtol << arma::endl;
            arma::cout << "  constant objective tolerance: " << optimizer.factr * 1E-13 << arma::endl;
            arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
            arma::cout << "  normalize: " << m_normalize << arma::endl;
            arma::cout << "  lower_bounds: " << theta_lower.t() << arma::endl;
            arma::cout << "  upper_bounds: " << theta_upper.t() << arma::endl;
            arma::cout << "  start_point: " << theta_start.t() << arma::endl;
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
              arma::cout << "  Start " << (start_idx + 1) << ", Retry " << (retry) << ": f_opt=" << opt_result.f_opt
                         << ", num_iters=" << opt_result.num_iters << ", task=" << opt_result.task << arma::endl;
            }

            if (opt_result.f_opt < best_f_opt) {
              best_f_opt = opt_result.f_opt;
              best_gamma = gamma_tmp;
            }

            // check theta part for distance to bounds
            arma::vec theta_part = (m_noise_model != NoiseModel::None && Optim::reparametrize)
                                       ? (m_noise_model == NoiseModel::Nugget ? nugget_reparam_from(gamma_tmp).head(d)
                                                                              : Optim::reparam_from(gamma_tmp).head(d))
                                       : (Optim::reparametrize ? Optim::reparam_from(gamma_tmp) : gamma_tmp.head(d));
            double sol_to_lb = arma::min(arma::abs(theta_part - theta_lower));
            double sol_to_ub = arma::min(arma::abs(theta_part - theta_upper));
            double sol_to_b = std::min(sol_to_ub, sol_to_lb);

            // Check abnormal termination or convergence at bounds to decide on restart
            if ((retry < Optim::max_restart)
                && ((opt_result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0)  // Check for abnormal termination
                    || (opt_result.num_iters <= 2)          // Start point is strangely quite optimal...
                    || (sol_to_lb < arma::datum::eps)       // Stuck at lower bound
                    || (opt_result.f_opt > best_f_opt))) {  // No improvement

              if (Optim::log_level > Optim::log_none) {
                arma::cout << "  Restarting BFGS (start " << (start_idx + 1) << ", retry " << (retry + 1)
                           << "): f_opt=" << opt_result.f_opt << ", sol_to_lb=" << sol_to_lb
                           << ", sol_to_ub=" << sol_to_ub << arma::endl;
              }

              // Restart with contracted bounds around initial point (theta part only)
              arma::vec restart_theta = (theta_start + theta_lower) / pow(2.0, retry + 1);
              gamma_tmp.head(d) = restart_theta;
              if (m_noise_model != NoiseModel::None)
                gamma_tmp.at(d) = extra0[start_idx % extra0.n_elem];
              if (Optim::reparametrize) {
                if (m_noise_model == NoiseModel::Nugget)
                  gamma_tmp = nugget_reparam_to(gamma_tmp);
                else if (m_noise_model == NoiseModel::Heterogeneous)
                  gamma_tmp = Optim::reparam_to(gamma_tmp);
                else
                  gamma_tmp.head(d) = Optim::reparam_to(restart_theta);
              }

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
          // Extract theta and extra_param from best_gamma
          if (m_noise_model == NoiseModel::Nugget) {
            arma::vec theta_alpha = Optim::reparametrize ? nugget_reparam_from(best_gamma) : best_gamma;
            result.theta = theta_alpha.head(d);
            result.extra_param = theta_alpha.at(d);
          } else if (m_noise_model == NoiseModel::Heterogeneous) {
            arma::vec theta_sigma2 = Optim::reparametrize ? Optim::reparam_from(best_gamma) : best_gamma;
            result.theta = theta_sigma2.head(d);
            result.extra_param = theta_sigma2.at(d);
          } else {
            result.theta = Optim::reparametrize ? Optim::reparam_from(best_gamma) : best_gamma;
          }

          // Copy (not move) since m is a reference to preallocated memory
          // Force DEEP copy to avoid any shared memory issues
          result.L = arma::mat(m.L);          // Force copy constructor
          result.R = arma::mat(m.R);          // Force copy constructor
          result.Fstar = arma::mat(m.Fstar);  // Force copy constructor
          result.Rstar = arma::mat(m.Rstar);  // Force copy constructor
          result.Qstar = arma::mat(m.Qstar);  // Force copy constructor
          result.Rinv = arma::mat(m.Rinv);    // Force copy constructor
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

      // Execute optimizations (sequential multistart in the calling thread)
      // Running multistart workers in std::threads even when serialized by a mutex causes MKL to
      // produce slightly different floating-point results compared to running on the main thread,
      // because MKL uses different code paths for secondary vs. primary threads. This breaks the
      // BFGS20 == best-of-20×BFGS1 invariant. Running sequentially in the calling thread ensures
      // every start uses the same BLAS/LAPACK context as a single-start BFGS1 run.
      std::vector<OptimizationResult> results(multistart);
      for (int task_id = 0; task_id < multistart; task_id++) {
        results[task_id] = optimize_worker(task_id);
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
        throw std::runtime_error("All " + std::to_string(multistart) + " optimization attempts failed");
      }

      if (Optim::log_level > Optim::log_none && successful_optimizations < multistart) {
        arma::cout << "\nOptimization summary: " << successful_optimizations << "/" << multistart << " succeeded"
                   << arma::endl;
      }

      // Update member variables with best result
      if (best_idx >= 0 && m_vecchia_m > 0 && !m_vecchia_exact_commit) {
        // light Vecchia commit: no exact factorization; theta from the
        // optimizer, beta/sigma2 profiled by the Vecchia likelihood
        const auto& best = results[best_idx];
        m_theta = best.theta;
        m_est_theta = true;
        arma::vec beta_v;
        double sigma2_v = -1;
        _logLikelihoodVecchia(m_theta, nullptr, &beta_v, &sigma2_v);
        if (m_est_beta)
          m_beta = beta_v;
        if (m_est_sigma2)
          m_sigma2 = sigma2_v;
        m_vecchia_light = true;
        m_is_empty = true;  // no committed factorization: predict routes to predictVecchia
      } else if (best_idx >= 0 && m_nystrom_k > 0) {
        // Nystrom commit: never an exact factorization; theta from the
        // optimizer, beta/sigma2/U/D profiled by the Nystrom likelihood at theta*
        const auto& best = results[best_idx];
        m_theta = best.theta;
        m_est_theta = true;
        arma::vec beta_v;
        double sigma2_v = -1;
        arma::mat U_v;
        arma::vec D_v;
        _logLikelihoodNystrom(m_theta, nullptr, &beta_v, &sigma2_v, &U_v, &D_v);
        if (m_est_beta)
          m_beta = beta_v;
        if (m_est_sigma2)
          m_sigma2 = sigma2_v;
        m_nystrom_U = U_v;
        m_nystrom_D = D_v;
        m_nystrom_light = true;
        m_is_empty = true;  // no committed factorization: predict routes to predictNystrom
      } else if (best_idx >= 0 && m_iterative_nprobe > 0) {
        // Iterative commit: never an exact factorization; theta from the
        // optimizer, beta/sigma2 profiled by the Iterative likelihood at theta*
        const auto& best = results[best_idx];
        m_theta = best.theta;
        m_est_theta = true;
        arma::vec beta_v;
        double sigma2_v = -1;
        _logLikelihoodIterative(m_theta, nullptr, &beta_v, &sigma2_v);
        if (m_est_beta)
          m_beta = beta_v;
        if (m_est_sigma2)
          m_sigma2 = sigma2_v;
        m_iterative_light = true;
        m_is_empty = true;  // no committed factorization: predict routes to predictCG
      } else if (best_idx >= 0) {
        const auto& best = results[best_idx];
        m_theta = best.theta;  // copy
        m_est_theta = true;
        m_is_empty = false;
        m_T = best.L;  // copy instead of move to avoid issues
        m_R = best.R;
        m_M = best.Fstar;
        m_circ = best.Rstar;
        m_star = best.Qstar;
        m_Rinv = best.Rinv;

        if (m_est_beta) {
          m_beta = best.betahat;
          m_z = best.Estar;
        } else {
          m_z = best.ystar - m_M * m_beta;
        }

        if (m_noise_model == NoiseModel::Nugget) {
          m_alpha = best.extra_param;
          if (m_est_sigma2) {
            if (m_est_nugget) {
              double total_var = best.SSEstar / n;
              m_sigma2 = m_alpha * total_var;
              if (m_objective.compare("LMP") == 0)
                m_sigma2 = m_sigma2 * n / (n - m_F.n_cols - 2);
              m_nugget = m_sigma2 / m_alpha - m_sigma2;
            } else {
              m_sigma2 = m_nugget * m_alpha / (1.0 - m_alpha);
            }
          } else {
            if (m_est_nugget)
              m_nugget = m_sigma2 * (1.0 - m_alpha) / m_alpha;
            // else: both fixed, keep existing m_sigma2 and m_nugget
          }
        } else if (m_noise_model == NoiseModel::Heterogeneous) {
          // Unconcentrated form: sigma2 is directly optimized
          if (m_est_sigma2)
            m_sigma2 = best.extra_param;
        } else {
          if (m_est_sigma2) {
            m_sigma2 = best.SSEstar / n;
            if (m_objective.compare("LMP") == 0)
              m_sigma2 = best.SSEstar / (n - m_F.n_cols);
          }
        }

        if (Optim::log_level > Optim::log_none) {
          arma::cout << "\nBest solution from start point " << (best_idx + 1) << " with objective: " << min_ofn
                     << arma::endl;
        }
      }

    } else
      throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS[#])");
  }

  // arma::cout << "theta:" << m_theta << arma::endl;
}

LIBKRIGING_EXPORT void Kriging::fit(const arma::vec& y,
                                    const arma::vec& noise,
                                    const arma::mat& X,
                                    const Trend::RegressionModel& regmodel,
                                    bool normalize,
                                    const std::string& optim,
                                    const std::string& objective,
                                    const Parameters& parameters) {
  if (m_noise_model != NoiseModel::Heterogeneous)
    throw std::runtime_error("fit(y, noise, X, ...) requires NoiseModel::Heterogeneous");
  if (noise.n_elem != y.n_elem)
    throw std::runtime_error("noise vector must have the same length as y");
  m_noise = noise;
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

/** Compute the prediction for given points X'
 * @param X_n is n_n*d matrix of points where to predict output
 * @param return_stdev is true if return also stdev column vector
 * @param return_cov is true if return also cov matrix between X_n
 * @param return_deriv is true if return also derivative of prediction wrt x
 * @return output prediction: n_n means, [n_n standard deviations], [n_n*n_n full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat>
Kriging::predict(const arma::mat& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  if (m_vecchia_light) {
    // light Vecchia fit: no exact factorization available; route mean/stdev
    // to the local Vecchia predictor
    if (return_cov || return_deriv)
      throw std::runtime_error(
          "predict: return_cov/return_deriv are not available on a light Vecchia fit "
          "(refit with set_vecchia_exact_commit(true), or use predictVecchia)");
    auto [mean, stdev] = predictVecchia(X_n, return_stdev);
    return {mean, stdev, arma::mat(), arma::mat(), arma::mat()};
  }
  if (m_nystrom_light) {
    // Nystrom fit: no exact factorization available; route mean/stdev to the
    // Woodbury-based Nystrom predictor
    if (return_cov || return_deriv)
      throw std::runtime_error(
          "predict: return_cov/return_deriv are not available on a Nystrom fit "
          "(refit with objective=\"LL\", or use predictNystrom)");
    auto [mean, stdev] = predictNystrom(X_n, return_stdev);
    return {mean, stdev, arma::mat(), arma::mat(), arma::mat()};
  }
  if (m_iterative_light) {
    // Iterative fit: no exact factorization available; route mean/stdev to
    // the matrix-free CG predictor
    if (return_cov || return_deriv)
      throw std::runtime_error(
          "predict: return_cov/return_deriv are not available on an Iterative fit "
          "(refit with objective=\"LL\", or use predictCG)");
    auto [mean, stdev] = predictCG(X_n, return_stdev);
    return {mean, stdev, arma::mat(), arma::mat(), arma::mat()};
  }
  const arma::uword n_o = m_X.n_rows;
  const double lmp_scale = (m_objective.compare("LMP") == 0) ? (n_o - m_F.n_cols) / (n_o - m_F.n_cols - 2.0) : 1.0;
  if (m_noise_model == NoiseModel::Nugget) {
    const double sigma2 = m_sigma2 * lmp_scale;
    const double alpha = m_alpha;
    return predict_impl(X_n,
                        return_stdev,
                        return_cov,
                        return_deriv,
                        /*R_on_factor=*/alpha,
                        /*R_nn_factor=*/alpha,
                        /*R_nn_diag=*/arma::vec(X_n.n_rows, arma::fill::ones),
                        /*var_scale=*/sigma2 / alpha);
  }
  const double sigma2 = m_sigma2 * lmp_scale;
  return predict_impl(X_n,
                      return_stdev,
                      return_cov,
                      return_deriv,
                      /*R_on_factor=*/1.0,
                      /*R_nn_factor=*/1.0,
                      /*R_nn_diag=*/arma::vec(),
                      /*var_scale=*/sigma2);
}

LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::predictCG(const arma::mat& X_n,
                                                                      bool return_stdev,
                                                                      arma::uword max_iter,
                                                                      double tol,
                                                                      bool use_nystrom_precond,
                                                                      arma::uword precond_rank) const {
  if (m_noise_model != NoiseModel::None)
    throw std::runtime_error("predictCG: only available for NoiseModel::None");
  if (m_X.n_rows == 0)
    throw std::runtime_error("predictCG: model was not fitted");
  const arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::invalid_argument("predictCG: X_n has wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                                + std::to_string(d));

  const arma::uword n = m_X.n_rows;
  if (max_iter == 0)
    // n is CG's exact-arithmetic convergence bound, but GP covariance
    // matrices are commonly ill-conditioned enough (smooth kernels, many
    // points) that round-off keeps the true error shrinking well past that
    // point in practice (see LinearAlgebra::conjugateGradient's periodic
    // residual-recompute comment) -- 2n is a more realistic default budget.
    max_iter = 2 * n;

  // Matrix-free matvec R*v: R(i,j) = _Cov(X_i - X_j, theta) for i != j, 1 on
  // the diagonal (correlation matrix, NoiseModel::None). O(n) memory (no R
  // ever materialized), O(n^2) time per call.
  const arma::mat Xt = m_X.t();  // d x n, contiguous columns for cache-friendly access
  const arma::vec& theta = m_theta;
  const auto& cov = _Cov;
  auto Rmul = [&Xt, &theta, &cov, n](const arma::vec& v) -> arma::vec {
    arma::vec out(n, arma::fill::none);
#ifdef _OPENMP
    if (n >= 200) {
      int optimal_threads = get_optimal_threads(2);
#pragma omp parallel for schedule(static) num_threads(optimal_threads) if (n >= 200)
      for (arma::sword i = 0; i < static_cast<arma::sword>(n); ++i) {
        double acc = v(static_cast<arma::uword>(i));  // diag = 1
        for (arma::uword j = 0; j < n; ++j) {
          if (j == static_cast<arma::uword>(i))
            continue;
          acc += cov(Xt.col(i) - Xt.col(j), theta) * v(j);
        }
        out(static_cast<arma::uword>(i)) = acc;
      }
    } else {
#endif
      for (arma::uword i = 0; i < n; ++i) {
        double acc = v(i);  // diag = 1
        for (arma::uword j = 0; j < n; ++j) {
          if (j == i)
            continue;
          acc += cov(Xt.col(i) - Xt.col(j), theta) * v(j);
        }
        out(i) = acc;
      }
#ifdef _OPENMP
    }
#endif
    return out;
  };

  // Optional Nystrom preconditioner: a rank-precond_rank factor of R at the
  // model's own (fixed, already-fitted) theta -- unlike the LLNystrom
  // objective's landmarks, no cross-theta smoothness constraint applies
  // here, so this can legitimately be built exactly at the theta being used
  // for prediction rather than at a generic reference. Pinv is left empty
  // (== plain CG) when disabled, matching the prior behavior exactly.
  arma::mat U_pc;
  arma::vec D_pc;
  std::function<arma::vec(const arma::vec&)> Pinv;
  if (use_nystrom_precond) {
    arma::vec diag_resid;
    U_pc = LinearAlgebra::nystromFactor(
        &diag_resid, m_X, m_theta, _Cov, /*factor=*/1.0, KrigingImpl::ones, std::min(precond_rank, n), 1e-12);
    D_pc = arma::clamp(diag_resid, LinearAlgebra::num_nugget, arma::datum::inf);
    Pinv = [&U_pc, &D_pc](const arma::vec& v) -> arma::vec {
      return LinearAlgebra::woodbury_solve(U_pc, D_pc, v).col(0);
    };
  }

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;
  const arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  const arma::uword n_n = X_n.n_rows;

  // One CG solve, reused for every prediction point's mean.
  const arma::vec resid = m_y - m_F * m_beta;
  const arma::mat w = LinearAlgebra::conjugateGradient(Rmul, resid, max_iter, tol, Pinv);

  arma::mat R_on = arma::mat(n, n_n, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, Xt, Xn_n.t(), m_theta, _Cov, 1.0);

  arma::vec mean = F_n * m_beta + R_on.t() * w.col(0);
  mean = mean * m_scaleY + m_centerY;

  arma::vec stdev;
  if (return_stdev) {
    // One CG solve PER prediction point (R_on's columns don't share a
    // Krylov subspace): O(n^2 * iters * n_n) total, hence opt-in.
    const arma::mat V = LinearAlgebra::conjugateGradient(Rmul, R_on, max_iter, tol, Pinv);
    const arma::vec quad = arma::sum(R_on % V, 0).t();
    stdev = arma::sqrt(arma::clamp(m_sigma2 * (1.0 - quad), 0.0, arma::datum::inf));
    stdev *= m_scaleY;
  }
  return {mean, stdev};
}

/** Draw sample trajectories of kriging at given points X'
 * @param X_n is n_n*d matrix of points where to simulate output
 * @param seed is seed for random number generator
 * @param nsim is number of simulations to draw
 * @param will_update is true if we want to keep simulations data for future update
 * @return output is n_n*nsim matrix of simulations at X_n
 */
LIBKRIGING_EXPORT arma::mat Kriging::simulate(const int nsim,
                                              const int seed,
                                              const arma::mat& X_n,
                                              const bool will_update) {
  check_not_vecchia_light("simulate");
  check_not_iterative_light("simulate");
  if (m_nystrom_light) {
    if (will_update)
      throw std::runtime_error(
          "simulate: will_update=true is not available on a Nystrom fit "
          "(refit with objective=\"LL\", or call simulate(..., false))");
    return simulateNystrom(nsim, seed, X_n);
  }
  if (m_noise_model == NoiseModel::Nugget)
    return simulate(nsim, seed, X_n, /*with_nugget=*/false, will_update);
  return simulate_impl(nsim,
                       seed,
                       X_n,
                       will_update,
                       /*R_on_factor=*/1.0,
                       /*R_on_coincident_to_one=*/false,
                       /*R_nn_factor=*/1.0,
                       /*R_nn_diag=*/arma::vec(),
                       /*Sigma_divisor=*/1.0,
                       /*use_qr_for_circ=*/true);
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const bool with_nugget,
                                              const bool will_update) {
  const double alpha = m_alpha;
  const arma::vec diag_nn = with_nugget ? arma::vec(X_n.n_rows, arma::fill::ones) : arma::vec();
  arma::mat y_n = simulate_impl(nsim,
                                seed,
                                X_n,
                                will_update,
                                /*R_on_factor=*/alpha,
                                /*R_on_coincident_to_one=*/with_nugget,
                                /*R_nn_factor=*/alpha,
                                /*R_nn_diag=*/diag_nn,
                                /*Sigma_divisor=*/alpha,
                                /*use_qr_for_circ=*/true);
  if (will_update)
    m_lastsim_with_nugget = with_nugget;
  return y_n;
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const arma::vec& with_noise,
                                              const bool will_update) {
  const arma::uword n_n = X_n.n_rows;
  if (with_noise.n_elem > 1 && with_noise.n_elem != n_n)
    throw std::runtime_error("Noise vector should have same length as X_n");
  arma::mat y_n = simulate_impl(nsim,
                                seed,
                                X_n,
                                will_update,
                                /*R_on_factor=*/1.0,
                                /*R_on_coincident_to_one=*/false,
                                /*R_nn_factor=*/1.0,
                                /*R_nn_diag=*/arma::vec(),
                                /*Sigma_divisor=*/1.0,
                                /*use_qr_for_circ=*/false);
  if (will_update)
    m_lastsim_with_noise = with_noise;
  arma::mat eps(n_n, nsim, arma::fill::none);
  if (with_noise.n_elem == 1)
    eps = with_noise.at(0) * Random::randn_mat(n_n, nsim);
  else if (with_noise.n_elem == n_n) {
    eps.each_col() = with_noise;
    eps = eps % Random::randn_mat(n_n, nsim);
  }
  return y_n + eps;
}

LIBKRIGING_EXPORT arma::mat Kriging::update_simulate(const arma::vec& y_u, const arma::mat& X_u) {
  check_not_vecchia_light("update_simulate");
  check_not_nystrom_light("update_simulate");
  check_not_iterative_light("update_simulate");
  if (m_noise_model == NoiseModel::Nugget) {
    const double alpha = m_alpha;
    return update_simulate_impl(y_u,
                                X_u,
                                /*allow_cache=*/true,
                                /*R_uu_factor=*/alpha,
                                /*R_uu_diag=*/arma::vec(y_u.n_elem, arma::fill::ones),
                                /*R_uo_factor=*/alpha,
                                /*R_un_factor=*/alpha,
                                /*R_un_coincident_to_one=*/false,
                                /*Sigma_divisor=*/alpha);
  }
  return update_simulate_impl(y_u,
                              X_u,
                              /*allow_cache=*/true,
                              /*R_uu_factor=*/1.0,
                              /*R_uu_diag=*/arma::vec(),
                              /*R_uo_factor=*/1.0,
                              /*R_un_factor=*/1.0,
                              /*R_un_coincident_to_one=*/false,
                              /*Sigma_divisor=*/1.0);
}

LIBKRIGING_EXPORT arma::mat Kriging::update_simulate(const arma::vec& y_u,
                                                     const arma::vec& noise_u,
                                                     const arma::mat& X_u) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension mismatch: y_u and X_u");
  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension mismatch: X_u cols vs X cols");
  if (noise_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Noise vector length must match X_u rows");
  const arma::vec diag_uu = 1.0 + noise_u / m_sigma2;
  const arma::uword n_n = lastsim_Xn_n.n_cols;
  arma::mat y_up = update_simulate_impl(y_u,
                                        X_u,
                                        /*allow_cache=*/false,
                                        /*R_uu_factor=*/1.0,
                                        /*R_uu_diag=*/diag_uu,
                                        /*R_uo_factor=*/1.0,
                                        /*R_un_factor=*/1.0,
                                        /*R_un_coincident_to_one=*/false,
                                        /*Sigma_divisor=*/1.0);
  arma::mat eps(n_n, lastsim_nsim, arma::fill::zeros);
  if (m_lastsim_with_noise.n_elem == 1)
    eps = m_lastsim_with_noise.at(0) * Random::randn_mat(n_n, lastsim_nsim);
  else if (m_lastsim_with_noise.n_elem == n_n) {
    eps.each_col() = m_lastsim_with_noise;
    eps = eps % Random::randn_mat(n_n, lastsim_nsim);
  }
  return y_up + eps;
}

/** Add new conditional data points to previous (X,y), then perform new fit.
 * @param y_u is n_u length column vector of new output
 * @param X_u is n_u*d matrix of new input
 * @param refit is true if we want to re-fit the model
 */
LIBKRIGING_EXPORT void Kriging::update(const arma::vec& y_u, const arma::mat& X_u, const bool refit) {
  check_not_vecchia_light("update");
  check_not_iterative_light("update");
  if (m_nystrom_light) {
    update_nystrom(y_u, X_u, refit);
    return;
  }
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  if (refit && m_optim != "none" && m_noise_model == NoiseModel::Nugget) {
    // For Nugget mode: full re-fit with de-normalized data (matches NuggetKriging::update behavior)
    const arma::vec y_all = arma::join_cols(m_y * m_scaleY + m_centerY, y_u);
    const arma::mat X_all = arma::join_cols((m_X.each_row() % m_scaleX).each_row() + m_centerX, X_u);
    Parameters params;
    if (m_est_beta && m_est_nugget && m_est_sigma2 && m_est_theta) {
      // All estimated: use default (null) starting points → full multistart BFGS
    } else {
      params.sigma2 = m_sigma2 * m_scaleY * m_scaleY;
      params.is_sigma2_estim = m_est_sigma2;
      params.theta = arma::trans(m_theta) % m_scaleX;
      params.is_theta_estim = m_est_theta;
      params.nugget = m_nugget * m_scaleY * m_scaleY;
      params.is_nugget_estim = m_est_nugget;
      if (!m_est_beta) {
        params.beta = m_beta * m_scaleY;
        params.is_beta_estim = false;
      }
    }
    fit(y_all, X_all, m_regmodel, m_normalize, m_optim, m_objective, params);
    return;
  }

  if (refit && m_optim != "none") {  // Warm restart: extend data and run single BFGS from current theta
    // Normalize new data using existing normalization
    arma::mat Xn_u = X_u;
    Xn_u.each_row() -= m_centerX;
    Xn_u.each_row() /= m_scaleX;
    arma::vec yn_u = (y_u - m_centerY) / m_scaleY;

    // Extend training data
    m_X = arma::join_cols(m_X, Xn_u);
    m_y = arma::join_cols(m_y, yn_u);

    const arma::uword n = m_X.n_rows;
    const arma::uword d = m_X.n_cols;

    // Update distance matrix
    m_dX = LinearAlgebra::compute_dX(m_X);
    m_maxdX = arma::max(arma::abs(m_dX), 1);

    // Update trend matrix
    m_F = Trend::regressionModelMatrix(m_regmodel, m_X);

    if (m_objective.rfind("LLVecchia", 0) == 0) {
      m_vecchia_m = parse_vll_m(m_objective);
      make_vecchia_sets();  // m_X was just extended
    }

    FitOfn fit_ofn = make_fit_objective(m_objective);

    // Compute theta bounds for the extended dataset
    auto theta_bounds_pair = Optim::theta_bounds(m_maxdX, m_dX, m_y, n);
    arma::vec theta_lower = theta_bounds_pair.first;
    arma::vec theta_upper = theta_bounds_pair.second;

    const arma::uword gd = gamma_dim();
    arma::vec gamma_start(gd), gamma_lower(gd), gamma_upper(gd);
    gamma_start.head(d) = m_theta;
    gamma_lower.head(d) = theta_lower;
    gamma_upper.head(d) = theta_upper;
    if (m_noise_model == NoiseModel::Nugget) {
      gamma_start.at(d) = m_alpha;
      gamma_lower.at(d) = nugget_alpha_lower;
      gamma_upper.at(d) = 1.0;
    } else if (m_noise_model == NoiseModel::Heterogeneous) {
      gamma_start.at(d) = m_sigma2;
      gamma_lower.at(d) = 0.0;
      gamma_upper.at(d) = 10.0 * m_sigma2;
    }
    if (Optim::reparametrize) {
      if (m_noise_model == NoiseModel::Nugget) {
        gamma_start = nugget_reparam_to(gamma_start);
        gamma_lower = nugget_reparam_to(gamma_lower);
        gamma_upper = nugget_reparam_to(gamma_upper);
      } else if (m_noise_model == NoiseModel::Heterogeneous) {
        gamma_start = Optim::reparam_to(gamma_start);
        gamma_lower = Optim::reparam_to(gamma_lower);
        gamma_upper = Optim::reparam_to(gamma_upper);
      } else {
        gamma_start.head(d) = Optim::reparam_to(m_theta);
        gamma_lower.head(d) = Optim::reparam_to(theta_lower);
        gamma_upper.head(d) = Optim::reparam_to(theta_upper);
      }
    }
    gamma_lower = arma::min(gamma_start, gamma_lower);
    gamma_upper = arma::max(gamma_start, gamma_upper);

    // Preallocate KModel
    arma::uword p = m_F.n_cols;
    Kriging::KModel km;
    km.R = arma::mat(n, n, arma::fill::none);
    km.L = arma::mat(n, n, arma::fill::none);
    km.Linv = arma::mat();
    km.Rinv = arma::mat(n, n, arma::fill::none);
    km.Fstar = arma::mat(n, p, arma::fill::none);
    km.ystar = arma::vec(n, arma::fill::none);
    km.Rstar = arma::mat(p, p, arma::fill::none);
    km.Qstar = arma::mat();
    km.Estar = arma::vec(n, arma::fill::none);
    km.betahat = arma::vec(p, arma::fill::none);
    km.SSEstar = 0.0;

    if (m_noise_model != NoiseModel::None)
      populate_Model(km, m_theta, (m_noise_model == NoiseModel::Nugget) ? m_alpha : m_sigma2, nullptr);
    else
      populate_Model(km, m_theta, nullptr);

    // Run single BFGS from current gamma (warm restart)
    lbfgsb::Optimizer optimizer{gd};
    optimizer.iprint = Optim::log_level - 2;
    optimizer.max_iter = Optim::max_iteration;
    optimizer.pgtol = m_objective == "LOO" ? Optim::gradient_tolerance / (n * n) : Optim::gradient_tolerance;
    optimizer.factr = m_objective == "LOO" ? Optim::objective_rel_tolerance / 1E-13 / (n * n)
                                           : Optim::objective_rel_tolerance / 1E-13;
    arma::ivec bounds_type{gd, arma::fill::value(2)};

    arma::vec gamma_tmp = gamma_start;
    optimizer.minimize([&km, &fit_ofn](const arma::vec& vals_inp,
                                       arma::vec& grad_out) -> double { return fit_ofn(vals_inp, &grad_out, &km); },
                       gamma_tmp,
                       gamma_lower.memptr(),
                       gamma_upper.memptr(),
                       bounds_type.memptr());

    // LLVecchia evaluations do not populate km during optimization; perform the
    // single exact factorization at the optimum before committing the model.
    if (m_objective.rfind("LLVecchia", 0) == 0)
      fit_ofn(gamma_tmp, nullptr, &km);

    // Extract theta and extra param from optimized gamma
    if (m_noise_model == NoiseModel::Nugget) {
      arma::vec theta_alpha = Optim::reparametrize ? nugget_reparam_from(gamma_tmp) : gamma_tmp;
      m_theta = theta_alpha.head(d);
      m_alpha = theta_alpha.at(d);
      if (m_est_sigma2) {
        if (m_est_nugget) {
          double total_var = km.SSEstar / n;
          m_sigma2 = m_alpha * total_var;
          if (m_objective.compare("LMP") == 0)
            m_sigma2 = m_sigma2 * n / (n - m_F.n_cols - 2);
          m_nugget = m_sigma2 / m_alpha - m_sigma2;
        } else {
          m_sigma2 = m_nugget * m_alpha / (1.0 - m_alpha);
        }
      } else {
        if (m_est_nugget)
          m_nugget = m_sigma2 * (1.0 - m_alpha) / m_alpha;
      }
    } else if (m_noise_model == NoiseModel::Heterogeneous) {
      arma::vec theta_sigma2 = Optim::reparametrize ? Optim::reparam_from(gamma_tmp) : gamma_tmp;
      m_theta = theta_sigma2.head(d);
      if (m_est_sigma2)
        m_sigma2 = theta_sigma2.at(d);
    } else {
      m_theta = Optim::reparametrize ? Optim::reparam_from(gamma_tmp) : gamma_tmp;
    }

    m_est_theta = true;
    m_is_empty = false;
    m_T = std::move(km.L);
    m_R = std::move(km.R);
    m_M = std::move(km.Fstar);
    m_circ = std::move(km.Rstar);
    m_star = std::move(km.Qstar);
    m_Rinv = std::move(km.Rinv);

    if (m_est_beta) {
      m_beta = std::move(km.betahat);
      m_z = std::move(km.Estar);
    } else {
      m_z = std::move(km.ystar) - m_M * m_beta;
    }

    if (m_noise_model == NoiseModel::None && m_est_sigma2) {
      m_sigma2 = km.SSEstar / n;
      if (m_objective == "LMP")
        m_sigma2 = km.SSEstar / (n - m_F.n_cols);
    }
  } else {  // incremental update without parameter re-optimization
    update_no_refit_impl(
        y_u,
        X_u,
        /*extend_class_data=*/[] {},
        /*build_model=*/[this] { return make_Model(m_theta, nullptr); });
  }
}

LIBKRIGING_EXPORT void Kriging::update(const arma::vec& y_u,
                                       const arma::vec& noise_u,
                                       const arma::mat& X_u,
                                       const bool refit) {
  if (m_noise_model != NoiseModel::Heterogeneous)
    throw std::runtime_error("update(y, noise, X) requires NoiseModel::Heterogeneous");
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");
  if (noise_u.n_elem != y_u.n_elem)
    throw std::runtime_error("noise_u must have the same length as y_u");
  // Rebuild de-normalized data and call fit() with the joined dataset
  const arma::vec y_all = arma::join_cols(m_y * m_scaleY + m_centerY, y_u);
  const arma::vec noise_all = arma::join_cols(m_noise * m_scaleY * m_scaleY, noise_u);
  const arma::mat X_all = arma::join_cols((m_X.each_row() % m_scaleX).each_row() + m_centerX, X_u);
  Kriging::Parameters params;
  params.sigma2 = m_sigma2 * m_scaleY * m_scaleY;
  params.is_sigma2_estim = m_est_sigma2;
  params.theta = trans(m_theta) % m_scaleX;
  params.is_theta_estim = m_est_theta;
  params.beta = m_est_beta ? std::optional<arma::vec>{} : std::optional<arma::vec>(trans(m_beta) * m_scaleY);
  params.is_beta_estim = m_est_beta;
  fit(y_all, noise_all, X_all, m_regmodel, m_normalize, refit ? m_optim : "none", m_objective, params);
}

LIBKRIGING_EXPORT std::string Kriging::summary() const {
  std::ostringstream oss;
  if (summary_top(oss)) {
    if (m_noise_model == NoiseModel::Nugget)
      oss << "  * nugget" << (m_est_nugget ? " (est.): " : ": ") << m_nugget << "\n";
    else if (m_noise_model == NoiseModel::Heterogeneous)
      oss << "  * noise (heterogeneous): " << m_noise.n_elem << " obs\n";
    summary_bottom(oss);
  }
  return oss.str();
}

static std::string noise_model_to_string(Kriging::NoiseModel nm) {
  switch (nm) {
    case Kriging::NoiseModel::Nugget:
      return "Nugget";
    case Kriging::NoiseModel::Heterogeneous:
      return "Heterogeneous";
    default:
      return "None";
  }
}

static Kriging::NoiseModel noise_model_from_string(const std::string& s) {
  if (s == "Nugget")
    return Kriging::NoiseModel::Nugget;
  if (s == "Heterogeneous")
    return Kriging::NoiseModel::Heterogeneous;
  return Kriging::NoiseModel::None;
}

void Kriging::save(const std::string filename) const {
  check_not_vecchia_light("save");
  check_not_iterative_light("save");
  nlohmann::json j;
  j["version"] = 2;
  j["content"] = "Kriging";
  dump_common_to_json(j);
  j["noise_model"] = noise_model_to_string(m_noise_model);
  if (m_noise_model == NoiseModel::Nugget) {
    j["nugget"] = m_nugget;
    j["est_nugget"] = m_est_nugget;
    j["alpha"] = m_alpha;
  }
  // Nystrom fits carry no m_T/m_R/... (dump_common_to_json serializes those
  // as empty, harmlessly): the committed low-rank factors/landmarks are the
  // extra state that reconstructs the model, stored only when present so
  // pre-Nystrom save files stay byte-identical.
  j["nystrom_light"] = m_nystrom_light;
  if (m_nystrom_light) {
    j["nystrom_k"] = m_nystrom_k;
    j["nystrom_landmarks"] = std::vector<arma::uword>(m_nystrom_landmarks.begin(), m_nystrom_landmarks.end());
    j["nystrom_U"] = to_json(m_nystrom_U);
    j["nystrom_D"] = to_json(m_nystrom_D);
  }

  std::ofstream f(filename);
  f << std::setw(4) << j;
}

Kriging Kriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version != 2)
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
  std::string content = j["content"].template get<std::string>();
  if (content != "Kriging")
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'Kriging'"));

  NoiseModel nm = j.contains("noise_model") ? noise_model_from_string(j["noise_model"].template get<std::string>())
                                            : NoiseModel::None;
  Kriging kr(j["covType"].template get<std::string>(), nm);  // _Cov_pow & std::function embedded by make_Cov
  kr.load_common_from_json(j);
  if (nm == NoiseModel::Nugget) {
    kr.m_nugget = j["nugget"].template get<double>();
    kr.m_est_nugget = j["est_nugget"].template get<bool>();
    kr.m_alpha = j["alpha"].template get<double>();
  }
  // Absent on pre-Nystrom save files: defaults to a normal (non-Nystrom) load.
  if (j.contains("nystrom_light") && j["nystrom_light"].template get<bool>()) {
    kr.m_nystrom_light = true;
    kr.m_nystrom_k = j["nystrom_k"].template get<arma::uword>();
    kr.m_nystrom_landmarks = arma::uvec(j["nystrom_landmarks"].template get<std::vector<arma::uword>>());
    kr.m_nystrom_U = mat_from_json(j["nystrom_U"]);
    kr.m_nystrom_D = colvec_from_json(j["nystrom_D"]);
  }
  return kr;
}
