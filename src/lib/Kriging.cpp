// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio

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

#include <atomic>
#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <map>
#include <mutex>
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
                               arma::mat* hess_out,
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

    arma::vec terme1(d);  // needed for Hessian: terme1[k] = term1_vec[k] / sigma2
    for (arma::uword k = 0; k < d; k++) {
      terme1.at(k) = term1_vec.at(k) / sigma2_grad;
      (*grad_out).at(k) = (terme1.at(k) + term2_vec.at(k)) / 2.0;
    }

    // Hessian computation (None noise model only, following O. Roustant's formula)
    if (hess_out != nullptr && m_noise_model == NoiseModel::None) {
      t0 = Bench::tic();
      arma::cube gradR(n, n, d, arma::fill::none);
      const arma::vec zeros_d = arma::vec(d, arma::fill::zeros);
      for (arma::uword i = 0; i < n; i++) {
        gradR.tube(i, i) = zeros_d;
        for (arma::uword j = 0; j < i; j++) {
          gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
          gradR.tube(j, i) = gradR.tube(i, j);
        }
      }
      t0 = Bench::toc(bench, "gradR cube for Hessian", t0);

      if ((m.Linv.memptr() == nullptr) || (arma::size(m.Linv) != arma::size(m.L)))
        m.Linv = LinearAlgebra::solve_lower(m.L, arma::mat(n, n, arma::fill::eye));

      // Compute Qstar for H = Qstar * Qstar^t
      arma::mat Qstar_h;
      arma::mat Rstar_h;
      LinearAlgebra::qr_econ(Qstar_h, Rstar_h, m.Fstar);

      hess_out->set_size(d, d);
      for (arma::uword k = 0; k < d; k++) {
        arma::mat gradR_k = gradR.slice(k);
        arma::mat H = LinearAlgebra::tcrossprod(Qstar_h);
        for (arma::uword l = 0; l <= k; l++) {
          arma::mat gradR_l = gradR.slice(l);
          arma::mat aux = gradR_k * Rinv * gradR_l;
          arma::mat hessR_k_l(n, n, arma::fill::none);
          if (k == l) {
            for (arma::uword i = 0; i < n; i++) {
              hessR_k_l.at(i, i) = 0;
              for (arma::uword j = 0; j < i; j++) {
                double dln_k = gradR_k.at(i, j);
                hessR_k_l.at(i, j) = hessR_k_l.at(j, i)
                    = dln_k * (dln_k / m.R.at(i, j) - (_Cov_pow + 1) / _theta.at(k));
              }
            }
          } else {
            for (arma::uword i = 0; i < n; i++) {
              hessR_k_l.at(i, i) = 0;
              for (arma::uword j = 0; j < i; j++)
                hessR_k_l.at(i, j) = hessR_k_l.at(j, i) = gradR_k.at(i, j) * gradR_l.at(i, j) / m.R.at(i, j);
            }
          }
          arma::mat xk = m.Linv * gradR_k * x;
          arma::mat xl = (k == l) ? xk : m.Linv * gradR_l * x;
          double h_lk = (2.0 * as_scalar(xk.t() * H * xl) / sigma2_grad
                         + as_scalar(x.t() * (hessR_k_l - 2.0 * aux) * x) / sigma2_grad + arma::trace(Rinv * aux)
                         - arma::trace(Rinv * hessR_k_l));
          if (m_est_sigma2)
            h_lk += terme1.at(k) * terme1.at(l) / n;
          (*hess_out).at(l, k) = (*hess_out).at(k, l) = h_lk / 2.0;
          t0 = Bench::toc(bench, "hess_ll[l,k] = ...", t0);
        }
      }
    }

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

LIBKRIGING_EXPORT std::tuple<double, arma::vec, arma::mat> Kriging::logLikelihoodFun(const arma::vec& _theta,
                                                                                     const bool _grad,
                                                                                     const bool _hess,
                                                                                     const bool _bench) {
  double val = -1;
  arma::vec grad;
  arma::mat hess;
  std::map<std::string, double> bench_map;
  std::map<std::string, double>* bench_ptr = _bench ? &bench_map : nullptr;
  if (_grad || _hess) {
    grad = arma::vec(_theta.n_elem);
    if (_hess) {
      hess = arma::mat(_theta.n_elem, _theta.n_elem, arma::fill::none);
      val = _logLikelihood(_theta, &grad, &hess, nullptr, bench_ptr);
    } else {
      val = _logLikelihood(_theta, &grad, nullptr, nullptr, bench_ptr);
    }
  } else {
    val = _logLikelihood(_theta, nullptr, nullptr, nullptr, bench_ptr);
  }
  if (_bench)
    print_bench(bench_map);
  return {val, std::move(grad), std::move(hess)};
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

Kriging::FitOfn Kriging::make_fit_objective(const std::string& objective) const {
  if (objective == "LL") {
    if (m_noise_model == NoiseModel::Nugget) {
      // d+1 gamma: composite reparam (theta part + alpha part)
      if (Optim::reparametrize) {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          const arma::vec _theta_alpha = nugget_reparam_from(_gamma);
          double ll = this->_logLikelihood(_theta_alpha, grad_out, nullptr, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -nugget_reparam_from_deriv(_theta_alpha, *grad_out);
          return -ll;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double ll = this->_logLikelihood(_gamma, grad_out, nullptr, km_data, nullptr);
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
          double ll = this->_logLikelihood(_theta_sigma2, grad_out, nullptr, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -Optim::reparam_from_deriv(_theta_sigma2, *grad_out);
          return -ll;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double ll = this->_logLikelihood(_gamma, grad_out, nullptr, km_data, nullptr);
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
          double ll = this->_logLikelihood(_theta, grad_out, nullptr, km_data, nullptr);
          if (grad_out != nullptr)
            *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
          return -ll;
        };
      } else {
        return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel* km_data) {
          double ll = this->_logLikelihood(_gamma, grad_out, nullptr, km_data, nullptr);
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
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LOO, LMP)");
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

  arma::mat theta0
      = fit_setup_impl(y, X, regmodel, normalize, parameters.is_beta_estim, parameters.beta, parameters.theta);
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
          arma::cout << "Threads per worker: " << threads_per_worker << " (total CPUs: " << n_cpu
                     << ", multistart: " << multistart << ")" << arma::endl;
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
          if (m_noise_model != NoiseModel::None)
            populate_Model(m, theta_start, extra0[start_idx % extra0.n_elem], nullptr);
          else
            populate_Model(m, theta_start, nullptr);

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
          arma::cout << "Thread pool: " << pool_size << " workers (ncpu=" << n_cpu << ", multistart=" << multistart
                     << ")" << arma::endl;
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
              if (task_id >= multistart)
                break;

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
        throw std::runtime_error("All " + std::to_string(multistart) + " optimization attempts failed");
      }

      if (Optim::log_level > Optim::log_none && successful_optimizations < multistart) {
        arma::cout << "\nOptimization summary: " << successful_optimizations << "/" << multistart << " succeeded"
                   << arma::endl;
      }

      // Update member variables with best result
      if (best_idx >= 0) {
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
  arma::mat eps(n_n, lastsim_nsim, arma::fill::none);
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
  return kr;
}
