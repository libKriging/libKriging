// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio

#include <algorithm>
#include <atomic>
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

#include "BlasThreads.hpp"
#include "TaskPool.hpp"
#include "ThreadBudget.hpp"

#include <cassert>
#include <cstdlib>
#include <string>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <map>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Helpers to get OpenBLAS thread control functions (if available)
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
typedef int (*openblas_get_num_threads_t)();
FARPROC get_openblas_proc(const char* name) {
  // Try to load from OpenBLAS DLL (used by numpy/scipy)
  HMODULE hModule = GetModuleHandleA("libopenblas.dll");
  if (!hModule)
    hModule = GetModuleHandleA("openblas.dll");
  return hModule ? GetProcAddress(hModule, name) : nullptr;
}
openblas_set_num_threads_t get_openblas_set_num_threads() {
  static openblas_set_num_threads_t func
      = reinterpret_cast<openblas_set_num_threads_t>(get_openblas_proc("openblas_set_num_threads"));
  return func;
}
openblas_get_num_threads_t get_openblas_get_num_threads() {
  static openblas_get_num_threads_t func
      = reinterpret_cast<openblas_get_num_threads_t>(get_openblas_proc("openblas_get_num_threads"));
  return func;
}
}  // namespace
#else
// GCC/Clang support weak symbols
extern "C" {
void openblas_set_num_threads(int num_threads) __attribute__((weak));
int openblas_get_num_threads() __attribute__((weak));
}
namespace {
typedef void (*openblas_set_num_threads_t)(int);
typedef int (*openblas_get_num_threads_t)();
openblas_set_num_threads_t get_openblas_set_num_threads() {
  return openblas_set_num_threads;
}
openblas_get_num_threads_t get_openblas_get_num_threads() {
  return openblas_get_num_threads;
}
}  // namespace
#endif
#endif

namespace libKriging {
namespace detail {
BlasThreadState get_blas_threads() {
  BlasThreadState s;
#if !defined(__APPLE__) || !defined(__arm64__)
  auto fn = get_openblas_get_num_threads();
  if (fn)
    s.blas = fn();
#endif
#ifdef _OPENMP
  s.omp = omp_get_max_threads();
#endif
  return s;
}

void set_blas_threads(unsigned int n) {
#if !defined(__APPLE__) || !defined(__arm64__)
  auto fn = get_openblas_set_num_threads();
  if (fn)
    fn(static_cast<int>(n));
#endif
#ifdef _OPENMP
  omp_set_num_threads(static_cast<int>(n));
#endif
}

int get_openblas_threads() {
#if !defined(__APPLE__) || !defined(__arm64__)
  auto fn = get_openblas_get_num_threads();
  if (fn)
    return fn();
#endif
  return 0;
}

void set_openblas_threads(int n) {
#if !defined(__APPLE__) || !defined(__arm64__)
  auto fn = get_openblas_set_num_threads();
  if (fn && n > 0)
    fn(n);
#endif
}

void restore_blas_threads(const BlasThreadState& s) {
#if !defined(__APPLE__) || !defined(__arm64__)
  auto fn = get_openblas_set_num_threads();
  if (fn && s.blas > 0)
    fn(s.blas);
#endif
#ifdef _OPENMP
  if (s.omp > 0)
    omp_set_num_threads(s.omp);
#endif
}
}  // namespace detail
}  // namespace libKriging

// Register pthread_atfork handlers so that forked child processes (e.g. R's
// parallel::mclapply) inherit a clean, single-threaded BLAS/OMP state instead
// of a locked idle thread pool that causes deadlocks.
//
// The prepare handler runs in the parent just before fork(): it records the
// current thread counts, then sets BLAS and OMP to 1 thread so no new parallel
// work can start during the fork window. The parent handler restores the
// recorded counts once fork() has returned, so the parent does not stay
// single-threaded after e.g. mclapply. The child handler re-asserts
// single-threading in case the runtime inherited stale state.
#if !defined(_WIN32) && defined(_POSIX_VERSION)
#include <pthread.h>
namespace {

libKriging::detail::BlasThreadState atfork_saved;

void libkriging_atfork_prepare() {
  atfork_saved = libKriging::detail::get_blas_threads();
  libKriging::detail::set_blas_threads(1);
}

void libkriging_atfork_parent() {
  libKriging::detail::restore_blas_threads(atfork_saved);
}

void libkriging_atfork_child() {
  libKriging::detail::set_blas_threads(1);
}

struct ForkSafeRegistrar {
  ForkSafeRegistrar() {
    pthread_atfork(libkriging_atfork_prepare,  // prepare: quiesce before fork
                   libkriging_atfork_parent,   // parent:  restore thread counts
                   libkriging_atfork_child);   // child:   ensure clean state
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

// -----------------------------------------------------------------------------
// Public API in the original scale of X and y (see Kriging.hpp). Internally the
// model is held on normalized data when normalize=true; these helpers convert.
// With normalize=false every conversion below is skipped (bit-for-bit identity).
// -----------------------------------------------------------------------------

LIBKRIGING_EXPORT arma::vec Kriging::theta() const {
  if (!m_normalize || m_theta.n_elem != m_scaleX.n_elem)
    return m_theta;
  return m_theta % m_scaleX.t();
}

LIBKRIGING_EXPORT double Kriging::sigma2() const {
  return m_normalize ? m_sigma2 * m_scaleY * m_scaleY : m_sigma2;
}

LIBKRIGING_EXPORT arma::vec Kriging::beta() const {
  return beta_to_original(m_beta);
}

LIBKRIGING_EXPORT double Kriging::nugget() const {
  return m_normalize ? m_nugget * m_scaleY * m_scaleY : m_nugget;
}

LIBKRIGING_EXPORT arma::vec Kriging::noise() const {
  return m_normalize ? arma::vec(m_noise * (m_scaleY * m_scaleY)) : m_noise;
}

LIBKRIGING_EXPORT arma::mat Kriging::covMat(const arma::mat& X1, const arma::mat& X2) {
  arma::mat C = KrigingImpl::covMat(X1, X2);  // inputs normalized inside, internal sigma2
  if (m_normalize)
    C *= m_scaleY * m_scaleY;
  return C;
}

arma::vec Kriging::gamma_from_original(const arma::vec& gamma) const {
  if (!m_normalize)
    return gamma;
  arma::vec g = gamma;
  const arma::uword d = std::min<arma::uword>(m_scaleX.n_elem, g.n_elem);
  for (arma::uword k = 0; k < d; k++)
    g.at(k) /= m_scaleX.at(k);
  if (m_noise_model == NoiseModel::Heterogeneous && g.n_elem > m_scaleX.n_elem)
    g.at(m_scaleX.n_elem) /= m_scaleY * m_scaleY;  // sigma2; a Nugget alpha is scale-free
  return g;
}

void Kriging::grad_to_original(arma::vec& grad) const {
  if (!m_normalize)
    return;
  // F(theta_n = theta / scaleX): dF/dtheta_i = dF/dtheta_n_i / scaleX_i
  const arma::uword d = std::min<arma::uword>(m_scaleX.n_elem, grad.n_elem);
  for (arma::uword k = 0; k < d; k++)
    grad.at(k) /= m_scaleX.at(k);
  if (m_noise_model == NoiseModel::Heterogeneous && grad.n_elem > m_scaleX.n_elem)
    grad.at(m_scaleX.n_elem) /= m_scaleY * m_scaleY;
}

double Kriging::ll_offset() const {
  // y = centerY + scaleY * y_n: log p(y) = log p(y_n) - n log(scaleY)
  return m_normalize ? -static_cast<double>(m_y.n_elem) * std::log(m_scaleY) : 0.0;
}

double Kriging::lmp_offset() const {
  if (!m_normalize)
    return 0.0;
  // log_marginal_lik = -log|L| - log|LX| - (n-p)/2 log(S2): R is unchanged,
  // S2 scales by scaleY^2 and F' R^-1 F by A^-T . A^-1 (F_n = F A); the
  // reference prior only involves range(X) / theta, which is scale-free.
  const double n = static_cast<double>(m_y.n_elem);
  const double p = static_cast<double>(m_F.n_cols);
  return -(n - p) * std::log(m_scaleY) + trend_log_abs_det_to_original();
}

LIBKRIGING_EXPORT Kriging::Kriging(const arma::vec& y,
                                   const arma::mat& X,
                                   const std::string& covType,
                                   const Trend::RegressionModel& regmodel,
                                   bool normalize,
                                   const std::string& optim,
                                   const std::string& objective,
                                   const Parameters& parameters,
                                   const std::string& method) {
  if (y.n_elem != X.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X.n_rows) + "x"
                             + std::to_string(X.n_cols) + "), y: (" + std::to_string(y.n_elem) + ")");

  make_Cov(covType);
  fit(y, X, regmodel, normalize, optim, objective, parameters, method);
}

LIBKRIGING_EXPORT Kriging::Kriging(const Kriging& other, ExplicitCopySpecifier) : Kriging{other} {}

void Kriging::populate_Model(KModel& m,
                             const arma::vec& theta,
                             double extra_param,
                             std::map<std::string, double>* bench) const {
  populate_Model_jitter(m, theta, extra_param, m_chol_jitter, bench);
}

void Kriging::populate_Model_jitter(KModel& m,
                                    const arma::vec& theta,
                                    double extra_param,
                                    double chol_jitter,
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
    // (m_T was factorized with m_model_chol_jitter)
    update_eligible = !m_is_empty && (m_theta.size() == theta.size()) && (theta - m_theta).is_zero()
                      && (m_T.memptr() != nullptr) && (m_X.n_rows > m_T.n_rows) && (chol_jitter == m_model_chol_jitter);
    // Noise-free: fixed jitter on the diagonal (objective: m_chol_jitter;
    // committed model: m_model_chol_jitter, see make_committed_Model); the
    // Nugget / Heterogeneous modes are regularized by their own diagonal.
    if (chol_jitter > 0.0)
      diag = arma::vec(m_X.n_rows, arma::fill::value(1.0 + chol_jitter));
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

Kriging::KModel Kriging::make_committed_Model(const arma::vec& theta, double extra_param) const {
  KModel m = allocate_KModel();
  populate_Model_jitter(
      m, theta, extra_param, (m_noise_model == NoiseModel::None) ? m_model_chol_jitter : 0.0, nullptr);
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
    // Lazy, on-demand R^-1 (see populate_Model's comment): only the gradient
    // needs it, so it's built here, the first (and only, across repeated
    // no-refit calls at the same theta/n) time it's actually asked for --
    // same size/null-check pattern _leaveOneOut already uses for Linv.
    if ((m.Rinv.memptr() == nullptr) || (arma::size(m.Rinv) != arma::size(m.L))) {
      m.Rinv = LinearAlgebra::inv_sympd(m.L);
      t0 = Bench::toc(bench, "R^-1 = L^-T * L^-1", t0);
    }
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
  const arma::vec gamma_n = gamma_from_original(_theta);
  auto res = eval_objective(gamma_n.n_elem, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _logLikelihood(gamma_n, g, nullptr, b);
  });
  if (m_normalize) {
    std::get<0>(res) += ll_offset();
    grad_to_original(std::get<1>(res));
  }
  return res;
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
  const arma::vec theta_n = gamma_from_original(_theta);
  auto res = eval_objective(theta_n.n_elem, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _leaveOneOut(theta_n, g, nullptr, nullptr, b);
  });
  if (m_normalize) {
    // mean squared LOO error, in y units: (scaleY * e_n)^2
    const double s2 = m_scaleY * m_scaleY;
    std::get<0>(res) *= s2;
    std::get<1>(res) *= s2;
    grad_to_original(std::get<1>(res));
  }
  return res;
}

LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::leaveOneOutVec(const arma::vec& _theta) {
  arma::mat yhat = arma::mat(m_y.n_elem, 2, arma::fill::none);
  _leaveOneOut(gamma_from_original(_theta), nullptr, &yhat, nullptr, nullptr);
  arma::vec mean = yhat.col(0);
  arma::vec stdev = yhat.col(1) * std::sqrt(m_sigma2);
  if (m_normalize) {  // back to the original scale of y, like predict()
    mean = mean * m_scaleY + m_centerY;
    stdev *= m_scaleY;
  }
  return std::make_tuple(std::move(mean), std::move(stdev));
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

double Kriging::_logMargPostAt(const arma::vec& gamma_n,
                               arma::vec* grad_out,
                               std::map<std::string, double>* bench) const {
  // For Nugget mode, gamma may be d-dim (theta only); augment with current alpha
  if (m_noise_model == NoiseModel::Nugget && gamma_n.n_elem == m_X.n_cols)
    return _logMargPost(arma::join_cols(gamma_n, arma::vec{m_alpha}), grad_out, nullptr, bench);
  return _logMargPost(gamma_n, grad_out, nullptr, bench);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logMargPostFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  const arma::vec gamma_n = gamma_from_original(_theta);
  const arma::uword n_params
      = (m_noise_model == NoiseModel::Nugget && gamma_n.n_elem == m_X.n_cols) ? gamma_n.n_elem + 1 : gamma_n.n_elem;
  auto res = eval_objective(n_params, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _logMargPostAt(gamma_n, g, b);
  });
  if (m_normalize) {
    std::get<0>(res) += lmp_offset();
    grad_to_original(std::get<1>(res));
  }
  return res;
}

// The no-argument objectives evaluate the internal implementation at the
// committed (internal-scale) parameters, then apply the same value offset as
// the public *Fun: logLikelihood() == logLikelihoodFun(theta()) etc.
LIBKRIGING_EXPORT double Kriging::logLikelihood() {
  return _logLikelihood(current_gamma(), nullptr, nullptr, nullptr) + ll_offset();
}

LIBKRIGING_EXPORT double Kriging::leaveOneOut() {
  const double loo = _leaveOneOut(m_theta, nullptr, nullptr, nullptr, nullptr);
  return m_normalize ? loo * m_scaleY * m_scaleY : loo;
}

LIBKRIGING_EXPORT double Kriging::logMargPost() {
  return _logMargPostAt(m_theta, nullptr, nullptr) + lmp_offset();
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
// Vecchia approximated log-likelihood (fit method "vecchia(m)", internal objective "LLVecchia(m)")
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

// Noise models ("response" Vecchia, Katzfuss & Guinness 2021): the
// conditionals are those of the NOISY responses, so the m x m blocks are
// blocks of the same noisy correlation matrix as the exact likelihood:
//   Nugget:        Rt = alpha * C + (1 - alpha) * I,  scale s = sigma2 + nugget
//   Heterogeneous: Rt = C + diag(noise / sigma2),     scale s = sigma2
// i.e. off-diagonal f * C (f = alpha or 1) and diagonal diag_j (1 or
// 1 + noise_j / sigma2). gamma = [theta] (None) or [theta, alpha] (Nugget) or
// [theta, sigma2] (Heterogeneous), and s is profiled / fixed exactly as in
// _logLikelihood. See docs/math/Vecchia.md.
double Kriging::_logLikelihoodVecchia(const arma::vec& _gamma,
                                      arma::vec* grad_out,
                                      arma::vec* beta_out,
                                      double* sigma2_out,
                                      double* q_over_n_out) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword p = m_F.n_cols;
  const bool with_grad = (grad_out != nullptr);
  constexpr double v_floor = 1e-15;
  const arma::vec _theta = _gamma.head(d);

  // --- noise model: extra parameter (alpha / sigma2), forced like _logLikelihood
  const bool is_nugget = (m_noise_model == NoiseModel::Nugget);
  const bool is_hetero = (m_noise_model == NoiseModel::Heterogeneous);
  double extra = 0.0;
  bool extra_fixed = true;  // extra forced to its committed value (zero gradient)
  if (is_nugget || is_hetero) {
    extra = (_gamma.n_elem > d) ? _gamma.at(d) : (is_nugget ? m_alpha : m_sigma2);
    extra_fixed = false;
    if (is_hetero && !m_est_sigma2) {
      extra = m_sigma2;
      extra_fixed = true;
    } else if (is_nugget && !m_est_sigma2 && !m_est_nugget) {
      extra = m_sigma2 / (m_sigma2 + m_nugget);
      extra_fixed = true;
    }
  }
  const double f = is_nugget ? extra : 1.0;  // off-diagonal factor of the noisy correlation
  // gradient directions: theta (d), plus the extra parameter if asked for
  const bool grad_extra = with_grad && (grad_out->n_elem > d) && (is_nugget || is_hetero);
  const arma::uword nd = grad_extra ? d + 1 : d;
  // diagonal of the noisy correlation (and its d/dsigma2 for Heterogeneous);
  // noise-free: 1 + the fixed Cholesky jitter of the exact path (populate_Model)
  const double jitter = m_chol_jitter;  // 0 unless noise-free (fit_impl)
  auto diag_of = [&](arma::uword j) { return is_hetero ? 1.0 + m_noise(j) / extra : 1.0 + jitter; };
  auto ddiag_of = [&](arma::uword j) { return is_hetero ? -m_noise(j) / (extra * extra) : 0.0; };

  arma::vec u(n);     // y_i - a' y_N  (trend-free residual numerator)
  arma::vec v(n);     // conditional (noisy) correlation variance Rt_ii - r'a
  arma::mat W(p, n);  // w_i = f_i - F_N' a  (per-conditional trend design)
  arma::mat du, dv;
  arma::cube dW;
  if (with_grad) {
    du.zeros(n, nd);
    dv.zeros(n, nd);
    dW.zeros(p, n, nd);
  }

  for (arma::uword t = 0; t < n; ++t) {
    const arma::uword oi = m_vecchia_order(t);
    const arma::uvec& Ni = m_vecchia_neighbors[t];
    const arma::uword mi = Ni.n_elem;
    const double Rii = diag_of(oi);
    if (mi == 0) {
      u(t) = m_y(oi);
      v(t) = std::max(Rii, v_floor);
      W.col(t) = m_F.row(oi).t();
      if (grad_extra && is_hetero)
        dv(t, d) = ddiag_of(oi);
      continue;
    }

    // correlations (and their theta-derivatives) within {N(i), i}
    arma::mat CN(mi, mi, arma::fill::zeros);  // off-diagonal correlations (zero diagonal)
    arma::vec c(mi);
    arma::cube dCN;
    arma::mat dc;
    if (with_grad) {
      dCN.zeros(mi, mi, d);
      dc.zeros(mi, d);
    }
    for (arma::uword a = 0; a < mi; ++a) {
      for (arma::uword b = a + 1; b < mi; ++b) {
        const arma::vec dx = (m_X.row(Ni(a)) - m_X.row(Ni(b))).t();
        const double cab = _Cov(dx, _theta);
        CN(a, b) = CN(b, a) = cab;
        if (with_grad) {
          const arma::vec dln = _DlnCovDtheta(dx, _theta);
          for (arma::uword k = 0; k < d; ++k)
            dCN(a, b, k) = dCN(b, a, k) = cab * dln(k);
        }
      }
      const arma::vec dx = (m_X.row(Ni(a)) - m_X.row(oi)).t();
      const double ca = _Cov(dx, _theta);
      c(a) = ca;
      if (with_grad) {
        const arma::vec dln = _DlnCovDtheta(dx, _theta);
        for (arma::uword k = 0; k < d; ++k)
          dc(a, k) = ca * dln(k);
      }
    }
    arma::mat RN = f * CN;
    for (arma::uword a = 0; a < mi; ++a)
      RN(a, a) = diag_of(Ni(a)) + LinearAlgebra::num_nugget;
    const arma::vec r = f * c;

    const arma::mat LN = LinearAlgebra::safe_chol_lower(RN);
    const arma::vec a_vec = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), r));

    v(t) = std::max(Rii - arma::dot(r, a_vec), v_floor);
    u(t) = m_y(oi) - arma::dot(a_vec, m_y(Ni));
    W.col(t) = m_F.row(oi).t() - m_F.rows(Ni).t() * a_vec;

    if (with_grad) {
      const arma::mat FN_t = m_F.rows(Ni).t();  // p x mi
      const arma::vec yN = m_y(Ni);
      for (arma::uword k = 0; k < nd; ++k) {
        // derivatives (dRN, dr, dRii) of the noisy blocks along direction k:
        // da = RN^-1 (dr - dRN a), dv = dRii - dr'a - r'da
        arma::vec dr;
        arma::vec rhs;
        double dRii = 0.0;
        if (k < d) {  // theta_k: f * dC (zero diagonal)
          dr = f * dc.col(k);
          rhs = dr - f * (dCN.slice(k) * a_vec);
        } else if (is_nugget) {  // alpha: dRt/dalpha = C - I (zero diagonal)
          dr = c;
          rhs = dr - CN * a_vec;
        } else {  // Heterogeneous sigma2: dRt/dsigma2 = -diag(noise) / sigma2^2
          dr.zeros(mi);
          arma::vec dd(mi);
          for (arma::uword a = 0; a < mi; ++a)
            dd(a) = ddiag_of(Ni(a));
          rhs = -(dd % a_vec);
          dRii = ddiag_of(oi);
        }
        const arma::vec da = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), rhs));
        dv(t, k) = dRii - (arma::dot(dr, a_vec) + arma::dot(r, da));
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

  // --- scale s (profiled, or fixed, exactly as in _logLikelihood) -----------
  //   None, or Nugget with sigma2 and nugget estimated: s = Q / n (concentrated)
  //   Nugget, nugget fixed:  s = nugget / (1 - alpha)
  //   Nugget, sigma2 fixed:  s = sigma2 / alpha
  //   Nugget, both fixed:    s = sigma2 + nugget
  //   Heterogeneous:         s = sigma2 (free parameter, not concentrated)
  const arma::vec e = u - W.t() * beta;
  const double Q = arma::accu(e % e / v);
  const double sum_logv = arma::accu(arma::log(v));
  bool concentrated = !(is_nugget || is_hetero);
  double s = Q / n;
  double ds_dextra = 0.0;
  if (is_nugget) {
    if (m_est_sigma2 && m_est_nugget) {
      concentrated = true;
    } else if (m_est_sigma2) {
      s = m_nugget / (1.0 - extra);
      ds_dextra = m_nugget / ((1.0 - extra) * (1.0 - extra));
    } else if (m_est_nugget) {
      s = m_sigma2 / extra;
      ds_dextra = -m_sigma2 / (extra * extra);
    } else {
      s = m_sigma2 + m_nugget;
    }
  } else if (is_hetero) {
    s = extra;
    ds_dextra = extra_fixed ? 0.0 : 1.0;
  }
  if (beta_out != nullptr)
    *beta_out = beta;
  if (sigma2_out != nullptr)
    *sigma2_out = s;
  if (q_over_n_out != nullptr)
    *q_over_n_out = Q / n;
  const double vll = concentrated ? -0.5 * n * std::log(2 * M_PI * s) - 0.5 * sum_logv - 0.5 * n
                                  : -0.5 * n * std::log(2 * M_PI * s) - 0.5 * sum_logv - 0.5 * Q / s;

  if (with_grad) {
    // envelope theorem: dQ/dbeta = 0 at beta_hat (and dLL/ds = 0 at the
    // concentrated s), so neither profile contributes to the gradient
    grad_out->zeros();
    for (arma::uword k = 0; k < nd; ++k) {
      const arma::vec de = du.col(k) - dW.slice(k).t() * beta;
      const double dQ = arma::accu(2.0 * e % de / v - (e % e / (v % v)) % dv.col(k));
      const double dlogv = arma::accu(dv.col(k) / v);
      if (concentrated) {
        (*grad_out)(k) = -0.5 * n * dQ / Q - 0.5 * dlogv;
      } else {
        double g = -0.5 * dlogv - 0.5 * dQ / s;
        if (k == d)
          g += (-0.5 * n / s + 0.5 * Q / (s * s)) * ds_dextra;
        (*grad_out)(k) = g;
      }
    }
    if (grad_extra && extra_fixed)
      (*grad_out)(d) = 0.0;
  }
  return vll;
}

// Variogram sill estimate 0.5 * mean (y_i - y_j)^2 over the pairs at least as
// far apart as the median pair distance: the formula of the Heterogeneous
// sigma2 bounds in fit(), without the O(n^2) m_dX (method "vecchia"), on an
// evenly strided subsample of at most n_max points.
static double variogram_sill(const arma::mat& X, const arma::vec& y, arma::uword n_max = 1000) {
  const arma::uword n_all = X.n_rows;
  const arma::uword ns = std::min<arma::uword>(n_all, n_max);
  arma::uvec idx(ns);
  for (arma::uword i = 0; i < ns; ++i)
    idx(i) = (i * n_all) / ns;
  const arma::mat Xs = X.rows(idx);
  const arma::vec ys = y(idx);
  arma::vec dy2(ns * ns, arma::fill::zeros);
  arma::vec dX2(ns * ns, arma::fill::zeros);
  for (arma::uword j = 0; j < ns; ++j) {
    for (arma::uword i = 0; i < ns; ++i) {
      if (i == j)
        continue;
      const double dy = ys(i) - ys(j);
      dy2(i + j * ns) = dy * dy;
      dX2(i + j * ns) = arma::accu(arma::square(Xs.row(i) - Xs.row(j)));
    }
  }
  return 0.5 * arma::mean(dy2.elem(arma::find(dX2 >= arma::median(dX2))));
}

void Kriging::commit_vecchia(const arma::vec& theta, double extra) {
  const arma::uword d = theta.n_elem;
  m_theta = theta;
  arma::vec gamma = theta;
  if (m_noise_model != NoiseModel::None) {
    gamma.resize(d + 1);
    gamma.at(d) = extra;
  }
  arma::vec beta_v;
  double s_v = -1;
  _logLikelihoodVecchia(gamma, nullptr, &beta_v, &s_v);
  if (m_est_beta)
    m_beta = beta_v;
  // same conventions as the exact BFGS commit in fit()
  if (m_noise_model == NoiseModel::Nugget) {
    if (!m_est_sigma2 && !m_est_nugget) {
      m_alpha = m_sigma2 / (m_sigma2 + m_nugget);  // forced, as in _logLikelihood
    } else {
      m_alpha = extra;
      if (m_est_sigma2 && m_est_nugget) {
        m_sigma2 = m_alpha * s_v;  // s_v = profiled total variance sigma2 + nugget
        m_nugget = m_sigma2 / m_alpha - m_sigma2;
      } else if (m_est_sigma2) {
        m_sigma2 = m_nugget * m_alpha / (1.0 - m_alpha);
      } else {
        m_nugget = m_sigma2 * (1.0 - m_alpha) / m_alpha;
      }
    }
  } else if (m_noise_model == NoiseModel::Heterogeneous) {
    if (m_est_sigma2)
      m_sigma2 = extra;  // unconcentrated: sigma2 is a parameter
  } else if (m_est_sigma2) {
    m_sigma2 = s_v;
  }
  m_vecchia_light = true;
  m_is_empty = true;  // no committed factorization: predict routes to predictVecchia
}

bool Kriging::y_in_trend_span() const {
  arma::vec r = m_y;
  if (m_F.n_cols > 0) {
    arma::vec b;
    if (!arma::solve(b, m_F, m_y, arma::solve_opts::no_approx))
      return false;
    r = m_y - m_F * b;
  }
  return arma::norm(r, "inf") <= 1e-10 * arma::norm(m_y, "inf");
}

void Kriging::check_not_vecchia_light(const char* what) const {
  if (m_vecchia_light)
    throw std::runtime_error(std::string(what)
                             + ": not available on a Vecchia fit "
                               "(fit with method=\"exact\")");
}

void Kriging::check_exact_available(const char* what, const char* alternative) const {
  if (m_vecchia_light)
    throw std::runtime_error(std::string(what) + ": no exact factorization on a Vecchia fit; use method \"vecchia\" ("
                             + alternative + "Vecchia), or fit with method=\"exact\"");
  if (m_nystrom_light)
    throw std::runtime_error(std::string(what) + ": no exact factorization on a Nystrom fit; use method \"nystrom\" ("
                             + alternative + "Nystrom), or fit with method=\"exact\"");
}

namespace {
// Explicit prediction / simulation / likelihood method: "exact", "vecchia",
// "vecchia(m)", "nystrom" or "nystrom(k)" (no implicit choice from the fit).
struct ApproxMethod {
  enum Kind { Exact, Vecchia, Nystrom } kind;
  arma::uword m;  // Vecchia neighbors / Nystrom rank (0 = default / fitted)
};

ApproxMethod parse_approx_method(const std::string& method, const char* what) {
  if (method == "exact")
    return {ApproxMethod::Exact, 0};
  for (const auto kind : {ApproxMethod::Vecchia, ApproxMethod::Nystrom}) {
    const std::string name = (kind == ApproxMethod::Vecchia) ? "vecchia" : "nystrom";
    if (method == name)
      return {kind, 0};
    if (method.rfind(name + "(", 0) == 0 && method.back() == ')') {
      try {
        const long m = std::stol(method.substr(name.size() + 1, method.size() - name.size() - 2));
        if (m >= 1)
          return {kind, static_cast<arma::uword>(m)};
      } catch (const std::exception&) {
        // fall through to the throw below
      }
    }
  }
  throw std::invalid_argument(std::string(what) + ": unknown method '" + method
                              + "': expected \"exact\", \"vecchia\", \"vecchia(m)\", \"nystrom\" or "
                                "\"nystrom(k)\" (m, k >= 1)");
}

// The Nystrom predictor / simulator / likelihood use the fitted rank-k
// factors: "nystrom(k)" is accepted only with the fitted k (so that the string
// returned by method() can be passed back as is).
void check_nystrom_rank(const ApproxMethod& am, arma::uword fitted_k, const char* what) {
  if (am.kind == ApproxMethod::Nystrom && am.m != 0 && am.m != fitted_k)
    throw std::invalid_argument(std::string(what) + ": the Nystrom method uses the fitted rank ("
                                + std::to_string(fitted_k) + "): use method \"nystrom\"");
}

// The Vecchia likelihood / LOO / LMP use the fitted neighbor sets: "vecchia(m)"
// is accepted only with the fitted m.
void check_vecchia_m(const ApproxMethod& am, arma::uword fitted_m, const char* what) {
  if (am.kind == ApproxMethod::Vecchia && am.m != 0 && am.m != fitted_m)
    throw std::invalid_argument(std::string(what) + ": the Vecchia criterion uses the fitted m ("
                                + std::to_string(fitted_m) + "): use method \"vecchia\"");
}
}  // namespace

LIBKRIGING_EXPORT double Kriging::logLikelihood(const std::string& method) {
  const ApproxMethod am = parse_approx_method(method, "logLikelihood");
  switch (am.kind) {
    case ApproxMethod::Vecchia:
      check_vecchia_m(am, m_vecchia_m, "logLikelihood");
      if (m_vecchia_m == 0)
        throw std::runtime_error("logLikelihoodVecchiaFun: model was not fitted with method=\"vecchia(m)\"");
      if (m_vecchia_order.n_elem != m_X.n_rows)
        make_vecchia_sets();  // not stored by save(): rebuilt on demand after load
      // internal members (theta and, with a noise model, alpha / sigma2) + offset
      return _logLikelihoodVecchia(current_gamma(), nullptr) + ll_offset();
    case ApproxMethod::Nystrom:
      check_nystrom_rank(am, m_nystrom_k, "logLikelihood");
      if (m_nystrom_k == 0)
        throw std::runtime_error("logLikelihoodNystromFun: model was not fitted with method=\"nystrom(k)\"");
      // internal members (theta and, with a noise model, alpha / sigma2) + offset
      return _logLikelihoodNystrom(current_gamma()) + ll_offset();
    case ApproxMethod::Exact:
    default:
      return logLikelihood();
  }
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logLikelihoodVecchiaFun(const arma::vec& theta,
                                                                                 bool return_grad) {
  if (m_vecchia_m == 0)
    throw std::runtime_error("logLikelihoodVecchiaFun: model was not fitted with method=\"vecchia(m)\"");
  if (theta.n_elem != m_X.n_cols && !(m_noise_model != NoiseModel::None && theta.n_elem == m_X.n_cols + 1))
    throw std::invalid_argument("logLikelihoodVecchiaFun: expected " + std::to_string(m_X.n_cols)
                                + " (theta) or, with a noise model, " + std::to_string(m_X.n_cols + 1)
                                + " (theta and alpha / sigma2) parameters, got " + std::to_string(theta.n_elem));
  if (m_vecchia_order.n_elem != m_X.n_rows)
    make_vecchia_sets();  // not stored by save(): rebuilt on demand after load
  const arma::vec theta_n = gamma_from_original(theta);
  arma::vec grad;
  if (return_grad) {
    grad.set_size(theta_n.n_elem);
    const double vll = _logLikelihoodVecchia(theta_n, &grad) + ll_offset();
    grad_to_original(grad);
    return {vll, grad};
  }
  return {_logLikelihoodVecchia(theta_n, nullptr) + ll_offset(), grad};
}

/* Vecchia (local) prediction: condition each prediction point on its m
 * nearest observations only (Katzfuss & Guinness 2021, response-only
 * conditioning). Per point: O(m^3); no cross-covariances between prediction
 * points (use predict() for the exact joint version). */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::predictVecchia(const arma::mat& X_n,
                                                                           bool return_stdev,
                                                                           arma::uword m) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword q = X_n.n_rows;
  if (X_n.n_cols != d)
    throw std::invalid_argument("X_n should have the same number of columns as X");
  if (m == 0)
    m = (m_vecchia_m > 0) ? m_vecchia_m : 30;
  m = std::min<arma::uword>(m, n);

  // noise model, with the predict() conventions: Nugget predicts a noisy
  // observation (variance scale sigma2 / alpha = sigma2 + nugget, cross
  // correlation alpha * C), Heterogeneous the latent process (variance scale
  // sigma2, cross correlation C); LMP fits rescale sigma2 like predict()
  const bool is_nugget = (m_noise_model == NoiseModel::Nugget);
  const bool is_hetero = (m_noise_model == NoiseModel::Heterogeneous);
  const double f = is_nugget ? m_alpha : 1.0;
  // noise-free observations: the fixed Cholesky jitter of the exact path
  const double jitter = m_chol_jitter;  // 0 unless noise-free (fit_impl)
  const double lmp_scale = (m_objective == "LMP") ? (n - m_F.n_cols) / (n - m_F.n_cols - 2.0) : 1.0;
  const double var_scale = m_sigma2 * lmp_scale / f;

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
#pragma omp parallel for schedule(dynamic, 16) num_threads(libKriging::detail::omp_threads()) if (q > 32 && n * m >= 40000)
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

    // local kriging on the neighborhood, with the noisy correlation block
    // (Nugget: alpha*C + (1-alpha)*I, Heterogeneous: C + diag(noise/sigma2))
    arma::mat RN(m, m);
    arma::vec r(m);
    for (arma::uword a = 0; a < m; ++a) {
      for (arma::uword b = a + 1; b < m; ++b) {
        const arma::vec dx = (m_X.row(Ni(a)) - m_X.row(Ni(b))).t();
        RN(a, b) = RN(b, a) = f * _Cov(dx, m_theta);
      }
      RN(a, a) = (is_hetero ? 1.0 + m_noise(Ni(a)) / m_sigma2 : 1.0 + jitter) + LinearAlgebra::num_nugget;
      const arma::vec dx = (m_X.row(Ni(a)) - x_t).t();
      // as predict(): a prediction point equal to an observation gets the
      // observation's (noisy) covariance 1 in Nugget mode
      r(a) = (is_nugget && dx.is_zero(arma::datum::eps)) ? 1.0 : f * _Cov(dx, m_theta);
    }
    const arma::mat LN = LinearAlgebra::safe_chol_lower(RN);
    const arma::vec a_vec = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), r));

    const arma::vec resid = m_y(Ni) - m_F.rows(Ni) * m_beta;
    mean(t) = arma::dot(F_n.row(static_cast<arma::uword>(t)), m_beta) + arma::dot(a_vec, resid);
    if (return_stdev)
      stdev(t) = std::sqrt(var_scale * std::max(0.0, 1.0 - arma::dot(r, a_vec)));
  }

  // de-normalize outputs
  mean = mean * m_scaleY + m_centerY;
  if (return_stdev)
    stdev = stdev * m_scaleY;
  return {mean, stdev};
}

// =============================================================================
/* Vecchia (local) simulation: sequential conditional simulation in the row
 * order of X_n. Point t is drawn from its simple-kriging conditional given
 * its m nearest neighbors among the observations and the points 0..t-1
 * already simulated (same draw shared by all nsim trajectories for the
 * neighbor selection and kriging weights; only the conditioning values
 * differ per trajectory). Chain rule => exact joint (beta known) as soon as
 * every point conditions on all its predecessors (m >= n + q - 1).
 *
 * Noise models, with the exact simulate() conventions: the observations are
 * noisy responses (Nugget: correlation alpha*C + (1-alpha)*I in units of
 * sigma2 / alpha; Heterogeneous: C + diag(noise / sigma2) in units of
 * sigma2), the simulated values are the latent process (cross correlation
 * alpha*C / C), or for Nugget with with_nugget the noisy response (variance
 * 1, i.e. sigma2 + nugget, and an observation at the same location is
 * reproduced, as simulate(..., with_nugget=true, ...)). with_noise (noise
 * variance, output scale: scalar or one per row of X_n) adds independent
 * Gaussian noise afterwards, as simulate(..., with_noise, ...). */
LIBKRIGING_EXPORT arma::mat Kriging::simulateVecchia(int nsim,
                                                     int seed,
                                                     const arma::mat& X_n,
                                                     arma::uword m,
                                                     bool with_nugget,
                                                     const arma::vec& with_noise) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword q = X_n.n_rows;
  if (X_n.n_cols != d)
    throw std::invalid_argument("simulateVecchia: X_n has wrong dimension: " + std::to_string(X_n.n_cols)
                                + " instead of " + std::to_string(d));
  if (nsim <= 0)
    throw std::invalid_argument("simulateVecchia: nsim should be > 0");
  if (with_noise.n_elem > 1 && with_noise.n_elem != q)
    throw std::invalid_argument("simulateVecchia: with_noise should be a scalar or have one value per row of X_n");
  check_noise_variances(with_noise, "simulateVecchia");
  if (m == 0)
    m = (m_vecchia_m > 0) ? m_vecchia_m : 30;
  const arma::uword m_obs = std::min<arma::uword>(m, n);

  const bool is_nugget = (m_noise_model == NoiseModel::Nugget);
  const bool is_hetero = (m_noise_model == NoiseModel::Heterogeneous);
  const bool noisy_sim = is_nugget && with_nugget;
  const double f = is_nugget ? m_alpha : 1.0;  // off-diagonal correlation factor
  const double sim_var = noisy_sim ? 1.0 : f;  // variance of a simulated value (units sigma2 / f)

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;
  const arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);

  // 1) m nearest OBSERVATIONS of each simulation point: independent across
  //    points, hence parallel (same search as predictVecchia).
  arma::umat Nobs(m_obs, q);
  arma::mat Dobs(m_obs, q);
  const arma::mat Xt = m_X.t();
  const double* xp = Xt.memptr();
#if defined(_OPENMP) && !defined(LK_NESTED_NO_OMP)
#pragma omp parallel for schedule(dynamic, 16) num_threads(libKriging::detail::omp_threads()) if (q > 32 && n * m_obs >= 40000)
#endif
  for (arma::sword t = 0; t < static_cast<arma::sword>(q); ++t) {
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
    if (m_obs < n)
      std::nth_element(cand.begin(), cand.begin() + m_obs, cand.end());
    for (arma::uword j = 0; j < m_obs; ++j) {
      Nobs(j, static_cast<arma::uword>(t)) = cand[j].second;
      Dobs(j, static_cast<arma::uword>(t)) = cand[j].first;
    }
  }

  // 2) sequential pass: the m nearest of (observations U previous simulation
  //    points) are among (m nearest observations U previous simulation points).
  //    Candidate index c < n is observation c, c >= n is simulation point c-n.
  const arma::vec resid_o = m_y - m_F * m_beta;  // observed residuals (shared by all trajectories)
  arma::mat Z(q, nsim, arma::fill::zeros);       // simulated residuals (normalized scale)
  Random::reset_seed(seed);
  const arma::mat eps = Random::randn_mat(q, nsim);
  const double sigma = std::sqrt(m_sigma2 / f);

  auto row_of = [&](arma::uword c) -> arma::rowvec { return (c < n) ? m_X.row(c) : Xn_n.row(c - n); };
  // correlation (units sigma2 / f) between two distinct candidates
  auto corr = [&](arma::uword ca, const arma::rowvec& xa, arma::uword cb, const arma::rowvec& xb) {
    const arma::vec dx = (xa - xb).t();
    if (noisy_sim && ((ca < n) != (cb < n)) && dx.is_zero(arma::datum::eps))
      return 1.0;  // noisy simulated response at an observed location
    return f * _Cov(dx, m_theta);
  };
  // variance (units sigma2 / f) of a candidate; observations: noise-free with
  // the fixed Cholesky jitter of the exact path (populate_Model)
  const double jitter = m_chol_jitter;  // 0 unless noise-free (fit_impl)
  auto var_of = [&](arma::uword c) {
    if (c >= n)
      return sim_var;
    return is_hetero ? 1.0 + m_noise(c) / m_sigma2 : 1.0 + jitter;
  };

  std::vector<std::pair<double, arma::uword>> cand;
  for (arma::uword t = 0; t < q; ++t) {
    cand.clear();
    for (arma::uword j = 0; j < m_obs; ++j)
      cand.emplace_back(Dobs(j, t), Nobs(j, t));
    const arma::rowvec x_t = Xn_n.row(t);
    for (arma::uword s = 0; s < t; ++s)
      cand.emplace_back(arma::accu(arma::square(x_t - Xn_n.row(s))), n + s);
    const arma::uword mt = std::min<arma::uword>(m, cand.size());
    if (mt < cand.size())
      std::nth_element(cand.begin(), cand.begin() + mt, cand.end());

    double cond_var = sim_var;
    arma::rowvec cond_mean(nsim, arma::fill::zeros);
    if (mt > 0) {
      arma::mat RN(mt, mt);
      arma::vec r(mt);
      arma::mat ZN(mt, nsim);
      for (arma::uword a = 0; a < mt; ++a) {
        const arma::uword ca = cand[a].second;
        const arma::rowvec xa = row_of(ca);
        for (arma::uword b = a + 1; b < mt; ++b) {
          const arma::uword cb = cand[b].second;
          RN(a, b) = RN(b, a) = corr(ca, xa, cb, row_of(cb));
        }
        RN(a, a) = var_of(ca) + LinearAlgebra::num_nugget;
        r(a) = corr(ca, xa, n + t, x_t);
        if (ca < n)
          ZN.row(a).fill(resid_o(ca));
        else
          ZN.row(a) = Z.row(ca - n);
      }
      const arma::mat LN = LinearAlgebra::safe_chol_lower(RN);
      const arma::vec a_vec = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), r));
      cond_mean = a_vec.t() * ZN;
      cond_var = std::max(0.0, sim_var - arma::dot(r, a_vec));
    }
    Z.row(t) = cond_mean + sigma * std::sqrt(cond_var) * eps.row(t);
  }

  arma::mat y_n = Z;
  y_n.each_col() += F_n * m_beta;
  y_n = m_centerY + m_scaleY * y_n;
  add_noise_draws(y_n, with_noise, "simulate");
  return y_n;
}

// Nystrom approximated log-likelihood (fit method "nystrom(k)", internal objective "LLNystrom(k)")
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
// the block factorization NystromChol (landmark rows first) -- the n x n
// matrix R is never materialized. Cost O(n*k^2) per evaluation instead of O(n^3).
//
// Profiling matches the exact "LL" objective: sigma2 and beta in closed form
// (same GLS formulas as _logLikelihood, just solved via NystromChol instead of a
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

namespace {

// Diagonal residual D0 = 1 - diag(U0 U0') of the Nystrom approximation
// (U0 U0' = R_ns (R_ss + eps I)^-1 R_ns', eps = num_nugget), floored at eps.
// At a landmark row i it is exactly eps - eps^2 [(R_ss + eps I)^-1]_ii, in
// [0, eps]: the floor always applies, so D0_i = eps. It is set so explicitly
// rather than computed as 1 - (~1): that difference only has an absolute
// accuracy ~1e-16, i.e. a relative error ~1e-6 at the eps = 1e-10 level, and
// whether the floor applies would be decided by rounding, which then gets
// amplified by the 1/D weights of the Woodbury solve and log-determinant
// (LL off by 1e-7..4e-6 with every row a landmark, depending on the CPU, #410).
arma::vec nystrom_residual_diag(const arma::mat& U0, const arma::uvec& landmarks) {
  arma::vec D0 = arma::clamp(1.0 - arma::sum(arma::square(U0), 1), LinearAlgebra::num_nugget, arma::datum::inf);
  D0.elem(landmarks).fill(LinearAlgebra::num_nugget);
  return D0;
}

// Upper triangular factor Rf of an economical QR of A, with a positive
// diagonal: A'A = Rf'Rf, obtained without forming the Gram matrix A'A.
arma::mat qr_r(const arma::mat& A) {
  arma::mat Q, Rf;
  arma::qr_econ(Q, Rf, A);
  Rf.each_col() %= arma::sign(Rf.diag());
  return Rf;
}

// Factorization of the Nystrom correlation matrix Rhat = U U' + diag(D),
// numerically stable when the landmark rows S have a diagonal D_S at the
// num_nugget level (#410). The Woodbury identity weights every row by 1/D
// and forms I + U' D^-1 U, a Gram matrix in which those 1/num_nugget weights
// square the conditioning: the predicted mean was off by up to 10% of sd(y)
// at an ill-conditioned R_ss, and by a CPU-dependent amount. Instead, block
// Cholesky with the landmark rows first (N: the other rows):
//   Rhat_SS = U_S U_S' + diag(D_S) = LA LA'         (QR of [U_S'; diag(D_S)^1/2])
//   Rhat_NS = U_N U_S' = LNS LA'                     (LNS = U_N U_S' LA^-T)
//   Schur   = Rhat_NN - LNS LNS' = diag(D_N) + U_N H U_N',
//   H = I - U_S' Rhat_SS^-1 U_S = (I + U_S' D_S^-1 U_S)^-1 = Rh^-1 Rh^-T
// (QR of [D_S^-1/2 U_S; I]), the Schur complement being solved by Woodbury
// (diag(D_N) is away from num_nugget but for near-duplicate rows). With every
// row a landmark, Rhat = LA LA' reproduces the exact Cholesky.
struct NystromChol {
  arma::uvec S, N;
  arma::mat LA;        // k x k lower
  arma::mat LNS;       // m x k
  arma::vec DNinv;     // m: Schur^-1 = diag(DNinv) - YN YN'
  arma::mat YN;        // m x k
  arma::mat PNS;       // m x k: block N x S of Rhat^-1   (with_inverse only)
  arma::vec diag_inv;  // n: diag(Rhat^-1)                  (with_inverse only)
  double logdet = 0;

  NystromChol(const arma::mat& U, const arma::vec& D, const arma::uvec& landmarks, bool with_inverse = false)
      : S(landmarks) {
    const arma::uword n = U.n_rows, k = U.n_cols;
    arma::uvec is_land(n, arma::fill::zeros);
    is_land.elem(S).ones();
    N = arma::find(is_land == 0);
    const arma::mat US = U.rows(S), UN = U.rows(N);
    const arma::vec DS = D.elem(S), DN = D.elem(N);
    const arma::vec sDS = arma::sqrt(DS);

    LA = qr_r(arma::join_cols(US.t(), arma::mat(arma::diagmat(sDS)))).t();
    logdet = 2.0 * arma::accu(arma::log(LA.diag()));
    DNinv = 1.0 / DN;
    if (N.is_empty()) {  // every row a landmark: Rhat = LA LA'
      LNS.zeros(0, k);
      YN.zeros(0, k);
      PNS.zeros(0, k);
      if (with_inverse) {
        diag_inv.set_size(n);
        diag_inv.elem(S) = arma::sum(arma::square(LinearAlgebra::solve_lower(LA, arma::eye<arma::mat>(k, k))), 0).t();
      }
      return;
    }

    LNS = UN * LinearAlgebra::solve_lower(LA, US).t();
    const arma::mat Rh = qr_r(arma::join_cols(US.each_col() / sDS, arma::eye<arma::mat>(k, k)));
    const arma::mat V = LinearAlgebra::solve_lower(Rh.t(), UN.t()).t();  // U_N Rh^-1: Schur = diag(D_N) + V V'
    const arma::mat DinvV = V.each_col() % DNinv;
    const arma::mat Lv = arma::chol(arma::eye<arma::mat>(k, k) + V.t() * DinvV, "lower");
    YN = LinearAlgebra::solve_lower(Lv, DinvV.t()).t();
    logdet += arma::accu(arma::log(DN)) + 2.0 * arma::accu(arma::log(Lv.diag()));

    if (!with_inverse)
      return;
    // Rhat^-1 = [[T'(I + LNS' Schur^-1 LNS) T, -T' LNS' Schur^-1], [., Schur^-1]], T = LA^-1
    const arma::mat T = LinearAlgebra::solve_lower(LA, arma::eye<arma::mat>(k, k));
    const arma::mat C = LNS * T;
    PNS = -schur_solve(C);
    diag_inv.set_size(n);
    diag_inv.elem(S) = arma::sum(arma::square(T), 0).t() - arma::sum(C % PNS, 0).t();
    diag_inv.elem(N) = DNinv - arma::sum(arma::square(YN), 1);
  }

  arma::mat schur_solve(const arma::mat& B) const { return (B.each_col() % DNinv) - YN * (YN.t() * B); }

  NystromChol() = default;

  arma::mat solve(const arma::mat& B) const {
    const arma::mat zS = LinearAlgebra::solve_lower(LA, B.rows(S));
    arma::mat x(B.n_rows, B.n_cols);
    if (N.is_empty()) {
      x.rows(S) = LinearAlgebra::solve_upper(LA.t(), zS);
      return x;
    }
    const arma::mat xN = schur_solve(B.rows(N) - LNS * zS);
    x.rows(S) = LinearAlgebra::solve_upper(LA.t(), zS - LNS.t() * xN);
    x.rows(N) = xN;
    return x;
  }
};

}  // namespace

double Kriging::_logLikelihoodNystrom(const arma::vec& _gamma,
                                      arma::vec* grad_out,
                                      arma::vec* beta_out,
                                      double* sigma2_out,
                                      arma::mat* U_out,
                                      arma::vec* D_out,
                                      double* sse_out) const {
  const arma::uword n = m_X.n_rows;
  const arma::uword d = m_X.n_cols;
  const arma::uword kL = m_nystrom_landmarks.n_elem;
  const arma::mat X_land = m_X.rows(m_nystrom_landmarks);  // k x d
  const arma::vec _theta = _gamma.head(d);

  // Extra parameter (alpha for Nugget, sigma2 for Heterogeneous), with the
  // same defaults / forcing rules as the exact _logLikelihood.
  double extra = 0.0;
  if (m_noise_model != NoiseModel::None) {
    extra = (_gamma.n_elem > d) ? _gamma.at(d) : ((m_noise_model == NoiseModel::Nugget) ? m_alpha : m_sigma2);
    if (m_noise_model == NoiseModel::Heterogeneous && !m_est_sigma2)
      extra = m_sigma2;
    else if (m_noise_model == NoiseModel::Nugget && !m_est_sigma2 && !m_est_nugget)
      extra = m_sigma2 / (m_sigma2 + m_nugget);
  }

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
  const arma::mat U0
      = arma::trans(LinearAlgebra::solve_lower(L_ss, R_ns.t()));  // n x k; U0*U0.t() = R_ns R_ss^-1 R_ns.t()

  const arma::vec D0 = nystrom_residual_diag(U0, m_nystrom_landmarks);

  // Correlation matrix of the observations, with Ct := U0*U0.t() + diag(D0) ~= C:
  //   None:          Rhat = Ct                     -> (U, D) = (U0, D0)
  //   Nugget:        Rhat = a*Ct + (1-a)*I          -> (U, D) = (sqrt(a)*U0, a*D0 + 1 - a)
  //   Heterogeneous: Rhat = Ct + diag(noise/sigma2) -> (U, D) = (U0, D0 + noise/sigma2)
  // i.e. always "low rank + diagonal", so the Woodbury solve/logdet apply as is.
  double c_scale = 1.0;  // dRhat/dtheta = c_scale * dCt/dtheta
  arma::mat U;
  arma::vec D;
  if (m_noise_model == NoiseModel::Nugget) {
    c_scale = extra;
    U = std::sqrt(extra) * U0;
    D = extra * D0 + (1.0 - extra);
  } else if (m_noise_model == NoiseModel::Heterogeneous) {
    U = U0;
    D = D0 + m_noise / extra;
  } else {
    // noise-free: the fixed Cholesky jitter of the exact path (populate_Model),
    // so that k = n reproduces the exact model
    U = U0;
    D = D0 + m_chol_jitter;
  }

  const NystromChol Rf(U, D, m_nystrom_landmarks, /*with_inverse=*/grad_out != nullptr);
  const arma::mat RinvF = Rf.solve(m_F);
  const arma::vec Rinvy = Rf.solve(m_y);

  const arma::mat A = m_F.t() * RinvF;
  const arma::vec b = m_F.t() * Rinvy;
  const arma::vec beta = arma::solve(A, b, arma::solve_opts::likely_sympd);

  const arma::vec e = m_y - m_F * beta;
  const arma::vec Rinve = Rinvy - RinvF * beta;  // Rinv is linear: Rinv(y - F*beta) = Rinv*y - Rinv*F*beta
  const double SSE = arma::dot(e, Rinve);
  const double logdetR = Rf.logdet;

  // Variance scale tv (Cov(y) = tv * Rhat) and its derivative wrt the extra
  // parameter, with exactly the profiling of the exact _logLikelihood:
  //   None:          tv = SSE/n (concentrated)
  //   Nugget:        tv = sigma2 + nugget: SSE/n when both are estimated
  //                  (concentrated), nugget/(1-a) when only sigma2 is,
  //                  sigma2/a when only the nugget is, sigma2 + nugget when
  //                  both are fixed (a then forced to sigma2/(sigma2+nugget))
  //   Heterogeneous: tv = sigma2 (NOT concentrated: an optimized parameter)
  double tv = SSE / n;
  double dtv = 0.0;  // d(tv)/d(extra); 0 when tv is concentrated (envelope theorem)
  bool extra_free = false;
  if (m_noise_model == NoiseModel::Nugget) {
    if (m_est_sigma2 && m_est_nugget) {
      extra_free = true;
    } else if (m_est_sigma2) {
      tv = m_nugget / (1.0 - extra);
      dtv = tv / (1.0 - extra);
      extra_free = true;
    } else if (m_est_nugget) {
      tv = m_sigma2 / extra;
      dtv = -tv / extra;
      extra_free = true;
    } else {
      tv = m_sigma2 + m_nugget;
    }
  } else if (m_noise_model == NoiseModel::Heterogeneous) {
    tv = extra;
    dtv = 1.0;
    extra_free = m_est_sigma2;
  }
  const double ll = -0.5 * (n * std::log(2 * M_PI * tv) + logdetR + SSE / tv);

  if (beta_out != nullptr)
    *beta_out = beta;
  if (sigma2_out != nullptr)
    *sigma2_out = tv;
  if (sse_out != nullptr)
    *sse_out = SSE;
  if (U_out != nullptr)
    *U_out = U;
  if (D_out != nullptr)
    *D_out = D;

  if (grad_out != nullptr) {
    // Analytic gradient, envelope theorem (beta_hat -- and tv when it is
    // concentrated -- held fixed, see the file-level comment above). For any
    // scalar parameter p of Rhat (and possibly of tv):
    //   d(ll)/dp = -0.5 * ( n tv'/tv + tr(Rinv dRhat/dp) - x' dRhat/dp x / tv - SSE tv'/tv^2 ),
    // x = Rinv*(y - F*beta_hat) = Rinve.
    //
    // theta: tv' = 0 and dRhat/dtheta_k = c_scale * dCt/dtheta_k, where
    // Ct = U0*U0.t() + diag(D0) with D0 = 1 - diag(U0*U0.t()) (constant on
    // the diagonal, by construction): so dCt/dtheta_k equals
    // d(U0*U0.t())/dtheta_k with its diagonal zeroed out. Writing U0*U0.t() =
    // M*K^-1*M.t() (M=R_ns, K=R_ss, both smooth in theta since the landmark
    // set is fixed), d(M K^-1 M.t())/dtheta_k = dM_k K^-1 M.t() + (same
    // transposed) - M K^-1 dK_k K^-1 M.t(); both the quadratic form x'(.)x
    // and the trace of Rinv*(.) reduce to O(n*k) / O(n*k^2) work per theta_k
    // via the identities below (derived directly in this function's body).
    //
    // extra parameter (index d of gamma):
    //   Nugget:        dRhat/da      = Ct - I = U0*U0.t() + diag(D0 - 1)
    //   Heterogeneous: dRhat/dsigma2 = -diag(noise)/sigma2^2
    // so x' dRhat x and tr(Rinv dRhat) only need U0.t()*x, U0.t()*Rinv*U0
    // and diag(Rinv).
    grad_out->set_size(_gamma.n_elem);

    const arma::vec& x = Rinve;

    const arma::mat Kinv = LinearAlgebra::inv_sympd(L_ss);           // R_ss^-1
    const arma::mat W = Kinv * R_ns.t();                             // k x n = R_ss^-1 * R_ns.t()
    const arma::mat RM = Rf.solve(R_ns);                             // n x k = Rinv * R_ns
    const arma::vec& diagRinv = Rf.diag_inv;

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
      const double term1_k = raw1 - diagcorr1;  // x' dCt x

      const arma::mat RMt_dMk = RM.t() * dM_k;  // k x k
      const double raw2 = 2.0 * arma::trace(Kinv * RMt_dMk) - arma::trace(Kinv * dK_k * Kinv * RMtM);
      const double diagcorr2 = arma::dot(diagRinv, 2.0 * rowdot - quad);
      const double term2_k = raw2 - diagcorr2;  // tr(Rinv dCt)

      (*grad_out)(kk) = 0.5 * c_scale * (term1_k / tv - term2_k);
    }

    if (_gamma.n_elem > d) {
      double g = 0.0;  // 0 when the extra parameter is fixed (as the exact gradient)
      if (extra_free && m_noise_model == NoiseModel::Nugget) {
        const arma::vec U0tx = U0.t() * x;
        const arma::mat RinvU0 = Rf.solve(U0);  // n x k
        const double xdRx = arma::dot(U0tx, U0tx) + arma::dot(x % x, D0 - 1.0);
        const double trRdR = arma::accu(U0 % RinvU0) + arma::dot(diagRinv, D0 - 1.0);
        g = -0.5 * (n * dtv / tv + trRdR - xdRx / tv - SSE * dtv / (tv * tv));
      } else if (extra_free && m_noise_model == NoiseModel::Heterogeneous) {
        const double s2sq = extra * extra;
        const double xdRx = -arma::dot(m_noise, x % x) / s2sq;
        const double trRdR = -arma::dot(m_noise, diagRinv) / s2sq;
        g = -0.5 * (n * dtv / tv + trRdR - xdRx / tv - SSE * dtv / (tv * tv));
      }
      (*grad_out)(d) = g;
    }
  }

  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logLikelihoodNystromFun(const arma::vec& gamma,
                                                                                 bool return_grad) {
  if (m_nystrom_k == 0)
    throw std::runtime_error("logLikelihoodNystromFun: model was not fitted with method=\"nystrom(k)\"");
  if (gamma.n_elem != m_X.n_cols && gamma.n_elem != gamma_dim())
    throw std::invalid_argument("logLikelihoodNystromFun: expected " + std::to_string(m_X.n_cols) + " (theta) or "
                                + std::to_string(gamma_dim()) + " (theta, alpha | sigma2) values, got "
                                + std::to_string(gamma.n_elem));
  const arma::vec gamma_n = gamma_from_original(gamma);
  arma::vec grad;
  if (return_grad) {
    const double ll = _logLikelihoodNystrom(gamma_n, &grad) + ll_offset();
    grad_to_original(grad);
    return {ll, grad};
  }
  return {_logLikelihoodNystrom(gamma_n) + ll_offset(), grad};
}

void Kriging::commit_nystrom(const arma::vec& gamma) {
  const arma::uword d = m_X.n_cols;
  m_theta = gamma.head(d);
  arma::vec beta_v;
  double tv = -1;
  arma::mat U_v;
  arma::vec D_v;
  _logLikelihoodNystrom(gamma, nullptr, &beta_v, &tv, &U_v, &D_v);
  if (m_est_beta)
    m_beta = beta_v;
  if (m_noise_model == NoiseModel::Nugget) {
    // same conventions as the exact Nugget BFGS commit in fit()
    if (!m_est_sigma2 && !m_est_nugget)
      m_alpha = m_sigma2 / (m_sigma2 + m_nugget);  // alpha forced by the likelihood
    else
      m_alpha = (gamma.n_elem > d) ? gamma.at(d) : m_alpha;
    if (m_est_sigma2) {
      if (m_est_nugget) {
        m_sigma2 = m_alpha * tv;  // tv = SSE/n (concentrated total variance)
        m_nugget = m_sigma2 / m_alpha - m_sigma2;
      } else {
        m_sigma2 = m_nugget * m_alpha / (1.0 - m_alpha);
      }
    } else if (m_est_nugget) {
      m_nugget = m_sigma2 * (1.0 - m_alpha) / m_alpha;
    }
  } else if (m_noise_model == NoiseModel::Heterogeneous) {
    // unconcentrated: sigma2 is the optimized parameter
    if (m_est_sigma2 && gamma.n_elem > d)
      m_sigma2 = gamma.at(d);
  } else if (m_est_sigma2) {
    m_sigma2 = tv;
  }
  m_nystrom_U = std::move(U_v);
  m_nystrom_D = std::move(D_v);
  m_nystrom_light = true;
  m_is_empty = true;  // no committed factorization: predict routes to predictNystrom
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
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (m_nystrom_U.is_empty())
    throw std::runtime_error("predictNystrom: model was not fitted with method=\"nystrom(k)\"");
  const arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::invalid_argument("predictNystrom: X_n has wrong dimension: " + std::to_string(X_n.n_cols)
                                + " instead of " + std::to_string(d));

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);

  // Same noise-model semantics as the exact predict (predict_impl): with a
  // nugget, the cross-correlations are alpha*C, except at a prediction point
  // coinciding with an observation (1: the observation itself), and the
  // variance scale is sigma2/alpha = sigma2 + nugget; with heterogeneous
  // noise, the cross-correlations are C and the variance scale sigma2 (the
  // noise only enters through the committed factors of Rhat).
  const bool nugget = (m_noise_model == NoiseModel::Nugget);
  const double r_factor = nugget ? m_alpha : 1.0;
  const double var_scale = nugget ? m_sigma2 / m_alpha : m_sigma2;

  arma::mat R_on = arma::mat(m_X.n_rows, X_n.n_rows, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, m_X.t(), Xn_n.t(), m_theta, _Cov, r_factor);
  if (nugget)
    for (arma::uword j = 0; j < Xn_n.n_rows; ++j)
      for (arma::uword i = 0; i < m_X.n_rows; ++i)
        if (arma::approx_equal(m_X.row(i), Xn_n.row(j), "absdiff", arma::datum::eps))
          R_on.at(i, j) = 1.0;

  const arma::mat RinvR_on = NystromChol(m_nystrom_U, m_nystrom_D, m_nystrom_landmarks).solve(R_on);  // n_o x n_n

  arma::vec mean = F_n * m_beta + RinvR_on.t() * (m_y - m_F * m_beta);
  mean = mean * m_scaleY + m_centerY;

  arma::vec stdev;
  if (return_stdev) {
    const arma::vec quad = arma::sum(R_on % RinvR_on, 0).t();  // r_j' Rinv r_j per prediction point j
    // LMP fits rescale sigma2 by (n-p)/(n-p-2), like predict() / predictVecchia()
    const double n_o = static_cast<double>(m_X.n_rows);
    const double lmp_scale = (m_objective == "LMP") ? (n_o - m_F.n_cols) / (n_o - m_F.n_cols - 2.0) : 1.0;
    stdev = arma::sqrt(arma::clamp(var_scale * lmp_scale * (1.0 - quad), 0.0, arma::datum::inf));
    stdev *= m_scaleY;
  }
  return {mean, stdev};
}

LIBKRIGING_EXPORT arma::mat Kriging::simulateNystrom(int nsim, int seed, const arma::mat& X_n) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (m_nystrom_U.is_empty())
    throw std::runtime_error("simulateNystrom: model was not fitted with method=\"nystrom(k)\"");
  const arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::invalid_argument("simulateNystrom: X_n has wrong dimension: " + std::to_string(X_n.n_cols)
                                + " instead of " + std::to_string(d));

  arma::mat Xn_n = X_n;
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  const arma::uword n_n = X_n.n_rows;

  // Same noise-model semantics as the exact simulate (with_nugget=false for
  // a nugget, no added noise for heterogeneous noise): the latent process,
  // with alpha-scaled correlations and variance scale sigma2/alpha for a
  // nugget (as simulate_impl's Sigma_divisor = alpha), sigma2 otherwise.
  const bool nugget = (m_noise_model == NoiseModel::Nugget);
  const double r_factor = nugget ? m_alpha : 1.0;
  const double var_scale = nugget ? m_sigma2 / m_alpha : m_sigma2;

  arma::mat R_on = arma::mat(m_X.n_rows, n_n, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_on, m_X.t(), Xn_n.t(), m_theta, _Cov, r_factor);

  const arma::mat RinvR_on = NystromChol(m_nystrom_U, m_nystrom_D, m_nystrom_landmarks).solve(R_on);  // n_o x n_n

  const arma::vec yhat_n = F_n * m_beta + RinvR_on.t() * (m_y - m_F * m_beta);

  // Joint (simple-kriging, beta treated as known -- like predictNystrom)
  // covariance among the n_n SIMULATION points: dense, but only n_n x n_n
  // (n_n is expected to be small; it is m_X, not X_n, that can be large).
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  LinearAlgebra::covMat_sym_X(&R_nn, Xn_n.t(), m_theta, _Cov, r_factor, arma::vec());
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
  y_n += L_sigma * Random::randn_mat(n_n, nsim) * std::sqrt(var_scale);

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
  // a heterogeneous-noise model never reaches here: update(y_u, X_u) throws
  // (the noise of the new observations is needed: update(y_u, noise_u, X_u)
  // refits on the joined data, as for the exact heterogeneous-noise model)

  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;
  const arma::vec yn_u = (y_u - m_centerY) / m_scaleY;

  // m_nystrom_landmarks holds row-indices into m_X; appending rows here
  // (never reordering/removing) keeps them valid without any adjustment.
  m_X = arma::join_cols(m_X, Xn_u);
  m_y = arma::join_cols(m_y, yn_u);
  m_F = Trend::regressionModelMatrix(m_regmodel, m_X);

  arma::vec gamma = current_gamma();
  if (refit && m_optim != "none") {
    // Warm restart: single BFGS from the current theta (and alpha for a
    // nugget), over the SAME (fixed) landmark set -- landmarks are only
    // re-picked by a full fit(), which is what keeps this
    // O((n_old+n_new)*k^2) rather than paying the O(n*k) landmark-ranking
    // pass (and losing the warm start) again.
    const FitOfn fit_ofn = make_fit_objective(internal_objective());
    const arma::uword d = m_X.n_cols;
    const arma::uword gd = gamma_dim();

    // theta bounds from the per-dimension range of the extended data --
    // O(n*d), NOT Optim::theta_bounds's variogram-slope heuristic, which
    // needs the O(n^2) dX cube this update path is built to avoid.
    const arma::vec maxdX_local = arma::trans(arma::max(m_X, 0) - arma::min(m_X, 0));
    arma::vec gamma_lower = gamma;
    arma::vec gamma_upper = gamma;
    if (m_est_theta) {  // else fixed theta: pinned at lower == upper
      gamma_lower.head(d) = arma::min(m_theta, Optim::theta_lower_factor * maxdX_local);
      gamma_upper.head(d) = arma::max(m_theta, Optim::theta_upper_factor * maxdX_local);
    }
    if (m_noise_model == NoiseModel::Nugget) {
      gamma_lower.at(d) = std::min(gamma.at(d), nugget_alpha_lower);
      gamma_upper.at(d) = 1.0;
    }

    arma::vec gamma_start = gamma;
    if (Optim::reparametrize) {
      if (m_noise_model == NoiseModel::Nugget) {
        gamma_start = nugget_reparam_to(gamma_start);
        gamma_lower = nugget_reparam_to(gamma_lower);
        gamma_upper = nugget_reparam_to(gamma_upper);
      } else {
        gamma_start = Optim::reparam_to(gamma_start);
        gamma_lower = Optim::reparam_to(gamma_lower);
        gamma_upper = Optim::reparam_to(gamma_upper);
      }
    }
    // L-BFGS-B does not project its starting point: keep it inside the bounds
    gamma_lower = arma::min(gamma_lower, gamma_start);
    gamma_upper = arma::max(gamma_upper, gamma_start);

    lbfgsb::Optimizer optimizer{static_cast<unsigned int>(gd)};
    optimizer.iprint = Optim::log_level - 2;
    optimizer.max_iter = Optim::max_iteration;
    const double loo_scale = (m_objective == "LOO") ? static_cast<double>(m_X.n_rows * m_X.n_rows) : 1.0;
    optimizer.pgtol = Optim::gradient_tolerance / loo_scale;
    optimizer.factr = Optim::objective_rel_tolerance / 1E-13 / loo_scale;
    const arma::ivec bounds_type{gd, arma::fill::value(2)};

    optimizer.minimize([&fit_ofn](const arma::vec& vals_inp,
                                  arma::vec& grad_out) -> double { return fit_ofn(vals_inp, &grad_out, nullptr); },
                       gamma_start,
                       gamma_lower.memptr(),
                       gamma_upper.memptr(),
                       bounds_type.memptr());

    if (!Optim::reparametrize)
      gamma = gamma_start;
    else if (m_noise_model == NoiseModel::Nugget)
      gamma = nugget_reparam_from(gamma_start);
    else
      gamma = Optim::reparam_from(gamma_start);
  }

  // commit at gamma (beta / sigma2 [/ nugget] / U / D profiled by the Nystrom
  // likelihood over the extended data; sigma2 = SSE / (n - p) for a LMP refit)
  const arma::uword d_all = m_X.n_cols;
  m_theta = gamma.head(d_all);
  commit_approx(/*lmp_dof=*/refit && m_optim != "none", gamma.n_elem > d_all ? gamma.at(d_all) : 0.0);
}

void Kriging::update_vecchia_light(const arma::vec& y_u, const arma::mat& X_u, bool refit) {
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

  m_X = arma::join_cols(m_X, Xn_u);
  m_y = arma::join_cols(m_y, yn_u);
  m_F = Trend::regressionModelMatrix(m_regmodel, m_X);
  make_vecchia_sets();  // ordering / neighbor sets over the extended design

  const bool is_nugget = (m_noise_model == NoiseModel::Nugget);
  double extra = m_alpha;  // Nugget: alpha (ignored for None)
  if (refit && m_optim != "none" && m_est_theta) {
    // Warm restart: single BFGS on the LLVecchia objective from the current
    // theta (and alpha for Nugget); bounds from the per-dimension range of
    // the extended data (O(n*d), as update_nystrom: no O(n^2)
    // pairwise-difference cube).
    const FitOfn fit_ofn = make_fit_objective(internal_objective());
    const arma::uword d = m_X.n_cols;
    const arma::uword gd = gamma_dim();
    const arma::vec maxdX_local = arma::trans(arma::max(m_X, 0) - arma::min(m_X, 0));
    arma::vec theta_lower = arma::min(m_theta, Optim::theta_lower_factor * maxdX_local);
    arma::vec theta_upper = arma::max(m_theta, Optim::theta_upper_factor * maxdX_local);

    arma::vec gamma_start(gd), gamma_lower(gd), gamma_upper(gd);
    gamma_start.head(d) = m_theta;
    gamma_lower.head(d) = theta_lower;
    gamma_upper.head(d) = theta_upper;
    if (is_nugget) {
      gamma_start.at(d) = std::min(std::max(m_alpha, nugget_alpha_lower), 1.0);
      gamma_lower.at(d) = nugget_alpha_lower;
      gamma_upper.at(d) = 1.0;
    }
    if (Optim::reparametrize) {
      if (is_nugget) {
        gamma_start = nugget_reparam_to(gamma_start);
        gamma_lower = nugget_reparam_to(gamma_lower);
        gamma_upper = nugget_reparam_to(gamma_upper);
      } else {
        gamma_start = Optim::reparam_to(gamma_start);
        gamma_lower = Optim::reparam_to(gamma_lower);
        gamma_upper = Optim::reparam_to(gamma_upper);
      }
    }

    lbfgsb::Optimizer optimizer{static_cast<unsigned int>(gd)};
    optimizer.iprint = Optim::log_level - 2;
    optimizer.max_iter = Optim::max_iteration;
    const double loo_scale = (m_objective == "LOO") ? static_cast<double>(m_X.n_rows * m_X.n_rows) : 1.0;
    optimizer.pgtol = Optim::gradient_tolerance / loo_scale;
    optimizer.factr = Optim::objective_rel_tolerance / 1E-13 / loo_scale;
    const arma::ivec bounds_type{gd, arma::fill::value(2)};

    optimizer.minimize([&fit_ofn](const arma::vec& vals_inp,
                                  arma::vec& grad_out) -> double { return fit_ofn(vals_inp, &grad_out, nullptr); },
                       gamma_start,
                       gamma_lower.memptr(),
                       gamma_upper.memptr(),
                       bounds_type.memptr());

    arma::vec gamma = gamma_start;
    if (Optim::reparametrize)
      gamma = is_nugget ? nugget_reparam_from(gamma_start) : Optim::reparam_from(gamma_start);
    m_theta = gamma.head(d);
    if (is_nugget)
      extra = gamma.at(d);
  }

  // light commit, as in fit(): beta / sigma2 (/ nugget) profiled by the Vecchia likelihood
  commit_approx(/*lmp_dof=*/refit && m_optim != "none" && m_est_theta, extra);
  // drop a legacy exact factorization (now stale): the model is a light Vecchia fit
  m_T.reset();
  m_R.reset();
  m_M.reset();
  m_circ.reset();
  m_star.reset();
  m_Rinv.reset();
  m_z.reset();
  m_dX.reset();
}

// =============================================================================
// LOO and LMP objectives with an approximate method (internal objectives
// "LOOVecchia(m)", "LMPVecchia(m)", "LOONystrom(k)", "LMPNystrom(k)").
//
// Both criteria only need the approximate correlation PRECISION P ~= R^-1,
// its diagonal, log det R, and, for the gradient, the bilinear forms
// X' dP/dtheta_k Z, the weighted diagonal sum_i w_i (dP/dtheta_k)_ii and
// d log det R / dtheta_k. They are written once (approx_loo / approx_lmp
// below) on top of two precision "back-ends":
//
//  * Vecchia: P = B' D^-1 B, B unit lower-triangular in the maxmin order with
//    row t = e_{o(t)} - a_t' e_{N(t)} (a_t = R_NN^-1 r_N), D = diag(v_t),
//    v_t = 1 - r_N' a_t. log det R = sum log v_t, and dP = dB' D^-1 B +
//    B' D^-1 dB - B' D^-1 dD D^-1 B, with da_t, dv_t from the same m x m
//    solves as the LLVecchia gradient. Everything is O(n m) per vector
//    besides the O(n m^3) block solves.
//  * Nystrom: P = Rhat^-1, Rhat = U U' + diag(D), U U' = M K^-1 M' (M = R_ns,
//    K = R_ss, fixed landmarks). P X = D^-1 X - Y (Y' X) with Y = D^-1 U
//    L_c^-T, L_c L_c' = I + U' D^-1 U (Woodbury), log det Rhat by the
//    determinant lemma. dP = -P dRhat P, and dRhat = dM W' + W dM' - W dK W'
//    - diag(delta * unclamped) (W = M K^-1, delta = diag of the low-rank
//    part: Rhat_ii == 1 while D_i is not clamped), so every needed quantity
//    is O(n k^2) per theta component.
//
// See docs/math/LOO.md and docs/math/LMP.md for the formulas.
// =============================================================================

namespace {

struct VecchiaPrecision {
  arma::uword n = 0;
  arma::uword d = 0;
  const arma::uvec* order = nullptr;
  const std::vector<arma::uvec>* nb = nullptr;
  std::vector<arma::vec> a;   // per ordered point t: kriging weights on N(t)
  arma::vec v;                // conditional variances
  std::vector<arma::mat> da;  // per t: (m_t x d) d a_t / d theta
  arma::mat dv;               // n x d

  // B X (rows indexed by the ordered position t)
  arma::mat B(const arma::mat& X) const {
    arma::mat out(n, X.n_cols);
    for (arma::uword t = 0; t < n; ++t) {
      const arma::uvec& N = (*nb)[t];
      out.row(t) = X.row((*order)(t));
      if (N.n_elem > 0)
        out.row(t) -= a[t].t() * X.rows(N);
    }
    return out;
  }
  // B' Z (Z rows indexed by the ordered position t)
  arma::mat Bt(const arma::mat& Z) const {
    arma::mat out(n, Z.n_cols, arma::fill::zeros);
    for (arma::uword t = 0; t < n; ++t) {
      const arma::uvec& N = (*nb)[t];
      out.row((*order)(t)) += Z.row(t);
      if (N.n_elem > 0)
        out.rows(N) -= a[t] * Z.row(t);
    }
    return out;
  }
  // (dB/dtheta_k) X
  arma::mat dB(arma::uword k, const arma::mat& X) const {
    arma::mat out(n, X.n_cols, arma::fill::zeros);
    for (arma::uword t = 0; t < n; ++t) {
      const arma::uvec& N = (*nb)[t];
      if (N.n_elem > 0)
        out.row(t) = -da[t].col(k).t() * X.rows(N);
    }
    return out;
  }
  arma::mat P(const arma::mat& X) const {
    arma::mat BX = B(X);
    BX.each_col() /= v;
    return Bt(BX);
  }
  arma::vec diagP() const {
    arma::vec dg(n, arma::fill::zeros);
    for (arma::uword t = 0; t < n; ++t) {
      const arma::uvec& N = (*nb)[t];
      dg((*order)(t)) += 1.0 / v(t);
      for (arma::uword j = 0; j < N.n_elem; ++j)
        dg(N(j)) += a[t](j) * a[t](j) / v(t);
    }
    return dg;
  }
  double logdetR() const { return arma::accu(arma::log(v)); }
  // X' dP_k Z
  arma::mat bil(arma::uword k, const arma::mat& X, const arma::mat& Z) const {
    const arma::mat BX = B(X);
    const arma::mat BZ = B(Z);
    const arma::mat dBX = dB(k, X);
    const arma::mat dBZ = dB(k, Z);
    const arma::vec iv = 1.0 / v;
    const arma::vec dvv = dv.col(k) / (v % v);
    return dBX.t() * (BZ.each_col() % iv) + (BX.each_col() % iv).t() * dBZ - BX.t() * (BZ.each_col() % dvv);
  }
  // sum_i w_i (dP_k)_ii, with (dP)_ii = sum_t 2 B_ti dB_ti / v_t - B_ti^2 dv_t / v_t^2
  double diag_term(arma::uword k, const arma::vec& w) const {
    double s = 0;
    for (arma::uword t = 0; t < n; ++t) {
      const arma::uvec& N = (*nb)[t];
      const double dvt = dv(t, k) / (v(t) * v(t));
      s -= w((*order)(t)) * dvt;
      for (arma::uword j = 0; j < N.n_elem; ++j) {
        const double aj = a[t](j);
        s += w(N(j)) * (2.0 * aj * da[t](j, k) / v(t) - aj * aj * dvt);
      }
    }
    return s;
  }
  double dlogdetR(arma::uword k) const { return arma::accu(dv.col(k) / v); }
};

VecchiaPrecision build_vecchia_precision(const arma::mat& X,
                                         const arma::uvec& order,
                                         const std::vector<arma::uvec>& nb,
                                         const arma::vec& theta,
                                         const std::function<double(const arma::vec&, const arma::vec&)>& Cov,
                                         const std::function<arma::vec(const arma::vec&, const arma::vec&)>& DlnCov,
                                         bool with_grad) {
  constexpr double v_floor = 1e-15;  // as in _logLikelihoodVecchia
  VecchiaPrecision vp;
  vp.n = X.n_rows;
  vp.d = X.n_cols;
  vp.order = &order;
  vp.nb = &nb;
  vp.a.resize(vp.n);
  vp.v.set_size(vp.n);
  if (with_grad) {
    vp.da.resize(vp.n);
    vp.dv.zeros(vp.n, vp.d);
  }
  const arma::uword d = vp.d;
  for (arma::uword t = 0; t < vp.n; ++t) {
    const arma::uword oi = order(t);
    const arma::uvec& Ni = nb[t];
    const arma::uword mi = Ni.n_elem;
    if (mi == 0) {
      vp.v(t) = 1.0;
      continue;
    }
    arma::mat RN(mi, mi, arma::fill::ones);
    arma::vec r(mi);
    arma::cube dRN;
    arma::mat dr;
    if (with_grad) {
      dRN.zeros(mi, mi, d);
      dr.zeros(mi, d);
    }
    for (arma::uword i = 0; i < mi; ++i) {
      for (arma::uword j = i + 1; j < mi; ++j) {
        const arma::vec dx = (X.row(Ni(i)) - X.row(Ni(j))).t();
        const double c = Cov(dx, theta);
        RN(i, j) = RN(j, i) = c;
        if (with_grad) {
          const arma::vec dln = DlnCov(dx, theta);
          for (arma::uword k = 0; k < d; ++k)
            dRN(i, j, k) = dRN(j, i, k) = c * dln(k);
        }
      }
      const arma::vec dx = (X.row(Ni(i)) - X.row(oi)).t();
      const double c = Cov(dx, theta);
      r(i) = c;
      if (with_grad) {
        const arma::vec dln = DlnCov(dx, theta);
        for (arma::uword k = 0; k < d; ++k)
          dr(i, k) = c * dln(k);
      }
    }
    RN.diag() += LinearAlgebra::num_nugget;
    const arma::mat LN = LinearAlgebra::safe_chol_lower(RN);
    const arma::vec a_vec = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), r));
    vp.a[t] = a_vec;
    vp.v(t) = std::max(1.0 - arma::dot(r, a_vec), v_floor);
    if (with_grad) {
      arma::mat dA(mi, d);
      for (arma::uword k = 0; k < d; ++k) {
        const arma::vec rhs = dr.col(k) - dRN.slice(k) * a_vec;
        dA.col(k) = arma::solve(arma::trimatu(LN.t()), arma::solve(arma::trimatl(LN), rhs));
        vp.dv(t, k) = -(arma::dot(dr.col(k), a_vec) + arma::dot(r, dA.col(k)));
      }
      vp.da[t] = std::move(dA);
    }
  }
  return vp;
}

struct NystromPrecision {
  arma::uword n = 0;
  arma::uword d = 0;
  NystromChol F;  // P = Rhat^-1
  // gradient data
  arma::mat Wt;                 // n x k: M K^-1
  arma::mat PW;                 // n x k: P W
  std::vector<arma::mat> dM;    // n x k per theta component
  std::vector<arma::mat> dK;    // k x k per theta component
  std::vector<arma::vec> dlow;  // n: diag of the low-rank part of dRhat
  std::vector<arma::vec> ddg;   // n: actual d Rhat_ii (0 where D is not clamped)

  arma::mat P(const arma::mat& X) const { return F.solve(X); }
  const arma::vec& diagP() const { return F.diag_inv; }
  double logdetR() const { return F.logdet; }
  // X' dRhat_k Z
  arma::mat bilR(arma::uword k, const arma::mat& X, const arma::mat& Z) const {
    const arma::mat WX = Wt.t() * X;
    const arma::mat WZ = Wt.t() * Z;
    const arma::mat dMX = dM[k].t() * X;
    const arma::mat dMZ = dM[k].t() * Z;
    const arma::vec corr = dlow[k] - ddg[k];  // diagonal removed from the low-rank part
    return dMX.t() * WZ + WX.t() * dMZ - WX.t() * dK[k] * WZ - X.t() * (Z.each_col() % corr);
  }
  // X' dP_k Z = -(P X)' dRhat_k (P Z)
  arma::mat bil(arma::uword k, const arma::mat& X, const arma::mat& Z) const { return -bilR(k, P(X), P(Z)); }
  // The two quantities below are written with P applied to the n x k
  // matrices W and dM (P W, P dM, like every P X here) rather than expanded
  // through a "diagonal - low rank" form of P: at the landmarks D_i is
  // ~num_nugget, and the expanded form is a difference of O(1/num_nugget^2)
  // terms (no correct digit left), whereas P itself stays moderate there.
  //
  // sum_i w_i (dP_k)_ii = -sum_i w_i p_i' dRhat_k p_i (p_i = P e_i)
  //   = -sum_i w_i [2 (dM' p_i).(W' p_i) - (W' p_i)' dK (W' p_i)] + sum_j corr_j (P diag(w) P)_jj
  // only needed where corr_j != 0, i.e. where D_j is not clamped: never at a
  // landmark row, so j is among the other rows N, where (NystromChol)
  // P_NN = diag(DNinv) - YN YN' and P_NS = PNS. Hence, for j in N,
  // (P diag(w) P)_jj = w_j P_jj^2 + y_j' (sum_{i in N} w_i y_i y_i') y_j - w_j |y_j|^4
  //                    + sum_{i in S} w_i PNS_ji^2   (y_j: row j of YN).
  double diag_term(arma::uword k, const arma::vec& w) const {
    const arma::mat PdM = P(dM[k]);
    const double t_low = 2.0 * arma::dot(w, arma::sum(PdM % PW, 1)) - arma::dot(w, arma::sum((PW * dK[k]) % PW, 1));
    const arma::vec corrN = arma::vec(dlow[k] - ddg[k]).elem(F.N);
    const arma::uvec J = arma::find(corrN != 0.0);  // positions in N
    double t_diag = 0;
    if (J.n_elem > 0) {
      const arma::vec wN = w.elem(F.N);
      const arma::mat Omega = F.YN.t() * (F.YN.each_col() % wN);
      const arma::mat YJ = F.YN.rows(J);
      const arma::vec yy = arma::sum(arma::square(YJ), 1);
      const arma::vec PJJ = F.DNinv(J) - yy;
      const arma::vec wJ = wN(J);
      const arma::vec pwp = wJ % arma::square(PJJ) + arma::sum((YJ * Omega) % YJ, 1) - wJ % arma::square(yy)
                            + arma::square(F.PNS.rows(J)) * w.elem(F.S);
      t_diag = arma::dot(corrN(J), pwp);
    }
    return -t_low + t_diag;
  }
  // d log det Rhat = tr(P dRhat) = 2 tr(W' P dM) - tr(dK W' P W) - sum_j corr_j P_jj
  double dlogdetR(arma::uword k) const {
    const arma::vec corr = dlow[k] - ddg[k];
    return 2.0 * arma::accu(Wt % P(dM[k])) - arma::accu(dK[k] % (Wt.t() * PW)) - arma::dot(corr, diagP());
  }
};

NystromPrecision build_nystrom_precision(const arma::mat& X,
                                         const arma::uvec& landmarks,
                                         const arma::vec& theta,
                                         const std::function<double(const arma::vec&, const arma::vec&)>& Cov,
                                         const std::function<arma::vec(const arma::vec&, const arma::vec&)>& DlnCov,
                                         bool with_grad) {
  NystromPrecision np;
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;
  const arma::uword kL = landmarks.n_elem;
  np.n = n;
  np.d = d;
  const arma::mat X_land = X.rows(landmarks);
  // same factorization as _logLikelihoodNystrom
  arma::mat R_ss(kL, kL, arma::fill::none);
  LinearAlgebra::covMat_sym_X(&R_ss, X_land.t(), theta, Cov, /*factor=*/1.0, KrigingImpl::ones);
  arma::mat R_ns(n, kL, arma::fill::none);
  LinearAlgebra::covMat_rect(&R_ns, X.t(), X_land.t(), theta, Cov, /*factor=*/1.0);
  R_ss.diag() += LinearAlgebra::num_nugget;
  const arma::mat L_ss = arma::chol(R_ss, "lower");
  const arma::mat U = arma::trans(LinearAlgebra::solve_lower(L_ss, R_ns.t()));
  const arma::vec D = nystrom_residual_diag(U, landmarks);
  np.F = NystromChol(U, D, landmarks, /*with_inverse=*/true);

  if (with_grad) {
    const arma::mat Kinv = LinearAlgebra::inv_sympd(L_ss);
    np.Wt = R_ns * Kinv;
    np.PW = np.P(np.Wt);
    const arma::uvec clamped = arma::find(D <= LinearAlgebra::num_nugget);
    np.dM.resize(d);
    np.dK.resize(d);
    np.dlow.resize(d);
    np.ddg.resize(d);
    for (arma::uword k = 0; k < d; ++k) {
      np.dM[k].set_size(n, kL);
      np.dK[k].zeros(kL, kL);
    }
    for (arma::uword i = 0; i < n; ++i)
      for (arma::uword j = 0; j < kL; ++j) {
        const arma::vec dln = DlnCov((X.row(i) - X_land.row(j)).t(), theta);
        for (arma::uword k = 0; k < d; ++k)
          np.dM[k](i, j) = R_ns(i, j) * dln(k);
      }
    for (arma::uword a = 0; a < kL; ++a)
      for (arma::uword b = a + 1; b < kL; ++b) {
        const arma::vec dln = DlnCov((X_land.row(a) - X_land.row(b)).t(), theta);
        for (arma::uword k = 0; k < d; ++k)
          np.dK[k](a, b) = np.dK[k](b, a) = R_ss(a, b) * dln(k);
      }
    for (arma::uword k = 0; k < d; ++k) {
      np.dlow[k] = 2.0 * arma::sum(np.dM[k] % np.Wt, 1) - arma::sum((np.Wt * np.dK[k]) % np.Wt, 1);
      np.ddg[k].zeros(n);
      np.ddg[k](clamped) = np.dlow[k](clamped);
    }
  }
  return np;
}

// Leave-one-out under the approximate joint law (precision P):
// Q = P - P F (F' P F)^-1 F' P (or Q = P with a known beta), e_i = (Q y)_i / Q_ii,
// LOO = |e|^2 / n. Gradient: with r = y - F beta, H = I - F A^-1 F' P,
// c = e / diag(Q), w = c o e (dQ = H' dP H),
//   d LOO = 2/n [ (H c)' dP r - tr(dP H diag(w) H') ]
// and H diag(w) H' = diag(w) + F C1' + C1 F' + F C2 F' (C1 = -diag(w) G A^-1,
// C2 = A^-1 G' diag(w) G A^-1, G = P F).
template <class Prec>
double approx_loo(const Prec& P,
                  const arma::vec& y,
                  const arma::mat& F,
                  bool est_beta,
                  const arma::vec& beta_fixed,
                  arma::vec* grad_out,
                  arma::mat* yhat_out) {
  const arma::uword n = y.n_elem;
  arma::vec r;
  arma::vec diagQ = P.diagP();
  arma::mat G, Ainv;
  if (est_beta) {
    G = P.P(F);
    const arma::mat A = arma::symmatu(F.t() * G);
    Ainv = LinearAlgebra::inv_sympd(LinearAlgebra::safe_chol_lower(A));
    const arma::vec beta = Ainv * (G.t() * y);
    r = y - F * beta;
    diagQ -= arma::sum((G * Ainv) % G, 1);
  } else {
    r = y - F * beta_fixed;
  }
  const arma::vec q = P.P(r);
  const arma::vec e = q / diagQ;
  const double loo = arma::dot(e, e) / n;
  if (yhat_out != nullptr) {
    yhat_out->set_size(n, 2);
    yhat_out->col(0) = y - e;
    yhat_out->col(1) = arma::sqrt(1.0 / diagQ);
  }
  if (grad_out != nullptr) {
    const arma::uword d = grad_out->n_elem;
    const arma::vec c = e / diagQ;
    const arma::vec w = c % e;
    arma::vec Hc = c;
    arma::mat C1, C2;
    if (est_beta) {
      Hc -= F * (Ainv * (G.t() * c));
      C1 = -(G.each_col() % w) * Ainv;
      C2 = Ainv * (G.t() * (G.each_col() % w)) * Ainv;
    }
    for (arma::uword k = 0; k < d; ++k) {
      double g = arma::as_scalar(P.bil(k, Hc, r)) - P.diag_term(k, w);
      if (est_beta)
        g -= 2.0 * arma::trace(P.bil(k, F, C1)) + arma::accu(C2 % P.bil(k, F, F));
      (*grad_out)(k) = 2.0 * g / n;
    }
  }
  return loo;
}

// Log-marginal posterior (docs/math/LMP.md) with the approximate R:
// -1/2 log det R - 1/2 log det(F' P F) - (n-p)/2 log(S2) + jointly robust prior,
// S2 = (y - F beta)' P (y - F beta) (= sigma2 (n-p) when sigma2 is estimated).
template <class Prec>
double approx_lmp(const Prec& P,
                  const arma::vec& y,
                  const arma::mat& F,
                  const arma::mat& X,
                  const arma::vec& theta,
                  bool est_sigma2,
                  double sigma2_fixed,
                  arma::vec* grad_out) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;
  const arma::uword p = F.n_cols;
  const arma::mat G = P.P(F);
  const arma::mat A = arma::symmatu(F.t() * G);
  const arma::mat LA = LinearAlgebra::safe_chol_lower(A);
  const arma::mat Ainv = LinearAlgebra::inv_sympd(LA);
  const arma::vec beta = Ainv * (G.t() * y);
  const arma::vec r = y - F * beta;
  const double S2 = arma::dot(r, P.P(r));
  const double sigma2 = est_sigma2 ? S2 / (n - p) : sigma2_fixed;
  const double log_S_2 = std::log(sigma2 * (n - p));
  const double log_marginal_lik = -0.5 * P.logdetR() - arma::accu(arma::log(LA.diag())) - (n - p) / 2.0 * log_S_2;

  // jointly robust prior, as in _logMargPost
  const double a = 0.2;
  const double b = 1.0 / std::pow(n, 1.0 / d) * (a + d);
  const arma::vec CL = arma::trans(arma::max(X, 0) - arma::min(X, 0)) / std::pow(n, 1.0 / d);
  const double t = arma::accu(CL / theta);
  const double log_approx_ref_prior = -b * t + a * std::log(t);

  if (grad_out != nullptr) {
    for (arma::uword k = 0; k < d; ++k) {
      double g = -0.5 * P.dlogdetR(k) - 0.5 * arma::accu(Ainv % P.bil(k, F, F));
      if (est_sigma2)
        g -= (n - p) / 2.0 * arma::as_scalar(P.bil(k, r, r)) / S2;
      (*grad_out)(k) = g;
    }
    grad_out->head(d) -= (a * CL / t - b * CL) / arma::square(theta);
  }
  return log_marginal_lik + log_approx_ref_prior;
}

}  // namespace

double Kriging::_leaveOneOutApprox(bool vecchia,
                                   const arma::vec& _theta,
                                   arma::vec* grad_out,
                                   arma::mat* yhat_out) const {
  const bool est_beta = m_est_beta || m_beta.n_elem != m_F.n_cols;
  if (grad_out != nullptr)
    grad_out->set_size(_theta.n_elem);
  if (vecchia) {
    const VecchiaPrecision P = build_vecchia_precision(
        m_X, m_vecchia_order, m_vecchia_neighbors, _theta, _Cov, _DlnCovDtheta, grad_out != nullptr);
    return approx_loo(P, m_y, m_F, est_beta, m_beta, grad_out, yhat_out);
  }
  const NystromPrecision P
      = build_nystrom_precision(m_X, m_nystrom_landmarks, _theta, _Cov, _DlnCovDtheta, grad_out != nullptr);
  return approx_loo(P, m_y, m_F, est_beta, m_beta, grad_out, yhat_out);
}

double Kriging::_logMargPostApprox(bool vecchia, const arma::vec& _theta, arma::vec* grad_out) const {
  if (grad_out != nullptr)
    grad_out->set_size(_theta.n_elem);
  if (vecchia) {
    const VecchiaPrecision P = build_vecchia_precision(
        m_X, m_vecchia_order, m_vecchia_neighbors, _theta, _Cov, _DlnCovDtheta, grad_out != nullptr);
    return approx_lmp(P, m_y, m_F, m_X, _theta, m_est_sigma2, m_sigma2, grad_out);
  }
  const NystromPrecision P
      = build_nystrom_precision(m_X, m_nystrom_landmarks, _theta, _Cov, _DlnCovDtheta, grad_out != nullptr);
  return approx_lmp(P, m_y, m_F, m_X, _theta, m_est_sigma2, m_sigma2, grad_out);
}

void Kriging::commit_approx(bool lmp_dof, double extra) {
  if (m_nystrom_k == 0) {
    // Vecchia: beta / sigma2 (/ nugget, with a noise model) profiled by the
    // Vecchia likelihood at m_theta (and extra = alpha | sigma2)
    commit_vecchia(m_theta, extra);
  } else {
    // Nystrom: beta (GLS under the approximate precision, the same for LL,
    // LOO and LMP, as in the exact commits), sigma2 (/ alpha / nugget with a
    // noise model) and the rank-k factors U / D, profiled by the Nystrom
    // likelihood at gamma = (m_theta [, extra = alpha | sigma2])
    commit_nystrom(m_noise_model == NoiseModel::None ? m_theta : arma::vec(arma::join_cols(m_theta, arma::vec{extra})));
  }
  // sigma2 = SSE / (n - p) for a LMP (re)fit, like the exact LMP commit (LOO /
  // LMP with an approximate method are restricted to noise model None)
  if (lmp_dof && m_est_sigma2 && m_objective == "LMP" && m_noise_model == NoiseModel::None)
    m_sigma2 *= static_cast<double>(m_X.n_rows) / static_cast<double>(m_X.n_rows - m_F.n_cols);
}

// Public approximate LOO / LMP: original-scale convention (as leaveOneOutFun /
// logMargPostFun): theta is read in the original scale of X, the value is the
// one of the original data (LOO * scaleY^2, LMP + lmp_offset()) and the
// gradient is chain-ruled back. The no-theta versions evaluate the internal
// implementation at the committed m_theta (no double conversion).
void Kriging::check_approx_theta(const arma::vec& theta, const char* what) const {
  if (theta.n_elem != m_X.n_cols)
    throw std::invalid_argument(std::string(what) + ": expected " + std::to_string(m_X.n_cols) + " theta values, got "
                                + std::to_string(theta.n_elem));
}

double Kriging::loo_scale2() const {
  return m_normalize ? m_scaleY * m_scaleY : 1.0;
}

void Kriging::ensure_approx_sets(bool vecchia, const char* what) {
  if (vecchia) {
    if (m_vecchia_m == 0)
      throw std::runtime_error(std::string(what) + ": model was not fitted with method=\"vecchia(m)\"");
    if (m_vecchia_order.n_elem != m_X.n_rows)
      make_vecchia_sets();  // not stored by save(): rebuilt on demand after load
  } else if (m_nystrom_k == 0) {
    throw std::runtime_error(std::string(what) + ": model was not fitted with method=\"nystrom(k)\"");
  }
}

void Kriging::check_approx_noise_free(bool loo, bool vecchia, const char* what) const {
  if (m_noise_model != NoiseModel::None)
    throw std::invalid_argument(std::string(what) + ": objective \"" + (loo ? "LOO" : "LMP") + "\" with method \""
                                + (vecchia ? "vecchia" : "nystrom")
                                + "\" is not implemented for Nugget/Heterogeneous noise models; use objective "
                                  "\"LL\" or method \"exact\"");
}

std::tuple<double, arma::vec> Kriging::approx_objective_fun(bool loo,
                                                            bool vecchia,
                                                            const arma::vec& theta,
                                                            bool return_grad,
                                                            const char* what) {
  check_approx_noise_free(loo, vecchia, what);
  ensure_approx_sets(vecchia, what);
  check_approx_theta(theta, what);
  const arma::vec theta_n = gamma_from_original(theta);
  arma::vec grad;
  double val = loo ? _leaveOneOutApprox(vecchia, theta_n, return_grad ? &grad : nullptr, nullptr)
                   : _logMargPostApprox(vecchia, theta_n, return_grad ? &grad : nullptr);
  if (loo) {
    val *= loo_scale2();
    if (return_grad)
      grad *= loo_scale2();
  } else {
    val += lmp_offset();
  }
  if (return_grad)
    grad_to_original(grad);
  return {val, grad};
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::leaveOneOutVecchiaFun(const arma::vec& theta,
                                                                               bool return_grad) {
  return approx_objective_fun(true, true, theta, return_grad, "leaveOneOutVecchiaFun");
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::leaveOneOutNystromFun(const arma::vec& theta,
                                                                               bool return_grad) {
  return approx_objective_fun(true, false, theta, return_grad, "leaveOneOutNystromFun");
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logMargPostVecchiaFun(const arma::vec& theta,
                                                                               bool return_grad) {
  return approx_objective_fun(false, true, theta, return_grad, "logMargPostVecchiaFun");
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logMargPostNystromFun(const arma::vec& theta,
                                                                               bool return_grad) {
  return approx_objective_fun(false, false, theta, return_grad, "logMargPostNystromFun");
}

LIBKRIGING_EXPORT double Kriging::leaveOneOut(const std::string& method) {
  const ApproxMethod am = parse_approx_method(method, "leaveOneOut");
  switch (am.kind) {
    case ApproxMethod::Vecchia:
      check_vecchia_m(am, m_vecchia_m, "leaveOneOut");
      check_approx_noise_free(true, true, "leaveOneOut");
      ensure_approx_sets(true, "leaveOneOutVecchiaFun");
      return _leaveOneOutApprox(true, m_theta, nullptr, nullptr) * loo_scale2();
    case ApproxMethod::Nystrom:
      check_nystrom_rank(am, m_nystrom_k, "leaveOneOut");
      check_approx_noise_free(true, false, "leaveOneOut");
      ensure_approx_sets(false, "leaveOneOutNystromFun");
      return _leaveOneOutApprox(false, m_theta, nullptr, nullptr) * loo_scale2();
    case ApproxMethod::Exact:
    default:
      return leaveOneOut();
  }
}

LIBKRIGING_EXPORT double Kriging::logMargPost(const std::string& method) {
  const ApproxMethod am = parse_approx_method(method, "logMargPost");
  switch (am.kind) {
    case ApproxMethod::Vecchia:
      check_vecchia_m(am, m_vecchia_m, "logMargPost");
      check_approx_noise_free(false, true, "logMargPost");
      ensure_approx_sets(true, "logMargPostVecchiaFun");
      return _logMargPostApprox(true, m_theta, nullptr) + lmp_offset();
    case ApproxMethod::Nystrom:
      check_nystrom_rank(am, m_nystrom_k, "logMargPost");
      check_approx_noise_free(false, false, "logMargPost");
      ensure_approx_sets(false, "logMargPostNystromFun");
      return _logMargPostApprox(false, m_theta, nullptr) + lmp_offset();
    case ApproxMethod::Exact:
    default:
      return logMargPost();
  }
}

LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::leaveOneOutVec(const arma::vec& theta,
                                                                           const std::string& method) {
  const ApproxMethod am = parse_approx_method(method, "leaveOneOutVec");
  bool vecchia = true;
  switch (am.kind) {
    case ApproxMethod::Vecchia:
      check_vecchia_m(am, m_vecchia_m, "leaveOneOutVec");
      break;
    case ApproxMethod::Nystrom:
      check_nystrom_rank(am, m_nystrom_k, "leaveOneOutVec");
      vecchia = false;
      break;
    case ApproxMethod::Exact:
    default:
      return leaveOneOutVec(theta);
  }
  check_approx_noise_free(true, vecchia, "leaveOneOutVec");
  ensure_approx_sets(vecchia, "leaveOneOutVec");
  check_approx_theta(theta, "leaveOneOutVec");
  arma::mat yhat;
  _leaveOneOutApprox(vecchia, gamma_from_original(theta), nullptr, &yhat);
  arma::vec mean = yhat.col(0);
  arma::vec stdev = yhat.col(1) * std::sqrt(m_sigma2);
  if (m_normalize) {  // back to the original scale of y, like the exact leaveOneOutVec
    mean = mean * m_scaleY + m_centerY;
    stdev *= m_scaleY;
  }
  return std::make_tuple(std::move(mean), std::move(stdev));
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
  } else if (objective.rfind("LOOVecchia", 0) == 0 || objective.rfind("LOONystrom", 0) == 0
             || objective.rfind("LMPVecchia", 0) == 0 || objective.rfind("LMPNystrom", 0) == 0) {
    // LOO / LMP under an approximate method (see _leaveOneOutApprox /
    // _logMargPostApprox); km_data is never populated, as for LLVecchia
    if (m_noise_model != NoiseModel::None)
      throw std::invalid_argument("approximate methods are not supported for Nugget/Heterogeneous noise models");
    const bool vecchia = objective.find("Vecchia") != std::string::npos;
    const bool loo = objective.rfind("LOO", 0) == 0;
    return [this, vecchia, loo](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
      const arma::vec _theta = Optim::reparametrize ? Optim::reparam_from(_gamma) : _gamma;
      arma::vec grad;
      arma::vec* g = (grad_out != nullptr) ? &grad : nullptr;
      // minimized: LOO as is, LMP negated
      const double val
          = loo ? this->_leaveOneOutApprox(vecchia, _theta, g, nullptr) : -this->_logMargPostApprox(vecchia, _theta, g);
      if (grad_out != nullptr) {
        if (!loo)
          grad = -grad;
        *grad_out = Optim::reparametrize ? Optim::reparam_from_deriv(_theta, grad) : grad;
      }
      return val;
    };
  } else if (objective.rfind("LLVecchia", 0) == 0) {
    parse_vll_m(objective);  // validate the spec early (throws on malformed)
    // km_data (the dense O(n^3) KModel) is never populated: a Vecchia fit
    // commits theta* with Vecchia-profiled beta/sigma2 only (see fit()).
    // Noise models: gamma = [theta, alpha] / [theta, sigma2], with the same
    // reparametrizations as the exact "LL" objective above.
    if (m_noise_model == NoiseModel::Nugget && Optim::reparametrize) {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const arma::vec _theta_alpha = nugget_reparam_from(_gamma);
        double vll = this->_logLikelihoodVecchia(_theta_alpha, grad_out);
        if (grad_out != nullptr)
          *grad_out = -nugget_reparam_from_deriv(_theta_alpha, *grad_out);
        return -vll;
      };
    } else if (Optim::reparametrize) {  // None: theta; Heterogeneous: [theta, sigma2]
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const arma::vec _theta = Optim::reparam_from(_gamma);
        double vll = this->_logLikelihoodVecchia(_theta, grad_out);
        if (grad_out != nullptr)
          *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
        return -vll;
      };
    } else {
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        double vll = this->_logLikelihoodVecchia(_gamma, grad_out);
        if (grad_out != nullptr)
          *grad_out = -*grad_out;
        return -vll;
      };
    }
  } else if (objective.rfind("LLNystrom", 0) == 0) {
    parse_nystrom_k(objective);  // validate the spec early (throws on malformed)
    // km_data (the dense O(n^3) KModel) is never populated, by design.
    if (m_noise_model != NoiseModel::None) {
      // d+1 gamma, same reparametrizations as the exact noise-model "LL":
      // composite (theta, alpha) one for a nugget, Optim::reparam_from on all
      // components (sigma2 as theta) for heterogeneous noise
      const bool nugget = (m_noise_model == NoiseModel::Nugget);
      if (Optim::reparametrize) {
        return [this, nugget](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
          const arma::vec _gamma_nat = nugget ? nugget_reparam_from(_gamma) : Optim::reparam_from(_gamma);
          arma::vec grad;
          const double ll = this->_logLikelihoodNystrom(_gamma_nat, grad_out != nullptr ? &grad : nullptr);
          if (grad_out != nullptr)
            *grad_out = nugget ? arma::vec(-nugget_reparam_from_deriv(_gamma_nat, grad))
                               : arma::vec(-Optim::reparam_from_deriv(_gamma_nat, grad));
          return -ll;
        };
      }
      return [this](const arma::vec& _gamma, arma::vec* grad_out, Kriging::KModel*) {
        const double ll = this->_logLikelihoodNystrom(_gamma, grad_out);
        if (grad_out != nullptr)
          *grad_out = -*grad_out;
        return -ll;
      };
    }
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
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LOO, LMP)");
}

namespace {
// Receiver of deprecation messages. Function-local static: safe to set from
// another translation unit's static initializer (bindings).
Kriging::DeprecationHandler& deprecation_handler() {
  static Kriging::DeprecationHandler handler;
  return handler;
}

// Default receiver: printed once per process
void print_deprecation_once(const std::string& message) {
  static std::atomic<bool> warned{false};
  if (!warned.exchange(true))
    arma::cout << "Warning: " << message << arma::endl;
}

// Canonical method string with its default size made explicit
std::string canonical_fit_method(const std::string& method) {
  if (method == "exact")
    return method;
  for (const char* name : {"vecchia", "nystrom"}) {
    const std::string nm(name);
    const arma::uword dflt = (nm == "vecchia") ? 30 : 50;
    if (method == nm)
      return nm + "(" + std::to_string(dflt) + ")";
    if (method.rfind(nm + "(", 0) == 0 && method.back() == ')') {
      try {
        const long k = std::stol(method.substr(nm.size() + 1, method.size() - nm.size() - 2));
        if (k >= 1)
          return nm + "(" + std::to_string(k) + ")";
      } catch (const std::exception&) {
        // fall through to the throw below
      }
    }
  }
  throw std::invalid_argument("fit: unknown method '" + method
                              + "': expected \"exact\", \"vecchia\", \"vecchia(m)\", \"nystrom\" or \"nystrom(k)\" "
                                "(m, k >= 1)");
}
}  // namespace

LIBKRIGING_EXPORT std::string Kriging::deprecated_objective_message(const std::string& objective) {
  const bool vll = objective.rfind("LLVecchia", 0) == 0;
  const bool nll = objective.rfind("LLNystrom", 0) == 0;
  if (!vll && !nll)
    return "";
  const std::string method = vll ? "vecchia(" + std::to_string(parse_vll_m(objective)) + ")"
                                 : "nystrom(" + std::to_string(parse_nystrom_k(objective)) + ")";
  return "objective \"" + objective + "\" is deprecated; use objective=\"LL\", method=\"" + method + "\"";
}

LIBKRIGING_EXPORT void Kriging::set_deprecation_handler(DeprecationHandler handler) {
  deprecation_handler() = std::move(handler);
}

LIBKRIGING_EXPORT std::pair<std::string, std::string> Kriging::fit_objective_method(const std::string& objective,
                                                                                    const std::string& method) {
  std::string objective_out = objective;
  std::string method_out = canonical_fit_method(method);
  // deprecated aliases: objective "LLVecchia(m)" / "LLNystrom(k)"
  const bool vll = objective.rfind("LLVecchia", 0) == 0;
  const bool nll = objective.rfind("LLNystrom", 0) == 0;
  if (vll || nll) {
    const std::string alias_method = vll ? "vecchia(" + std::to_string(parse_vll_m(objective)) + ")"
                                         : "nystrom(" + std::to_string(parse_nystrom_k(objective)) + ")";
    if (method_out != "exact" && method_out != alias_method)
      throw std::invalid_argument("fit: objective \"" + objective + "\" (deprecated alias of method \"" + alias_method
                                  + "\") conflicts with method \"" + method + "\"");
    const DeprecationHandler& handler = deprecation_handler();
    const std::string message = deprecated_objective_message(objective);
    if (handler)
      handler(message);
    else
      print_deprecation_once(message);
    objective_out = "LL";
    method_out = alias_method;
  }
  if (objective_out != "LL" && objective_out != "LOO" && objective_out != "LMP")
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LOO, LMP)");
  return {objective_out, method_out};
}

std::string Kriging::internal_fit_objective(const std::string& objective, const std::string& method) const {
  if (method == "exact")
    return objective;
  if (m_noise_model != NoiseModel::None && objective != "LL")
    throw std::invalid_argument("fit: objective \"" + objective + "\" with method \"" + method
                                + "\" is not implemented for Nugget/Heterogeneous noise models; use objective "
                                  "\"LL\" or method \"exact\"");
  // every noise model is supported by both approximate methods with objective "LL"
  // internal objective name of the approximate criterion, e.g. "LLVecchia(30)", "LOONystrom(50)"
  return objective + (method.rfind("vecchia", 0) == 0 ? "Vecchia" : "Nystrom") + method.substr(7);
}

std::string Kriging::internal_objective() const {
  return internal_fit_objective(m_objective, m_method.empty() ? std::string("exact") : m_method);
}

LIBKRIGING_EXPORT void Kriging::set_vecchia_exact_commit(bool b) {
  if (b)
    throw std::invalid_argument(
        "set_vecchia_exact_commit(true) is no longer supported: a Vecchia fit never keeps the exact factorization; "
        "fit the exact model with method=\"exact\" (optim=\"none\" and the Vecchia theta to reuse it)");
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LL' (default), 'LOO' or 'LMP' ('LLVecchia(m)' / 'LLNystrom(k)':
 *        deprecated aliases of 'LL' with method 'vecchia(m)' / 'nystrom(k)').
 *        Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 * @param method 'exact' (default), 'vecchia(m)' or 'nystrom(k)': see Kriging.hpp.
 */
LIBKRIGING_EXPORT void Kriging::fit(const arma::vec& y,
                                    const arma::mat& X,
                                    const Trend::RegressionModel& regmodel,
                                    bool normalize,
                                    const std::string& optim,
                                    const std::string& objective,
                                    const Parameters& parameters,
                                    const std::string& method) {
  fit_impl(y, nullptr, X, regmodel, normalize, optim, objective, parameters, method);
}

void Kriging::commit_model_chol_jitter() {
  if (m_noise_model != NoiseModel::None || m_is_empty)
    return;
  m_model_chol_jitter = LinearAlgebra::model_chol_jitter(m_covType);
  if (m_model_chol_jitter == m_chol_jitter)
    return;  // the optimizer's last model is already the committed one
  KModel m = make_committed_Model(m_theta, m_sigma2);
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
    m_z = std::move(m.ystar) - m_M * m_beta;
  }
  if (m_est_sigma2) {
    const arma::uword n = m_X.n_rows;
    m_sigma2 = (m_objective == "LMP") ? m.SSEstar / (n - m_F.n_cols) : m.SSEstar / n;
  }
}

void Kriging::fit_impl(const arma::vec& y,
                       const arma::vec* noise,
                       const arma::mat& X,
                       const Trend::RegressionModel& regmodel,
                       bool normalize,
                       const std::string& optim,
                       const std::string& objective_in,
                       const Parameters& parameters,
                       const std::string& method) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const arma::uword n = X.n_rows;
  // Heterogeneous noise in the ORIGINAL scale of y: the given one, or (refit
  // without new noise) the current one, read before the normalization
  // constants change below.
  arma::vec noise_orig;
  if (m_noise_model == NoiseModel::Heterogeneous)
    noise_orig = (noise != nullptr) ? *noise : this->noise();
  const arma::uword d = X.n_cols;

  // public (objective, method) pair; `objective` below is the internal name of
  // the evaluated criterion ("LLVecchia(m)" / "LLNystrom(k)" for approximate LL)
  const auto objective_method = fit_objective_method(objective_in, method);
  const std::string objective = internal_fit_objective(objective_method.first, objective_method.second);
  // the last simulate(..., will_update=true) was built for the previous model
  clear_simulate_state();
  // Fixed Cholesky jitter (noise-free only): m_chol_jitter, the kernel's
  // value, for the objective (optimization, logLikelihood(Fun); reproducible
  // theta); m_model_chol_jitter, at most num_nugget, for the committed exact
  // factorization (commit_model_chol_jitter; predict / simulate / update).
  // Vecchia / Nystrom keep the objective value (their committed state comes
  // from the objective itself).
  m_chol_jitter = (m_noise_model == NoiseModel::None) ? LinearAlgebra::kernel_chol_jitter(m_covType) : 0.0;
  m_model_chol_jitter = m_chol_jitter;
  m_optim = optim;
  m_objective = objective_method.first;
  m_method = objective_method.second;
  FitOfn fit_ofn = make_fit_objective(objective);

  // Nystrom and Vecchia never touch m_dX (they work straight from m_X) and
  // never build the exact factorization that would need it.
  const bool vecchia_method = m_method.rfind("vecchia", 0) == 0;
  const bool nystrom_method = m_method.rfind("nystrom", 0) == 0;
  const bool build_dX = !vecchia_method && !nystrom_method;
  arma::mat theta0 = fit_setup_impl(
      y, X, regmodel, normalize, parameters.is_beta_estim, parameters.beta, parameters.theta, build_dX);
  // The per-observation noise variances live in the same (normalized-y)
  // space as sigma2 in R = C + diag(noise / sigma2).
  if (m_noise_model == NoiseModel::Heterogeneous)
    m_noise = m_normalize ? arma::vec(noise_orig / (m_scaleY * m_scaleY)) : noise_orig;

  m_vecchia_light = false;
  m_nystrom_light = false;
  m_nystrom_k = 0;
  m_nystrom_U.reset();
  m_nystrom_D.reset();
  if (vecchia_method) {
    m_vecchia_m = parse_vll_m("LLVecchia" + m_method.substr(7));
    make_vecchia_sets();
  } else {
    m_vecchia_m = 0;
  }
  if (nystrom_method) {
    m_nystrom_k = parse_nystrom_k("LLNystrom" + m_method.substr(7));
    make_nystrom_landmarks();
  }

  const double scaleY = m_scaleY;
  const arma::rowvec& scaleX = m_scaleX;

  // Degenerate data: y is (to rounding) in the span of the trend (e.g. a
  // constant y with a constant trend). The residual y - F*beta is then zero
  // for every theta, so sigma2 = 0 and the likelihood is unbounded: theta is
  // not identifiable and an optimizer only drives it to a bound where R is
  // singular. Skip the optimization instead and commit the model at the given
  // theta (or the default starting point): beta = least squares, sigma2 = 0,
  // i.e. predictions equal to the trend with zero variance.
  // With a nugget, the estimated total variance is zero too (sigma2 = nugget
  // = 0). With a known heterogeneous noise, the sigma2 estimate is 0, which
  // the R = C + diag(noise / sigma2) parameterization cannot represent:
  // reject such data explicitly rather than with an opaque optimizer failure.
  const bool degenerate
      = (optim != "none") && (m_noise_model != NoiseModel::Heterogeneous) && !parameters.sigma2.has_value()
        && !(m_noise_model == NoiseModel::Nugget && parameters.nugget.has_value()) && y_in_trend_span();
  if (optim != "none" && m_noise_model == NoiseModel::Heterogeneous && !parameters.sigma2.has_value()
      && y_in_trend_span())
    throw std::invalid_argument(
        "fit: y lies in the span of the trend (no residual variability), so the process variance sigma2 is 0, "
        "which a heterogeneous-noise model cannot represent; use noise=\"nugget\" or no noise, or give sigma2");
  if (degenerate && Optim::log_level > Optim::log_none)
    arma::cout << "Warning: y lies in the span of the trend (no residual variability): theta is not identifiable; "
                  "fit skips the optimization (sigma2 = 0)"
               << arma::endl;

  // just keep given theta, no optimisation of ll (but estim sigma2  &beta still possible)
  if (optim == "none" || degenerate) {
    if (!parameters.theta.has_value() && !degenerate)
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");

    if (!theta0.is_empty()) {
      m_theta = trans(theta0.row(0));
    } else {  // degenerate data, no theta given: deterministic default (typical
              // inter-point spacing, which keeps R well conditioned); theta has
              // no effect on predictions since sigma2 = 0
      const double n_pow = std::pow(static_cast<double>(n), -1.0 / static_cast<double>(d));
      m_theta = arma::vec(m_maxdX) * n_pow;
      m_theta.replace(0.0, 1.0);
    }
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
      // (any objective: like the exact optim="none" commit, sigma2 = SSE / n)
      //
      // Noise models: same conventions as the exact optim="none" branch
      // below (alpha from the given sigma2/nugget, or 1 - alpha_lower;
      // heterogeneous sigma2 given, or the previous one), with the profiled
      // quantities computed from the Nystrom Rhat instead of the dense R.
      arma::vec gamma = m_theta;
      double nugget_param = 0.0;
      if (m_noise_model == NoiseModel::Nugget) {
        m_est_nugget = parameters.is_nugget_estim;
        if (parameters.nugget.has_value()) {
          nugget_param = parameters.nugget.value();
          if (m_normalize)
            nugget_param /= (scaleY * scaleY);
        }
        m_alpha
            = (sigma2 > 0 && (sigma2 + nugget_param) > 0) ? sigma2 / (sigma2 + nugget_param) : 1.0 - nugget_alpha_lower;
        if (!m_est_sigma2)
          m_sigma2 = sigma2;
        m_nugget = nugget_param;
        gamma = arma::join_cols(m_theta, arma::vec{m_alpha});
      } else if (m_noise_model == NoiseModel::Heterogeneous) {
        // the noisy correlation needs a sigma2 (internal scale): the given
        // one, else the variogram sill (as method "vecchia" and the BFGS bounds)
        const double s2 = (sigma2 > 0) ? sigma2 : std::max(variogram_sill(m_X, m_y), 1e-10);
        m_sigma2 = s2;
        gamma = arma::join_cols(m_theta, arma::vec{s2});
      }
      arma::vec beta_v;
      double tv = -1;
      double sse = -1;
      arma::mat U_v;
      arma::vec D_v;
      _logLikelihoodNystrom(gamma, nullptr, &beta_v, &tv, &U_v, &D_v, &sse);
      if (m_est_beta)
        m_beta = beta_v;
      if (m_noise_model == NoiseModel::Nugget) {
        if (m_est_sigma2) {
          const double total_var = sse / n;
          m_sigma2 = m_alpha * total_var;
          m_nugget = m_est_nugget ? (1.0 - m_alpha) * total_var : nugget_param;
        } else {
          m_sigma2 = sigma2;
          m_nugget = m_est_nugget ? 0.0 : nugget_param;
        }
      } else if (m_est_sigma2)
        m_sigma2 = sse / n;
      else
        m_sigma2 = sigma2;
      m_nystrom_U = std::move(U_v);
      m_nystrom_D = std::move(D_v);
      m_nystrom_light = true;
      m_is_empty = true;  // no committed factorization: predict routes to predictNystrom
      return;
    }

    if (m_vecchia_m > 0) {
      // method "vecchia" with optim="none": same rationale as the Nystrom
      // branch above -- commit beta/sigma2 from the Vecchia likelihood at
      // the given (fixed) theta, without any exact factorization. Noise
      // models follow the exact optim="none" conventions below, with the
      // Vecchia Q/n in place of the exact SSE/n.
      arma::vec gamma = m_theta;
      double nugget_param = 0.0;
      if (m_noise_model == NoiseModel::Nugget) {
        m_est_nugget = parameters.is_nugget_estim;
        if (parameters.nugget.has_value()) {
          nugget_param = parameters.nugget.value();
          if (m_normalize)
            nugget_param /= (scaleY * scaleY);
        }
        m_alpha
            = (sigma2 > 0 && (sigma2 + nugget_param) > 0) ? sigma2 / (sigma2 + nugget_param) : 1.0 - nugget_alpha_lower;
        m_nugget = nugget_param;
        if (!m_est_sigma2)
          m_sigma2 = sigma2;
        gamma = arma::join_cols(m_theta, arma::vec{m_alpha});
      } else if (m_noise_model == NoiseModel::Heterogeneous) {
        // the noisy correlation needs a sigma2: the given one, else the
        // variogram sill (as the BFGS bounds)
        const double s2 = (sigma2 > 0) ? sigma2 : std::max(variogram_sill(m_X, m_y), 1e-10);
        if (!m_est_sigma2)
          m_sigma2 = sigma2;
        gamma = arma::join_cols(m_theta, arma::vec{s2});
      }
      arma::vec beta_v;
      double s_v = -1;
      double q_over_n = -1;
      _logLikelihoodVecchia(gamma, nullptr, &beta_v, &s_v, &q_over_n);
      if (m_est_beta)
        m_beta = beta_v;
      if (m_noise_model == NoiseModel::Nugget) {
        if (m_est_sigma2) {
          m_sigma2 = m_alpha * q_over_n;
          m_nugget = m_est_nugget ? (1.0 - m_alpha) * q_over_n : nugget_param;
        } else {
          m_nugget = m_est_nugget ? m_sigma2 * (1.0 - m_alpha) / m_alpha : nugget_param;
        }
      } else if (m_est_sigma2) {
        m_sigma2 = q_over_n;  // None: the concentrated sigma2; Heterogeneous: SSE/n as the exact path
      } else {
        m_sigma2 = sigma2;
      }
      m_vecchia_light = true;
      m_is_empty = true;  // no committed factorization: predict routes to predictVecchia
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
      extra_param = m_sigma2;  // ignored for None
      m_model_chol_jitter = LinearAlgebra::model_chol_jitter(m_covType);
    }

    Kriging::KModel m = make_committed_Model(m_theta, extra_param);
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

    const bool theta_fixed = !parameters.is_theta_estim && parameters.theta.has_value();
    if (theta_fixed && theta0.n_rows != 1)
      throw std::invalid_argument("fit: is_theta_estim = false needs a single theta (1x" + std::to_string(d)
                                  + " matrix), got " + std::to_string(theta0.n_rows) + " rows");

    if (optim.rfind("BFGS", 0) == 0) {
      // Nugget warm start (#383): from a random theta and alpha, a single
      // start often ends in the all-noise region (small theta, nugget >>
      // sigma2), far below the multistart optimum. The noise-free fit on the
      // same data is cheap and well conditioned, and its theta with a small
      // nugget share (alpha = 0.99) lands near the optimum on most problems:
      // seed the first start with it, keep the others random. Done before
      // Random::init() below, as the sub-fit reseeds the shared engine.
      arma::mat theta0_nugget_seed;
      if (m_noise_model == NoiseModel::Nugget && !parameters.theta.has_value() && build_dX) {
        try {
          Kriging nf_fit(m_y, m_X, m_covType, m_regmodel, /*normalize=*/false, "BFGS", "LL");
          theta0_nugget_seed = nf_fit.theta().t();  // 1 x d, in m_X's (normalized) scale
        } catch (const std::exception&) {
          // fall back to plain random multistart
        }
      }

      Random::init();

      auto parsed_bfgs = Optim::parse_method(optim, "BFGS");
      int multistart = parsed_bfgs.second;

      // The starts run in a thread pool (see run_task_pool below): the cores
      // go to the starts first, the remaining share to their BLAS / OpenMP.

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
      } else if (!theta0_nugget_seed.is_empty()) {
        theta0 = arma::join_cols(theta0_nugget_seed, theta0_rand);  // noise-free fit theta first, then random
        theta0.resize(multistart, theta0.n_cols);
      } else {
        theta0 = theta0_rand;
      }
      // Fixed theta (given with is_theta_estim = false): pin it with
      // lower == upper bounds (L-BFGS-B keeps such a variable fixed), so the
      // other hyperparameters (alpha / sigma2) are still optimized according
      // to their own is_*_estim flags, and sigma2 / beta are profiled as for
      // an estimated theta.
      if (theta_fixed) {
        theta_lower = theta0.row(0).t();
        theta_upper = theta_lower;
        theta0 = arma::repmat(theta0.row(0), multistart, 1);
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
          if (!theta0_nugget_seed.is_empty())
            extra0.at(0) = 0.99;  // small nugget share for the noise-free-fit seed
        }
      } else if (m_noise_model == NoiseModel::Heterogeneous) {
        // sigma2 bounds from variogram (on a subsample without the O(n^2)
        // m_dX, i.e. for method "vecchia")
        double sigma2_variogram = 0.0;
        if (m_dX.is_empty()) {
          sigma2_variogram = variogram_sill(m_X, m_y);
        } else {
          arma::vec dy2(n * n, arma::fill::zeros);
          for (arma::uword ij = 0; ij < dy2.n_elem; ij++) {
            arma::uword i = ij % n, j = ij / n;
            if (i != j) {
              dy2[ij] = m_y[i] - m_y[j];
              dy2[ij] *= dy2[ij];
            }
          }
          arma::vec dX2 = arma::sum(m_dX % m_dX, 0).t();
          sigma2_variogram = 0.5 * arma::mean(dy2.elem(arma::find(dX2 >= arma::median(dX2))));
        }
        extra_lower_val = 0.1 * (sigma2_variogram - arma::max(m_noise));
        extra_upper_val = 10.0 * (sigma2_variogram - arma::min(m_noise));
        if (parameters.sigma2.has_value()) {
          extra0 = arma::vec{parameters.sigma2.value()};
          if (m_normalize)
            extra0 /= (scaleY * scaleY);
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

      // approximate methods never use the dense KModel: no n x n allocation
      for (int i = 0; i < multistart && build_dX; i++) {
        auto& m = preallocated_models[i];
        m.R = arma::mat(n_data, n_data, arma::fill::none);
        m.L = arma::mat(n_data, n_data, arma::fill::none);
        m.Linv = arma::mat();  // Empty matrix
        m.Rinv = arma::mat();  // Empty matrix -- computed lazily on demand, like Linv; must NOT be
                               // preallocated at the right size (fill::none) the way the other KModel
                               // fields below are, or the lazy size-check in _logLikelihood would
                               // mistake this uninitialized block for an already-computed R^-1
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
          // fit_ofn evaluation for objectives that touch km_data (exact
          // LL/LOO/LMP). It always builds a dense R via m_dX, so it must be
          // skipped whenever m_dX was never built (methods "vecchia" and
          // "nystrom"): both ignore km_data entirely, so skipping this call
          // changes nothing for them beyond avoiding the now-empty m_dX.
          if (build_dX) {
            if (m_noise_model != NoiseModel::None)
              populate_Model(m, theta_start, extra0[start_idx % extra0.n_elem], nullptr);
            else
              populate_Model(m, theta_start, nullptr);
          }

          lbfgsb::Optimizer optimizer{gd};
          optimizer.iprint = -1;  // Disable output in parallel mode. was Optim::log_level - 2;
          optimizer.max_iter = Optim::max_iteration;
          const bool loo_objective = objective.rfind("LOO", 0) == 0;  // exact or approximate LOO
          optimizer.pgtol = loo_objective ? Optim::gradient_tolerance / (n * n)
                                          : Optim::gradient_tolerance;  // scale by: n^2 for LOO vs. LL, and /10
                                                                        // because LOO is usually more smooth
          optimizer.factr = loo_objective ? Optim::objective_rel_tolerance / 1E-13 / (n * n)
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
            // over free dimensions only: a pinned one (lower == upper, e.g. a
            // constant input) is always "at its bound"
            const arma::uvec free_dims = arma::find(theta_upper > theta_lower);
            double sol_to_lb = free_dims.is_empty()
                                   ? arma::datum::inf
                                   : arma::min(arma::abs(theta_part(free_dims) - theta_lower(free_dims)));
            double sol_to_ub = free_dims.is_empty()
                                   ? arma::datum::inf
                                   : arma::min(arma::abs(theta_part(free_dims) - theta_upper(free_dims)));
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
              // kept inside [theta_lower, theta_upper]: L-BFGS-B does not
              // project its starting point, so an outside one gave a theta
              // outside the bounds
              arma::vec restart_theta = arma::max(
                  arma::min(arma::vec((theta_start + theta_lower) / pow(2.0, retry + 1)), theta_upper), theta_lower);
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

      // Execute optimizations: the starts run concurrently in a thread pool
      // (TaskPool.hpp), which gives the cores to the starts first and the
      // remaining share to each start's BLAS / OpenMP calls; a single start
      // runs inline with all of them. Each start works on its own
      // preallocated KModel and the objectives are const. With MKL, BLAS calls
      // from secondary threads may round differently from the calling
      // thread's: BFGS20 matches the best of 20 BFGS1 up to that rounding
      // (KrigingTest checks it to 1e-3 relative).
      std::vector<OptimizationResult> results(multistart);
      libKriging::detail::run_task_pool(
          multistart,
          [&](int task_id) { results[task_id] = optimize_worker(task_id); },
          "multistart",
          /*stagger=*/false);

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
      if (best_idx >= 0 && m_vecchia_m > 0) {
        // light Vecchia commit: no exact factorization; theta (and alpha /
        // sigma2) from the optimizer, beta/sigma2 profiled by the Vecchia likelihood
        const auto& best = results[best_idx];
        m_theta = best.theta;
        m_est_theta = !theta_fixed;
        commit_approx(/*lmp_dof=*/true, best.extra_param);
      } else if (best_idx >= 0 && m_nystrom_k > 0) {
        // Nystrom commit: never an exact factorization; theta from the
        // optimizer, beta/sigma2 (alpha/nugget)/U/D profiled by the Nystrom
        // likelihood at gamma* = (theta* [, alpha* | sigma2*])
        const auto& best = results[best_idx];
        m_theta = best.theta;
        m_est_theta = !theta_fixed;
        commit_approx(/*lmp_dof=*/true, best.extra_param);
      } else if (best_idx >= 0) {
        const auto& best = results[best_idx];
        m_theta = best.theta;  // copy
        m_est_theta = !theta_fixed;
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
        commit_model_chol_jitter();
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
                                    const Parameters& parameters,
                                    const std::string& method) {
  if (m_noise_model != NoiseModel::Heterogeneous)
    throw std::runtime_error("fit(y, noise, X, ...) requires NoiseModel::Heterogeneous");
  if (noise.n_elem != y.n_elem)
    throw std::runtime_error("noise vector must have the same length as y");
  if (!noise.is_finite() || noise.min() < 0)
    throw std::invalid_argument("noise variances must be finite and non-negative");
  fit_impl(y, &noise, X, regmodel, normalize, optim, objective, parameters, method);
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
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  // exact prediction only: no implicit switch to an approximate predictor
  check_exact_available("predict", "predict");
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

LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat> Kriging::predict(
    const arma::mat& X_n,
    bool return_stdev,
    bool return_cov,
    bool return_deriv,
    const std::string& method) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const ApproxMethod am = parse_approx_method(method, "predict");
  check_nystrom_rank(am, m_nystrom_k, "predict");
  if (am.kind == ApproxMethod::Exact)
    return predict(X_n, return_stdev, return_cov, return_deriv);
  if (return_cov || return_deriv)
    throw std::invalid_argument("predict: return_cov / return_deriv are only available with method \"exact\"");
  auto [mean, stdev] = (am.kind == ApproxMethod::Vecchia) ? predictVecchia(X_n, return_stdev, am.m)
                                                          : predictNystrom(X_n, return_stdev);
  return {mean, stdev, arma::mat(), arma::mat(), arma::mat()};
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const std::string& method,
                                              const bool will_update) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const ApproxMethod am = parse_approx_method(method, "simulate");
  check_nystrom_rank(am, m_nystrom_k, "simulate");
  if (am.kind == ApproxMethod::Exact)
    return simulate(nsim, seed, X_n, will_update);
  if (will_update)
    throw std::invalid_argument(
        "simulate: will_update=true (for update_simulate) is only available with method \"exact\"");
  return (am.kind == ApproxMethod::Vecchia) ? simulateVecchia(nsim, seed, X_n, am.m) : simulateNystrom(nsim, seed, X_n);
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const bool with_nugget,
                                              const bool will_update,
                                              const std::string& method) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const ApproxMethod am = parse_approx_method(method, "simulate");
  check_nystrom_rank(am, m_nystrom_k, "simulate");
  if (am.kind == ApproxMethod::Exact)
    return simulate(nsim, seed, X_n, with_nugget, will_update);
  if (will_update)
    throw std::invalid_argument(
        "simulate: will_update=true (for update_simulate) is only available with method \"exact\"");
  if (am.kind == ApproxMethod::Nystrom) {
    // simulateNystrom draws the latent process only (no observation noise)
    if (with_nugget)
      throw std::invalid_argument(
          "simulate: with_nugget=true is not available with method \"nystrom\" (it simulates the latent process "
          "only); use method \"exact\" or \"vecchia\"");
    return simulateNystrom(nsim, seed, X_n);
  }
  return simulateVecchia(nsim, seed, X_n, am.m, with_nugget);
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const arma::vec& with_noise,
                                              const bool will_update,
                                              const std::string& method) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const ApproxMethod am = parse_approx_method(method, "simulate");
  check_nystrom_rank(am, m_nystrom_k, "simulate");
  if (am.kind == ApproxMethod::Exact)
    return simulate(nsim, seed, X_n, with_noise, will_update);
  if (will_update)
    throw std::invalid_argument(
        "simulate: will_update=true (for update_simulate) is only available with method \"exact\"");
  if (with_noise.n_elem > 1 && with_noise.n_elem != X_n.n_rows)
    throw std::runtime_error("Noise vector should have same length as X_n");
  if (am.kind == ApproxMethod::Nystrom) {
    // simulateNystrom draws the latent process only (no observation noise)
    if (arma::any(with_noise != 0.0))
      throw std::invalid_argument(
          "simulate: a non-zero with_noise is not available with method \"nystrom\" (it simulates the latent "
          "process only); use method \"exact\" or \"vecchia\"");
    return simulateNystrom(nsim, seed, X_n);
  }
  return simulateVecchia(nsim, seed, X_n, am.m, false, with_noise);
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
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  check_exact_available("simulate", "simulate");
  if (m_noise_model == NoiseModel::Nugget)
    return simulate(nsim, seed, X_n, /*with_nugget=*/false, will_update);
  arma::mat y_n = simulate_impl(nsim,
                                seed,
                                X_n,
                                will_update,
                                /*R_on_factor=*/1.0,
                                /*R_on_coincident_to_one=*/false,
                                /*R_nn_factor=*/1.0,
                                /*R_nn_diag=*/arma::vec(),
                                /*Sigma_divisor=*/1.0);
  if (will_update) {  // latent simulation: no with_noise of an earlier simulate to re-add
    m_lastsim_with_nugget = false;
    m_lastsim_with_noise.reset();
    m_lastsimup_noise_u.reset();
  }
  return y_n;
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const bool with_nugget,
                                              const bool will_update) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  check_exact_available("simulate", "simulate");
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
                                /*Sigma_divisor=*/alpha);
  if (will_update) {  // replaces the state of an earlier simulate(..., will_update=true)
    m_lastsim_with_nugget = with_nugget;
    m_lastsim_with_noise.reset();
    m_lastsimup_noise_u.reset();
  }
  return y_n;
}

LIBKRIGING_EXPORT arma::mat Kriging::simulate(int nsim,
                                              int seed,
                                              const arma::mat& X_n,
                                              const arma::vec& with_noise,
                                              const bool will_update) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  const arma::uword n_n = X_n.n_rows;
  if (with_noise.n_elem > 1 && with_noise.n_elem != n_n)
    throw std::runtime_error("Noise vector should have same length as X_n");
  check_noise_variances(with_noise, "simulate");
  check_exact_available("simulate", "simulate");
  arma::mat y_n = simulate_impl(nsim,
                                seed,
                                X_n,
                                will_update,
                                /*R_on_factor=*/1.0,
                                /*R_on_coincident_to_one=*/false,
                                /*R_nn_factor=*/1.0,
                                /*R_nn_diag=*/arma::vec(),
                                /*Sigma_divisor=*/1.0);
  if (will_update) {  // replaces the state of an earlier simulate(..., will_update=true)
    m_lastsim_with_nugget = false;
    m_lastsim_with_noise = with_noise;
    m_lastsimup_noise_u.reset();
  }
  add_noise_draws(y_n, with_noise, "simulate");
  return y_n;
}

void Kriging::clear_simulate_state() {
  KrigingImpl::clear_simulate_state();
  m_lastsim_with_nugget = false;
  m_lastsim_with_noise.reset();
  m_lastsimup_noise_u.reset();
}

LIBKRIGING_EXPORT arma::mat Kriging::update_simulate(const arma::vec& y_u, const arma::mat& X_u) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  // the noise of the new observations is needed (checked before any state change)
  if (m_noise_model == NoiseModel::Heterogeneous)
    throw std::invalid_argument("heterogeneous noise model: use update_simulate(y_u, noise_u, X_u)");
  check_not_vecchia_light("update_simulate");
  check_not_nystrom_light("update_simulate");
  // reuse the update-points cache only if it was built by the noise-free form
  const bool allow_cache = m_lastsimup_noise_u.is_empty();
  m_lastsimup_noise_u.reset();
  if (m_noise_model == NoiseModel::Nugget) {
    const double alpha = m_alpha;
    return update_simulate_impl(y_u,
                                X_u,
                                allow_cache,
                                /*R_uu_factor=*/alpha,
                                /*R_uu_diag=*/arma::vec(y_u.n_elem, arma::fill::ones),
                                /*R_uo_factor=*/alpha,
                                /*R_un_factor=*/alpha,
                                /*R_un_coincident_to_one=*/false,
                                /*Sigma_divisor=*/alpha);
  }
  // noise-free: the new observations are conditioned on exactly (no fixed
  // Cholesky jitter on R_uu): an update point on the simulation design is
  // reproduced exactly; update() adds the jitter (populate_Model), a 1e-10..1e-8
  // relative difference only (docs/dev/CholJitter-TODO.md)
  return update_simulate_impl(y_u,
                              X_u,
                              allow_cache,
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
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension mismatch: y_u and X_u");
  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension mismatch: X_u cols vs X cols");
  if (noise_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Noise vector length must match X_u rows");
  const arma::vec diag_uu = 1.0 + noise_u / (m_scaleY * m_scaleY * m_sigma2);
  const arma::uword n_n = lastsim_Xn_n.n_cols;
  // The update-points cache (lastsimup_*, keyed on X_u by the base) holds R_uu
  // with the noise of the call that built it: reuse it only for the same noise_u
  // (e.g. several y_u at the same (X_u, noise_u)), never for the noise-free form.
  const bool allow_cache = !noise_u.is_empty() && m_lastsimup_noise_u.n_elem == noise_u.n_elem
                           && arma::approx_equal(m_lastsimup_noise_u, noise_u, "absdiff", arma::datum::eps);
  m_lastsimup_noise_u = noise_u;
  arma::mat y_up = update_simulate_impl(y_u,
                                        X_u,
                                        allow_cache,
                                        /*R_uu_factor=*/1.0,
                                        /*R_uu_diag=*/diag_uu,
                                        /*R_uo_factor=*/1.0,
                                        /*R_un_factor=*/1.0,
                                        /*R_un_coincident_to_one=*/false,
                                        /*Sigma_divisor=*/1.0);
  if (m_lastsim_with_noise.n_elem == 1 || m_lastsim_with_noise.n_elem == n_n)
    add_noise_draws(y_up, m_lastsim_with_noise, "update_simulate");
  return y_up;
}

/** Add new conditional data points to previous (X,y), then perform new fit.
 * @param y_u is n_u length column vector of new output
 * @param X_u is n_u*d matrix of new input
 * @param refit is true if we want to re-fit the model
 */
LIBKRIGING_EXPORT void Kriging::update(const arma::vec& y_u, const arma::mat& X_u, const bool refit) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  // the noise of the new observations is needed (checked before any state change)
  if (m_noise_model == NoiseModel::Heterogeneous)
    throw std::invalid_argument("heterogeneous noise model: use update(y_u, noise_u, X_u)");
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  // input validated: the last simulate(..., will_update=true) was built for
  // the model before this update (the Nugget refit below goes through fit(),
  // which clears it again)
  clear_simulate_state();

  // a Vecchia model (including a legacy one still carrying an exact
  // factorization, see load_from_json) is updated as a Vecchia fit
  if (m_vecchia_m > 0) {
    update_vecchia_light(y_u, X_u, refit);
    return;
  }
  if (m_nystrom_light) {
    update_nystrom(y_u, X_u, refit);
    return;
  }

  if (refit && m_optim != "none" && m_noise_model == NoiseModel::Nugget) {
    // For Nugget mode: full re-fit with de-normalized data (matches NuggetKriging::update behavior)
    const arma::vec y_all = arma::join_cols(m_y * m_scaleY + m_centerY, y_u);
    const arma::mat X_all = arma::join_cols((m_X.each_row() % m_scaleX).each_row() + m_centerX, X_u);
    Parameters params;
    if (m_est_beta && m_est_nugget && m_est_sigma2 && m_est_theta) {
      // All estimated: use default (null) starting points → full multistart BFGS
    } else {
      // Parameters are read in the original scale: use the accessors
      params.sigma2 = sigma2();
      params.is_sigma2_estim = m_est_sigma2;
      params.theta = arma::trans(theta());
      params.is_theta_estim = m_est_theta;
      params.nugget = nugget();
      params.is_nugget_estim = m_est_nugget;
      if (!m_est_beta) {
        params.beta = beta();
        params.is_beta_estim = false;
      }
    }
    fit(y_all, X_all, m_regmodel, m_normalize, m_optim, m_objective, params, m_method);
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
    arma::vec theta_lower = m_est_theta ? theta_bounds_pair.first : m_theta;  // fixed theta: pinned
    arma::vec theta_upper = m_est_theta ? theta_bounds_pair.second : m_theta;

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
    km.Rinv = arma::mat();  // computed lazily on demand, like Linv -- must not be preallocated at the
                            // right size or the lazy size-check in _logLikelihood would mistake this
                            // uninitialized block for an already-computed R^-1
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
    commit_model_chol_jitter();
  } else {  // incremental update without parameter re-optimization
    update_no_refit_impl(
        y_u,
        X_u,
        /*extend_class_data=*/[] {},
        /*build_model=*/[this] {
          return make_committed_Model(m_theta, (m_noise_model == NoiseModel::Nugget) ? m_alpha : m_sigma2);
        });
  }
}

LIBKRIGING_EXPORT void Kriging::update(const arma::vec& y_u,
                                       const arma::vec& noise_u,
                                       const arma::mat& X_u,
                                       const bool refit) {
  libKriging::detail::BudgetScope budget_scope;  // BLAS at the thread budget for the call
  if (m_noise_model != NoiseModel::Heterogeneous)
    throw std::runtime_error("update(y, noise, X) requires NoiseModel::Heterogeneous");
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");
  if (noise_u.n_elem != y_u.n_elem)
    throw std::runtime_error("noise_u must have the same length as y_u");
  // Rebuild de-normalized data and call fit() with the joined dataset (fit()
  // also drops the state of the last simulate(..., will_update=true))
  const arma::vec y_all = arma::join_cols(m_y * m_scaleY + m_centerY, y_u);
  const arma::vec noise_all = arma::join_cols(noise(), noise_u);
  const arma::mat X_all = arma::join_cols((m_X.each_row() % m_scaleX).each_row() + m_centerX, X_u);
  // Parameters are read in the original scale: use the accessors
  Kriging::Parameters params;
  params.sigma2 = sigma2();
  params.is_sigma2_estim = m_est_sigma2;
  params.theta = arma::trans(theta());
  params.is_theta_estim = m_est_theta;
  params.beta = m_est_beta ? std::optional<arma::vec>{} : std::optional<arma::vec>(beta());
  params.is_beta_estim = m_est_beta;
  fit(y_all, noise_all, X_all, m_regmodel, m_normalize, refit ? m_optim : "none", m_objective, params, m_method);
}

LIBKRIGING_EXPORT std::string Kriging::summary() const {
  std::ostringstream oss;
  // parameters displayed in the original scale, like the accessors
  const SummaryValues values{beta(), sigma2(), theta(), noise()};
  if (summary_top(oss, nullptr, &values)) {
    if (m_noise_model == NoiseModel::Nugget)
      oss << "  * nugget" << (m_est_nugget ? " (est.): " : ": ") << nugget() << "\n";
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
  dump_to_json(j);
  std::ofstream f(filename);
  f << std::setw(4) << j;
}

void Kriging::dump_to_json(nlohmann::json& j) const {
  j["version"] = 2;
  j["content"] = "Kriging";
  dump_common_to_json(j);
  j["noise_model"] = noise_model_to_string(m_noise_model);
  if (m_noise_model == NoiseModel::Nugget) {
    j["nugget"] = m_nugget;
    j["est_nugget"] = m_est_nugget;
    j["alpha"] = m_alpha;
  }
  // Only when non-zero, so files without a fixed jitter stay byte-identical
  if (m_chol_jitter > 0.0)
    j["chol_jitter"] = m_chol_jitter;
  if (m_model_chol_jitter > 0.0)
    j["model_chol_jitter"] = m_model_chol_jitter;
  // Nystrom fits carry no m_T/m_R/... (dump_common_to_json serializes those
  // as empty, harmlessly): the committed low-rank factors/landmarks are the
  // extra state that reconstructs the model, stored only when present so
  // pre-Nystrom save files stay byte-identical.
  // Light Vecchia fits carry no factorization either: theta / beta / sigma2
  // (common part) are the whole state; the Vecchia sets are rebuilt on demand.
  if (m_vecchia_light)
    j["vecchia_light"] = true;
  j["nystrom_light"] = m_nystrom_light;
  if (m_nystrom_light) {
    j["nystrom_k"] = m_nystrom_k;
    j["nystrom_landmarks"] = std::vector<arma::uword>(m_nystrom_landmarks.begin(), m_nystrom_landmarks.end());
    j["nystrom_U"] = to_json(m_nystrom_U);
    j["nystrom_D"] = to_json(m_nystrom_D);
  }
}

Kriging Kriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);
  return load_from_json(j, filename);
}

Kriging Kriging::load_from_json(const nlohmann::json& j, const std::string& filename) {
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
  // Absent on files without a fixed jitter (all files before it existed)
  kr.m_chol_jitter = j.contains("chol_jitter") ? j["chol_jitter"].template get<double>() : 0.0;
  kr.m_model_chol_jitter = j.contains("model_chol_jitter") ? j["model_chol_jitter"].template get<double>() : 0.0;
  // Absent on pre-Nystrom save files: defaults to a normal (non-Nystrom) load.
  if (j.contains("nystrom_light") && j["nystrom_light"].template get<bool>()) {
    kr.m_nystrom_light = true;
    kr.m_nystrom_k = j["nystrom_k"].template get<arma::uword>();
    kr.m_nystrom_landmarks = arma::uvec(j["nystrom_landmarks"].template get<std::vector<arma::uword>>());
    kr.m_nystrom_U = mat_from_json(j["nystrom_U"]);
    kr.m_nystrom_D = colvec_from_json(j["nystrom_D"]);
    kr.m_is_empty = true;  // no exact factorization
  }
  // Files written before the fit `method` existed store the approximation in
  // the objective ("LLVecchia(m)" / "LLNystrom(k)"): map it to (LL, method).
  if (kr.m_objective.rfind("LLVecchia", 0) == 0) {
    kr.m_method = "vecchia(" + std::to_string(parse_vll_m(kr.m_objective)) + ")";
    kr.m_objective = "LL";
  } else if (kr.m_objective.rfind("LLNystrom", 0) == 0) {
    kr.m_method = "nystrom(" + std::to_string(parse_nystrom_k(kr.m_objective)) + ")";
    kr.m_objective = "LL";
  }
  // Vecchia neighbor count (vecchia_neighbors(), logLikelihood("vecchia");
  // the sets themselves are rebuilt on demand)
  if (kr.m_method.rfind("vecchia", 0) == 0) {
    kr.m_vecchia_m = parse_vll_m("LLVecchia" + kr.m_method.substr(7));
    // A Vecchia file without exact factorization is a light fit. Files of
    // libKriging <= 1.2 (exact commit) keep their factorization, hence the
    // exact methods, until the next update (a light Vecchia refit).
    if (kr.m_T.is_empty()) {
      kr.m_vecchia_light = true;
      kr.m_is_empty = true;  // no exact factorization
    }
  }
  return kr;
}
