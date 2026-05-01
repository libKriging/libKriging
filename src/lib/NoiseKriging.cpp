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
#include "libKriging/NoiseKriging.hpp"
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
#include <mutex>
#include <queue>
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
/**      NoiseKriging implementation        **/
/************************************************/

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT NoiseKriging::NoiseKriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT NoiseKriging::NoiseKriging(const arma::vec& y,
                                             const arma::vec& noise,
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
  fit(y, noise, X, regmodel, normalize, optim, objective, parameters);
}

LIBKRIGING_EXPORT NoiseKriging::NoiseKriging(const NoiseKriging& other, ExplicitCopySpecifier) : NoiseKriging{other} {}

void NoiseKriging::populate_Model(KModel& m,
                                  const arma::vec& theta,
                                  const double sigma2,
                                  std::map<std::string, double>* bench) const {
  // Normalized covariance matrix: C̃_ij = ρ(x_i,x_j) off-diagonal, C̃_ii = 1 + noise_i/σ².
  // Full covariance is σ²·C̃; σ² is factored out and reapplied post-hoc.
  const arma::vec diag_norm = 1.0 + m_noise / sigma2;
  const bool update_eligible = !m_is_empty && (m_sigma2 == sigma2) && (m_theta.size() == theta.size())
                               && (theta - m_theta).is_zero() && (m_T.memptr() != nullptr) && (m_X.n_rows > m_T.n_rows);
  KrigingImpl::populate_Model(m, theta, 1.0, diag_norm, update_eligible, bench);
  if (!m_est_beta) {
    // NoiseKriging's historical behavior: when β is fixed externally, SSE is
    // computed against raw y (i.e. as if β̂ = 0) rather than against GLS β̂.
    m.betahat = arma::vec(m_F.n_cols, arma::fill::zeros);
    m.Estar = LinearAlgebra::solve_lower(m.L, m_y);
    m.SSEstar = arma::dot(m.Estar, m.Estar);
  }
}

NoiseKriging::KModel NoiseKriging::make_Model(const arma::vec& theta,
                                              const double sigma2,
                                              std::map<std::string, double>* bench) const {
  KModel m = allocate_KModel();

  // Populate the model
  populate_Model(m, theta, sigma2, bench);

  return m;
}

// Objective function for fit : -logLikelihood

double NoiseKriging::_logLikelihood(const arma::vec& _theta_sigma2,
                                    arma::vec* grad_out,
                                    NoiseKriging::KModel* model,
                                    std::map<std::string, double>* bench) const {
  // arma::cout << " theta, sigma2: " << _theta_sigma2.t() << arma::endl;

  arma::uword d = m_X.n_cols;
  double _sigma2 = _theta_sigma2.at(d);
  if (!m_est_sigma2) {  // Force sigma2 to fixed value, if defined
    _sigma2 = m_sigma2;
  }
  arma::vec _theta = _theta_sigma2.head(d);

  NoiseKriging::KModel m_local;
  if (model != nullptr) {
    populate_Model(*model, _theta, _sigma2, bench);
  } else {
    m_local = make_Model(_theta, _sigma2, bench);
  }
  NoiseKriging::KModel& m = (model != nullptr) ? *model : m_local;

  arma::uword n = m_X.n_rows;

  // Normalized form: full C = σ²·C̃, so log|C| = n·log(σ²) + log|C̃| and
  // residual'·C⁻¹·residual = SSEstar / σ² where SSEstar = Ẽstar'·Ẽstar is computed
  // on the normalized L̃ stored in m.L.
  double ll = -0.5 * (n * log(2 * M_PI * _sigma2) + 2 * sum(log(m.L.diag())) + m.SSEstar / _sigma2);

  if (grad_out != nullptr) {
    auto t0 = Bench::tic();
    arma::vec terme1 = arma::vec(d);

    const arma::mat& Rinv = m.Rinv;  // cached C̃⁻¹ from populate_Model

    arma::mat x = LinearAlgebra::solve_upper(m.L.t(), m.Estar);  // x = C̃⁻¹·residual
    t0 = Bench::toc(bench, "x = tL \\ z", t0);

    arma::vec term1_vec(d, arma::fill::zeros);
    arma::vec term2_vec(d, arma::fill::zeros);
    t0 = Bench::tic();
    compute_ll_grad_theta_vecs(m.R, Rinv, x, _theta, term1_vec, term2_vec);
    t0 = Bench::toc(bench, "gradC computation [optimized]", t0);

    for (arma::uword k = 0; k < d; k++) {
      terme1.at(k) = term1_vec.at(k) / _sigma2;
      (*grad_out).at(k) = (terme1.at(k) + term2_vec.at(k)) / 2.0;
    }

    if (m_est_sigma2) {
      // ∂C̃/∂σ² = -diag(noise)/σ⁴, off-diagonal = 0
      // ∂ll/∂σ² = -0.5·[ n/σ² - sum(noise_i·C̃⁻¹_ii)/σ⁴
      //                  + sum(noise_i·x_i²)/σ⁶ - SSEstar/σ⁴ ]
      double sigma2_sq = _sigma2 * _sigma2;
      double noise_Rinv = arma::dot(m_noise, Rinv.diag());
      double noise_x2 = arma::dot(m_noise, x % x);
      (*grad_out).at(d)
          = -0.5 * (n / _sigma2 - noise_Rinv / sigma2_sq + noise_x2 / (sigma2_sq * _sigma2) - m.SSEstar / sigma2_sq);
    } else
      (*grad_out).at(d) = 0;  // if sigma2 is defined & fixed by user

    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> NoiseKriging::logLikelihoodFun(const arma::vec& _theta_sigma2,
                                                                               const bool _grad,
                                                                               const bool _bench) {
  return eval_objective(_theta_sigma2.n_elem, _grad, _bench, [&](arma::vec* g, std::map<std::string, double>* b) {
    return _logLikelihood(_theta_sigma2, g, nullptr, b);
  });
}

LIBKRIGING_EXPORT double NoiseKriging::logLikelihood() {
  int d = m_theta.n_elem;
  arma::vec _theta_sigma2 = arma::vec(d + 1);
  _theta_sigma2.head(d) = m_theta;
  _theta_sigma2.at(d) = m_sigma2;
  return std::get<0>(NoiseKriging::logLikelihoodFun(_theta_sigma2, false, false));
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
LIBKRIGING_EXPORT void NoiseKriging::fit(const arma::vec& y,
                                         const arma::vec& noise,
                                         const arma::mat& X,
                                         const Trend::RegressionModel& regmodel,
                                         bool normalize,
                                         const std::string& optim,
                                         const std::string& objective,
                                         const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<double(const arma::vec& _gamma, arma::vec* grad_out, NoiseKriging::KModel* km_data)> fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NoiseKriging::KModel* km_data) {
        // Change variable for opt: . -> 1/exp(.)
        const arma::vec _theta_sigma2 = Optim::reparam_from(_gamma);
        double ll = this->_logLikelihood(_theta_sigma2, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -Optim::reparam_from_deriv(_theta_sigma2, *grad_out);
        }
        return -ll;
      };
    } else {
      fit_ofn = [this](const arma::vec& _gamma, arma::vec* grad_out, NoiseKriging::KModel* km_data) {
        const arma::vec _theta_sigma2 = _gamma;
        double ll = this->_logLikelihood(_theta_sigma2, grad_out, km_data, nullptr);
        if (grad_out != nullptr) {
          *grad_out = -*grad_out;
        }
        return -ll;
      };
    }
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported is: LL)");

  arma::mat theta0
      = fit_setup_impl(y, X, regmodel, normalize, parameters.is_beta_estim, parameters.beta, parameters.theta);
  const double scaleY = m_scaleY;
  const arma::rowvec& scaleX = m_scaleX;
  m_noise = noise / (scaleY * scaleY);

  if (optim == "none") {  // just keep given theta, no optimisation of ll (but estim beta still possible)
    if (!parameters.theta.has_value())
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");
    if (!parameters.sigma2.has_value())
      throw std::runtime_error("Sigma2 should be given, when optim=none");

    m_theta = trans(theta0.row(0));
    m_est_theta = false;

    m_sigma2 = parameters.sigma2.value()[0];
    if (m_normalize)
      m_sigma2 /= (scaleY * scaleY);
    m_est_sigma2 = false;

    NoiseKriging::KModel m = make_Model(m_theta, m_sigma2, nullptr);
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

  } else if (optim.rfind("BFGS", 0) == 0) {
    Random::init();

    arma::vec theta_lower = Optim::theta_lower_factor * m_maxdX;
    arma::vec theta_upper = Optim::theta_upper_factor * m_maxdX;

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

    if (Optim::variogram_bounds_heuristic) {
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

    auto parsed = Optim::parse_method(optim, "BFGS");
    int multistart = parsed.second;

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

    arma::vec dX2 = arma::sum(m_dX % m_dX, 0).t();

    // see https://github.com/cran/DiceKriging/blob/547135515e32fa0a37260b9cd01631c1b7a69a5b/R/kmNuggets.init.R#L30
    double sigma2_variogram = 0.5 * arma::mean(dy2.elem(arma::find(dX2 >= arma::median(dX2))));
    double sigma2_lower = 0.1 * (sigma2_variogram - arma::max(m_noise));
    double sigma2_upper = 10 * (sigma2_variogram - arma::min(m_noise));
    arma::vec sigma20;
    if (parameters.sigma2.has_value()) {
      sigma20 = arma::vec(parameters.sigma2.value());
      if (m_normalize)
        sigma20 /= scaleY;
    } else {
      sigma20 = sigma2_lower + (sigma2_upper - sigma2_lower) * Random::randu_vec(theta0.n_rows);
    }
    // arma::cout << "sigma20:" << sigma20 << arma::endl;

    arma::vec gamma_lower = arma::vec(d + 1);
    gamma_lower.head(d) = theta_lower;
    gamma_lower.at(d) = sigma2_lower;
    arma::vec gamma_upper = arma::vec(d + 1);
    gamma_upper.head(d) = theta_upper;
    gamma_upper.at(d) = sigma2_upper;
    if (Optim::reparametrize) {
      gamma_lower = Optim::reparam_to(gamma_lower);
      gamma_upper = Optim::reparam_to(gamma_upper);
    }

    // Set estimation flags before parallel execution
    m_est_sigma2 = parameters.is_sigma2_estim;
    if ((!m_est_sigma2) && (parameters.sigma2.has_value())) {
      m_sigma2 = parameters.sigma2.value()[0];
      if (m_normalize)
        m_sigma2 /= (scaleY * scaleY);
    } else {
      m_est_sigma2 = true;  // force estim if no value given
    }

    double min_ofn = std::numeric_limits<double>::infinity();

    // Pre-allocate KModel structures (one per multistart)
    arma::uword n_data = n;
    arma::uword p_data = m_F.n_cols;
    std::vector<NoiseKriging::KModel> preallocated_models(multistart);

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

    // Prepare gamma bounds for all starts
    std::vector<arma::vec> all_gamma_lower(multistart);
    std::vector<arma::vec> all_gamma_upper(multistart);

    for (arma::uword i = 0; i < multistart; i++) {
      arma::vec gamma_tmp = arma::vec(d + 1);
      gamma_tmp.head(d) = theta0.row(i % multistart).t();
      gamma_tmp.at(d) = sigma20[i % sigma20.n_elem];
      if (Optim::reparametrize) {
        gamma_tmp = Optim::reparam_to(gamma_tmp);
      }

      all_gamma_lower[i] = arma::min(gamma_tmp, gamma_lower);
      all_gamma_upper[i] = arma::max(gamma_tmp, gamma_upper);
    }

    // Thread pool configuration will be handled in the parallel execution section below

    // Multi-threading implementation for BFGS multistart
    // Each thread uses its own preallocated KModel, so no mutex needed

    // Structure to hold optimization results from each thread
    struct OptimizationResult {
      arma::uword start_index;
      double objective_value;
      arma::vec gamma;
      arma::vec theta_sigma2;
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

      OptimizationResult() : start_index(0), objective_value(std::numeric_limits<double>::infinity()), success(false) {}
    };

    // Worker function returns OptimizationResult
    auto optimize_worker = [&](arma::uword start_idx) -> OptimizationResult {
      OptimizationResult result;
      result.start_index = start_idx;

      try {
        arma::vec gamma_tmp = arma::vec(d + 1);
        gamma_tmp.head(d) = theta0.row(start_idx % multistart).t();
        gamma_tmp.at(d) = sigma20[start_idx % sigma20.n_elem];
        if (Optim::reparametrize) {
          gamma_tmp = Optim::reparam_to(gamma_tmp);
        }

        arma::vec gamma_lower_local = all_gamma_lower[start_idx];
        arma::vec gamma_upper_local = all_gamma_upper[start_idx];

        // Use pre-allocated KModel for this thread (thread-safe)
        if (start_idx >= preallocated_models.size()) {
          throw std::runtime_error("Preallocated model index out of bounds");
        }

        NoiseKriging::KModel& m = preallocated_models[start_idx];
        populate_Model(m, theta0.row(start_idx % multistart).t(), sigma20[start_idx % sigma20.n_elem], nullptr);

        lbfgsb::Optimizer optimizer{d + 1};
        optimizer.iprint = -1;  // Suppress LBFGSB output (we handle logging)
        optimizer.max_iter = Optim::max_iteration;
        optimizer.pgtol = Optim::gradient_tolerance;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
        arma::ivec bounds_type{d + 1, arma::fill::value(2)};

        if (Optim::log_level > Optim::log_none) {
          arma::cout << "BFGS (start " << (start_idx + 1) << "/" << multistart << "):" << arma::endl;
          arma::cout << "  objective: " << m_objective << arma::endl;
          arma::cout << "  max iterations: " << optimizer.max_iter << arma::endl;
          arma::cout << "  null gradient tolerance: " << optimizer.pgtol << arma::endl;
          arma::cout << "  constant objective tolerance: " << optimizer.factr * 1E-13 << arma::endl;
          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower.t() << "";
          arma::cout << "                " << sigma2_lower << arma::endl;
          arma::cout << "  upper_bounds: " << theta_upper.t() << "";
          arma::cout << "                " << sigma2_upper << arma::endl;
          arma::cout << "  start_point: " << theta0.row(start_idx % multistart) << "";
          arma::cout << "               " << sigma20[start_idx % sigma20.n_elem] << arma::endl;
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

          // check theta for distance to bounds
          double sol_to_lb_theta = Optim::reparametrize
                                       ? arma::min(arma::abs(Optim::reparam_from(gamma_tmp.head(d)) - theta_lower))
                                       : arma::min(arma::abs(gamma_tmp.head(d) - theta_upper));
          double sol_to_ub_theta = Optim::reparametrize
                                       ? arma::min(arma::abs(Optim::reparam_from(gamma_tmp.head(d)) - theta_upper))
                                       : arma::min(arma::abs(gamma_tmp.head(d) - theta_lower));
          double sol_to_lb_sigma2 = Optim::reparametrize ? Optim::reparam_from_(gamma_tmp.at(d)) - sigma2_lower
                                                         : gamma_tmp.at(d) - sigma2_lower;
          double sol_to_ub_sigma2 = Optim::reparametrize ? sigma2_upper - Optim::reparam_from_(gamma_tmp.at(d))
                                                         : sigma2_upper - gamma_tmp.at(d);

          // Check abnormal termination or convergence at bounds to decide on restart
          if ((retry < Optim::max_restart)
              && ((opt_result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0)  // Check for abnormal termination
                  || (opt_result.num_iters <= 2)           // Start point is strangely quite optimal...
                  || (sol_to_lb_theta < arma::datum::eps)  // Stuck at lower bound
                  || (opt_result.f_opt > best_f_opt))) {   // No improvement

            if (Optim::log_level > Optim::log_none) {
              arma::cout << "  Restarting BFGS (start " << (start_idx + 1) << ", retry " << (retry + 1)
                         << "): f_opt=" << opt_result.f_opt << ", sol_to_lb=" << sol_to_lb_theta
                         << ", sol_to_ub=" << sol_to_ub_theta << arma::endl;
            }

            gamma_tmp.head(d) = (theta0.row(start_idx % multistart).t() + theta_lower) / pow(2.0, retry + 1);
            gamma_tmp.at(d) = sigma20[start_idx % sigma20.n_elem];

            if (Optim::reparametrize)
              gamma_tmp = Optim::reparam_to(gamma_tmp);

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
        result.theta_sigma2 = Optim::reparametrize ? Optim::reparam_from(best_gamma) : best_gamma;

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
        arma::cout << "Thread pool: " << pool_size << " workers (ncpu=" << n_cpu << ", multistart=" << multistart << ")"
                   << arma::endl;
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
      m_theta = best.theta_sigma2.head(d);  // copy
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

      if (m_est_sigma2) {
        m_sigma2 = best.theta_sigma2.at(d);
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
NoiseKriging::predict(const arma::mat& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  return predict_impl(X_n,
                      return_stdev,
                      return_cov,
                      return_deriv,
                      /*R_on_factor=*/1.0,
                      /*R_nn_factor=*/1.0,
                      /*R_nn_diag=*/arma::vec(),
                      /*var_scale=*/m_sigma2);
}

/** Draw sample trajectories of kriging at given points X'
 * @param X_n is n_n*d matrix of points where to simulate output
 * @param seed is seed for random number generator
 * @param nsim is number of simulations to draw
 * @param with_noise is n_n (or 1) vector of noise to add to simulations
 * @param will_update is true if we want to keep simulations data for future update
 * @return output is n_n*nsim matrix of simulations at X_n
 */
LIBKRIGING_EXPORT arma::mat NoiseKriging::simulate(const int nsim,
                                                   const int seed,
                                                   const arma::mat& X_n,
                                                   const arma::vec& with_noise,
                                                   const bool will_update) {
  const arma::uword n_n = X_n.n_rows;
  if (with_noise.n_elem > 1 && with_noise.n_elem != n_n)
    throw std::runtime_error("Noise vector should have same length as X_n: " + std::to_string(with_noise.n_elem)
                             + " instead of " + std::to_string(n_n) + " (or 0 if no noise)");

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
  if (will_update) {
    lastsim_with_noise = with_noise;
  }

  arma::mat eps = arma::mat(n_n, nsim, arma::fill::none);
  if (with_noise.n_elem == 1)
    eps = with_noise.at(0) * Random::randn_mat(n_n, nsim);
  else if (with_noise.n_elem == n_n) {
    eps.each_col() = with_noise;
    eps = eps % Random::randn_mat(n_n, nsim);
  }

  return y_n + eps;
}

LIBKRIGING_EXPORT arma::mat NoiseKriging::update_simulate(const arma::vec& y_u,
                                                          const arma::vec& noise_u,
                                                          const arma::mat& X_u) {
  // Preserve original error-check ordering (y/X dim → X/m_X dim → noise dim)
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  if (noise_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Noise vector should have same length as X_u: " + std::to_string(noise_u.n_elem)
                             + " instead of " + std::to_string(X_u.n_rows));

  const arma::vec diag_uu = 1.0 + noise_u / m_sigma2;
  arma::mat y_up = update_simulate_impl(y_u,
                                        X_u,
                                        /*allow_cache=*/false,
                                        /*R_uu_factor=*/1.0,
                                        /*R_uu_diag=*/diag_uu,
                                        /*R_uo_factor=*/1.0,
                                        /*R_un_factor=*/1.0,
                                        /*R_un_coincident_to_one=*/false,
                                        /*Sigma_divisor=*/1.0);

  // Add noise (same semantics as NoiseKriging::simulate: RNG stream continues
  // from the reset_seed inside update_simulate_impl).
  const arma::uword n_n = lastsim_Xn_n.n_cols;
  arma::mat eps = arma::mat(n_n, lastsim_nsim, arma::fill::none);
  if (lastsim_with_noise.n_elem == 1)
    eps = lastsim_with_noise.at(0) * Random::randn_mat(n_n, lastsim_nsim);
  else if (lastsim_with_noise.n_elem == n_n) {
    eps.each_col() = lastsim_with_noise;
    eps = eps % Random::randn_mat(n_n, lastsim_nsim);
  }
  return y_up + eps;
}

/** Add new conditional data points to previous (X,y), then perform new fit.
 * @param y_u is n_u length column vector of new output
 * @param noise_u is n_u length column vector of new output variance
 * @param X_u is n_u*d matrix of new input
 * @param refit is true if we want to re-fit the model
 */
LIBKRIGING_EXPORT void NoiseKriging::update(const arma::vec& y_u,
                                            const arma::vec& noise_u,
                                            const arma::mat& X_u,
                                            const bool refit) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (noise_u.n_elem != y_u.n_elem)
    throw std::runtime_error("Dimension of new data should be the same:\n noise: (" + std::to_string(noise_u.n_elem)
                             + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x" + std::to_string(m_X.n_cols)
                             + "), new X: (...x" + std::to_string(X_u.n_cols) + ")");

  // rebuild starting parameters
  Parameters parameters;
  if (refit) {  // re-fit with parameter optimization
    if (m_est_beta)
      parameters
          = Parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY * arma::ones<arma::vec>(1)),
                       this->m_est_sigma2,
                       std::make_optional(trans(this->m_theta) % this->m_scaleX),
                       this->m_est_theta,
                       std::make_optional(arma::ones<arma::vec>(0)),
                       true};
    else
      parameters
          = Parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY * arma::ones<arma::vec>(1)),
                       this->m_est_sigma2,
                       std::make_optional(trans(this->m_theta) % this->m_scaleX),
                       this->m_est_theta,
                       std::make_optional(trans(this->m_beta) * this->m_scaleY),
                       false};
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                              y_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols(m_noise * this->m_scaleY * this->m_scaleY,
                              noise_u),  // de-normalize previous data according to suite unnormed new data
              arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
              m_regmodel,
              m_normalize,
              m_optim,
              m_objective,
              parameters);
  } else {  // incremental update without parameter re-optimization
    const arma::vec noise_n_u = noise_u / (m_scaleY * m_scaleY);
    update_no_refit_impl(
        y_u,
        X_u,
        /*extend_class_data=*/
        [this, noise_n_u] { m_noise = arma::join_cols(m_noise, noise_n_u); },
        /*build_model=*/[this] { return make_Model(m_theta, m_sigma2, nullptr); });
  }
}

LIBKRIGING_EXPORT std::string NoiseKriging::summary() const {
  std::ostringstream oss;
  if (summary_top(oss))  // auto-emits the `* noise:` line because m_noise is non-empty
    summary_bottom(oss);
  return oss.str();
}

void NoiseKriging::save(const std::string filename) const {
  nlohmann::json j;
  j["version"] = 2;
  j["content"] = "NoiseKriging";
  dump_common_to_json(j);  // writes m_noise too (NoiseKriging always has it set)

  std::ofstream f(filename);
  f << std::setw(4) << j;
}

NoiseKriging NoiseKriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version != 2)
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
  std::string content = j["content"].template get<std::string>();
  if (content != "NoiseKriging")
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'NoiseKriging'"));

  NoiseKriging kr(j["covType"].template get<std::string>());  // _Cov_pow & std::function embedded by make_Cov
  kr.load_common_from_json(j);
  return kr;
}
