// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Optim.hpp"

#include <cstdlib>
#include <string>
#include "libKriging/utils/lk_armadillo.hpp"

// Helper function to read environment variables and convert to appropriate type
namespace {
template <typename T>
T get_env_or_default(const char* var_name, T default_value) {
  const char* env_value = std::getenv(var_name);
  if (env_value == nullptr) {
    return default_value;
  }

  try {
    if constexpr (std::is_same_v<T, int>) {
      return std::stoi(env_value);
    } else if constexpr (std::is_same_v<T, double>) {
      return std::stod(env_value);
    } else if constexpr (std::is_same_v<T, bool>) {
      std::string val(env_value);
      return (val == "1" || val == "true" || val == "TRUE" || val == "True");
    }
  } catch (...) {
    // If parsing fails, return default
    return default_value;
  }
  return default_value;
}
}  // namespace

bool Optim::reparametrize = get_env_or_default("LK_REPARAMETRIZE", true);

LIBKRIGING_EXPORT void Optim::use_reparametrize(bool do_reparametrize) {
  Optim::reparametrize = do_reparametrize;
};

LIBKRIGING_EXPORT bool Optim::is_reparametrized() {
  return Optim::reparametrize;
};

// Unified reparametrization: gamma = log(theta), theta = exp(gamma).
// Chain rule (theta-space grad -> gamma-space grad): dL/dgamma = dL/dtheta * theta.
// Hessian (theta-space hess -> gamma-space hess):
//   d2L/dgamma_k dgamma_l = theta_k * theta_l * d2L/dtheta_k dtheta_l + delta_kl * theta_k * dL/dtheta_k
std::function<double(const double&)> Optim::reparam_to_ = [](const double& _theta) { return std::log(_theta); };
std::function<arma::vec(const arma::vec&)> Optim::reparam_to
    = [](const arma::vec& _theta) { return arma::conv_to<arma::colvec>::from(arma::log(_theta)); };
std::function<double(const double&)> Optim::reparam_from_ = [](const double& _gamma) { return std::exp(_gamma); };
std::function<arma::vec(const arma::vec&)> Optim::reparam_from
    = [](const arma::vec& _gamma) { return arma::conv_to<arma::colvec>::from(arma::exp(_gamma)); };
std::function<arma::vec(const arma::vec&, const arma::vec&)> Optim::reparam_from_deriv
    = [](const arma::vec& _theta, const arma::vec& _grad) { return arma::conv_to<arma::colvec>::from(_grad % _theta); };
std::function<arma::mat(const arma::vec&, const arma::vec&, const arma::mat&)> Optim::reparam_from_deriv2
    = [](const arma::vec& _theta, const arma::vec& _grad_theta, const arma::mat& _hess_theta) {
        // Returns gamma-space Hessian given theta-space grad and Hessian.
        arma::mat H = _hess_theta % (_theta * _theta.t());
        H.diag() += _grad_theta % _theta;
        return H;
      };

double Optim::theta_lower_factor = get_env_or_default("LK_THETA_LOWER_FACTOR", 0.02);

LIBKRIGING_EXPORT void Optim::set_theta_lower_factor(double _theta_lower_factor) {
  Optim::theta_lower_factor = _theta_lower_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_lower_factor() {
  return Optim::theta_lower_factor;
};

double Optim::theta_upper_factor = get_env_or_default("LK_THETA_UPPER_FACTOR", 10.0);

LIBKRIGING_EXPORT void Optim::set_theta_upper_factor(double _theta_upper_factor) {
  Optim::theta_upper_factor = _theta_upper_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_upper_factor() {
  return Optim::theta_upper_factor;
};

bool Optim::variogram_bounds_heuristic = get_env_or_default("LK_VARIOGRAM_BOUNDS_HEURISTIC", true);

LIBKRIGING_EXPORT void Optim::use_variogram_bounds_heuristic(bool _variogram_bounds_heuristic) {
  Optim::variogram_bounds_heuristic = _variogram_bounds_heuristic;
};

LIBKRIGING_EXPORT bool Optim::variogram_bounds_heuristic_used() {
  return Optim::variogram_bounds_heuristic;
};

int Optim::log_level = get_env_or_default("LK_LOG_LEVEL", 0);

LIBKRIGING_EXPORT void Optim::set_log_level(int l) {
  Optim::log_level = l;
};

LIBKRIGING_EXPORT int Optim::get_log_level() {
  return Optim::log_level;
};

int Optim::max_restart = get_env_or_default("LK_MAX_RESTART", 10);

int Optim::max_iteration = get_env_or_default("LK_MAX_ITERATION", 20);

LIBKRIGING_EXPORT void Optim::set_max_iteration(int max_iteration_val) {
  Optim::max_iteration = max_iteration_val;
};

LIBKRIGING_EXPORT int Optim::get_max_iteration() {
  return Optim::max_iteration;
};

double Optim::gradient_tolerance = get_env_or_default("LK_GRADIENT_TOLERANCE", 0.001);

LIBKRIGING_EXPORT void Optim::set_gradient_tolerance(double gradient_tolerance_val) {
  Optim::gradient_tolerance = gradient_tolerance_val;
};

LIBKRIGING_EXPORT double Optim::get_gradient_tolerance() {
  return Optim::gradient_tolerance;
};

double Optim::objective_rel_tolerance = get_env_or_default("LK_OBJECTIVE_REL_TOLERANCE", 0.001);

LIBKRIGING_EXPORT void Optim::set_objective_rel_tolerance(double objective_rel_tolerance_val) {
  Optim::objective_rel_tolerance = objective_rel_tolerance_val;
};

LIBKRIGING_EXPORT double Optim::get_objective_rel_tolerance() {
  return Optim::objective_rel_tolerance;
};

int Optim::thread_start_delay_ms = get_env_or_default("LK_THREAD_START_DELAY_MS", 10);

LIBKRIGING_EXPORT void Optim::set_thread_start_delay_ms(int delay_ms) {
  Optim::thread_start_delay_ms = delay_ms;
};

LIBKRIGING_EXPORT int Optim::get_thread_start_delay_ms() {
  return Optim::thread_start_delay_ms;
};

LIBKRIGING_EXPORT std::pair<std::string, int> Optim::parse_method(const std::string& method,
                                                                  const std::string& prefix) {
  if (method.rfind(prefix, 0) != 0)
    return {method, 1};

  // Find end of digits after prefix
  size_t pos = prefix.size();
  size_t num_end = pos;
  while (num_end < method.size() && std::isdigit(static_cast<unsigned char>(method[num_end])))
    ++num_end;

  if (num_end == pos)
    return {method, 1};

  int multistart = 1;
  try {
    multistart = std::stoi(method.substr(pos, num_end - pos));
  } catch (...) {
    multistart = 1;
  }
  if (multistart < 1)
    multistart = 1;

  // Reconstruct base method: prefix + everything after the digits
  std::string base = prefix + method.substr(num_end);
  return {base, multistart};
}

LIBKRIGING_EXPORT std::pair<arma::vec, arma::vec> Optim::theta_bounds(const arma::vec& maxdX,
                                                                      const arma::mat& dX,
                                                                      const arma::vec& y,
                                                                      arma::uword n) {
  arma::vec theta_lower = Optim::theta_lower_factor * maxdX;
  arma::vec theta_upper = Optim::theta_upper_factor * maxdX;

  if (Optim::variogram_bounds_heuristic && n > 0 && dX.n_cols == n * n) {
    arma::vec dy2(n * n, arma::fill::zeros);
    for (arma::uword ij = 0; ij < dy2.n_elem; ++ij) {
      arma::uword i = ij / n;
      arma::uword j = ij % n;
      if (i < j) {
        double d = y.at(i) - y.at(j);
        dy2[ij] = d * d;
        dy2[j * n + i] = dy2[ij];
      }
    }
    arma::vec dy2dX2_slope = dy2 / arma::sum(dX % dX, 0).t();
    dy2dX2_slope.replace(arma::datum::nan, 0.0);
    double wsum = arma::sum(dy2dX2_slope);
    if (wsum > 0.0) {
      arma::vec w = dy2dX2_slope / wsum;
      arma::vec steepest_dX_mean = arma::abs(dX) * w;
      theta_lower = arma::max(theta_lower, Optim::theta_lower_factor * steepest_dX_mean);
      theta_lower = arma::min(theta_lower, theta_upper);
      theta_upper = arma::max(theta_lower, theta_upper);
    }
  }
  return {theta_lower, theta_upper};
}

int Optim::thread_pool_size = get_env_or_default("LK_THREAD_POOL_SIZE", 0);  // 0 means auto-detect (ncpu/4)

LIBKRIGING_EXPORT void Optim::set_thread_pool_size(int pool_size) {
  Optim::thread_pool_size = pool_size;
};

LIBKRIGING_EXPORT int Optim::get_thread_pool_size() {
  return Optim::thread_pool_size;
};
