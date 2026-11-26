// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Optim.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include "EnvSettings.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

// Defaults from the LK_* environment variables, read at load (EnvSettings.hpp)
using libKriging::detail::env_bool;
using libKriging::detail::env_double;
using libKriging::detail::env_int;

bool Optim::reparametrize = env_bool("LK_REPARAMETRIZE", true);

LIBKRIGING_EXPORT void Optim::use_reparametrize(bool do_reparametrize) {
  Optim::reparametrize = do_reparametrize;
};

LIBKRIGING_EXPORT bool Optim::is_reparametrized() {
  return Optim::reparametrize;
};

// Unified reparametrization: gamma = log(theta), theta = exp(gamma).
// Chain rule (theta-space grad -> gamma-space grad): dL/dgamma = dL/dtheta * theta.
std::function<double(const double&)> Optim::reparam_to_ = [](const double& _theta) { return std::log(_theta); };
std::function<arma::vec(const arma::vec&)> Optim::reparam_to
    = [](const arma::vec& _theta) { return arma::conv_to<arma::colvec>::from(arma::log(_theta)); };
std::function<double(const double&)> Optim::reparam_from_ = [](const double& _gamma) { return std::exp(_gamma); };
std::function<arma::vec(const arma::vec&)> Optim::reparam_from
    = [](const arma::vec& _gamma) { return arma::conv_to<arma::colvec>::from(arma::exp(_gamma)); };
std::function<arma::vec(const arma::vec&, const arma::vec&)> Optim::reparam_from_deriv
    = [](const arma::vec& _theta, const arma::vec& _grad) { return arma::conv_to<arma::colvec>::from(_grad % _theta); };

double Optim::theta_lower_factor = env_double("LK_THETA_LOWER_FACTOR", 0.02, 0.0);

LIBKRIGING_EXPORT void Optim::set_theta_lower_factor(double _theta_lower_factor) {
  Optim::theta_lower_factor = _theta_lower_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_lower_factor() {
  return Optim::theta_lower_factor;
};

double Optim::theta_upper_factor = env_double("LK_THETA_UPPER_FACTOR", 10.0, 0.0);

LIBKRIGING_EXPORT void Optim::set_theta_upper_factor(double _theta_upper_factor) {
  Optim::theta_upper_factor = _theta_upper_factor;
};

LIBKRIGING_EXPORT double Optim::get_theta_upper_factor() {
  return Optim::theta_upper_factor;
};

bool Optim::variogram_bounds_heuristic = env_bool("LK_VARIOGRAM_BOUNDS_HEURISTIC", true);

LIBKRIGING_EXPORT void Optim::use_variogram_bounds_heuristic(bool _variogram_bounds_heuristic) {
  Optim::variogram_bounds_heuristic = _variogram_bounds_heuristic;
};

LIBKRIGING_EXPORT bool Optim::variogram_bounds_heuristic_used() {
  return Optim::variogram_bounds_heuristic;
};

int Optim::log_level = env_int("LK_LOG_LEVEL", 0, 0);

LIBKRIGING_EXPORT void Optim::set_log_level(int l) {
  Optim::log_level = l;
};

LIBKRIGING_EXPORT int Optim::get_log_level() {
  return Optim::log_level;
};

int Optim::max_restart = env_int("LK_MAX_RESTART", 10, 0);

LIBKRIGING_EXPORT void Optim::set_max_restart(int max_restart_val) {
  if (max_restart_val < 0)
    throw std::invalid_argument("set_max_restart: expected max_restart >= 0");
  Optim::max_restart = max_restart_val;
};

LIBKRIGING_EXPORT int Optim::get_max_restart() {
  return Optim::max_restart;
};

int Optim::max_iteration = env_int("LK_MAX_ITERATION", 20, 0);

LIBKRIGING_EXPORT void Optim::set_max_iteration(int max_iteration_val) {
  Optim::max_iteration = max_iteration_val;
};

LIBKRIGING_EXPORT int Optim::get_max_iteration() {
  return Optim::max_iteration;
};

double Optim::gradient_tolerance = env_double("LK_GRADIENT_TOLERANCE", 0.001, 0.0);

LIBKRIGING_EXPORT void Optim::set_gradient_tolerance(double gradient_tolerance_val) {
  Optim::gradient_tolerance = gradient_tolerance_val;
};

LIBKRIGING_EXPORT double Optim::get_gradient_tolerance() {
  return Optim::gradient_tolerance;
};

double Optim::objective_rel_tolerance = env_double("LK_OBJECTIVE_REL_TOLERANCE", 0.001, 0.0);

LIBKRIGING_EXPORT void Optim::set_objective_rel_tolerance(double objective_rel_tolerance_val) {
  Optim::objective_rel_tolerance = objective_rel_tolerance_val;
};

LIBKRIGING_EXPORT double Optim::get_objective_rel_tolerance() {
  return Optim::objective_rel_tolerance;
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
  // A constant input (zero range) carries no information on its range
  // parameter, and [0, 0] bounds would give theta = 0, i.e. a division by
  // zero in the kernel: pin it to 1 instead (lower = upper).
  for (arma::uword k = 0; k < maxdX.n_elem; ++k) {
    if (maxdX.at(k) == 0.0) {
      theta_lower.at(k) = 1.0;
      theta_upper.at(k) = 1.0;
    }
  }
  return {theta_lower, theta_upper};
}
