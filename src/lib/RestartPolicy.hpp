#ifndef LIBKRIGING_SRC_LIB_RESTARTPOLICY_HPP
#define LIBKRIGING_SRC_LIB_RESTARTPOLICY_HPP

// Private (not installed) helper shared by Kriging.cpp and WarpKriging.cpp
// (joint BFGS, and the inner BFGS of AdamBFGS through its inner_restart).

#include "libKriging/utils/lk_armadillo.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace libKriging {
namespace detail {

/// When to restart an L-BFGS-B run, and from where (docs/dev/rationalization/T2-restart-policy.md).
///
/// A run is restarted (at most max_restart times) when it
///  - ended with ABNORMAL_TERMINATION_IN_LNSRCH,
///  - stopped in <= 2 iterations (start point strangely optimal, or a corner of
///    the bounds where the gradient points outward),
///  - ended at the lower bound of a free theta,
///  - ended at a long-range local optimum (decided by the caller; at most
///    max_long_range_restart of these restarts),
///  - is worse than the best run so far by more than the optimizer's own
///    tolerance factr * epsmch (if worse_criterion), unless this run was itself
///    started only because the previous one was worse: a worse restart does not
///    trigger another one by itself.
/// The restart point is the start point contracted toward the lower bound,
/// (theta_start + theta_lower) / 2^retry, kept inside the bounds.
///
/// Usage, after each run: record(f_opt), then restart(...); when it returns
/// true, start the next run from restart_theta(...).
class RestartPolicy {
 public:
  enum Reason : unsigned { None = 0, Abnormal = 1, FewIterations = 2, AtLowerBound = 4, LongRange = 8, Worse = 16 };

  struct Config {
    int max_restart = 0;
    double factr = 1e7;  // L-BFGS-B factr: "no improvement" below factr * epsmch is no change
    bool worse_criterion = false;
    int max_long_range_restart = 0;
  };

  explicit RestartPolicy(const Config& config) : m_config(config) {}

  /// Records the objective value reached by the last run (minimized); returns
  /// true if it is strictly the best so far.
  bool record(double f_opt) {
    // "No improvement" below the optimizer's own relative tolerance is no
    // change: an exact comparison of two equal optima made the restart
    // decision depend on the rounding (BLAS thread count, starting points)
    const double f_tol = m_config.factr * arma::datum::eps * std::max(1.0, std::abs(std::min(m_best, f_opt)));
    m_worse = (m_best < arma::datum::inf) && (f_opt > m_best + f_tol);
    m_previous_best = m_best;
    if (f_opt < m_best) {
      m_best = f_opt;
      return true;
    }
    return false;
  }

  /// Decides on a restart after the last recorded run; on true, the retry
  /// number is incremented (restart_theta then gives the restart point).
  /// long_range: the caller's long-range criterion (ignored past max_long_range_restart).
  bool restart(const std::string& task, int num_iters, bool at_lower_bound, bool long_range = false) {
    if (m_retry >= m_config.max_restart)
      return false;
    unsigned reasons = None;
    if (task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0)
      reasons |= Abnormal;
    if (num_iters <= 2)
      reasons |= FewIterations;
    if (at_lower_bound)
      reasons |= AtLowerBound;
    if (long_range && m_long_range_restarts < m_config.max_long_range_restart)
      reasons |= LongRange;
    if (m_config.worse_criterion && m_worse && m_last_reasons != Worse)
      reasons |= Worse;
    if (reasons == None)
      return false;
    if (reasons & LongRange)
      m_long_range_restarts++;
    m_last_reasons = reasons;
    m_retry++;
    return true;
  }

  /// Counts a restart decided by the caller from the point reached (e.g.
  /// iteration budget exhausted while still improving): not a "worse" one.
  bool resume() {
    if (m_retry >= m_config.max_restart)
      return false;
    m_last_reasons = None;
    m_retry++;
    return true;
  }

  /// Start point of the current restart: theta_start contracted toward
  /// theta_lower, kept inside [theta_lower, theta_upper] (as L-BFGS-B projects
  /// its starting point onto fixed bounds).
  arma::vec restart_theta(const arma::vec& theta_start,
                          const arma::vec& theta_lower,
                          const arma::vec& theta_upper) const {
    return arma::max(arma::min(arma::vec((theta_start + theta_lower) / std::pow(2.0, m_retry)), theta_upper),
                     theta_lower);
  }

  /// Smallest distance to the lower bound over the free dimensions (a pinned
  /// one, lower == upper, is always "at its bound"); inf if none is free.
  static double distance_to_lower(const arma::vec& x, const arma::vec& lower, const arma::vec& upper) {
    const arma::uvec free_dims = arma::find(upper > lower);
    return free_dims.is_empty() ? arma::datum::inf : arma::min(arma::abs(x(free_dims) - lower(free_dims)));
  }

  /// Number of restarts (and resumes) so far.
  int retry() const { return m_retry; }
  /// Reasons of the last restart (bitwise or of Reason).
  unsigned last_reasons() const { return m_last_reasons; }
  double best() const { return m_best; }
  /// Best value before the last record().
  double previous_best() const { return m_previous_best; }
  bool long_range_allowed() const { return m_long_range_restarts < m_config.max_long_range_restart; }

 private:
  Config m_config;
  int m_retry = 0;
  int m_long_range_restarts = 0;
  unsigned m_last_reasons = None;
  bool m_worse = false;
  double m_best = std::numeric_limits<double>::infinity();
  double m_previous_best = std::numeric_limits<double>::infinity();
};

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_RESTARTPOLICY_HPP
