#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPP

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/libKriging_exports.h"

class Optim {
 public:
  static bool reparametrize;
  LIBKRIGING_EXPORT static void use_reparametrize(bool do_reparametrize);
  LIBKRIGING_EXPORT static bool is_reparametrized();
  static std::function<double(const double&)> reparam_to_;
  static std::function<arma::vec(const arma::vec&)> reparam_to;
  static std::function<double(const double&)> reparam_from_;
  static std::function<arma::vec(const arma::vec&)> reparam_from;
  static std::function<double(const double&, const double&)> reparam_from_deriv_;
  static std::function<arma::vec(const arma::vec&, const arma::vec&)> reparam_from_deriv;
  static std::function<arma::mat(const arma::vec&, const arma::vec&, const arma::mat&)> reparam_from_deriv2;

  static double theta_lower_factor;
  LIBKRIGING_EXPORT static void set_theta_lower_factor(double _theta_lower_factor);
  LIBKRIGING_EXPORT static double get_theta_lower_factor();

  static double theta_upper_factor;
  LIBKRIGING_EXPORT static void set_theta_upper_factor(double _theta_upper_factor);
  LIBKRIGING_EXPORT static double get_theta_upper_factor();

  static bool variogram_bounds_heuristic;
  LIBKRIGING_EXPORT static void use_variogram_bounds_heuristic(bool _variogram_bounds_heuristic);
  LIBKRIGING_EXPORT static bool variogram_bounds_heuristic_used();

  // Log levels
  static constexpr int log_none = 0;
  static constexpr int log_error = 1;
  static constexpr int log_warning = 2;
  static constexpr int log_info = 3;
  static constexpr int log_debug = 4;
  static constexpr int log_trace = 5;

  static int log_level;
  LIBKRIGING_EXPORT static void set_log_level(int t);
  LIBKRIGING_EXPORT static int get_log_level();

  static int max_restart;  // eg. for wrong convergence to bounds

  static int max_iteration;
  LIBKRIGING_EXPORT static void set_max_iteration(int max_iteration_val);
  LIBKRIGING_EXPORT static int get_max_iteration();

  static double gradient_tolerance;
  LIBKRIGING_EXPORT static void set_gradient_tolerance(double gradient_tolerance_val);
  LIBKRIGING_EXPORT static double get_gradient_tolerance();

  static double objective_rel_tolerance;
  LIBKRIGING_EXPORT static void set_objective_rel_tolerance(double objective_rel_tolerance_val);
  LIBKRIGING_EXPORT static double get_objective_rel_tolerance();

  // Thread startup delay for BFGS multistart optimization (in milliseconds)
  // Purpose: Stagger worker thread initialization to avoid race conditions
  // 
  // Background:
  //   When multiple threads start simultaneously in BFGS multistart, they can
  //   encounter race conditions during initialization, particularly with:
  //   - Armadillo matrix memory allocation
  //   - Thread-local RNG initialization
  //   - Internal library state setup
  //
  // Investigation results (with exact equivalence test):
  //   - 1ms delay:  FAILED - threads too close, race conditions observed
  //                 Small differences in results (sigma2 diff ~1.3)
  //   - 10ms delay: SUCCESS - exact equivalence achieved (diff == 0.0)
  //   - 100ms delay: SUCCESS - exact equivalence, but unnecessary overhead
  //
  // Recommendation: 10ms provides the minimum safe delay to ensure:
  //   - Deterministic behavior (BFGS20 == best of 20×BFGS1)
  //   - Minimal overhead (max 190ms for 20 threads)
  //   - Perfect reproducibility across runs
  //
  // Usage: Each worker thread i waits (i × thread_start_delay_ms) milliseconds
  //        before beginning optimization work.
  static int thread_start_delay_ms;
  LIBKRIGING_EXPORT static void set_thread_start_delay_ms(int delay_ms);
  LIBKRIGING_EXPORT static int get_thread_start_delay_ms();

  // Thread pool size for BFGS multistart optimization
  // Purpose: Limit concurrent threads to avoid oversubscription
  //
  // Default: ncpu/8 (conservative to allow nested BLAS parallelism)
  // - Each worker thread can use multiple CPU cores for BLAS operations
  // - Prevents thread thrashing and memory bandwidth saturation
  // - Can be set to 0 to use unlimited threads (one per multistart)
  //
  // Example: On a 20-core system:
  //   - pool_size = 20/8 = 2 workers at a time
  //   - Each worker uses ~10 BLAS threads (20/2)
  //   - BFGS20 runs as 10 batches of 2 workers each
  static int thread_pool_size;
  LIBKRIGING_EXPORT static void set_thread_pool_size(int pool_size);
  LIBKRIGING_EXPORT static int get_thread_pool_size();
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINLIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPPEARALGEBRA_HPP
