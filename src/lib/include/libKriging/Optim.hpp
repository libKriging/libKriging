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

  // Parse an optimizer method string of the form "<prefix><N>[<suffix>]".
  // Returns {base_method_without_N, multistart_count}.
  //   parse_method("BFGS")         -> {"BFGS", 1}
  //   parse_method("BFGS10")       -> {"BFGS", 10}
  //   parse_method("BFGS5+Adam")   -> {"BFGS+Adam", 5}
  //   parse_method("Newton3")      -> {"Newton", 3}   (with prefix="Newton")
  LIBKRIGING_EXPORT static std::pair<std::string, int> parse_method(const std::string& method,
                                                                    const std::string& prefix = "BFGS");

  // Data-driven θ bounds with optional variogram-slope heuristic (gated by
  // Optim::variogram_bounds_heuristic). The dX matrix holds pairwise input
  // (or feature) differences laid out as d × n² with column (i*n+j) = x_i − x_j.
  LIBKRIGING_EXPORT static std::pair<arma::vec, arma::vec> theta_bounds(const arma::vec& maxdX,
                                                                        const arma::mat& dX,
                                                                        const arma::vec& y,
                                                                        arma::uword n);

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

  // Thread budget of the library: the share of the physical cores it uses,
  // x <= 1 a ratio (0.5), x > 1 a percentage (50), 0 = automatic = 100 %.
  // Default from LK_NUM_THREADS_PERCENT. Never more than an explicit
  // OMP_NUM_THREADS / omp_set_num_threads of the calling thread nor the
  // OpenBLAS count (OPENBLAS_NUM_THREADS) in effect when the library is first
  // used. The multistart pool gets it first (one worker per start, up to the
  // budget), each worker's BLAS / OpenMP calls the rest (budget / workers).
  static double num_threads_percent;
  LIBKRIGING_EXPORT static void set_num_threads_percent(double x);
  LIBKRIGING_EXPORT static double get_num_threads_percent();
  /// Resolved thread budget (number of threads), read only.
  LIBKRIGING_EXPORT static int get_num_threads();
  /// Diagnostics, read only: the OpenBLAS thread count currently set (0: not
  /// OpenBLAS, or its count cannot be read) and the OpenMP limit of the calling
  /// thread (omp_get_max_threads(); 0 without OpenMP). In a child forked by a
  /// process that loaded the library, the OpenBLAS count and get_num_threads()
  /// are 1 (pthread_atfork handlers); the OpenMP limit may not be (LLVM libomp
  /// resets it), it does not bound the library's loops there.
  LIBKRIGING_EXPORT static int get_blas_num_threads();
  LIBKRIGING_EXPORT static int get_omp_num_threads();

  // Number of multistart workers (Kriging, WarpKriging, MLPKriging), at most
  // the number of starts; 0 = automatic = the thread budget (one thread per
  // worker when the starts outnumber it). Default from LK_THREAD_POOL_SIZE.
  static int thread_pool_size;
  LIBKRIGING_EXPORT static void set_thread_pool_size(int pool_size);
  LIBKRIGING_EXPORT static int get_thread_pool_size();
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_LINLIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_OPTIM_HPPEARALGEBRA_HPP
