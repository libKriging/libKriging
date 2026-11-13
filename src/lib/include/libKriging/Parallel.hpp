#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_PARALLEL_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_PARALLEL_HPP

#include <string>

#include "libKriging/libKriging_exports.h"

// Parallelism of the library (docs/dev/parallel.md): one thread budget bounds
// its three levels, the multistart workers (std::thread pool), the OpenMP loops
// (covariance, gradient, predict / simulate cross-covariances) and the BLAS
// threads. The settings below only shape how that budget is used: there is no
// separate OpenMP or BLAS thread count, which would let the levels together
// exceed the budget (oversubscription).
//
// Process-wide, configuration-before-use as Optim's settings (Optim.hpp): set
// them before the calls, not while calls run in other threads.
//
// The thread settings of Optim (set_num_threads_percent, get_num_threads,
// set_thread_pool_size, set_thread_start_delay_ms, set_blas_threads_min_n) are
// the same settings, kept for compatibility.
class Parallel {
 public:
  // --- budget
  /// Share of the physical cores used by the library: x <= 1 a ratio (0.5),
  /// x > 1 a percentage (50), 0 automatic (100 %). Default from
  /// LK_NUM_THREADS_PERCENT. Never more than an explicit OMP_NUM_THREADS /
  /// omp_set_num_threads of the calling thread.
  LIBKRIGING_EXPORT static void set_num_threads_percent(double x);
  LIBKRIGING_EXPORT static double get_num_threads_percent();
  /// Resolved thread budget (number of threads), read only.
  LIBKRIGING_EXPORT static int get_num_threads();

  // --- how the budget is distributed
  /// Number of multistart workers (0: automatic, the budget); each worker's
  /// BLAS and OpenMP loops get budget / workers. Default from LK_THREAD_POOL_SIZE.
  LIBKRIGING_EXPORT static void set_thread_pool_size(int pool_size);
  LIBKRIGING_EXPORT static int get_thread_pool_size();
  /// Stagger of the WarpKriging multistart workers, in ms. Default from
  /// LK_THREAD_START_DELAY_MS.
  LIBKRIGING_EXPORT static void set_thread_start_delay_ms(int delay_ms);
  LIBKRIGING_EXPORT static int get_thread_start_delay_ms();
  /// Small-problem BLAS rule (OpenBLAS): calls on fewer training observations
  /// than this run the BLAS on 1 thread. -1 automatic (default), 0 off, n > 0
  /// this threshold. Default from LK_BLAS_THREADS_MIN_N.
  LIBKRIGING_EXPORT static void set_blas_threads_min_n(int min_n);
  LIBKRIGING_EXPORT static int get_blas_threads_min_n();
  /// Number of rows from which the covariance and gradient loops run on the
  /// OpenMP threads (below: serial). Default: the platform value of the build
  /// (200 on Linux, 800 elsewhere unless measured), or LK_OMP_MIN_ROWS. A
  /// performance setting: the results do not depend on it.
  LIBKRIGING_EXPORT static void set_omp_min_rows(int n);
  LIBKRIGING_EXPORT static int get_omp_min_rows();

  // --- read only, per level
  /// BLAS whose thread count the library reads: "mkl", "openblas",
  /// "accelerate" (macOS, neither of them loaded) or "unknown".
  LIBKRIGING_EXPORT static std::string blas_backend();
  /// Current thread count of that BLAS in the calling thread (MKL: its
  /// thread-local count if set), 0 if unknown.
  LIBKRIGING_EXPORT static int blas_num_threads();
  /// True when the OpenMP loops are compiled in (OpenMP build, not compiled
  /// out under ThreadSanitizer).
  LIBKRIGING_EXPORT static bool omp_enabled();
};

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_PARALLEL_HPP
