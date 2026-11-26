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
// Since 2.0 the thread settings are here only: Optim's set_num_threads_percent,
// get_num_threads, set_thread_pool_size, set_thread_start_delay_ms and
// set_blas_threads_min_n (and their getters) were moved to Parallel (#468).
// Defaults from the LK_* environment variables, read when the library is
// loaded; a setter call overrides them (docs/Settings.md).
class Parallel {
 public:
  // --- budget
  /// Share of the physical cores used by the library: x <= 1 a ratio (0.5),
  /// x > 1 a percentage (50), 0 automatic (100 %). Default from
  /// LK_NUM_THREADS_PERCENT. Never more than an explicit OMP_NUM_THREADS /
  /// omp_set_num_threads of the calling thread nor the OpenBLAS count
  /// (OPENBLAS_NUM_THREADS) in effect when the library is first used. 1 thread
  /// in a forked child (fork(), R's parallel::mclapply) until
  /// set_num_threads_percent is called in it.
  LIBKRIGING_EXPORT static void set_num_threads_percent(double x);
  LIBKRIGING_EXPORT static double get_num_threads_percent();
  /// Resolved thread budget (number of threads), read only.
  LIBKRIGING_EXPORT static int get_num_threads();

  // --- how the budget is distributed
  /// Number of multistart workers (Kriging, WarpKriging, MLPKriging), at most
  /// the number of starts (0: automatic, the budget); each worker's BLAS and
  /// OpenMP loops get budget / workers. n >= 0. Default from LK_THREAD_POOL_SIZE.
  LIBKRIGING_EXPORT static void set_thread_pool_size(int pool_size);
  LIBKRIGING_EXPORT static int get_thread_pool_size();
  /// Stagger of the WarpKriging / MLPKriging multistart workers: the first start
  /// of worker k waits k times this, in ms (default 10, >= 0). Default from
  /// LK_THREAD_START_DELAY_MS.
  LIBKRIGING_EXPORT static void set_thread_start_delay_ms(int delay_ms);
  LIBKRIGING_EXPORT static int get_thread_start_delay_ms();
  /// Small-problem BLAS rule (OpenBLAS): calls on fewer training observations
  /// than this run the BLAS on 1 thread. -1 automatic (default), 0 off, n > 0
  /// this threshold (min_n >= -1). Default from LK_BLAS_THREADS_MIN_N. The
  /// automatic threshold is 500 when the BLAS threads and the OpenMP team
  /// outnumber the logical CPUs (no SMT headroom: VMs, CI runners), off
  /// otherwise, except with MSVC on Windows where it is always 500 (#473).
  LIBKRIGING_EXPORT static void set_blas_threads_min_n(int min_n);
  LIBKRIGING_EXPORT static int get_blas_threads_min_n();
  /// Number of rows from which the covariance and gradient loops run on the
  /// OpenMP threads (below: serial). Default: the platform value of the build
  /// (200 on Linux and with MSVC, 800 elsewhere until measured), or LK_OMP_MIN_ROWS. A
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
