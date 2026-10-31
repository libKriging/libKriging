#ifndef LIBKRIGING_SRC_LIB_THREADBUDGET_HPP
#define LIBKRIGING_SRC_LIB_THREADBUDGET_HPP

// Private (not installed): single source of truth for the number of threads the
// library uses (docs/dev/parallel.md). Three levels share it:
// the multistart / task pool (TaskPool.hpp) first, then, inside each pool worker
// or in the calling thread, the OpenMP loops and the BLAS, each at the remaining
// share: BLAS calls and OpenMP loops run one after the other, so they do not
// split it (bench/thread-budget/RESULTS.md).

namespace libKriging {
namespace detail {

/// Physical cores available to the process: the cores of the CPUs in the
/// affinity mask (Linux: sysfs topology, capped by the cgroup CPU quota; macOS:
/// hw.physicalcpu; Windows: GetLogicalProcessorInformationEx), at least 1;
/// std::thread::hardware_concurrency() when the topology cannot be read.
/// Computed once.
int physical_cores();

/// Thread budget of the library: round(share × physical_cores()), share =
/// Optim::num_threads_percent (x <= 1: ratio, x > 1: percentage, 0 = 100 %), at
/// least 1, never more than the calling thread's OpenMP limit
/// (omp_get_max_threads(): OMP_NUM_THREADS, omp_set_num_threads). The OpenBLAS
/// count bounds the BLAS only. Optim::get_num_threads().
int budget_full();

/// Budget of a call: budget_full() divided by the number of public calls
/// running concurrently in other user threads (concurrent_calls(), counted by
/// BudgetScope), at least 1; budget_full() inside a pool worker (its share is
/// set by the pool) and for a single call. Read when the call sizes its pool
/// or an OpenMP loop, so a call started alone keeps the pool it sized.
int budget_total();

/// Public calls (BudgetScope) running now in user threads, nested calls of a
/// thread counted once.
int concurrent_calls();

/// OpenBLAS count for a call with `share` threads, 0 = leave it as it is: no
/// OpenBLAS, or OpenBLAS loaded with at most `share` threads (an explicit
/// OPENBLAS_NUM_THREADS below the budget is kept); otherwise `share`.
int blas_threads_for(int share);

/// Threads for an OpenMP loop of the calling thread: its pool share inside a
/// pool worker, budget_total() otherwise. Use as num_threads(omp_threads()).
int omp_threads();

/// True in a TaskPool worker.
bool in_pool_worker();

/// Set by the task pool in each worker for its lifetime: marks it as a pool
/// worker and gives its OpenMP loops `omp` threads. Never touches the OpenMP
/// ICV of any thread.
struct ScopedShare {
  explicit ScopedShare(int omp);
  ~ScopedShare();
  ScopedShare(const ScopedShare&) = delete;
  ScopedShare& operator=(const ScopedShare&) = delete;

 private:
  int saved_share;
  bool saved_in_pool;
};

/// Requests an OpenBLAS thread count of `n` for its lifetime (no-op for
/// n <= 0, and without OpenBLAS: Accelerate, MKL, reference BLAS). The count is
/// process-global: the scopes of all threads share one mutex-protected state,
/// the count in effect is the smallest live request, the first scope saves the
/// count it finds and the last one restores it, whatever the order in which
/// concurrent scopes close.
struct ScopedBlasThreads {
  explicit ScopedBlasThreads(int n);
  ~ScopedBlasThreads();
  ScopedBlasThreads(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;

 private:
  int requested = 0;
  unsigned generation = 0;
};

/// Counts a public call of a user thread in concurrent_calls() for its
/// lifetime; no-op inside a pool worker.
struct CallCount {
  CallCount();
  ~CallCount();
  CallCount(const CallCount&) = delete;
  CallCount& operator=(const CallCount&) = delete;

 private:
  bool counted = false;
  bool outermost = false;
  unsigned generation = 0;
};

/// Entry-point guard of the public fit / predict / simulate / update methods:
/// counts the call, then requests OpenBLAS at blas_threads_for(budget_total())
/// (OpenBLAS defaults to the logical CPUs, which costs 1.3-2x on a
/// hyperthreaded machine) until it returns; no-op inside a pool worker (the
/// pool has set the worker's share).
struct BudgetScope {
  BudgetScope();
  BudgetScope(const BudgetScope&) = delete;
  BudgetScope& operator=(const BudgetScope&) = delete;

 private:
  CallCount call;  // first: budget_total() of `blas` counts this call
  ScopedBlasThreads blas;
};

/// pthread_atfork handlers of the OpenBLAS count state (Kriging.cpp): prepare
/// locks it so no scope is half-way through an update in the child; parent
/// unlocks it; child unlocks it after dropping the scopes of the parent's
/// threads (they never close in the child).
void blas_count_fork_prepare();
void blas_count_fork_parent();
void blas_count_fork_child();

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_THREADBUDGET_HPP
