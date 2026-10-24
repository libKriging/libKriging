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

/// Thread budget: round(share × physical_cores()), share = Optim::num_threads_percent
/// (x <= 1: ratio, x > 1: percentage, 0 = 100 %), at least 1, never more than the
/// calling thread's OpenMP limit (omp_get_max_threads(): OMP_NUM_THREADS,
/// omp_set_num_threads). The OpenBLAS count bounds the BLAS only. 1 in a
/// forked child (set_forked_child()).
int budget_total();

/// Called by the pthread_atfork child handler (Kriging.cpp): caps the budget
/// of the forked child at 1 thread until Optim::set_num_threads_percent() is
/// called in it. The OpenMP ICV set to 1 by that handler is not enough: LLVM
/// libomp registers its own child handler after it (at its first use) and
/// resets the ICV to OMP_NUM_THREADS / the CPU count (#420).
void set_forked_child();

/// Clears the cap of set_forked_child() (Optim::set_num_threads_percent).
void clear_forked_child();

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

/// Sets the (process-global) OpenBLAS thread count to `n` for its lifetime and
/// restores it after; no-op when OpenBLAS is absent (Accelerate, MKL, reference
/// BLAS), when the count is already `n`, or for n <= 0.
struct ScopedBlasThreads {
  explicit ScopedBlasThreads(int n);
  ~ScopedBlasThreads();
  ScopedBlasThreads(const ScopedBlasThreads&) = delete;
  ScopedBlasThreads& operator=(const ScopedBlasThreads&) = delete;

 private:
  int saved = 0;
};

/// Entry-point guard of the public fit / predict / simulate / update methods:
/// OpenBLAS at blas_threads_for(budget) for the call
/// (OpenBLAS defaults to the logical CPUs, which costs 1.3-2x on a
/// hyperthreaded machine), restored after; no-op inside a pool worker (the
/// pool has set the worker's share).
struct BudgetScope {
  BudgetScope();
  BudgetScope(const BudgetScope&) = delete;
  BudgetScope& operator=(const BudgetScope&) = delete;

 private:
  ScopedBlasThreads blas;
};

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_THREADBUDGET_HPP
