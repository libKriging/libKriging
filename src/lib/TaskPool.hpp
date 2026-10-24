#ifndef LIBKRIGING_SRC_LIB_TASKPOOL_HPP
#define LIBKRIGING_SRC_LIB_TASKPOOL_HPP

// Private (not installed) helper shared by Kriging.cpp and WarpKriging.cpp.

#include <functional>

namespace libKriging {
namespace detail {

/// Runs task(0), …, task(n_tasks − 1) in a pool of std::thread workers and
/// returns once all have run. The threads go to the tasks first: pool =
/// min(n_tasks, Optim::thread_pool_size or all the cores), and each worker gets
/// the remaining share (cores / pool, at least 1) for its BLAS and OpenMP calls.
/// Runs inline (in the calling thread, thread counts untouched) when
/// n_tasks == 1 or when called from a pool worker (no nested pools).
///
/// Each task must work on its own model / buffers, catch its own exceptions
/// and not draw from the global RNG (draw the starting points before). This
/// requires a reentrant L-BFGS-B: lbfgsb_cpp >= b6eff96, whose f2c BLAS /
/// LINPACK helpers (Lbfgsb.3.0/blas.c, linpack.c) no longer keep `static`
/// locals that concurrent optimizers would race on.
///
/// stagger: delay the first task of worker k by k × Optim::thread_start_delay_ms
/// (WarpKriging); off for short tasks such as a Kriging start, where 20 workers
/// waited up to 190 ms, longer than a whole start at n = 100. Defined in TaskPool.cpp.
void run_task_pool(int n_tasks, const std::function<void(int)>& task, const char* label, bool stagger = true);

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_TASKPOOL_HPP
