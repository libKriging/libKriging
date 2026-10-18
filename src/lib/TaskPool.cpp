#include "TaskPool.hpp"

#include "libKriging/Optim.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include "BlasThreads.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace libKriging {
namespace detail {

namespace {
// Set in the pool workers: a pool started from a worker runs inline instead of
// multiplying the threads (e.g. a fit with multistart inside a parallel task).
thread_local bool in_pool_worker = false;
}  // namespace

void run_task_pool(int n_tasks, const std::function<void(int)>& task, const char* label) {
  if (n_tasks <= 1 || in_pool_worker) {
    for (int i = 0; i < n_tasks; i++)
      task(i);
    return;
  }
  // Determine thread pool size
  unsigned int n_cpu = std::thread::hardware_concurrency();
  int pool_size = Optim::thread_pool_size;
  if (pool_size <= 0)
    pool_size = std::max(1u, n_cpu);
  pool_size = std::min(pool_size, n_tasks);

  // Give each worker its share of the BLAS/OMP threads: left at the
  // default, every worker's BLAS calls spawn all-core thread teams that
  // contend with the other workers (≈2x slower at n = 500, mostly in
  // system time). Declared before the joiner below, so that it is reset
  // only once all workers have joined.
  detail::BlasThreadGuard blas_guard;
  const unsigned int blas_threads = std::max(1u, n_cpu / static_cast<unsigned int>(pool_size));
  blas_guard.set(blas_threads);

  if (Optim::log_level > Optim::log_none) {
    arma::cout << "Thread pool: " << pool_size << " workers x " << blas_threads
               << " BLAS threads (ncpu=" << n_cpu << ", " << label << "=" << n_tasks << ")" << arma::endl;
  }

  // Thread pool with atomic task counter
  std::atomic<int> next_task(0);
  std::vector<std::thread> threads;
  threads.reserve(pool_size);

  // RAII guard to ensure threads are always joined (also when spawning
  // a later thread throws)
  struct ThreadJoiner {
    std::vector<std::thread>& threads_ref;
    explicit ThreadJoiner(std::vector<std::thread>& t) : threads_ref(t) {}
    ~ThreadJoiner() {
      for (auto& t : threads_ref)
        if (t.joinable())
          t.join();
    }
  } joiner(threads);

  for (int worker_id = 0; worker_id < pool_size; worker_id++) {
    threads.emplace_back([&]() {
      in_pool_worker = true;
#ifdef _OPENMP
      // OpenMP thread counts are per thread: without this, the OpenMP loops
      // of each worker (covariance matrix, ...) would use all the cores.
      omp_set_num_threads(static_cast<int>(blas_threads));
#endif
      while (true) {
        int task_id = next_task.fetch_add(1);
        if (task_id >= n_tasks)
          break;

        // Staggered startup delay: only for each worker's first task
        // (applied to every task, it added up to ~(n_tasks/pool)^2
        // delays with no benefit once the pool is running).
        if (task_id < pool_size) {
          int delay_ms = task_id * Optim::thread_start_delay_ms;
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        task(task_id);
      }
    });
  }
}
}  // namespace detail
}  // namespace libKriging
