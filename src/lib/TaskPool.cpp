#include "TaskPool.hpp"

#include "libKriging/Optim.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include "ThreadBudget.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace libKriging {
namespace detail {

void run_task_pool(int n_tasks, const std::function<void(int)>& task, const char* label, bool stagger) {
  // A pool started from a worker runs inline instead of multiplying the
  // threads (e.g. a fit with multistart inside a parallel task).
  if (n_tasks <= 1 || in_pool_worker()) {
    for (int i = 0; i < n_tasks; i++)
      task(i);
    return;
  }
  // The budget goes to the tasks first: one worker per task up to the
  // budget (or Optim::thread_pool_size), then each worker gets the rest for
  // its BLAS and OpenMP calls.
  const int total = budget_total();
  int pool_size = Optim::thread_pool_size > 0 ? Optim::thread_pool_size : total;
  pool_size = std::max(1, std::min(pool_size, n_tasks));
  const int share = std::max(1, total / pool_size);

  // Left at the default, every worker's BLAS calls spawn all-core thread
  // teams that contend with the other workers (≈2x slower at n = 500, mostly
  // in system time). The OpenBLAS count is process-global: set once here,
  // restored once all workers have joined (declared before the joiner).
  ScopedBlasThreads blas_share(blas_threads_for(share));

  if (Optim::log_level > Optim::log_none) {
    arma::cout << "Thread pool: " << pool_size << " workers x " << share << " BLAS/OpenMP threads (budget=" << total
               << ", " << label << "=" << n_tasks << ")" << arma::endl;
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
      // The OpenMP loops of the worker use its share (num_threads(omp_threads())),
      // and a pool started from it runs inline.
      ScopedShare worker_share(share);
      while (true) {
        int task_id = next_task.fetch_add(1);
        if (task_id >= n_tasks)
          break;

        // Staggered startup delay: only for each worker's first task
        // (applied to every task, it added up to ~(n_tasks/pool)^2
        // delays with no benefit once the pool is running).
        if (stagger && task_id < pool_size) {
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
