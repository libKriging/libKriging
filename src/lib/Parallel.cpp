#include "libKriging/Parallel.hpp"

#include <stdexcept>

#include "BlasThreads.hpp"
#include "OmpLoops.hpp"
#include "EnvSettings.hpp"
#include "ThreadBudget.hpp"

// The thread settings of the library, stored here since 2.0 (they were in
// Optim before, #468); the OpenMP threshold in ThreadBudget.cpp (OmpLoops.hpp).
// Defaults from the environment, read when the library is loaded (the
// initializer below), whatever the order of the static initializers: the
// storage is a function-local static, also safe for an earlier first use.

namespace {
struct ThreadSettings {
  double num_threads_percent = libKriging::detail::env_double("LK_NUM_THREADS_PERCENT", 0.0, 0.0);  // 0: 100 %
  int thread_pool_size = libKriging::detail::env_int("LK_THREAD_POOL_SIZE", 0, 0);  // 0: the thread budget
  int thread_start_delay_ms = libKriging::detail::env_int("LK_THREAD_START_DELAY_MS", 10, 0);
  int blas_threads_min_n = libKriging::detail::env_int("LK_BLAS_THREADS_MIN_N", -1, -1);  // -1: automatic
};

ThreadSettings& settings() {
  static ThreadSettings s;
  return s;
}

const bool settings_read_at_load = (settings(), true);
}  // namespace

LIBKRIGING_EXPORT void Parallel::set_num_threads_percent(double x) {
  if (!(x >= 0.0))
    throw std::invalid_argument("set_num_threads_percent: expected x >= 0 (x <= 1: ratio, x > 1: percentage)");
  settings().num_threads_percent = x;
  libKriging::detail::clear_forked_child();  // a forked child may opt back in
}

LIBKRIGING_EXPORT double Parallel::get_num_threads_percent() {
  return settings().num_threads_percent;
}

LIBKRIGING_EXPORT int Parallel::get_num_threads() {
  return libKriging::detail::budget_full();
}

LIBKRIGING_EXPORT void Parallel::set_thread_pool_size(int pool_size) {
  if (pool_size < 0)
    throw std::invalid_argument("set_thread_pool_size: expected pool_size >= 0 (0: automatic)");
  settings().thread_pool_size = pool_size;
}

LIBKRIGING_EXPORT int Parallel::get_thread_pool_size() {
  return settings().thread_pool_size;
}

LIBKRIGING_EXPORT void Parallel::set_thread_start_delay_ms(int delay_ms) {
  if (delay_ms < 0)
    throw std::invalid_argument("set_thread_start_delay_ms: expected delay_ms >= 0");
  settings().thread_start_delay_ms = delay_ms;
}

LIBKRIGING_EXPORT int Parallel::get_thread_start_delay_ms() {
  return settings().thread_start_delay_ms;
}

LIBKRIGING_EXPORT void Parallel::set_blas_threads_min_n(int min_n) {
  if (min_n < -1)
    throw std::invalid_argument("set_blas_threads_min_n: expected min_n >= -1 (-1: automatic, 0: off)");
  settings().blas_threads_min_n = min_n;
}

LIBKRIGING_EXPORT int Parallel::get_blas_threads_min_n() {
  return settings().blas_threads_min_n;
}

LIBKRIGING_EXPORT void Parallel::set_omp_min_rows(int n) {
  if (n < 0)
    throw std::invalid_argument("set_omp_min_rows: expected n >= 0");
  libKriging::detail::set_omp_min_rows(static_cast<unsigned long long>(n));
}

LIBKRIGING_EXPORT int Parallel::get_omp_min_rows() {
  return static_cast<int>(libKriging::detail::omp_min_rows());
}

LIBKRIGING_EXPORT std::string Parallel::blas_backend() {
  // MKL first: an MKL preloaded over a linked OpenBLAS serves the calls
  if (libKriging::detail::get_mkl_threads() > 0)
    return "mkl";
  if (libKriging::detail::get_openblas_threads() > 0)
    return "openblas";
#if defined(__APPLE__)
  return "accelerate";  // neither OpenBLAS nor MKL: the system BLAS
#else
  return "unknown";
#endif
}

LIBKRIGING_EXPORT int Parallel::blas_num_threads() {
  const int mkl = libKriging::detail::get_mkl_threads();
  if (mkl > 0)
    return mkl;
  return libKriging::detail::get_openblas_threads();
}

LIBKRIGING_EXPORT bool Parallel::omp_enabled() {
#ifdef LK_OMP_LOOPS
  return true;
#else
  return false;
#endif
}
