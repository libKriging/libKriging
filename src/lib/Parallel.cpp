#include "libKriging/Parallel.hpp"

#include <stdexcept>

#include "BlasThreads.hpp"
#include "OmpLoops.hpp"
#include "libKriging/Optim.hpp"

// The budget, pool and BLAS settings are stored in Optim (Optim.cpp), which
// keeps its setters for compatibility; the OpenMP threshold in OmpLoops.hpp.

LIBKRIGING_EXPORT void Parallel::set_num_threads_percent(double x) {
  Optim::set_num_threads_percent(x);
}

LIBKRIGING_EXPORT double Parallel::get_num_threads_percent() {
  return Optim::get_num_threads_percent();
}

LIBKRIGING_EXPORT int Parallel::get_num_threads() {
  return Optim::get_num_threads();
}

LIBKRIGING_EXPORT void Parallel::set_thread_pool_size(int pool_size) {
  Optim::set_thread_pool_size(pool_size);
}

LIBKRIGING_EXPORT int Parallel::get_thread_pool_size() {
  return Optim::get_thread_pool_size();
}

LIBKRIGING_EXPORT void Parallel::set_thread_start_delay_ms(int delay_ms) {
  Optim::set_thread_start_delay_ms(delay_ms);
}

LIBKRIGING_EXPORT int Parallel::get_thread_start_delay_ms() {
  return Optim::get_thread_start_delay_ms();
}

LIBKRIGING_EXPORT void Parallel::set_blas_threads_min_n(int min_n) {
  Optim::set_blas_threads_min_n(min_n);
}

LIBKRIGING_EXPORT int Parallel::get_blas_threads_min_n() {
  return Optim::get_blas_threads_min_n();
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
