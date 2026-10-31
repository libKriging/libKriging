#ifndef LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
#define LIBKRIGING_SRC_LIB_BLASTHREADS_HPP

// Private (not installed): OpenBLAS / OpenMP thread-count access, defined in
// Kriging.cpp (fork handlers, ThreadBudget.cpp).

#include <mutex>

namespace libKriging {
namespace detail {

/// OpenBLAS and OpenMP thread counts; 0 for a backend whose count cannot be
/// read (no openblas_get_num_threads, no OpenMP).
struct BlasThreadState {
  int blas = 0;
  int omp = 0;
};

/// Current OpenBLAS and OpenMP thread counts. Defined in Kriging.cpp.
BlasThreadState get_blas_threads();

/// Set the OpenBLAS and OpenMP thread counts (no-op for a backend without
/// openblas_set_num_threads). Defined in Kriging.cpp.
void set_blas_threads(unsigned int n);

/// Restore counts read by get_blas_threads() (0 entries are left untouched).
/// Defined in Kriging.cpp.
void restore_blas_threads(const BlasThreadState& s);

/// OpenBLAS thread count only (0 / no-op without openblas_*_num_threads); the
/// OpenMP ICV is left untouched. Defined in Kriging.cpp.
int get_openblas_threads();
void set_openblas_threads(int n);

/// Serializes arma::rcond (LinearAlgebra::rcond_chol). Also held by the fork
/// handlers, so that the child of a fork() made while another thread is in
/// rcond_chol does not inherit it locked. Defined in LinearAlgebra.cpp.
std::mutex& rcond_mutex();

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
