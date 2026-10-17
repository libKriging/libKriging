#ifndef LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
#define LIBKRIGING_SRC_LIB_BLASTHREADS_HPP

// Private (not installed) helper shared by Kriging.cpp and WarpKriging.cpp.

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

/// Balances nested parallelism during a parallel multistart: each worker gets
/// `n` BLAS/OMP threads instead of all of them. Restores the thread counts in
/// effect before the first set() on scope exit, so that later calls (predict,
/// simulate, ...) keep the caller's setting. Fork safety does not rely on
/// this guard: the pthread_atfork handlers in Kriging.cpp quiesce BLAS/OMP
/// before every fork().
struct BlasThreadGuard {
  bool active = false;
  BlasThreadState saved;
  void set(unsigned int n) {
    if (!active) {
      saved = get_blas_threads();
      active = true;
    }
    set_blas_threads(n);
  }
  ~BlasThreadGuard() {
    if (active)
      restore_blas_threads(saved);
  }
};

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
