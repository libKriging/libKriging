#ifndef LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
#define LIBKRIGING_SRC_LIB_BLASTHREADS_HPP

// Private (not installed) helper shared by Kriging.cpp and WarpKriging.cpp.

namespace libKriging {
namespace detail {

/// Set the OpenBLAS and OpenMP thread counts (no-op for a backend without
/// openblas_set_num_threads). Defined in Kriging.cpp.
void set_blas_threads(unsigned int n);

/// Balances nested parallelism during a parallel multistart: each worker gets
/// `n` BLAS/OMP threads instead of all of them. Resets to 1 thread on scope
/// exit, which also collapses the idle BLAS pool so that a later fork() (e.g.
/// R's mclapply) does not inherit locked mutexes and deadlock.
struct BlasThreadGuard {
  bool active = false;
  void set(unsigned int n) {
    active = true;
    set_blas_threads(n);
  }
  ~BlasThreadGuard() {
    if (active)
      set_blas_threads(1);
  }
};

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
