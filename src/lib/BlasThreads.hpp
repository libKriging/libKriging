#ifndef LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
#define LIBKRIGING_SRC_LIB_BLASTHREADS_HPP

// Private (not installed) helper shared by Kriging.cpp and WarpKriging.cpp.

namespace libKriging {
namespace detail {

/// Set the OpenBLAS and OpenMP thread counts (no-op for a backend without
/// openblas_set_num_threads). Defined in Kriging.cpp.
void set_blas_threads(unsigned int n);

/// Current OpenBLAS thread count, or 0 when the backend does not expose
/// openblas_get_num_threads (Accelerate, reference BLAS, MKL...).
int get_blas_threads();

/// Current OpenMP thread count of the calling thread (0 without OpenMP).
int get_omp_threads();

/// Balances nested parallelism during a parallel multistart: each worker gets
/// `n` BLAS/OMP threads instead of all of them. Restores, on scope exit, the
/// BLAS and OMP thread counts found at the first set() (it used to reset them
/// to 1, which left the whole process single-threaded after any multistart).
/// Fork-safety (e.g. R's mclapply) does not depend on this reset: the
/// pthread_atfork handlers in Kriging.cpp quiesce BLAS/OMP around each fork.
struct BlasThreadGuard {
  bool active = false;
  int saved_blas = 0;
  int saved_omp = 0;
  void set(unsigned int n) {
    if (!active) {
      saved_blas = get_blas_threads();
      saved_omp = get_omp_threads();
      active = true;
    }
    set_blas_threads(n);
  }
  ~BlasThreadGuard();
};

/// Number of physical cores (Linux: distinct thread-sibling sets in sysfs;
/// macOS: hw.physicalcpu), std::thread::hardware_concurrency() elsewhere or
/// when unknown. Computed once.
unsigned int physical_cores();

/// OpenBLAS thread count that Optim's BLAS thread policy gives to an
/// operation on n training observations, from the current count `current`
/// (see Optim::blas_threads_min_n / blas_threads_max). Never above current.
int policy_blas_threads(unsigned long n, int current);

/// Applies the BLAS thread policy for the duration of a fit / predict /
/// update on n observations, and restores the OpenBLAS thread count on scope
/// exit. Only the outermost scope of the process acts: nested calls (a
/// WarpKriging fit refitting a Kriging, multistart workers calling predict)
/// leave the thread count alone, so that it is not changed while worker
/// threads are running BLAS calls. Leaves OpenMP alone. No-op when the
/// backend is not OpenBLAS or the policy is disabled.
class BlasThreadScope {
 public:
  explicit BlasThreadScope(unsigned long n);
  ~BlasThreadScope();
  BlasThreadScope(const BlasThreadScope&) = delete;
  BlasThreadScope& operator=(const BlasThreadScope&) = delete;

 private:
  bool m_outer = false;
  int m_saved = 0;  // OpenBLAS thread count to restore, 0 if unchanged
};

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
