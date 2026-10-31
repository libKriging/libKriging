#ifndef LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
#define LIBKRIGING_SRC_LIB_BLASTHREADS_HPP

// Private (not installed): OpenBLAS / MKL / OpenMP thread-count access, defined
// in Kriging.cpp (fork handlers, ThreadBudget.cpp).

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

/// Intel MKL / oneMKL thread count of the calling thread (MKL_Get_Max_Threads:
/// its thread-local count if set, else the global one, MKL_NUM_THREADS or the
/// physical cores), 0 without MKL. Defined in Kriging.cpp.
int get_mkl_threads();

/// Set the MKL thread count of the calling thread only
/// (MKL_Set_Num_Threads_Local; 0 = back to the global count) and return the
/// previous thread-local count (0: none was set); -1 without MKL. Thread-local:
/// each pool worker sets its own, without touching the other threads.
/// Defined in Kriging.cpp.
int set_mkl_threads_local(int n);

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_BLASTHREADS_HPP
