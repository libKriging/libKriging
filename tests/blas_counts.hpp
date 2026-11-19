#ifndef LIBKRIGING_TESTS_BLAS_COUNTS_HPP
#define LIBKRIGING_TESTS_BLAS_COUNTS_HPP

// Thread counts of the BLAS the test process runs, read as the library reads
// them (src/lib/Kriging.cpp): weak symbols on ELF, GetProcAddress on Windows,
// dlsym for MKL elsewhere on POSIX. 0 when that BLAS is absent.
//
// Tests comparing results across thread budgets use it to know whether the BLAS
// count the library sets can change between the runs: OpenBLAS and MKL may then
// round differently (MKL: T9 log; OpenBLAS 0.3.33 on Windows: #473), and only a
// tolerance can be required. Bit-identity is checked by the ctest variants that
// load the BLAS with 1 thread (OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1).

#include <initializer_list>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__) || defined(__unix__)
#include <dlfcn.h>
#endif

#if defined(__ELF__)
extern "C" {
int openblas_get_num_threads() __attribute__((weak));
int MKL_Get_Max_Threads() __attribute__((weak));
}
#endif

namespace lk_test {

using count_fn = int (*)();

#if defined(_WIN32)
inline FARPROC blas_proc(std::initializer_list<const char*> dlls, const char* name) {
  for (const char* dll : dlls)
    if (HMODULE m = GetModuleHandleA(dll))
      if (FARPROC f = GetProcAddress(m, name))
        return f;
  return nullptr;
}
#endif

/// OpenBLAS count (process-global), 0 without OpenBLAS.
inline int openblas_threads() {
#if defined(__ELF__)
  const count_fn f = openblas_get_num_threads;
#elif defined(_WIN32)
  static const count_fn f
      = reinterpret_cast<count_fn>(blas_proc({"libopenblas.dll", "openblas.dll"}, "openblas_get_num_threads"));
#elif defined(__APPLE__) && !defined(__arm64__)
  static const count_fn f = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
#else
  const count_fn f = nullptr;
#endif
  return f ? f() : 0;
}

/// MKL count of the calling thread, 0 without MKL.
inline int mkl_threads() {
#if defined(__ELF__)
  const count_fn f = MKL_Get_Max_Threads;
#elif defined(_WIN32)
  static const count_fn f
      = reinterpret_cast<count_fn>(blas_proc({"mkl_rt.2.dll", "mkl_rt.dll", "mkl_rt.1.dll"}, "MKL_Get_Max_Threads"));
#elif defined(__APPLE__) || defined(__unix__)
  static const count_fn f = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT, "MKL_Get_Max_Threads"));
#else
  const count_fn f = nullptr;
#endif
  return f ? f() : 0;
}

/// True when the library may run the BLAS with different counts for different
/// budgets: OpenBLAS or MKL loaded with more than 1 thread.
inline bool blas_count_can_change() {
  return openblas_threads() > 1 || mkl_threads() > 1;
}

}  // namespace lk_test

#endif  // LIBKRIGING_TESTS_BLAS_COUNTS_HPP
