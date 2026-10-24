#ifndef LIBKRIGING_SRC_LIB_OMPLOOPS_HPP
#define LIBKRIGING_SRC_LIB_OMPLOOPS_HPP

// Private (not installed) helper: LK_OMP_LOOPS is 1 when the OpenMP loops of
// the library are compiled in. ThreadSanitizer reports false-positive races on
// them with a runtime it cannot see through (libgomp is not TSan-instrumented,
// so the implicit barrier closing a parallel region is not seen): they are
// compiled out under TSan, as NestedKriging's pair loop, unless LK_TSAN_OPENMP
// is defined (CMake: SANITIZE=THREAD with Clang and LLVM libomp, whose Archer
// tool, libarcher, reports the OpenMP synchronization to TSan; #420).

#if defined(_OPENMP)
#define LK_OMP_LOOPS 1
#if defined(LK_TSAN_OPENMP)
// TSan-aware OpenMP runtime: the loops stay in and are checked
#elif defined(__SANITIZE_THREAD__)
#undef LK_OMP_LOOPS
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#undef LK_OMP_LOOPS
#endif
#endif
#endif

namespace libKriging {
namespace detail {
/// Smallest n for which the O(n^2) loops of an objective evaluation (covariance
/// matrix, log-likelihood gradient) run in parallel. With BLAS and OpenMP both
/// at the thread budget (physical cores, ThreadBudget.hpp) they pay off from
/// n = 200 on Linux (i9-10900K, OpenBLAS pthreads / OpenMP, libgomp: x3 at
/// n = 200-500, bench/thread-budget/RESULTS.md). Elsewhere (libomp, Accelerate,
/// Windows OpenBLAS) not measured yet: kept at 800, see #416 (macOS) and #417
/// (Windows). LK_OMP_MIN_ROWS overrides it at compile time.
#ifndef LK_OMP_MIN_ROWS
#if defined(__linux__)
#define LK_OMP_MIN_ROWS 200
#else
#define LK_OMP_MIN_ROWS 800
#endif
#endif
constexpr unsigned long long kOmpMinRows = LK_OMP_MIN_ROWS;
}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_OMPLOOPS_HPP
