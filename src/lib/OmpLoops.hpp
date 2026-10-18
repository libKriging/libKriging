#ifndef LIBKRIGING_SRC_LIB_OMPLOOPS_HPP
#define LIBKRIGING_SRC_LIB_OMPLOOPS_HPP

// Private (not installed) helper: LK_OMP_LOOPS is 1 when the OpenMP loops of
// the library are compiled in. ThreadSanitizer reports false-positive races on
// them (libgomp is not TSan-instrumented, so the implicit barrier closing a
// parallel region is not seen): they are compiled out under TSan, as
// NestedKriging's pair loop.

#if defined(_OPENMP)
#define LK_OMP_LOOPS 1
#if defined(__SANITIZE_THREAD__)
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
/// matrix, log-likelihood gradient) run in parallel. Below it, the OpenMP team
/// competes for the cores with the OpenBLAS threads still spinning after the
/// previous BLAS call and is slower than one thread (4 cores, OpenBLAS pthreads,
/// both at 4 threads, d = 4: log-likelihood + gradient 3.9 -> 17 ms at n = 200,
/// break-even around n = 700, 75 -> 48 ms at n = 800). See
/// docs/dev/ThreadBudget-TODO.md for the shared thread budget meant to lift it.
/// LK_OMP_MIN_ROWS overrides it at compile time (thread-budget benchmarks).
#ifndef LK_OMP_MIN_ROWS
#define LK_OMP_MIN_ROWS 800
#endif
constexpr unsigned long long kOmpMinRows = LK_OMP_MIN_ROWS;
}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_OMPLOOPS_HPP
