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
/// matrix, log-likelihood gradient) run in parallel. A performance setting only
/// once the LL gradient sums in one fixed order on the parallel and sequential
/// paths (#441, T1); the covariance loops write independent entries. Before
/// that, the gradient's summation order, hence the fitted model, changed at
/// this threshold: with the per-OS values below it depended on the OS for
/// 200 <= n < 800 (docs/dev/rationalization/T5-omp-threshold-blas.md). With
/// BLAS and OpenMP both at the thread budget (physical cores, ThreadBudget.hpp)
/// the loops pay off from n = 200 on Linux (i9-10900K, OpenBLAS pthreads / OpenMP, libgomp:
/// x3 at n = 200-500, bench/thread-budget/RESULTS.md; Xeon 4 cores: T5 log) and
/// with MSVC on Windows (same CPU, vcomp, OpenBLAS 0.3.33: x1.5-1.6 per
/// evaluation at n = 200-400, x1.12-1.20 per fit at n = 500, no hang with the
/// Python suite; #473, branch windows-session-1:
/// docs/dev/rationalization/Windows-results.md).
/// Elsewhere (libomp / Accelerate on macOS, MinGW / libgomp on Windows: Octave,
/// R) not measured yet: kept at 800, see #416 (macOS). LK_OMP_MIN_ROWS overrides
/// it at compile time.
#ifndef LK_OMP_MIN_ROWS
#if defined(__linux__) || defined(_MSC_VER)
#define LK_OMP_MIN_ROWS 200
#else
#define LK_OMP_MIN_ROWS 800
#endif
#endif
constexpr unsigned long long kOmpMinRows = LK_OMP_MIN_ROWS;
}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_OMPLOOPS_HPP
