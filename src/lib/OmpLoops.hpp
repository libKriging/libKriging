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

#endif  // LIBKRIGING_SRC_LIB_OMPLOOPS_HPP
