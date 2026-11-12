#ifndef LIBKRIGING_SRC_LIB_PARALLELFOR_HPP
#define LIBKRIGING_SRC_LIB_PARALLELFOR_HPP

// Private (not installed): parallel loop used by the O(n^2) loops of the library.
//
//   parallel_for(begin, end, chunk, nthreads, parallel, [&](long lo, long hi) { ... });
//
// calls the body on consecutive sub-ranges [lo, hi) of [begin, end) of at most
// `chunk` indices, dynamically distributed over at most `nthreads` OpenMP
// threads. The body runs once on the whole range, on the calling thread, when
// `parallel` is false, nthreads <= 1, the range is a single chunk, or the
// OpenMP loops are not compiled in (no OpenMP, or TSan: OmpLoops.hpp). The body
// must not throw and the sub-ranges must be independent (disjoint outputs; any
// reduction is done by the caller after the loop, in a fixed order).

#include "OmpLoops.hpp"

#include <algorithm>

namespace libKriging {
namespace detail {

template <class F>
void parallel_for(long begin, long end, long chunk, int nthreads, bool parallel, F&& body) {
  if (end <= begin)
    return;
  chunk = std::max(1L, chunk);
#ifdef LK_OMP_LOOPS
  const long nchunks = (end - begin + chunk - 1) / chunk;
  if (parallel && nthreads > 1 && nchunks > 1) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (long k = 0; k < nchunks; k++) {
      const long lo = begin + k * chunk;
      body(lo, std::min(end, lo + chunk));
    }
    return;
  }
#else
  (void)nthreads;
  (void)parallel;
#endif
  body(begin, end);
}

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_PARALLELFOR_HPP
