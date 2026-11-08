#ifndef LIBKRIGING_SRC_LIB_PARALLELFOR_HPP
#define LIBKRIGING_SRC_LIB_PARALLELFOR_HPP

// Private (not installed): parallel loop used by the O(n^2) loops of the library.
//
//   parallel_for(begin, end, chunk, nthreads, parallel, [&](long lo, long hi) { ... });
//
// calls the body on consecutive sub-ranges [lo, hi) of [begin, end) of at most
// `chunk` indices, dynamically distributed over at most `nthreads` threads. The
// body runs once on the whole range, on the calling thread, when `parallel` is
// false, nthreads <= 1, the range is a single chunk, parallel loops are not
// compiled in (OmpLoops.hpp) or this process is a fork() child. The body must
// not throw and the sub-ranges must be independent.
//
// Backends: OpenMP when _OPENMP is defined; otherwise, on Apple, Grand Central
// Dispatch. libdispatch is not safe in a child created by fork() without exec
// (R's mclapply): the fork child handler (Kriging.cpp) switches the loops off
// there.

#include "OmpLoops.hpp"

#include <algorithm>
#include <atomic>
#include <type_traits>

#ifdef LK_GCD_LOOPS
#include <dispatch/dispatch.h>
#endif

namespace libKriging {
namespace detail {

/// True in a fork() child: parallel loops then run sequentially.
inline std::atomic<bool>& parallel_loops_off() {
  static std::atomic<bool> off{false};
  return off;
}

#ifdef LK_GCD_LOOPS
template <class F>
struct GcdLoop {
  F* body;
  long begin, end, chunk, nchunks;
  std::atomic<long> next;
};

template <class F>
void gcd_worker(void* p, size_t) {
  GcdLoop<F>* c = static_cast<GcdLoop<F>*>(p);
  for (long k = c->next.fetch_add(1); k < c->nchunks; k = c->next.fetch_add(1)) {
    const long lo = c->begin + k * c->chunk;
    (*c->body)(lo, std::min(c->end, lo + c->chunk));
  }
}
#endif

template <class F>
void parallel_for(long begin, long end, long chunk, int nthreads, bool parallel, F&& body) {
  if (end <= begin)
    return;
  chunk = std::max(1L, chunk);
#ifdef LK_OMP_LOOPS
  const long nchunks = (end - begin + chunk - 1) / chunk;
  if (parallel && nthreads > 1 && nchunks > 1 && !parallel_loops_off().load(std::memory_order_relaxed)) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (long k = 0; k < nchunks; k++) {
      const long lo = begin + k * chunk;
      body(lo, std::min(end, lo + chunk));
    }
    return;
#else  // LK_GCD_LOOPS
    GcdLoop<typename std::remove_reference<F>::type> c{&body, begin, end, chunk, nchunks, {0}};
    dispatch_apply_f(static_cast<size_t>(std::min<long>(nthreads, nchunks)),
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &c,
                     gcd_worker<typename std::remove_reference<F>::type>);
    return;
#endif
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
