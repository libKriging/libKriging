#ifndef LIBKRIGING_SRC_LIB_ITERATIVEMATVEC_HPP
#define LIBKRIGING_SRC_LIB_ITERATIVEMATVEC_HPP
// Private (not installed): matrix-free products of the iterative method
// (method "iterative": CG solves, SLQ log-determinant, Hutchinson gradient).
//
//   symmetric_pair_sum(n, dimX, ncols, [&](i, j, dx, acc) { ... });
//
// visits every unordered pair i < j once (dx = X_i - X_j, from the d x n
// matrix Xt given to pair_delta) and lets the body add its contributions to
// rows i AND j of `acc` (n x ncols): the symmetric R_ij = R_ji is evaluated
// once per pair. The pairs are split into a fixed number of strips (rows i
// taken round-robin, which balances the triangular work), each with its own
// accumulator, summed in a fixed order: the result does not depend on the
// number of threads nor on the scheduling. The strips run in parallel under
// the thread budget (parallel_for, OmpLoops.hpp), from omp_min_rows() rows on.
#include "libKriging/utils/lk_armadillo.hpp"

#include "OmpLoops.hpp"
#include "ParallelFor.hpp"
#include "ThreadBudget.hpp"

#include <algorithm>
#include <vector>

namespace libKriging {
namespace detail {

constexpr arma::uword kIterativeStrips = 16;

// dx = Xt.col(i) - Xt.col(j), Xt being d x n (observations in columns)
inline void pair_delta(const double* Xt_mem, arma::uword dimX, arma::uword i, arma::uword j, double* dx_mem) {
  for (arma::uword q = 0; q < dimX; ++q)
    dx_mem[q] = Xt_mem[q + i * dimX] - Xt_mem[q + j * dimX];
}

template <class PairBody>
arma::mat symmetric_pair_sum(arma::uword n, arma::uword dimX, arma::uword ncols, PairBody&& body) {
  const arma::uword nstrips = std::max<arma::uword>(1, std::min<arma::uword>(kIterativeStrips, n));
  std::vector<arma::mat> acc(nstrips);
  const bool parallel = n >= static_cast<arma::uword>(omp_min_rows());
  parallel_for(0, static_cast<long>(nstrips), 1, omp_threads(), parallel, [&](long lo, long hi) {
    arma::vec dx(dimX, arma::fill::none);
    for (long s = lo; s < hi; ++s) {
      arma::mat& a = acc[static_cast<std::size_t>(s)];
      a.zeros(n, ncols);
      for (arma::uword i = static_cast<arma::uword>(s); i < n; i += nstrips)
        for (arma::uword j = i + 1; j < n; ++j)
          body(i, j, dx, a);
    }
  });
  arma::mat out = std::move(acc[0]);
  for (arma::uword s = 1; s < nstrips; ++s)
    out += acc[s];
  return out;
}

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_ITERATIVEMATVEC_HPP
