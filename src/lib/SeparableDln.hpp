#ifndef LIBKRIGING_SRC_LIB_SEPARABLEDLN_HPP
#define LIBKRIGING_SRC_LIB_SEPARABLEDLN_HPP

// Private (not installed) helpers of the theta gradients (KrigingImpl.cpp,
// docs/dev/rationalization/T1-vectorized-gradients.md).
//
// One-dimension factors of the kernels'
// d ln k / d theta (Covariance::DlnCovDtheta_*, Covariance.cpp). The kernels
// are separable: component k of DlnCovDtheta_*(dX, theta) only depends on
// (dX[k], theta[k]). Each factor is built once per dimension (per gradient
// evaluation) with its theta-only constants, so that evaluating it on a pair
// needs at most one division, against one to three in Covariance.cpp. Same
// formulas, rounded differently (a few ulps per value).

#include "libKriging/utils/lk_armadillo.hpp"

#include "OmpLoops.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace libKriging {
namespace detail {

struct DlnGauss {  // dx^2 / theta^3
  double c;
  explicit DlnGauss(double theta) : c(1.0 / (theta * theta * theta)) {}
  double operator()(double dx) const { return dx * dx * c; }
};

struct DlnExp {  // |dx| / theta^2
  double c;
  explicit DlnExp(double theta) : c(1.0 / (theta * theta)) {}
  double operator()(double dx) const { return std::abs(dx) * c; }
};

struct DlnMatern32 {  // u^2 / (1 + u) / theta, u = sqrt(3) |dx| / theta
  double s, inv_theta;
  explicit DlnMatern32(double theta) : s(std::sqrt(3.0) / theta), inv_theta(1.0 / theta) {}
  double operator()(double dx) const {
    const double u = s * std::abs(dx);
    return (u * u) / (1.0 + u) * inv_theta;
  }
};

struct DlnMatern52 {  // a b / (a + b) / theta, a = 1 + u, b = u^2 / 3, u = sqrt(5) |dx| / theta
  double s, inv_theta;
  explicit DlnMatern52(double theta) : s(std::sqrt(5.0) / theta), inv_theta(1.0 / theta) {}
  double operator()(double dx) const {
    const double u = s * std::abs(dx);
    const double a = 1.0 + u;
    const double b = (u * u) / 3.0;
    return (a * b) / (a + b) * inv_theta;
  }
};

// D = dlnCov/dtheta(dx, theta) for a separable kernel: one factor per
// dimension, built once per evaluation.
template <typename Dln1>
struct SeparableDln {
  std::vector<Dln1> f;
  explicit SeparableDln(const arma::vec& theta) {
    f.reserve(theta.n_elem);
    for (arma::uword k = 0; k < theta.n_elem; k++)
      f.emplace_back(theta.at(k));
  }
  void operator()(const double* dx, double* D) const {
    for (std::size_t k = 0; k < f.size(); k++)
      D[k] = f[k](dx[k]);
  }
};

struct ZeroDln {
  arma::uword d;
  void operator()(const double* /*dx*/, double* D) const {
    for (arma::uword k = 0; k < d; k++)
      D[k] = 0.0;
  }
};

// Calls body(dln) with the dlnCov/dtheta functor of the kernel covType (the
// kernels accepted by Covariance::resolve).
template <typename Body>
void with_dln(const std::string& covType, const arma::vec& theta, Body&& body) {
  if (covType == "gauss")
    body(SeparableDln<DlnGauss>(theta));
  else if (covType == "exp")
    body(SeparableDln<DlnExp>(theta));
  else if (covType == "matern3_2")
    body(SeparableDln<DlnMatern32>(theta));
  else if (covType == "matern5_2")
    body(SeparableDln<DlnMatern52>(theta));
  else if (covType == "whitenoise")
    body(ZeroDln{theta.n_elem});
  else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);
}

// Pair loop shared by the LL and LMP theta gradients. dX is d x n^2 (column
// i * n + j: X_i - X_j, LinearAlgebra::compute_dX). For W(i, j) an array of NW
// per-pair weights (pairs j < i) and D_k(i, j) = dlnCov/dtheta_k(dX_ij),
// writes out[l * d + k] = sum_{j < i} W(i, j)[l] * D_k(i, j).
// The sum is taken row by row (j ascending), then over the rows (i ascending),
// whatever `threads` (rows spread over that many OpenMP threads when > 1): the
// result depends neither on the thread count nor on the n from which the
// caller goes parallel (kOmpMinRows, OS dependent), tested by
// tests/GradientThreadsTest.cpp. Extra memory: NW * d * n doubles (row partial
// sums).
template <std::size_t NW, typename DlnFn, typename WeightFn>
void sum_pairs_dln(const arma::mat& dX,
                   arma::uword n,
                   const arma::vec& theta,
                   const DlnFn& dln,
                   int threads,
                   const WeightFn& weights,
                   double* out) {
  const arma::uword d = theta.n_elem;
  arma::mat rows(NW * d, n, arma::fill::zeros);
  auto add_row = [&](arma::uword i) {
    double* acc = rows.colptr(i);
    arma::vec D(d);
    for (arma::uword j = 0; j < i; j++) {
      dln(dX.colptr(i * n + j), D.memptr());
      const std::array<double, NW> w = weights(i, j);
      for (std::size_t l = 0; l < NW; l++)
        for (arma::uword k = 0; k < d; k++)
          acc[l * d + k] += w[l] * D.at(k);
    }
  };
#ifdef LK_OMP_LOOPS
#pragma omp parallel for schedule(dynamic, 8) num_threads(threads) if (threads > 1)
#endif
  for (arma::sword si = 0; si < static_cast<arma::sword>(n); si++)
    add_row(static_cast<arma::uword>(si));
  (void)threads;
  for (arma::uword r = 0; r < NW * d; r++)
    out[r] = 0.0;
  for (arma::uword i = 0; i < n; i++)
    for (arma::uword r = 0; r < NW * d; r++)
      out[r] += rows.at(r, i);
}

}  // namespace detail
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_SEPARABLEDLN_HPP
