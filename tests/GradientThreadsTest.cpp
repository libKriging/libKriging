// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "SeparableDln.hpp"  // private (src/lib)
#include "blas_counts.hpp"
#include <array>
#include <cmath>
#include <string>
#include <vector>
// clang-format on

// Reproducibility of the LL / LMP theta gradients (KrigingImpl.cpp,
// docs/dev/rationalization/T1-vectorized-gradients.md).
//
// 1. White box: their pair sums (sum_pairs_dln, src/lib/SeparableDln.hpp) are
//    taken row by row, then over the rows in order, and give the same bits for
//    any thread count, 1 being the sequential path the gradients take below
//    kOmpMinRows (200 on Linux, 800 elsewhere, OmpLoops.hpp): the gradients
//    depend neither on the thread count nor on kOmpMinRows, hence not on the OS.
// 2. End to end: the same bits with a 1-thread budget and with the whole
//    budget, below and above kOmpMinRows. The gradients also go through the
//    BLAS / LAPACK (R^-1, triangular solves), whose count follows the budget
//    (Optim::set_num_threads_percent) and which may round differently with it
//    (MKL; OpenBLAS 0.3.33 on Windows, #473): with a BLAS loaded with more than
//    1 thread, with Apple Accelerate (thread count not settable by the
//    library, on macOS CI two identical calls differ), or a BLAS found to give
//    different bits for two identical calls (e.g. MKL without CNR), the
//    comparison falls back to 1e-10 relative (case 1 keeps the bitwise check of
//    the pair sums; GradientThreadsTest/BLAS_NUM_THREADS=1 the bitwise
//    end-to-end check).

namespace {

struct Budget {
  const double saved = Optim::get_num_threads_percent();
  ~Budget() { Optim::set_num_threads_percent(saved); }
};

std::vector<arma::vec> gradients(Kriging& k, const arma::vec& gamma) {
  std::vector<arma::vec> g;
  g.push_back(std::get<1>(k.logLikelihoodFun(gamma, true, false)));
  g.push_back(std::get<1>(k.logMargPostFun(gamma, true, false)));
  return g;
}

bool bitwise_equal(const std::vector<arma::vec>& a, const std::vector<arma::vec>& b) {
  for (std::size_t i = 0; i < a.size(); i++)
    for (arma::uword k = 0; k < a[i].n_elem; k++)
      if (a[i].at(k) != b[i].at(k))
        return false;
  return true;
}

// rel_tol = 0: same bits; otherwise |a - b| <= rel_tol * max|a| per vector
void require_equal(const std::vector<arma::vec>& a, const std::vector<arma::vec>& b, double rel_tol) {
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); i++) {
    CAPTURE(i, a[i], b[i], rel_tol);
    REQUIRE(a[i].n_elem == b[i].n_elem);
    const double scale = arma::abs(a[i]).max();
    for (arma::uword k = 0; k < a[i].n_elem; k++) {
      if (rel_tol == 0.0)
        REQUIRE(a[i].at(k) == b[i].at(k));
      else
        REQUIRE(std::abs(a[i].at(k) - b[i].at(k)) <= rel_tol * scale);
    }
  }
}

}  // namespace

TEST_CASE("LL / LMP theta gradients do not depend on the OpenMP thread count", "[gradient][threads]") {
  // below / just above kOmpMinRows (200 Linux, 800 elsewhere): not larger, the
  // case must stay short under valgrind (Memcheck CI, ctest timeout 1500 s)
  const arma::uword n = GENERATE(arma::uword(100), static_cast<arma::uword>(libKriging::detail::kOmpMinRows + 10));
  const std::string kernel = GENERATE("gauss", "exp", "matern3_2", "matern5_2");
  const bool nugget = GENERATE(false, true);
  const arma::uword d = 4;
  CAPTURE(n, kernel, nugget);

  arma::arma_rng::set_seed(123);
  const arma::mat X(n, d, arma::fill::randu);
  arma::vec y = arma::sum(arma::sin(3 * X), 1);
  arma::vec theta(d);
  for (arma::uword k = 0; k < d; k++)
    theta(k) = (0.05 + 0.1 * k / (d - 1)) * 2.0;

  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;
  arma::vec gamma = theta;
  Kriging k = nugget ? Kriging(kernel, Kriging::NoiseModel::Nugget) : Kriging(kernel);
  if (nugget) {
    // alpha = sigma2 / (sigma2 + nugget) is estimated: the gradients have d + 1 components
    y += 0.05 * arma::randn<arma::vec>(n);
    par.nugget = 0.01;
    gamma = arma::join_cols(theta, arma::vec{0.8});
  }
  k.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", par);

  Budget restore;
  Optim::set_num_threads_percent(1e-9);  // budget: 1 thread
  REQUIRE(Optim::get_num_threads() == 1);
  const std::vector<arma::vec> g1 = gradients(k, gamma);
#if defined(__APPLE__)
  // Accelerate: thread count not settable by the library, results not
  // reproducible from one call to the next (macOS CI, n = 100)
  const bool blas_reproducible = false;
#else
  const bool blas_reproducible = !lk_test::blas_count_can_change() && bitwise_equal(g1, gradients(k, gamma));
#endif

  Optim::set_num_threads_percent(0);  // whole budget
  INFO("thread budget: " << Optim::get_num_threads());
  const std::vector<arma::vec> gn = gradients(k, gamma);
  if (!blas_reproducible)
    WARN(
        "BLAS count may change with the budget, or two identical gradient calls differ: compared to 1e-10 "
        "relative");
  require_equal(g1, gn, blas_reproducible ? 0.0 : 1e-10);
  for (const auto& g : g1)
    REQUIRE(g.is_finite());
}

TEST_CASE("Gradient pair sums: same bits whatever the thread count, documented order", "[gradient][threads]") {
  const std::string kernel = GENERATE("gauss", "exp", "matern3_2", "matern5_2");
  const arma::uword n = 300, d = 3;
  CAPTURE(kernel);
  arma::arma_rng::set_seed(7);
  const arma::mat X(n, d, arma::fill::randu);
  const arma::mat dX = LinearAlgebra::compute_dX(X);
  const arma::vec theta{0.2, 0.5, 1.3};
  const arma::mat W1(n, n, arma::fill::randn), W2(n, n, arma::fill::randn);
  auto weights = [&](arma::uword i, arma::uword j) { return std::array<double, 2>{W1.at(j, i), W2.at(j, i)}; };

  libKriging::detail::with_dln(kernel, theta, [&](const auto& dln) {
    // reference: the documented order, rows j ascending, then rows i ascending
    arma::mat rows(2 * d, n, arma::fill::zeros);
    arma::vec D(d);
    for (arma::uword i = 0; i < n; i++)
      for (arma::uword j = 0; j < i; j++) {
        dln(dX.colptr(i * n + j), D.memptr());
        const auto w = weights(i, j);
        for (arma::uword l = 0; l < 2; l++)
          for (arma::uword k = 0; k < d; k++)
            rows.at(l * d + k, i) += w[l] * D.at(k);
      }
    arma::vec ref(2 * d, arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++)
      for (arma::uword r = 0; r < 2 * d; r++)
        ref.at(r) += rows.at(r, i);

    for (int threads : {1, 2, 3, 4, 7}) {
      CAPTURE(threads);
      arma::vec out(2 * d);
      libKriging::detail::sum_pairs_dln<2>(dX, n, theta, dln, threads, weights, out.memptr());
      for (arma::uword r = 0; r < 2 * d; r++)
        REQUIRE(out.at(r) == ref.at(r));
    }
  });
}
