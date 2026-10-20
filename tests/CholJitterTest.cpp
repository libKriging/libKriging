// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include <cmath>
#include <cstdlib>
#include <string>
// clang-format on

/**
 * @file CholJitterTest.cpp
 * @brief Fixed Cholesky jitter of noise-free fits (LinearAlgebra::kernel_chol_jitter,
 *        docs/dev/CholJitter-TODO.md): the fit no longer depends on the thread
 *        budget, the objective has exactly the jitter on its diagonal, and
 *        set_chol_jitter(0) / LK_CHOL_JITTER=0 give the exact objective back.
 */

namespace {

// sum_j sin(2 pi f_j x_j) on n random points of [0,1]^d (bench/thread-budget studies)
void sin_data(int seed, arma::uword n, arma::uword d, arma::mat& X, arma::vec& y) {
  arma::arma_rng::set_seed(seed);
  X = arma::randu(n, d);
  const arma::rowvec f = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
  y.zeros(n);
  for (arma::uword j = 0; j < d; ++j)
    y += arma::sin(2 * arma::datum::pi * f(j) * X.col(j));
}

// Concentrated log-likelihood, constant trend, gauss kernel, with `jitter`
// added to the diagonal of R: computed independently of the library.
double gauss_ll(const arma::mat& X, const arma::vec& y, const arma::vec& theta, double jitter) {
  const arma::uword n = X.n_rows;
  arma::mat R(n, n);
  for (arma::uword i = 0; i < n; ++i)
    for (arma::uword j = 0; j < n; ++j)
      R(i, j) = std::exp(-0.5 * arma::accu(arma::square((X.row(i) - X.row(j)).t() / theta)));
  R.diag() += jitter;
  const arma::mat L = arma::chol(R, "lower");
  const arma::vec one(n, arma::fill::ones);
  const arma::vec Liy = arma::solve(arma::trimatl(L), y), Li1 = arma::solve(arma::trimatl(L), one);
  const double beta = arma::dot(Li1, Liy) / arma::dot(Li1, Li1);
  const arma::vec z = Liy - beta * Li1;
  const double sigma2 = arma::dot(z, z) / n;
  return -0.5 * n * std::log(2 * arma::datum::pi * sigma2) - arma::accu(arma::log(L.diag())) - 0.5 * n;
}

}  // namespace

TEST_CASE("fixed Cholesky jitter: the fit does not depend on the thread budget", "[jitter]") {
  // Single-start fits in the near-singular region of smooth kernels. Without
  // the jitter these cases end in a different basin depending on the thread
  // budget on some machines (gauss d = 4 seed 8: theta spread 31 %, matern5_2
  // seed 1: 0.3 %); with it, the fits agree to the optimizer tolerance (the
  // rounding still moves theta by ~1e-8, not bit-identical).
  struct Case {
    std::string kernel;
    arma::uword n, d;
    int seed;
  };
  const Case c = GENERATE(Case{"gauss", 200, 2, 1}, Case{"gauss", 200, 4, 8}, Case{"matern5_2", 200, 4, 1});
  arma::mat X;
  arma::vec y;
  sin_data(c.seed, c.n, c.d, X, y);

  const double saved_pct = Optim::get_num_threads_percent();
  LinearAlgebra::set_chol_jitter(-1);  // per-kernel default
  arma::mat theta(3, c.d);
  arma::vec ll(3);
  int k = 0;
  for (const double pct : {10.0, 100.0, 200.0}) {
    Optim::set_num_threads_percent(pct);
    Kriging m(c.kernel);
    m.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
    theta.row(k) = m.theta().t();
    ll(k++) = m.logLikelihood();
  }
  Optim::set_num_threads_percent(saved_pct);

  INFO(c.kernel << " n " << c.n << " d " << c.d << " seed " << c.seed << "\ntheta\n" << theta << "LL " << ll.t());
  for (int i = 1; i < 3; ++i) {
    CHECK(arma::max(arma::abs(theta.row(i) - theta.row(0)) / theta.row(0)) < 1e-4);
    CHECK(std::abs(ll(i) - ll(0)) < 1e-4 * std::max(1.0, std::abs(ll(0))));
  }
}

TEST_CASE("fixed Cholesky jitter: exact diagonal, set_chol_jitter(0) restores the exact objective", "[jitter]") {
  arma::mat X;
  arma::vec y;
  sin_data(7, 20, 2, X, y);
  const arma::vec theta{0.15, 0.2};  // well conditioned: the jitter is the only difference
  Kriging::Parameters fixed;
  fixed.theta = arma::mat(theta.t());
  fixed.is_theta_estim = false;

  const double saved = LinearAlgebra::get_chol_jitter();
  for (const double setting : {-1.0, 0.0, 1e-6}) {
    LinearAlgebra::set_chol_jitter(setting);
    const double expected_jitter = setting < 0 ? LinearAlgebra::kernel_chol_jitter("gauss") : setting;
    Kriging m("gauss");
    m.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", fixed);
    INFO("set_chol_jitter(" << setting << ")");
    CHECK(m.chol_jitter() == expected_jitter);
    const double ll = std::get<0>(m.logLikelihoodFun(theta, false, false));
    const double ll_ref = gauss_ll(X, y, theta, expected_jitter);
    CHECK(std::abs(ll - ll_ref) < 1e-10 * std::max(1.0, std::abs(ll_ref)));
  }
  LinearAlgebra::set_chol_jitter(saved);
  CHECK(LinearAlgebra::kernel_chol_jitter("gauss") == 1e-9);
  CHECK(LinearAlgebra::kernel_chol_jitter("matern5_2") == 1e-10);
  CHECK(LinearAlgebra::kernel_chol_jitter("matern3_2") == 0.0);
  CHECK(LinearAlgebra::kernel_chol_jitter("exp") == 0.0);
}

// Hidden: run by the CholJitterTest/LK_CHOL_JITTER test with LK_CHOL_JITTER=0
// in its environment (tests/CMakeLists.txt).
TEST_CASE("fixed Cholesky jitter: LK_CHOL_JITTER is read", "[.env]") {
  const char* v = std::getenv("LK_CHOL_JITTER");
  REQUIRE(v != nullptr);
  CHECK(LinearAlgebra::get_chol_jitter() == std::stod(v));
  arma::mat X;
  arma::vec y;
  sin_data(7, 20, 2, X, y);
  Kriging m("gauss");
  m.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
  CHECK(m.chol_jitter() == std::stod(v));
}
