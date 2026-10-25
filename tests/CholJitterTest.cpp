// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/WarpKriging.hpp"
#include <cmath>
#include <cstdlib>
#include <string>
// clang-format on

/**
 * @file CholJitterTest.cpp
 * @brief Fixed Cholesky jitter of noise-free fits (LinearAlgebra::kernel_chol_jitter,
 *        docs/dev/CholJitter.md): the fit no longer depends on the thread
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

TEST_CASE("fixed Cholesky jitter: the fitted model interpolates the observations", "[jitter]") {
  // The jitter is only in the objective: the committed model is factorized
  // exactly. With 1e-10 on its diagonal this case (theta ~ (1, 9), R near
  // singular, the python-sdist CI check) missed y by 1e-5.
  const arma::mat X{{0.63696168732145431, 0.26978671376387031},
                    {0.040973523936194689, 0.016527635528529094},
                    {0.81327023920027242, 0.91275557727772172},
                    {0.60663577576717986, 0.7294965609839984},
                    {0.54362499146542287, 0.93507242378776823},
                    {0.81585355412153215, 0.0027385001701480949},
                    {0.85740427658756935, 0.033585575305464355},
                    {0.72965544642994407, 0.17565562060255901},
                    {0.86317892234988658, 0.54146122024909171},
                    {0.29971189053738478, 0.42268722119765845}};
  // numpy.random.default_rng(0).random((10, 2))
  const arma::vec y = arma::sin(3 * X.col(0)) + X.col(1);
  const double saved = LinearAlgebra::get_chol_jitter();
  LinearAlgebra::set_chol_jitter(-1);
  for (const std::string kernel : {"gauss", "matern5_2"}) {
    Kriging m(kernel);
    m.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
    INFO(kernel << " theta " << m.theta().t());
    CHECK(m.chol_jitter() == LinearAlgebra::kernel_chol_jitter(kernel));
    CHECK(m.model_chol_jitter() == 0.0);
    const arma::vec yp = std::get<0>(m.predict(X, false, false, false));
    CHECK(arma::abs(yp - y).max() < 1e-8);
  }
  LinearAlgebra::set_chol_jitter(saved);
}

TEST_CASE("WarpKriging: the multistart searches the jittered objective", "[jitter][warp]") {
  // The parallel starts run on clones (clone_for_thread), which must carry
  // the jitter: they optimized the objective without it, so the result was
  // not a maximum of the reported logLikelihood (4 to 19 lower on such data).
  arma::arma_rng::set_seed(4);
  const arma::mat X = arma::randu(60, 2);
  const arma::vec y = arma::sin(6 * X.col(0)) + arma::cos(4 * X.col(1));
  const double saved = LinearAlgebra::get_chol_jitter();
  LinearAlgebra::set_chol_jitter(-1);
  libKriging::WarpKriging wk(y, X, {"none", "none"}, "gauss", Trend::RegressionModel::Constant, true, "BFGS8");
  const arma::vec theta = wk.theta();
  const double ll = wk.logLikelihood();
  CHECK(std::abs(ll - std::get<0>(wk.logLikelihoodFun(theta, false))) < 1e-10 * std::max(1.0, std::abs(ll)));
  // a maximum of that objective along each theta axis
  for (arma::uword k = 0; k < theta.n_elem; ++k)
    for (const double f : {0.99, 1.01}) {
      arma::vec t = theta;
      t(k) *= f;
      INFO("theta " << theta.t() << "axis " << k << " factor " << f);
      CHECK(std::get<0>(wk.logLikelihoodFun(t, false)) <= ll + 1e-6 * std::abs(ll));
    }
  LinearAlgebra::set_chol_jitter(saved);
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
