// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

static double f2d(double x1, double x2) {
  return std::sin(3.0 * x1) + std::cos(5.0 * x2) + x1 * x2;
}

static void make_data(arma::uword n, arma::mat& X, arma::vec& y, unsigned seed = 123) {
  arma::arma_rng::set_seed(seed);
  X = arma::mat(n, 2, arma::fill::randu);
  y = arma::vec(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f2d(X(i, 0), X(i, 1));
}

// Builds a Kriging model with a FIXED, moderate theta (via optim="none"),
// rather than letting BFGS free-fit it. This test function is deterministic
// (noise-free), and its exact MLE is known to drift toward a very large
// theta (a well-documented GP-MLE degeneracy on noise-free data, unrelated
// to predictCG -- see KrigingNystromTest.cpp's history for the same issue).
// A large theta makes the correlation matrix extremely ill-conditioned,
// which is a genuinely hard regime for ANY iterative solver (not a
// predictCG-specific weakness); fixing theta at a moderate, well-conditioned
// value isolates predictCG's own correctness from that unrelated fit issue.
static Kriging make_fixed_theta_model(const arma::vec& y,
                                      const arma::mat& X,
                                      const std::string& cov,
                                      Trend::RegressionModel regmodel,
                                      double theta_val) {
  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(theta_val));
  params.is_theta_estim = false;
  return Kriging(y, X, cov, regmodel, false, "none", "LL", params);
}

// -----------------------------------------------------------------------------

TEST_CASE("predictCG mean/stdev match exact predict at a moderate theta", "[predictcg][kriging]") {
  // n/n_n kept small on purpose: predictCG's return_stdev path runs one CG
  // solve PER prediction point (O(n^2 * iters * n_n), see predictCG's own
  // comment in Kriging.cpp), which is fine at these sizes natively but adds
  // up to real minutes under Valgrind/TSan's 20-100x instrumentation
  // overhead -- large enough here to previously time out the CI memcheck/tsan
  // jobs. Kept just large enough to exercise the multi-RHS CG path.
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictCG(Xt, true);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  // predictCG solves the same exact system as predict() (same
  // objective/theta), via CG instead of a stored Cholesky factor -- at this
  // well-conditioned theta, default settings converge to ~1e-9.
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predictCG defaults to mean only (stdev empty)", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt(10, 2, arma::fill::randu);
  auto [mean, stdev] = k.predictCG(Xt);
  CHECK(mean.n_elem == 10);
  CHECK(stdev.n_elem == 0);
}

TEST_CASE("predictCG interpolates the training data", "[predictcg][kriging]") {
  // Small n on purpose -- see the sizing comment on the first predictCG test
  // case above (predicts at X itself here, so n_n = n too).
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  auto [mean, stdev] = k.predictCG(X, true);
  CHECK(arma::abs(mean - y).max() < 0.05 * arma::stddev(y));
  CHECK(stdev.max() < 0.05 * arma::stddev(y));
}

TEST_CASE("predictCG matches predict for a different kernel/trend", "[predictcg][kriging]") {
  // Small n/n_n on purpose -- see the sizing comment on the first predictCG
  // test case above.
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);
  Kriging k = make_fixed_theta_model(y, X, "gauss", Trend::RegressionModel::Linear, 0.15);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 789);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  // gauss kernel + linear trend at theta=0.15 is more ill-conditioned than
  // the matern5_2/constant case above; give CG a generous budget (still
  // <1s at n=40) so it actually converges before checking the mean/stdev
  // formulas match across kernels/trends -- the default budget's own
  // accuracy is covered separately by "predictCG accuracy improves with a
  // larger iteration budget" below.
  auto [m_cg, s_cg] = k.predictCG(Xt, true, /*max_iter=*/5000, /*tol=*/1e-12);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predictCG rejects wrong dimension and Nugget models", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(50, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xbad(5, 3, arma::fill::randu);
  CHECK_THROWS_AS(k.predictCG(Xbad), std::invalid_argument);

  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
  arma::mat Xt(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(knug.predictCG(Xt), std::runtime_error);
}

TEST_CASE("predictCG accuracy improves with a larger iteration budget", "[predictcg][kriging]") {
  // Sanity check that the CG loop is doing meaningful work: an artificially
  // tiny iteration budget should be measurably less accurate than the
  // default (max_iter=0 => 2n). Small n/n_n on purpose -- this was the
  // slowest case (n=150, n_n=40 originally), the one that actually timed
  // out CI's memcheck job at 1500s under Valgrind; see the sizing comment
  // on the first predictCG test case above.
  arma::mat X;
  arma::vec y;
  make_data(50, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_default, s_default] = k.predictCG(Xt, true);
  auto [m_tiny, s_tiny] = k.predictCG(Xt, true, 2, 1e-8);  // only 2 CG iterations

  const double err_default = arma::abs(m_default - m_ex).max();
  const double err_tiny = arma::abs(m_tiny - m_ex).max();
  INFO("err with default budget (2n) = " << err_default << ", err with max_iter=2 = " << err_tiny);
  CHECK(err_default < err_tiny);

  // Same sanity check for stdev, including its own GLS-correction solve.
  const double err_default_s = arma::abs(s_default - s_ex).max();
  const double err_tiny_s = arma::abs(s_tiny - s_ex).max();
  INFO("stdev err with default budget (2n) = " << err_default_s << ", err with max_iter=2 = " << err_tiny_s);
  CHECK(err_default_s < err_tiny_s);
}

TEST_CASE("predictCG with Nystrom preconditioning matches exact predict", "[predictcg][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictCG(Xt,
                                  true,
                                  /*max_iter=*/0,
                                  /*tol=*/1e-8,
                                  /*use_nystrom_precond=*/true,
                                  /*precond_rank=*/20);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  // Nystrom-preconditioned CG converges to the same exact result as plain
  // CG, just faster -- ~1e-9 here at default tol=1e-8.
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predictCG Nystrom preconditioning converges faster on a tight iteration budget", "[predictcg][kriging]") {
  // A larger theta makes R more strongly correlated/ill-conditioned (see the
  // sizing rationale in make_fixed_theta_model's comment) -- exactly the
  // regime a preconditioner should help with. Compare plain vs
  // Nystrom-preconditioned CG at the SAME, deliberately tight iteration
  // budget (tol set unreachably small so both runs use the full budget,
  // isolating the effect of the preconditioner from early stopping).
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.5);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);

  const arma::uword tight_budget = 6;
  auto [m_plain, s_plain] = k.predictCG(Xt, false, tight_budget, 1e-12);
  auto [m_pc, s_pc] = k.predictCG(Xt, false, tight_budget, 1e-12, /*use_nystrom_precond=*/true, /*precond_rank=*/30);

  const double err_plain = arma::abs(m_plain - m_ex).max();
  const double err_pc = arma::abs(m_pc - m_ex).max();
  INFO("err plain CG (budget=" << tight_budget << ") = " << err_plain
                               << ", err Nystrom-preconditioned CG = " << err_pc);
  CHECK(err_pc < err_plain);
}

TEST_CASE("predictCG benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt(100, 2, arma::fill::randu);

  BENCHMARK("Kriging::predict (exact, stored Cholesky) n=300") {
    return k.predict(Xt, true, false, false);
  };
  BENCHMARK("Kriging::predictCG (matrix-free) n=300") {
    return k.predictCG(Xt, true);
  };
}
