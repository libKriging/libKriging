// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
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

// A CG solve over `nprobe` right-hand sides (each up to max_iter=2n Krylov
// iterations, each an O(n^2) matvec) is the expensive part of every
// LLIterative gradient evaluation. Free BFGS optimization on top of that
// multiplies it by however many iterations BFGS needs, which explodes fast.
// Every test below therefore fits with optim="none" and a fixed theta
// (isolating the objective/gradient's own correctness from Optim.cpp's
// convergence behaviour) and keeps n/nprobe small, except the one dedicated
// smoke test that deliberately exercises a real free BFGS fit.
static Kriging make_fixed_theta_iterative(const arma::vec& y,
                                          const arma::mat& X,
                                          const std::string& objective,
                                          double theta_val = 0.3) {
  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(theta_val));
  params.is_theta_estim = false;
  return Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", objective, params);
}

// -----------------------------------------------------------------------------

TEST_CASE("LLIterative objective spec parsing and validation", "[iterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(25, X, y);

  // valid specs fit fine (fixed theta: only spec parsing + a single cheap
  // no-grad evaluation happen, no CG-heavy optimization)
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative"));
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8)"));

  // malformed specs throw
  for (const std::string bad :
       {"LLIterative()", "LLIterative(x)", "LLIterative(0)", "LLIterative(-3)", "LLIterative(10"}) {
    CHECK_THROWS_AS(make_fixed_theta_iterative(y, X, bad), std::invalid_argument);
  }

  // LLIterative is not available with a nugget/noise channel
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLIterative(8)", {}),
                  std::invalid_argument);
}

TEST_CASE("LLIterative(m) approximates the exact concentrated log-likelihood", "[iterative][kriging]") {
  // A moderate probe count keeps the SLQ/Hutchinson estimators' inherent
  // stochastic error small enough for a meaningful (not exact -- there is
  // no finite-m exact limit here, unlike LLNystrom's k=n case) comparison.
  // Probes are drawn with a FIXED seed (make_iterative_probes), so this is
  // fully reproducible run to run, not flaky.
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);

  Kriging k = make_fixed_theta_iterative(y, X, "LLIterative(40)");

  for (const arma::vec& theta : {arma::vec{0.2, 0.2}, arma::vec{0.4, 0.3}}) {
    auto [ll_it, gv] = k.logLikelihoodIterativeFun(theta, false);
    auto [ll, gl] = k.logLikelihoodFun(theta, false, false);
    INFO("theta=" << theta.t() << ": LLIterative(40) = " << ll_it << " vs exact LL = " << ll);
    CHECK(std::abs(ll_it - ll) < 0.15 * std::abs(ll) + 1.0);
  }
}

TEST_CASE("LLIterative analytic gradient points in an ascending direction", "[iterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);

  Kriging k = make_fixed_theta_iterative(y, X, "LLIterative(20)");

  for (const arma::vec& theta : {arma::vec{0.2, 0.3}, arma::vec{0.5, 0.15}, arma::vec{0.3, 0.3}}) {
    auto [ll0, grad] = k.logLikelihoodIterativeFun(theta, true);
    REQUIRE(grad.n_elem == theta.n_elem);
    REQUIRE(arma::norm(grad) > 0.0);
    const arma::vec dir = arma::normalise(grad);
    const double step = 1e-3;
    const double ll_up = std::get<0>(k.logLikelihoodIterativeFun(theta + step * dir, false));
    INFO("theta=" << theta.t() << ": ll0=" << ll0 << " ll_up=" << ll_up);
    CHECK(ll_up > ll0);
  }
}

TEST_CASE("LLIterative analytic gradient approximately matches finite differences", "[iterative][kriging]") {
  // Unlike LLNystrom's exact Woodbury identities, the analytic gradient here
  // is an *independent* stochastic estimator from the log-det it accompanies:
  // stochasticLogDet approximates z^T log(A) z via a *truncated* Lanczos
  // quadrature, while the analytic gradient's trace term is a plain
  // Hutchinson estimate of z^T A^-1 dA/dtheta z using the exact (CG-solved)
  // A^-1. Both are unbiased estimators of the same true quantities in
  // expectation, but for one FIXED set of probes their FD relationship is
  // only approximate, not exact -- so this checks order-of-magnitude and
  // sign agreement, not a tight numerical match.
  arma::mat X;
  arma::vec y;
  make_data(35, X, y);

  Kriging k = make_fixed_theta_iterative(y, X, "LLIterative(60)");

  const double h = 1e-3;
  const arma::vec theta{0.4, 0.4};
  auto [ll0, grad] = k.logLikelihoodIterativeFun(theta, true);
  REQUIRE(grad.n_elem == 2);

  for (arma::uword kk = 0; kk < 2; ++kk) {
    arma::vec theta_p = theta;
    arma::vec theta_m = theta;
    theta_p(kk) += h;
    theta_m(kk) -= h;
    const double ll_p = std::get<0>(k.logLikelihoodIterativeFun(theta_p, false));
    const double ll_m = std::get<0>(k.logLikelihoodIterativeFun(theta_m, false));
    const double fd = (ll_p - ll_m) / (2 * h);
    INFO("dim=" << kk << " analytic=" << grad(kk) << " fd=" << fd);
    CHECK(grad(kk) * fd > 0.0);  // same sign
    CHECK(std::abs(grad(kk) - fd) < 0.5 * (std::abs(grad(kk)) + std::abs(fd)) + 3.0);
  }
}

// The "light fit" flag (and everything gated behind it: predictCG routing,
// blocking simulate/update/save) is only set on the actual multistart-BFGS
// commit path -- exactly like m_nystrom_light/m_vecchia_light. optim="none"
// bypasses that path entirely and does a plain exact factorization (mirrors
// existing LLNystrom/LLVecchia behaviour), so these tests need a real BFGS
// fit. n and nprobe are kept deliberately tiny (each gradient evaluation
// does a CG solve over `nprobe` right-hand sides, and BFGS calls it many
// times) to keep runtime bounded.
TEST_CASE("LLIterative fit is a permanent light fit: predict routes to predictCG", "[iterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(15, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLIterative(6)");
  CHECK(k.iterative_nprobe() == 6);
  CHECK(k.is_iterative_light());

  arma::mat Xt;
  arma::vec yt;
  make_data(8, Xt, yt, 456);

  auto [m_pred, s_pred, cov, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictCG(Xt, true);
  CHECK(arma::approx_equal(m_pred, m_cg, "absdiff", 1e-8));
  CHECK(arma::approx_equal(s_pred, s_cg, "absdiff", 1e-8));

  CHECK_THROWS_AS(k.predict(Xt, true, true, false), std::runtime_error);  // return_cov
  CHECK_THROWS_AS(k.predict(Xt, true, false, true), std::runtime_error);  // return_deriv
}

TEST_CASE("LLIterative fit blocks simulate/update/update_simulate/save", "[iterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(15, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLIterative(6)");
  REQUIRE(k.is_iterative_light());

  CHECK_THROWS_AS(k.simulate(5, 123, X), std::runtime_error);
  CHECK_THROWS_AS(k.update(y.head(2), X.head_rows(2), true), std::runtime_error);
  CHECK_THROWS_AS(k.update_simulate(y.head(2), X.head_rows(2)), std::runtime_error);
  CHECK_THROWS_AS(k.save("unused.json"), std::runtime_error);
}

TEST_CASE("LLIterative(m) at a fixed theta: predictCG is broadly consistent with the exact MLE",
          "[iterative][kriging]") {
  // Fixed theta (optim="none" doesn't set the light flag, so call
  // predictCG directly to force the CG-based prediction path regardless):
  // isolates the CG-based beta/sigma2 estimation + predictCG's own accuracy
  // (already covered in isolation by KrigingPredictCGTest) from Optim.cpp's
  // free-fit convergence behaviour, which on this deterministic test
  // function is prone to the well-documented GP-MLE degeneracy (see
  // docs/math/Nystrom.md's limitations section).
  // grad_out is null on this path (theta fixed => no optimizer gradient
  // calls), so this stays cheap even at n=80.
  arma::mat X;
  arma::vec y;
  make_data(80, X, y);

  Kriging k = make_fixed_theta_iterative(y, X, "LLIterative(80)");
  Kriging k_exact = make_fixed_theta_iterative(y, X, "LL");

  arma::mat Xt;
  arma::vec yt;
  make_data(20, Xt, yt, 789);

  auto [m_it, s_it] = k.predictCG(Xt, true);
  auto [m_ex, s_ex, c2, d3, d4] = k_exact.predict(Xt, true, false, false);

  INFO("max |mean diff| = " << arma::abs(m_it - m_ex).max());
  CHECK(arma::abs(m_it - m_ex).max() < 0.1 * arma::stddev(y));
}
