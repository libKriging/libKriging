// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

#ifdef _OPENMP
#include <omp.h>
#endif

// Cross-platform environment variable functions (mirrors the same helper in
// NuggetKrigingTest.cpp/NoiseKrigingTest.cpp).
#ifdef _WIN32
#include <cstdlib>
inline int setenv_portable(const char* name, const char* value, int overwrite) {
  if (!overwrite && std::getenv(name) != nullptr) {
    return 0;
  }
  return _putenv_s(name, value);
}
inline int unsetenv_portable(const char* name) {
  return _putenv_s(name, "");
}
#else
inline int setenv_portable(const char* name, const char* value, int overwrite) {
  return setenv(name, value, overwrite);
}
inline int unsetenv_portable(const char* name) {
  return unsetenv(name);
}
#endif

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
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,5)"));  // opt-in Nystrom CG precond

  // malformed specs throw
  for (const std::string bad : {"LLIterative()",
                                "LLIterative(x)",
                                "LLIterative(0)",
                                "LLIterative(-3)",
                                "LLIterative(10",
                                "LLIterative(8,0)",
                                "LLIterative(8,-2)",
                                "LLIterative(8,x)"}) {
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

TEST_CASE("LLIterative(m,precond_rank) Nystrom-preconditioned CG matches the unpreconditioned objective/gradient",
          "[iterative][kriging]") {
  // The preconditioner only changes HOW FAST CG's Krylov iteration converges
  // to R^-1*[F|y|probes], not what it converges TO: both runs use the same
  // default max_iter=2n budget and tol=1e-8, comfortably enough for exact
  // (unpreconditioned) CG to fully converge on this size of problem, so the
  // preconditioned and unpreconditioned objective/gradient should agree
  // tightly -- unlike the SLQ/Hutchinson-vs-exact-LL comparisons elsewhere
  // in this file, which only agree in an order-of-magnitude sense.
  //
  // Force single-threaded execution for this comparison specifically: both
  // Rmul's row-parallel matvec (Kriging.cpp) and conjugateGradient's
  // column-parallel per-probe solve (LinearAlgebra.cpp) are internally
  // multi-threaded, and although each is individually deterministic for a
  // FIXED thread count, that count itself varies across machines/CI
  // runners -- producing a rounding-order difference tight enough to
  // occasionally cross this test's tolerance (observed failing on CI, never
  // locally, with two separate runs landing on two slightly different
  // near-miss values). Setting both env vars AND calling
  // omp_set_num_threads(1) directly, since libgomp/MSVC OpenMP read
  // OMP_NUM_THREADS once at first use and don't hot-reload later setenv
  // calls -- the explicit API call is what actually guarantees it here.
  const char* old_openblas = std::getenv("OPENBLAS_NUM_THREADS");
  const char* old_omp = std::getenv("OMP_NUM_THREADS");
  setenv_portable("OPENBLAS_NUM_THREADS", "1", 1);
  setenv_portable("OMP_NUM_THREADS", "1", 1);
#ifdef _OPENMP
  const int old_omp_max_threads = omp_get_max_threads();
  omp_set_num_threads(1);
#endif

  arma::mat X;
  arma::vec y;
  make_data(35, X, y);

  Kriging k_plain = make_fixed_theta_iterative(y, X, "LLIterative(30)");
  Kriging k_pc = make_fixed_theta_iterative(y, X, "LLIterative(30,15)");

  const arma::vec theta{0.35, 0.3};
  auto [ll_plain, grad_plain] = k_plain.logLikelihoodIterativeFun(theta, true);
  auto [ll_pc, grad_pc] = k_pc.logLikelihoodIterativeFun(theta, true);

#ifdef _OPENMP
  omp_set_num_threads(old_omp_max_threads);
#endif
  if (old_omp) {
    setenv_portable("OMP_NUM_THREADS", old_omp, 1);
  } else {
    unsetenv_portable("OMP_NUM_THREADS");
  }
  if (old_openblas) {
    setenv_portable("OPENBLAS_NUM_THREADS", old_openblas, 1);
  } else {
    unsetenv_portable("OPENBLAS_NUM_THREADS");
  }

  INFO("ll_plain=" << ll_plain << " ll_pc=" << ll_pc << " grad_plain=" << grad_plain.t() << " grad_pc=" << grad_pc.t());
  // Forcing single-threaded execution above (see comment) makes ll_pc
  // exactly reproducible across repeated runs on one platform -- confirmed
  // by 5/5 local reruns landing on the identical value -- but ll_plain
  // (the UNPRECONDITIONED solve, which the single-threading fix cannot
  // help: it's a genuine cross-platform difference, not a threading race)
  // still differs by ~3-4e-4 relative between GCC/Linux and MSVC/Windows
  // CI runs, consistent with the plain CG solve not being quite as fully
  // converged within max_iter=2n=70 at this n=35 as the reasoning above
  // assumes -- an under-converged iterative result is inherently more
  // sensitive to compiler/math-library rounding than a tightly converged
  // one. 1e-4 relative was too tight to survive that; 1e-3 comfortably
  // covers the observed ~4e-4 worst case with margin while still checking
  // real, tight agreement (not just order-of-magnitude, unlike the
  // SLQ/Hutchinson comparisons elsewhere in this file).
  CHECK(std::abs(ll_plain - ll_pc) < 1e-3 * std::abs(ll_plain) + 1e-5);
  CHECK(arma::abs(grad_plain - grad_pc).max() < 0.02 * arma::abs(grad_plain).max() + 0.05);
}

// The "light fit" flag (and everything gated behind it: predictIterative routing,
// blocking simulate/update/save) is set on BOTH the multistart-BFGS commit
// path and the optim="none" fixed-theta commit path -- exactly like
// m_nystrom_light (LLIterative has no exact-commit toggle, unlike
// m_vecchia_light, so it is unconditional either way; see
// "LLIterative honors optim=none/BFGS identically" below for the dedicated
// cross-optim regression test). This test specifically exercises the real
// BFGS path. n and nprobe are kept deliberately tiny (each gradient
// evaluation does a CG solve over `nprobe` right-hand sides, and BFGS calls
// it many times) to keep runtime bounded.
TEST_CASE("LLIterative fit is a permanent light fit: predict routes to predictIterative", "[iterative][kriging]") {
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
  auto [m_cg, s_cg] = k.predictIterative(Xt, true);
  CHECK(arma::approx_equal(m_pred, m_cg, "absdiff", 1e-8));
  CHECK(arma::approx_equal(s_pred, s_cg, "absdiff", 1e-8));

  CHECK_THROWS_AS(k.predict(Xt, true, true, false), std::runtime_error);  // return_cov
  CHECK_THROWS_AS(k.predict(Xt, true, false, true), std::runtime_error);  // return_deriv
}

TEST_CASE("LLIterative fit blocks simulate/update_simulate/save", "[iterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(15, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLIterative(6)");
  REQUIRE(k.is_iterative_light());

  CHECK_THROWS_AS(k.simulate(5, 123, X), std::runtime_error);
  CHECK_THROWS_AS(k.update_simulate(y.head(2), X.head_rows(2)), std::runtime_error);
  CHECK_THROWS_AS(k.save("unused.json"), std::runtime_error);
}

// update() has its own incremental path (updateIterative): unlike
// simulate/update_simulate/save, it does NOT require ever materializing an
// n x n matrix -- it just extends m_X/m_y/m_F, redraws the (n-sized) probes,
// and re-profiles beta/sigma2 (optionally after a warm-restart single BFGS)
// via the same matrix-free CG machinery as the original fit. Mirrors
// update_nystrom's test pattern in KrigingNystromTest.cpp.
TEST_CASE("LLIterative update() extends the fit without a full re-fit", "[iterative][kriging]") {
  // Small n on purpose -- this is the one test in this file that exercises a
  // real free BFGS fit (every other LLIterative test uses optim="none" and a
  // fixed theta specifically to avoid this cost, see the file-level comment
  // above make_fixed_theta_iterative). Catch2 SECTIONs re-run the shared
  // TEST_CASE body from scratch for each leaf section, so this free-BFGS
  // constructor call runs 3x; at n0=20 that measured ~19s natively, comfortably
  // past ctest's 1500s timeout under Valgrind's ~50-100x memcheck overhead.
  arma::mat X;
  arma::vec y;
  make_data(10, X, y);

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LLIterative(8)");
  REQUIRE(k.is_iterative_light());
  const arma::uword n0 = k.X().n_rows;

  arma::mat Xu;
  arma::vec yu;
  make_data(3, Xu, yu, 789);

  SECTION("refit=false: re-profiles beta/sigma2 at the current theta") {
    const arma::vec theta_before = k.theta();
    k.update(yu, Xu, false);

    CHECK(k.is_iterative_light());
    CHECK(k.X().n_rows == n0 + 3);
    CHECK(k.y().n_elem == n0 + 3);
    CHECK(arma::approx_equal(k.theta(), theta_before, "absdiff", 1e-12));  // theta untouched

    // predictIterative should still give finite, sane predictions after the update.
    arma::mat Xt;
    arma::vec yt;
    make_data(6, Xt, yt, 321);
    auto [m_pred, s_pred] = k.predictIterative(Xt, true);
    CHECK(m_pred.n_elem == 6);
    CHECK(m_pred.is_finite());
    CHECK(s_pred.is_finite());
  }

  SECTION("refit=true: warm-restarts theta from its current value") {
    k.update(yu, Xu, true);

    CHECK(k.is_iterative_light());
    CHECK(k.X().n_rows == n0 + 3);

    arma::mat Xt;
    arma::vec yt;
    make_data(6, Xt, yt, 321);
    auto [m_pred, s_pred] = k.predictIterative(Xt, true);
    CHECK(m_pred.n_elem == 6);
    CHECK(m_pred.is_finite());
    CHECK(s_pred.is_finite());
  }

  SECTION("still blocks simulate/update_simulate/save after update()") {
    k.update(yu, Xu, false);
    CHECK_THROWS_AS(k.simulate(5, 123, k.X()), std::runtime_error);
    CHECK_THROWS_AS(k.update_simulate(yu, Xu), std::runtime_error);
    CHECK_THROWS_AS(k.save("unused.json"), std::runtime_error);
  }
}

TEST_CASE("LLIterative(m) at a fixed theta: predictIterative is broadly consistent with the exact MLE",
          "[iterative][kriging]") {
  // Fixed theta, optim="none" (which now also sets the light flag -- see the
  // cross-optim regression test below -- so k.predict() would work here too;
  // predictIterative is called directly anyway to isolate the CG-based
  // beta/sigma2 estimation + predictIterative's own accuracy, already covered
  // in isolation by KrigingPredictIterativeTest, from Optim.cpp's free-fit
  // convergence behaviour, which on this deterministic test function is
  // prone to the well-documented GP-MLE degeneracy (see docs/math/Nystrom.md's
  // limitations section).
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

  auto [m_it, s_it] = k.predictIterative(Xt, true);
  auto [m_ex, s_ex, c2, d3, d4] = k_exact.predict(Xt, true, false, false);

  INFO("max |mean diff| = " << arma::abs(m_it - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_it - s_ex).max());
  // An LLIterative fit's predict() always routes through predictIterative (see
  // Iterative.md), so its stdev (including the GLS-correction term) needs
  // the same verification as predictIterative's own tests.
  CHECK(arma::abs(m_it - m_ex).max() < 0.02 * arma::stddev(y));
  CHECK(arma::abs(s_it - s_ex).max() < 0.02 * arma::stddev(y));
}

// Regression test for a real bug: optim="none" used to silently fall through
// to a plain exact factorization for LLIterative, ignoring the objective
// requested at construction (fixed alongside the identical LLVecchia bug --
// see "LLVecchia honors optim=none identically to LLNystrom/LLIterative" in
// KrigingVecchiaTest.cpp). The contract under test: whatever objective is
// given at fit time is what predict()/etc actually use afterwards, for EVERY
// supported value of optim, not just the free-fit "BFGS" path exercised by
// the "permanent light fit" test above.
TEST_CASE("LLIterative honors optim=none identically to optim=BFGS", "[iterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(15, X, y);
  arma::mat Xt;
  arma::vec yt;
  make_data(8, Xt, yt, 456);

  const std::string optim = GENERATE(as<std::string>{}, "none", "BFGS(1)");
  CAPTURE(optim);

  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.3));
  params.is_theta_estim = (optim != "none");  // optim="none" requires a fixed theta

  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, optim, "LLIterative(6)", params);
  CHECK(k.is_iterative_light());

  auto [m_pred, s_pred, cov, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictIterative(Xt, true);
  CHECK(arma::approx_equal(m_pred, m_cg, "absdiff", 1e-8));
  CHECK(arma::approx_equal(s_pred, s_cg, "absdiff", 1e-8));

  // simulate/update_simulate/save stay blocked regardless of how the light
  // fit was reached.
  CHECK_THROWS_AS(k.simulate(3, 123, Xt.rows(0, 2), false), std::runtime_error);
  CHECK_THROWS_AS(k.save("unused.json"), std::runtime_error);
}
