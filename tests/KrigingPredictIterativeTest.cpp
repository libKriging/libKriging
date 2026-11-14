// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
// clang-format on

// Cross-platform environment variable functions (mirrors the same helper in
// KrigingIterativeTest.cpp).
#ifdef _WIN32
#include <cstdlib>
#include <string>
#include <tuple>
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

// Builds a Kriging model with a FIXED, moderate theta (via optim="none"),
// rather than letting BFGS free-fit it. This test function is deterministic
// (noise-free), and its exact MLE is known to drift toward a very large
// theta (a well-documented GP-MLE degeneracy on noise-free data, unrelated
// to predict method "iterative" -- see KrigingNystromTest.cpp's history for the same issue).
// A large theta makes the correlation matrix extremely ill-conditioned,
// which is a genuinely hard regime for ANY iterative solver (not a
// predict method "iterative"-specific weakness); fixing theta at a moderate, well-conditioned
// value isolates predict method "iterative"'s own correctness from that unrelated fit issue.
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

// predict(..., method) with an iterative spec: (mean, stdev)
static std::tuple<arma::vec, arma::vec> pit(Kriging& k,
                                            const arma::mat& Xt,
                                            bool return_stdev,
                                            const std::string& method = "iterative") {
  auto [mean, stdev, cov, dm, ds] = k.predict(Xt, return_stdev, false, false, method);
  return {mean, stdev};
}

// -----------------------------------------------------------------------------

TEST_CASE("predict method iterative mean/stdev match exact predict at a moderate theta",
          "[predictiterative][kriging]") {
  // n/n_n kept small on purpose: predict method "iterative"'s return_stdev path runs one CG
  // solve PER prediction point (O(n^2 * iters * n_n), see predict method "iterative"'s own
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
  auto [m_cg, s_cg] = pit(k, Xt, true);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  // predict method "iterative" solves the same exact system as predict() (same
  // objective/theta), via CG instead of a stored Cholesky factor -- at this
  // well-conditioned theta, default settings converge to ~1e-9.
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predict method iterative: the dense fast path matches the matrix-free path", "[predictiterative][kriging]") {
  // For a separable kernel and small enough n, predictIterative_impl
  // materializes R once (KrigingImpl::build_separable_cov, shared with
  // Kriging::_logLikelihoodIterative) and solves via BLAS-3 R*V instead of
  // the matrix-free per-pair matvec. LK_ITERATIVE_DENSE_MAX_MB=0 forces the
  // matrix-free path. Same theta, same CG budget => the two must agree
  // tightly (the only difference is BLAS vs pair-loop summation order).
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt;
  arma::vec yt;
  make_data(15, Xt, yt, 456);

  const char* old_env = std::getenv("LK_ITERATIVE_DENSE_MAX_MB");

  // Both paths must be compared at CONVERGED solves: two non-converged
  // iterates sharing a residual norm need not share digits beyond it. The
  // default tol=1e-8 bounds the mean error only to ~cond(R)*1e-8 (cond ~4.5e3
  // here) and the stdev solves run at sqrt(tol), so the comparison uses a
  // tol at the round-off floor (5th field of the spec) and a generous
  // iteration budget (10 n, 4th field) instead.
  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "0", 1);  // force matrix-free
  auto [m_mf, s_mf] = pit(k, Xt, true, "iterative(30,0,20,10,1e-14)");

  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "4096", 1);  // allow dense
  auto [m_de, s_de] = pit(k, Xt, true, "iterative(30,0,20,10,1e-14)");

  // and the preconditioned solve too
  auto [m_mf_pc, s_mf_pc] = ([&] {
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "0", 1);
    return pit(k, Xt, true, "iterative(30,20,20,10,1e-14)");
  })();
  auto [m_de_pc, s_de_pc] = ([&] {
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "4096", 1);
    return pit(k, Xt, true, "iterative(30,20,20,10,1e-14)");
  })();

  if (old_env)
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", old_env, 1);
  else
    unsetenv_portable("LK_ITERATIVE_DENSE_MAX_MB");

  INFO("max |mean diff| = " << arma::abs(m_mf - m_de).max());
  INFO("max |stdev diff| = " << arma::abs(s_mf - s_de).max());
  CHECK(arma::abs(m_mf - m_de).max() < 1e-8 * arma::stddev(y));
  // stdev solves run at sqrt(tol_cmp) = 1e-7 by design (see
  // predictIterative_impl's stdev_tol), hence a correspondingly looser bound.
  CHECK(arma::abs(s_mf - s_de).max() < 1e-6 * arma::stddev(y));
  CHECK(arma::abs(m_mf_pc - m_de_pc).max() < 1e-8 * arma::stddev(y));
  CHECK(arma::abs(s_mf_pc - s_de_pc).max() < 1e-6 * arma::stddev(y));
}

TEST_CASE("predict method iterative defaults to mean only (stdev empty)", "[predictiterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt(10, 2, arma::fill::randu);
  auto [mean, stdev] = pit(k, Xt, false);
  CHECK(mean.n_elem == 10);
  CHECK(stdev.n_elem == 0);
}

TEST_CASE("predict method iterative interpolates the training data", "[predictiterative][kriging]") {
  // Small n on purpose -- see the sizing comment on the first predict method "iterative" test
  // case above (predicts at X itself here, so n_n = n too).
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  auto [mean, stdev] = pit(k, X, true);
  CHECK(arma::abs(mean - y).max() < 0.05 * arma::stddev(y));
  CHECK(stdev.max() < 0.05 * arma::stddev(y));
}

TEST_CASE("predict method iterative matches predict for a different kernel/trend", "[predictiterative][kriging]") {
  // Small n/n_n on purpose -- see the sizing comment on the first predict method "iterative"
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
  auto [m_cg, s_cg] = pit(k, Xt, true, "iterative(30,0,20,125,1e-12)");

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predict method iterative rejects wrong dimension and Nugget models", "[predictiterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(50, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xbad(5, 3, arma::fill::randu);
  CHECK_THROWS_AS(pit(k, Xbad, false), std::invalid_argument);

  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
  arma::mat Xt(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(pit(knug, Xt, false), std::invalid_argument);
}

TEST_CASE("predict method iterative with Nystrom preconditioning matches exact predict",
          "[predictiterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = pit(k, Xt, true, "iterative(30,20)");

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  // Nystrom-preconditioned CG converges to the same exact result as plain
  // CG, just faster -- ~1e-9 here at default tol=1e-8.
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predict method iterative with an iterative fit matches exact predict (cache warm start)",
          "[predictiterative][kriging][iterative]") {
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);

  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.1));
  params.is_theta_estim = false;

  Kriging k_exact(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params);
  Kriging k_iter(
      y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params, "iterative(10,0,20,4,1e-10)");
  REQUIRE(k_iter.is_iterative_light());

  arma::mat Xt1, Xt2;
  arma::vec yt1, yt2;
  make_data(10, Xt1, yt1, 456);
  make_data(8, Xt2, yt2, 789);

  auto [m_ex1, s_ex1, c1, dm1, ds1] = k_exact.predict(Xt1, true, false, false);
  auto [m_cg1, s_cg1] = pit(k_iter, Xt1, true, "iterative(10,0,20,2,1e-10)");
  INFO("call 1: max |mean diff| = " << arma::abs(m_cg1 - m_ex1).max()
                                    << ", max |stdev diff| = " << arma::abs(s_cg1 - s_ex1).max());
  CHECK(arma::abs(m_cg1 - m_ex1).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg1 - s_ex1).max() < 1e-5 * arma::stddev(y));

  // Second call, different X_n, same model/cache -- the cache is read-only
  // from predict method "iterative"'s point of view (only fit/update write it), so
  // this must be just as accurate as the first call.
  auto [m_ex2, s_ex2, c2, dm2, ds2] = k_exact.predict(Xt2, true, false, false);
  auto [m_cg2, s_cg2] = pit(k_iter, Xt2, true, "iterative(10,0,20,2,1e-10)");
  INFO("call 2: max |mean diff| = " << arma::abs(m_cg2 - m_ex2).max()
                                    << ", max |stdev diff| = " << arma::abs(s_cg2 - s_ex2).max());
  CHECK(arma::abs(m_cg2 - m_ex2).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg2 - s_ex2).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predict method iterative benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt(100, 2, arma::fill::randu);

  BENCHMARK("Kriging::predict (exact, stored Cholesky) n=300") {
    return k.predict(Xt, true, false, false);
  };
  BENCHMARK("Kriging::predict method iterative (matrix-free) n=300") {
    return pit(k, Xt, true);
  };
}
