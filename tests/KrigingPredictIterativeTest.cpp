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
// to predictIterative -- see KrigingNystromTest.cpp's history for the same issue).
// A large theta makes the correlation matrix extremely ill-conditioned,
// which is a genuinely hard regime for ANY iterative solver (not a
// predictIterative-specific weakness); fixing theta at a moderate, well-conditioned
// value isolates predictIterative's own correctness from that unrelated fit issue.
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

TEST_CASE("predictIterative mean/stdev match exact predict at a moderate theta", "[predictiterative][kriging]") {
  // n/n_n kept small on purpose: predictIterative's return_stdev path runs one CG
  // solve PER prediction point (O(n^2 * iters * n_n), see predictIterative's own
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
  auto [m_cg, s_cg] = k.predictIterative(Xt, true);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  // predictIterative solves the same exact system as predict() (same
  // objective/theta), via CG instead of a stored Cholesky factor -- at this
  // well-conditioned theta, default settings converge to ~1e-9.
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predictIterative: the dense fast path matches the matrix-free path", "[predictiterative][kriging]") {
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

  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "0", 1);  // force matrix-free
  auto [m_mf, s_mf] = k.predictIterative(Xt, true);

  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "4096", 1);  // allow dense
  auto [m_de, s_de] = k.predictIterative(Xt, true);

  // and the preconditioned solve too
  auto [m_mf_pc, s_mf_pc] = ([&] {
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "0", 1);
    return k.predictIterative(Xt, true, 0, 1e-8, true, 20);
  })();
  auto [m_de_pc, s_de_pc] = ([&] {
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "4096", 1);
    return k.predictIterative(Xt, true, 0, 1e-8, true, 20);
  })();

  if (old_env)
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", old_env, 1);
  else
    unsetenv_portable("LK_ITERATIVE_DENSE_MAX_MB");

  INFO("max |mean diff| = " << arma::abs(m_mf - m_de).max());
  INFO("max |stdev diff| = " << arma::abs(s_mf - s_de).max());
  CHECK(arma::abs(m_mf - m_de).max() < 1e-8 * arma::stddev(y));
  CHECK(arma::abs(s_mf - s_de).max() < 1e-8 * arma::stddev(y));
  CHECK(arma::abs(m_mf_pc - m_de_pc).max() < 1e-8 * arma::stddev(y));
  CHECK(arma::abs(s_mf_pc - s_de_pc).max() < 1e-8 * arma::stddev(y));
}

TEST_CASE("predictIterative defaults to mean only (stdev empty)", "[predictiterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(60, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt(10, 2, arma::fill::randu);
  auto [mean, stdev] = k.predictIterative(Xt);
  CHECK(mean.n_elem == 10);
  CHECK(stdev.n_elem == 0);
}

TEST_CASE("predictIterative interpolates the training data", "[predictiterative][kriging]") {
  // Small n on purpose -- see the sizing comment on the first predictIterative test
  // case above (predicts at X itself here, so n_n = n too).
  arma::mat X;
  arma::vec y;
  make_data(30, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  auto [mean, stdev] = k.predictIterative(X, true);
  CHECK(arma::abs(mean - y).max() < 0.05 * arma::stddev(y));
  CHECK(stdev.max() < 0.05 * arma::stddev(y));
}

TEST_CASE("predictIterative matches predict for a different kernel/trend", "[predictiterative][kriging]") {
  // Small n/n_n on purpose -- see the sizing comment on the first predictIterative
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
  // accuracy is covered separately by "predictIterative accuracy improves with a
  // larger iteration budget" below.
  auto [m_cg, s_cg] = k.predictIterative(Xt, true, /*max_iter=*/5000, /*tol=*/1e-12);

  INFO("max |mean diff| = " << arma::abs(m_cg - m_ex).max());
  INFO("max |stdev diff| = " << arma::abs(s_cg - s_ex).max());
  CHECK(arma::abs(m_cg - m_ex).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg - s_ex).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predictIterative rejects wrong dimension and Nugget models", "[predictiterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(50, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xbad(5, 3, arma::fill::randu);
  CHECK_THROWS_AS(k.predictIterative(Xbad), std::invalid_argument);

  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", {});
  arma::mat Xt(5, 2, arma::fill::randu);
  CHECK_THROWS_AS(knug.predictIterative(Xt), std::runtime_error);
}

TEST_CASE("predictIterative accuracy improves with a larger iteration budget", "[predictiterative][kriging]") {
  // Sanity check that the CG loop is doing meaningful work: an artificially
  // tiny iteration budget should be measurably less accurate than the
  // default (max_iter=0 => 2n). Small n/n_n on purpose -- this was the
  // slowest case (n=150, n_n=40 originally), the one that actually timed
  // out CI's memcheck job at 1500s under Valgrind; see the sizing comment
  // on the first predictIterative test case above.
  arma::mat X;
  arma::vec y;
  make_data(50, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_default, s_default] = k.predictIterative(Xt, true);
  auto [m_tiny, s_tiny] = k.predictIterative(Xt, true, 2, 1e-8);  // only 2 CG iterations

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

TEST_CASE("predictIterative with Nystrom preconditioning matches exact predict", "[predictiterative][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.1);

  arma::mat Xt;
  arma::vec yt;
  make_data(10, Xt, yt, 456);

  auto [m_ex, s_ex, c, dm, ds] = k.predict(Xt, true, false, false);
  auto [m_cg, s_cg] = k.predictIterative(Xt,
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

TEST_CASE("predictIterative Nystrom preconditioning converges faster on a tight iteration budget",
          "[predictiterative][kriging]") {
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
  auto [m_plain, s_plain] = k.predictIterative(Xt, false, tight_budget, 1e-12);
  auto [m_pc, s_pc]
      = k.predictIterative(Xt, false, tight_budget, 1e-12, /*use_nystrom_precond=*/true, /*precond_rank=*/30);

  const double err_plain = arma::abs(m_plain - m_ex).max();
  const double err_pc = arma::abs(m_pc - m_ex).max();
  INFO("err plain CG (budget=" << tight_budget << ") = " << err_plain
                               << ", err Nystrom-preconditioned CG = " << err_pc);
  CHECK(err_pc < err_plain);
}

// predictIterative_impl's mean solve (right-hand side m_y - m_F*m_beta) and
// the stdev branch's GLS-correction F-solve (right-hand side m_F) don't
// depend on X_n at all -- when the model was fit with LLIterative,
// Kriging::predictIterative now warm-starts both from
// m_iterative_RinvFY_cache (the [R^-1*F | R^-1*y] left behind by that fit),
// see docs/math/predictiterative_cg_warmstart.ipynb for the measured
// iteration savings. This must never change WHAT predictIterative converges
// to: compare against a model fit with the exact ("LL") objective at the
// SAME fixed theta, and exercise the cache across two separate calls (with
// different X_n) on the same LLIterative-fitted model.
TEST_CASE("predictIterative with an LLIterative fit matches exact predict (cache warm start)",
          "[predictiterative][kriging][iterative]") {
  arma::mat X;
  arma::vec y;
  make_data(40, X, y);

  Kriging::Parameters params;
  params.theta = arma::mat(1, X.n_cols, arma::fill::value(0.1));
  params.is_theta_estim = false;

  Kriging k_exact(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", params);
  Kriging k_iter(
      y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LLIterative(10,0,20,4,1e-10)", params);
  REQUIRE(k_iter.is_iterative_light());

  arma::mat Xt1, Xt2;
  arma::vec yt1, yt2;
  make_data(10, Xt1, yt1, 456);
  make_data(8, Xt2, yt2, 789);

  auto [m_ex1, s_ex1, c1, dm1, ds1] = k_exact.predict(Xt1, true, false, false);
  auto [m_cg1, s_cg1] = k_iter.predictIterative(Xt1, true, /*max_iter=*/0, /*tol=*/1e-10);
  INFO("call 1: max |mean diff| = " << arma::abs(m_cg1 - m_ex1).max()
                                    << ", max |stdev diff| = " << arma::abs(s_cg1 - s_ex1).max());
  CHECK(arma::abs(m_cg1 - m_ex1).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg1 - s_ex1).max() < 1e-5 * arma::stddev(y));

  // Second call, different X_n, same model/cache -- the cache is read-only
  // from predictIterative's point of view (only fit/update write it), so
  // this must be just as accurate as the first call.
  auto [m_ex2, s_ex2, c2, dm2, ds2] = k_exact.predict(Xt2, true, false, false);
  auto [m_cg2, s_cg2] = k_iter.predictIterative(Xt2, true, /*max_iter=*/0, /*tol=*/1e-10);
  INFO("call 2: max |mean diff| = " << arma::abs(m_cg2 - m_ex2).max()
                                    << ", max |stdev diff| = " << arma::abs(s_cg2 - s_ex2).max());
  CHECK(arma::abs(m_cg2 - m_ex2).max() < 1e-5 * arma::stddev(y));
  CHECK(arma::abs(s_cg2 - s_ex2).max() < 1e-5 * arma::stddev(y));
}

TEST_CASE("predictIterative benchmark", "[.benchmark]") {
  arma::mat X;
  arma::vec y;
  make_data(300, X, y);
  Kriging k = make_fixed_theta_model(y, X, "matern5_2", Trend::RegressionModel::Constant, 0.15);

  arma::mat Xt(100, 2, arma::fill::randu);

  BENCHMARK("Kriging::predict (exact, stored Cholesky) n=300") {
    return k.predict(Xt, true, false, false);
  };
  BENCHMARK("Kriging::predictIterative (matrix-free) n=300") {
    return k.predictIterative(Xt, true);
  };
}
