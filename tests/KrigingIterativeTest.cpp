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
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,5)"));    // opt-in Nystrom CG precond
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,0)"));    // 0 = preconditioning off (== "LLIterative(8)")
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,5,30)"));  // + explicit SLQ Lanczos steps
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,0,40)"));  // Lanczos steps without a preconditioner
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,5,30,4)"));  // + explicit CG max_iter multiplier
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,0,40,1)"));  // multiplier without a preconditioner
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,0,40,2,1e-6)"));  // + explicit CG tolerance
  CHECK_NOTHROW(make_fixed_theta_iterative(y, X, "LLIterative(8,0,40,2,0.01)"));  // non-scientific notation too
  CHECK_NOTHROW(
      make_fixed_theta_iterative(y, X, "LLIterative(8,0,40,2,1e-4,1e-2)"));  // + separate probe-solve CG tolerance
  CHECK_NOTHROW(
      make_fixed_theta_iterative(y, X, "LLIterative(8,0,40,2,0.01,0.01)"));  // probes_cg_tol == cg_tol, explicit

  // malformed specs throw
  for (const std::string bad : {"LLIterative()",
                                "LLIterative(x)",
                                "LLIterative(0)",
                                "LLIterative(-3)",
                                "LLIterative(10",
                                "LLIterative(8x)",
                                "LLIterative(8,-2)",
                                "LLIterative(8,x)",
                                "LLIterative(8,5,1)",       // lanczos_steps must be >= 2
                                "LLIterative(8,5,x)",
                                "LLIterative(8,5,-4)",
                                "LLIterative(8,5,30,0)",    // cg_max_iter_mult must be >= 1
                                "LLIterative(8,5,30,-2)",
                                "LLIterative(8,5,30,x)",
                                "LLIterative(8,5,30,4,1)",       // cg_tol must be in (0,1)
                                "LLIterative(8,5,30,4,0)",
                                "LLIterative(8,5,30,4,-1e-4)",
                                "LLIterative(8,5,30,4,x)",
                                "LLIterative(8,5,30,4,1e-4,1)",     // probes_cg_tol must be in (0,1)
                                "LLIterative(8,5,30,4,1e-4,0)",
                                "LLIterative(8,5,30,4,1e-4,-1e-2)",
                                "LLIterative(8,5,30,4,1e-4,x)",
                                "LLIterative(8,5,30,4,1e-4,1e-2,7)"}) {  // at most 6 arguments
    CHECK_THROWS_AS(make_fixed_theta_iterative(y, X, bad), std::invalid_argument);
  }

  // LLIterative is not available with a nugget/noise channel
  Kriging knug("matern5_2", Kriging::NoiseModel::Nugget);
  CHECK_THROWS_AS(knug.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LLIterative(8)", {}),
                  std::invalid_argument);
}

TEST_CASE("LLIterative(m,precond_rank,lanczos_steps,cg_max_iter_mult,cg_tol): looser CG costs no accuracy", "[iterative][kriging]") {
  // The default CG tolerance is 1e-4, not 1e-8, because the log-determinant
  // it is paired with is a stochastic SLQ estimate whose bias dominates: the
  // linear solves are NOT the accuracy-limiting step, so tightening them
  // only buys iterations. Guard that property -- if a future change makes
  // the log-det exact enough that cg_tol starts to matter, this test is
  // where that shows up.
  arma::mat X;
  arma::vec y;
  make_data(120, X, y);
  const arma::vec theta{0.3, 0.3};

  const double ll_tight
      = std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24,0,60,2,1e-10)").logLikelihoodIterativeFun(theta, false));
  const double ll_default
      = std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24,0,60)").logLikelihoodIterativeFun(theta, false));

  INFO("ll(cg_tol=1e-10) = " << ll_tight << ", ll(default cg_tol) = " << ll_default);
  CHECK(std::abs(ll_default - ll_tight) <= 1e-3 * std::abs(ll_tight) + 1e-6);
}

TEST_CASE("LLIterative(...,cg_tol,probes_cg_tol): probes_cg_tol defaults to cg_tol and can be loosened on its own",
          "[iterative][kriging]") {
  // probes_cg_tol (6th field) only governs the gradient's Hutchinson-probe
  // CG solve (W = R^-1*probes); the [F|y] solve always uses cg_tol (5th
  // field). Two properties to guard:
  //  1. omitting the 6th field must behave EXACTLY as if probes_cg_tol ==
  //     cg_tol (backward compatibility with pre-6-field specs);
  //  2. loosening ONLY probes_cg_tol (keeping cg_tol tight) must cost no
  //     meaningful accuracy -- same "buys iterations, not accuracy"
  //     argument as cg_tol itself (docs/math/Iterative.md), since the probe
  //     solve only feeds a stochastic Hutchinson trace estimate.
  arma::mat X;
  arma::vec y;
  make_data(120, X, y);
  const arma::vec theta{0.3, 0.3};

  const auto [ll_omitted, g_omitted]
      = make_fixed_theta_iterative(y, X, "LLIterative(24,0,60,2,1e-6)").logLikelihoodIterativeFun(theta, true);
  const auto [ll_explicit_same, g_explicit_same]
      = make_fixed_theta_iterative(y, X, "LLIterative(24,0,60,2,1e-6,1e-6)").logLikelihoodIterativeFun(theta, true);

  INFO("ll(5-field, cg_tol=1e-6) = " << ll_omitted << ", ll(6-field, probes_cg_tol=cg_tol=1e-6) = " << ll_explicit_same);
  CHECK(ll_omitted == ll_explicit_same);  // same probe seed, same tol -> bit-identical CG trajectory
  CHECK(arma::approx_equal(g_omitted, g_explicit_same, "absdiff", 0.0));

  // probes_cg_tol only feeds W = R^-1*probes, used solely by the gradient's
  // Hutchinson trace term -- it does NOT touch the log-likelihood VALUE at
  // all (beta/sigma2/SSE/logdetR all come from the [F|y] solve and the SLQ
  // matvec, never from W). Guard that invariant, then check the gradient
  // itself degrades negligibly when only probes_cg_tol is loosened.
  const auto [ll_tight_probes, g_tight_probes] = make_fixed_theta_iterative(y, X, "LLIterative(24,0,60,2,1e-6,1e-6)")
                                                     .logLikelihoodIterativeFun(theta, true);
  const auto [ll_loose_probes, g_loose_probes] = make_fixed_theta_iterative(y, X, "LLIterative(24,0,60,2,1e-6,1e-2)")
                                                     .logLikelihoodIterativeFun(theta, true);

  INFO("ll(probes_cg_tol=1e-6) = " << ll_tight_probes << ", ll(probes_cg_tol=1e-2) = " << ll_loose_probes);
  CHECK(ll_tight_probes == ll_loose_probes);  // value is independent of probes_cg_tol by construction

  const double g_err = arma::norm(g_loose_probes - g_tight_probes);
  const double g_norm = arma::norm(g_tight_probes);
  INFO("grad(probes_cg_tol=1e-6) = " << g_tight_probes.t() << "grad(probes_cg_tol=1e-2) = " << g_loose_probes.t());
  CHECK(g_err <= 1e-3 * g_norm + 1e-6);
}

TEST_CASE("LLIterative(m,precond_rank,lanczos_steps): more SLQ Lanczos steps tighten the log-det estimate",
          "[iterative][kriging]") {
  // On an ill-conditioned R the default 20-step Lanczos quadrature
  // under-resolves the spectrum and the concentrated log-likelihood is
  // biased; raising lanczos_steps must move it toward the exact value
  // (Lanczos quadrature -> exact as steps -> n). Fixed probe seed, fixed
  // theta => fully deterministic.
  arma::mat X;
  arma::vec y;
  make_data(120, X, y);
  const arma::vec theta{0.3, 0.3};

  const double ll_exact
      = std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24)").logLikelihoodFun(theta, false, false));
  const double err_default = std::abs(
      std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24)").logLikelihoodIterativeFun(theta, false))
      - ll_exact);
  const double err_more = std::abs(
      std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24,0,120)").logLikelihoodIterativeFun(theta, false))
      - ll_exact);

  INFO("|ll_iter - ll_exact|: default 20 steps = " << err_default << ", 120 steps = " << err_more);
  CHECK(err_more <= err_default + 1e-6);  // more Lanczos steps are never meaningfully worse
  if (err_default > 0.02 * std::abs(ll_exact) + 1.0)
    CHECK(err_more < 0.75 * err_default);  // and materially better when the default is biased
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
  // The preconditioner changes TWO things, not one:
  //  1. HOW FAST CG's Krylov iteration converges to R^-1*[F|y|probes] --
  //     both runs converging (to the SAME solve) makes their CG-derived
  //     quantities (SSE/beta/sigma2, and hence the gradient's envelope-
  //     theorem term1) agree tightly, which is what the gradient CHECK
  //     below tests.
  //  2. WHICH stochastic estimator the log-determinant term uses:
  //     unpreconditioned SLQ(R) directly vs. logdetP_pc + SLQ(Rtilde) on the
  //     Nystrom-whitened operator. These are two DIFFERENT Monte Carlo
  //     estimators of the same true log|R| -- unbiased in expectation, but
  //     for one fixed probe realization at this test's small n=35/nprobe=30
  //     they need not (and empirically do not) land close together, even
  //     once every CG solve involved is fully converged (confirmed with
  //     LLIterative's cg_max_iter_mult field pushed to 64x: the ll_plain/
  //     ll_pc gap barely moves, so it is NOT a CG under-convergence
  //     artifact -- it is genuinely the SLQ-vs-whitened-SLQ estimator
  //     choice). So `ll` only agrees in an order-of-magnitude sense here,
  //     same as the SLQ/Hutchinson-vs-exact-LL comparisons elsewhere in
  //     this file -- it is the gradient (driven by the tightly-converged CG
  //     solves) that gets the tight check.
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

  Kriging k_plain = make_fixed_theta_iterative(y, X, "LLIterative(30,0,20,2,1e-10)");
  Kriging k_pc = make_fixed_theta_iterative(y, X, "LLIterative(30,15,20,2,1e-10)");

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
  // Order-of-magnitude only, per the comment above -- ll is dominated by
  // whichever SLQ estimator ran, not by the (tightly-converged either way)
  // CG solves.
  CHECK(std::abs(ll_plain - ll_pc) < 0.5 * (std::abs(ll_plain) + std::abs(ll_pc)) + 1.0);
  // Tight: both are the same envelope-theorem gradient built from CG-solved
  // quantities that converge to the SAME answer regardless of preconditioning.
  CHECK(arma::abs(grad_plain - grad_pc).max() < 0.02 * arma::abs(grad_plain).max() + 0.05);
}

TEST_CASE("LLIterative(m,precond_rank): the Nystrom preconditioner is applied to the SLQ log-determinant",
          "[iterative][kriging]") {
  // With objective "LLIterative(m,precond_rank)" the SLQ Lanczos runs on the
  // whitened Rtilde = L^-1 R L^-T (L L' = P = D + U U', LinearAlgebra::
  // WoodburyFactorization::whitenL/whitenLt), and log|P| is added back
  // exactly (LinearAlgebra::woodbury_logdet). The identity
  //   log|R| = log|P| + log|P^-1 R|
  // must hold, so the preconditioned concentrated log-likelihood tracks the
  // exact one at least as closely as the unpreconditioned SLQ estimate at
  // the same Lanczos-step budget -- it is never made worse by preconditioning
  // (and on a well-approximated R it is markedly better; see
  // docs/comparisons/libKriging_vs_GPyTorch.ipynb / bench/gpu for the
  // theta=0.15 sweep where LLIterative(30,50) beats LLIterative(30,0,40)).
  arma::mat X;
  arma::vec y;
  make_data(150, X, y);
  const double th = 0.2;
  const arma::vec theta{th, th};

  const double ll_exact
      = std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24)", th).logLikelihoodFun(theta, false, false));
  const double err_plain = std::abs(
      std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24,0,16)", th).logLikelihoodIterativeFun(theta, false))
      - ll_exact);
  const double err_pc = std::abs(
      std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(24,80,16)", th).logLikelihoodIterativeFun(theta, false))
      - ll_exact);

  INFO("|ll_iter - ll_exact|: unpreconditioned = " << err_plain << ", preconditioned = " << err_pc);
  CHECK(err_pc <= err_plain + 0.02 * std::abs(ll_exact) + 1.0);  // never materially worse
}

TEST_CASE("LLIterative: the dense fast path matches the matrix-free path", "[iterative][kriging]") {
  // For a separable kernel and small enough n, _logLikelihoodIterative
  // materializes R (and the dR/dtheta_k blocks) once and runs every matvec
  // as a BLAS-3 R*V. LK_ITERATIVE_DENSE_MAX_MB=0 forces the strictly
  // matrix-free per-pair path instead. Same probes, same theta => the two
  // must agree on both the concentrated log-likelihood and its gradient to
  // well within the SLQ/Hutchinson stochastic-estimator noise (the only
  // difference is BLAS vs pair-loop summation order).
  //
  // cg_tol is pinned at 1e-10 rather than left at the default so that the
  // comparison stays about summation order even if that default moves: an
  // under-converged solve stops at an iterate, not at a solution, and two
  // iterates sharing a residual norm need not share any digits beyond it.
  //
  // KNOWN FAILURE (pre-existing on this branch, reproducible at the branch
  // HEAD without any of the changes around it, and bit-identical at
  // cg_tol=1e-8 and 1e-10 -- so it is NOT a convergence artefact): on the
  // CPU backend ll_mf and ll_de differ by ~3e-5 relative and the gradients
  // by ~2.6e-2 relative, both well past these 1e-6/1e-5 bounds. Note the
  // test is vacuous when a GPU backend is live, since
  // LK_ITERATIVE_DENSE_MAX_MB only gates the CPU dense path (the CUDA one
  // has its own LK_ITERATIVE_CUDA_DENSE_MAX_MB), so both halves then run
  // the same device code and agree trivially -- which is why this went
  // unnoticed. Tracked in todo_reach_gpytorch.md.
  arma::mat X;
  arma::vec y;
  make_data(160, X, y);
  const arma::vec theta{0.25, 0.3};

  const char* old_env = std::getenv("LK_ITERATIVE_DENSE_MAX_MB");

  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "0", 1);  // force matrix-free
  auto [ll_mf, g_mf] = make_fixed_theta_iterative(y, X, "LLIterative(30,0,24,2,1e-10)").logLikelihoodIterativeFun(theta, true);

  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "4096", 1);  // allow dense
  auto [ll_de, g_de] = make_fixed_theta_iterative(y, X, "LLIterative(30,0,24,2,1e-10)").logLikelihoodIterativeFun(theta, true);

  if (old_env)
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", old_env, 1);
  else
    unsetenv_portable("LK_ITERATIVE_DENSE_MAX_MB");

  INFO("ll matrix-free = " << ll_mf << ", ll dense = " << ll_de);
  CHECK(std::abs(ll_mf - ll_de) < 1e-6 * std::abs(ll_mf) + 1e-6);
  CHECK(arma::abs(g_mf - g_de).max() < 1e-5 * arma::abs(g_mf).max() + 1e-5);

  // and the preconditioned path (whitened SLQ + separate probe solve) too
  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "0", 1);
  const double llp_mf
      = std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(30,40,24,2,1e-10)").logLikelihoodIterativeFun(theta, false));
  setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", "4096", 1);
  const double llp_de
      = std::get<0>(make_fixed_theta_iterative(y, X, "LLIterative(30,40,24,2,1e-10)").logLikelihoodIterativeFun(theta, false));
  if (old_env)
    setenv_portable("LK_ITERATIVE_DENSE_MAX_MB", old_env, 1);
  else
    unsetenv_portable("LK_ITERATIVE_DENSE_MAX_MB");
  CHECK(std::abs(llp_mf - llp_de) < 1e-6 * std::abs(llp_mf) + 1e-6);
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

// The final CG solve inside updateIterative is warm-started from the
// PREVIOUS commit's R^-1*[F|y] (see m_iterative_RinvFY_cache), zero-padded
// for the newly appended rows -- a pure iteration-count optimization
// (docs/math/updateiterative_cg_warmstart.ipynb measures 76% fewer
// iterations for a single-point append, ~32% cumulative over 30 sequential
// updates). It must never change WHAT update() converges to: several
// updates chained in a row (each one reusing the cache the previous one
// left behind) should land on the same beta/sigma2 -- to CG tolerance -- as
// a single fresh fit on the fully combined data.
TEST_CASE("LLIterative update() warm start does not change the converged answer", "[iterative][kriging]") {
  arma::mat X0, X1, X2, X3;
  arma::vec y0, y1, y2, y3;
  make_data(15, X0, y0, 111);
  make_data(4, X1, y1, 222);
  make_data(3, X2, y2, 333);
  make_data(5, X3, y3, 444);

  Kriging k_seq = make_fixed_theta_iterative(y0, X0, "LLIterative(10,0,20,4,1e-8)");
  k_seq.update(y1, X1, false);  // 1st update: cache is empty -> cold start
  k_seq.update(y2, X2, false);  // 2nd update: warm-started from the 1st update's cache
  k_seq.update(y3, X3, false);  // 3rd update: warm-started from the 2nd update's cache

  const arma::mat X_all = arma::join_cols(arma::join_cols(arma::join_cols(X0, X1), X2), X3);
  const arma::vec y_all = arma::join_cols(arma::join_cols(arma::join_cols(y0, y1), y2), y3);
  Kriging k_direct = make_fixed_theta_iterative(y_all, X_all, "LLIterative(10,0,20,4,1e-8)");

  CHECK(k_seq.X().n_rows == k_direct.X().n_rows);
  CHECK(arma::approx_equal(k_seq.beta(), k_direct.beta(), "absdiff", 1e-4));
  CHECK(std::abs(k_seq.sigma2() - k_direct.sigma2()) < 1e-4 * k_direct.sigma2());

  // predictIterative from the warm-started sequential model should also
  // agree with the directly-fitted one at a handful of query points.
  arma::mat Xt;
  arma::vec yt;
  make_data(5, Xt, yt, 555);
  auto [m_seq, s_seq] = k_seq.predictIterative(Xt, true);
  auto [m_direct, s_direct] = k_direct.predictIterative(Xt, true);
  CHECK(arma::approx_equal(m_seq, m_direct, "absdiff", 1e-3));
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
