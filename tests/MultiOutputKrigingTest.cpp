// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/MultiOutputKriging.hpp"
// clang-format on

// Temporal-output toy code: x = (frequency, damping) in [0,1]^2 -> curve on t
static arma::rowvec curve(const arma::rowvec& x, const arma::vec& t) {
  const double f = 0.5 + 1.5 * x(0), a = 0.1 + 0.5 * x(1);
  return arma::trans(arma::exp(-a * t) % arma::cos(2 * arma::datum::pi * f * t / 5));
}

static arma::mat curves(const arma::mat& X, const arma::vec& t) {
  arma::mat Y(X.n_rows, t.n_elem);
  for (arma::uword i = 0; i < X.n_rows; ++i)
    Y.row(i) = curve(X.row(i), t);
  return Y;
}

// Exactly rank-2 outputs: Y = a1(x) phi1' + a2(x) phi2' (+ mean)
static arma::mat rank2(const arma::mat& X, arma::uword q) {
  const arma::vec s = arma::linspace(0, 1, q);
  const arma::vec phi1 = arma::sin(3 * s), phi2 = arma::cos(7 * s);
  arma::mat Y(X.n_rows, q);
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    const double a1 = std::sin(3 * X(i, 0)) + X(i, 1), a2 = std::cos(4 * X(i, 1)) * X(i, 0);
    Y.row(i) = arma::trans(1.0 + a1 * phi1 + a2 * phi2);
  }
  return Y;
}

// -----------------------------------------------------------------------------

TEST_CASE("MultiOutputKriging output model parsing", "[multioutput]") {
  CHECK(MultiOutputKriging("gauss", "pca").output_model_string() == "pca(0.99)");
  CHECK(MultiOutputKriging("gauss", "pca(5)").output_model_string() == "pca(5)");
  CHECK(MultiOutputKriging("gauss", "pca(0.9)").output_model_string() == "pca(0.9)");
  CHECK(MultiOutputKriging("gauss", "shared").output_model() == MultiOutputKriging::OutputModel::Shared);
  CHECK(MultiOutputKriging("gauss", "separable").output_model() == MultiOutputKriging::OutputModel::Separable);
  CHECK(MultiOutputKriging("gauss", "separable(matern5_2)").output_model_string() == "separable(matern5_2)");

  for (const char* bad : {"pca(0)", "pca(1.5)", "pca(-2)", "pca(x)", "pca(", "pca()", "foo", "shared(1)"})
    CHECK_THROWS_AS(MultiOutputKriging("gauss", bad), std::invalid_argument);

  arma::arma_rng::set_seed(1);
  arma::mat X(10, 2, arma::fill::randu), Y(10, 3, arma::fill::randu);
  MultiOutputKriging sep("gauss", "separable");
  CHECK_THROWS_AS(sep.fit(Y, X), std::runtime_error);  // not implemented yet
  MultiOutputKriging pc("gauss", "pca");
  CHECK_THROWS_AS(pc.predict(X), std::runtime_error);  // not fitted
  CHECK_THROWS_AS(pc.fit(Y, X.rows(0, 8)), std::invalid_argument);
}

TEST_CASE("MultiOutputKriging pca with q = 1 reduces to Kriging", "[multioutput]") {
  arma::arma_rng::set_seed(123);
  const arma::mat X(30, 2, arma::fill::randu);
  const arma::vec y = arma::sin(3 * X.col(0)) + arma::cos(5 * X.col(1));
  const arma::mat Xt(20, 2, arma::fill::randu);

  MultiOutputKriging mo(arma::mat(y), X, "matern5_2", "pca");
  Kriging kr(y, X, "matern5_2");
  REQUIRE(mo.nb_components() == 1);
  CHECK(arma::abs(mo.pca_residual()).max() < 1e-10);

  auto [m1, s1, c1, d1] = mo.predict(Xt, true, true, true);
  auto [m2, s2, c2, dm2, ds2] = kr.predict(Xt, true, true, true);
  CHECK(arma::abs(m1.col(0) - m2).max() < 1e-6);
  CHECK(arma::abs(s1.col(0) - s2).max() < 1e-6);
  CHECK(arma::abs(c1 - c2).max() < 1e-6);
  CHECK(arma::abs(d1.slice(0) - dm2).max() < 1e-5);

  auto [l1, ls1] = mo.leaveOneOutMat();
  auto [l2, ls2] = kr.leaveOneOutVec(kr.theta());
  CHECK(arma::abs(l1.col(0) - l2).max() < 1e-6);
  CHECK(arma::abs(ls1.col(0) - ls2).max() < 1e-6);
}

TEST_CASE("MultiOutputKriging pca recovers an exactly low-rank output", "[multioutput]") {
  arma::arma_rng::set_seed(42);
  const arma::mat X(40, 2, arma::fill::randu);
  const arma::mat Y = rank2(X, 50);

  for (bool normalize : {false, true}) {
    INFO("normalize = " << normalize);
    MultiOutputKriging mo(Y, X, "matern5_2", "pca(0.999)", Trend::RegressionModel::Constant, normalize);
    CHECK(mo.nb_components() == 2);
    CHECK(mo.pca_explained()(1) > 1 - 1e-9);
    CHECK(arma::abs(mo.pca_residual()).max() < 1e-8);
    CHECK(arma::norm(mo.pca_basis().t() * mo.pca_basis() - arma::eye(2, 2)) < 1e-10);

    // interpolation at design points
    auto [m, s, c, d] = mo.predict(X, true);
    CHECK(arma::abs(m - Y).max() < 1e-5);
    CHECK(s.max() < 1e-3);

    // pca(K) explicit, capped by the rank
    MultiOutputKriging mo5(Y, X, "matern5_2", "pca(5)", Trend::RegressionModel::Constant, normalize);
    CHECK(mo5.nb_components() == 2);
  }
}

TEST_CASE("MultiOutputKriging pca on temporal curves", "[multioutput]") {
  const arma::vec t = arma::linspace(0, 10, 200);
  arma::arma_rng::set_seed(0);
  const arma::mat X(40, 2, arma::fill::randu);
  const arma::mat Y = curves(X, t);
  const arma::mat Xt(15, 2, arma::fill::randu);
  const arma::mat Yt = curves(Xt, t);

  MultiOutputKriging mo(Y, X, "matern5_2", "pca(0.999)");
  INFO(mo.summary());
  CHECK(mo.nb_components() >= 2);
  CHECK(mo.nb_components() < 40);
  CHECK(mo.pca_explained()(mo.nb_components() - 1) >= 0.999);

  auto [m, s, c, d] = mo.predict(Xt, true, true, true);
  REQUIRE(m.n_rows == 15);
  REQUIRE(m.n_cols == 200);
  const double rmse = std::sqrt(arma::accu(arma::square(m - Yt)) / Yt.n_elem);
  CHECK(rmse < 0.1 * arma::stddev(arma::vectorise(Yt)));

  // stdev is consistent with the diagonal of the joint covariance (vec order)
  CHECK(arma::abs(arma::vectorise(s) - arma::sqrt(c.diag())).max() < 1e-8);
  CHECK(c.n_rows == 15 * 200);

  // standardized errors are not wildly off
  const arma::mat z = (m - Yt) / s;
  CHECK(arma::mean(arma::vectorise(arma::square(z))) < 10.0);

  // derivative vs finite differences of the mean
  const double h = 1e-5;
  for (arma::uword j = 0; j < 2; ++j) {
    arma::mat Xp = Xt, Xm = Xt;
    Xp.col(j) += h;
    Xm.col(j) -= h;
    const arma::mat fd = (std::get<0>(mo.predict(Xp, false)) - std::get<0>(mo.predict(Xm, false))) / (2 * h);
    arma::mat dj(15, 200);
    for (arma::uword o = 0; o < 200; ++o)
      dj.col(o) = d.slice(o).col(j);
    CHECK(arma::abs(dj - fd).max() < 1e-3 * std::max(1.0, arma::abs(fd).max()));
  }

  CHECK(std::isfinite(mo.leaveOneOut()));
  CHECK(mo.leaveOneOut() < arma::var(arma::vectorise(Y)));
}

TEST_CASE("MultiOutputKriging pca simulate", "[multioutput]") {
  const arma::vec t = arma::linspace(0, 10, 60);
  arma::arma_rng::set_seed(7);
  const arma::mat X(30, 2, arma::fill::randu);
  const arma::mat Y = curves(X, t);
  const arma::mat Xt(4, 2, arma::fill::randu);

  MultiOutputKriging mo(Y, X, "matern5_2", "pca(0.95)");  // keeps a non-zero truncation residual
  REQUIRE(arma::abs(mo.pca_residual()).max() > 0);

  const int nsim = 4000;
  const arma::cube S = mo.simulate(nsim, 123, Xt);
  REQUIRE(S.n_rows == 4);
  REQUIRE(S.n_cols == 60);
  REQUIRE(S.n_slices == static_cast<arma::uword>(nsim));

  // deterministic for a given seed
  CHECK(arma::approx_equal(S, mo.simulate(nsim, 123, Xt), "absdiff", 0.0));

  // empirical moments vs predict
  auto [m, s, c, d] = mo.predict(Xt, true);
  const arma::mat emean = arma::mean(S, 2);
  const arma::mat esd = arma::sqrt(arma::mean(arma::square(S.each_slice() - emean), 2));
  CHECK(arma::abs(emean - m).max() < 5 * s.max() / std::sqrt(nsim) + 1e-8);
  // (t = 0 is a constant output, s ~ 0 there: compare where variance is not negligible)
  const arma::uvec active = arma::find(s > 1e-3 * s.max());
  REQUIRE(active.n_elem > s.n_elem / 2);
  const arma::mat ratio = esd / s;
  CHECK(arma::abs(ratio.elem(active) - 1).max() < 0.1);
}

TEST_CASE("MultiOutputKriging pca update and update_simulate", "[multioutput]") {
  const arma::vec t = arma::linspace(0, 10, 40);
  arma::arma_rng::set_seed(11);
  const arma::mat X(25, 2, arma::fill::randu);
  const arma::mat Y = curves(X, t);
  const arma::mat Xu(3, 2, arma::fill::randu);
  const arma::mat Yu = curves(Xu, t);
  const arma::mat Xt(5, 2, arma::fill::randu);

  SECTION("update without refit keeps the basis and interpolates the projected new data") {
    MultiOutputKriging mo(Y, X, "matern5_2", "pca(0.999)");
    const arma::mat basis = mo.pca_basis();
    mo.update(Yu, Xu, false);
    CHECK(mo.X().n_rows == 28);
    CHECK(mo.pca_residual().n_rows == 28);
    CHECK(arma::approx_equal(mo.pca_basis(), basis, "absdiff", 0.0));
    auto [m, s, c, d] = mo.predict(Xu, false);
    CHECK(arma::abs(m - (Yu - mo.pca_residual().tail_rows(3))).max() < 1e-5);
  }

  SECTION("update with refit equals a fit on all data") {
    MultiOutputKriging mo(Y, X, "matern5_2", "pca(0.999)");
    mo.update(Yu, Xu, true);
    MultiOutputKriging all(arma::join_cols(Y, Yu), arma::join_cols(X, Xu), "matern5_2", "pca(0.999)");
    CHECK(mo.nb_components() == all.nb_components());
    CHECK(arma::abs(std::get<0>(mo.predict(Xt, false)) - std::get<0>(all.predict(Xt, false))).max() < 1e-8);
  }

  SECTION("update_simulate conditions the last trajectories") {
    // matern3_2: Kriging::update_simulate has a ~1e-3 absolute stdev floor that
    // dominates the tiny posterior stdev of smoother kernels (pre-existing)
    MultiOutputKriging mo(Y, X, "matern3_2", "pca(0.999)");
    CHECK_THROWS_AS(mo.update_simulate(Yu, Xu), std::runtime_error);
    const int nsim = 2000;
    const arma::cube S0 = mo.simulate(nsim, 5, Xt, true);
    const arma::cube S1 = mo.update_simulate(Yu, Xu);
    REQUIRE(S1.n_rows == 5);
    REQUIRE(S1.n_cols == 40);
    REQUIRE(S1.n_slices == static_cast<arma::uword>(nsim));

    // reference: model actually updated (no refit), then predicted
    MultiOutputKriging ref(Y, X, "matern3_2", "pca(0.999)");
    ref.update(Yu, Xu, false);
    auto [m, s, c, d] = ref.predict(Xt, true);
    const arma::mat emean = arma::mean(S1, 2);
    CHECK(arma::abs(emean - m).max() < 5 * s.max() / std::sqrt(nsim) + 1e-8);
  }
}

// =============================================================================
// "shared" output model
// =============================================================================

TEST_CASE("MultiOutputKriging shared with q = 1 reduces to Kriging", "[multioutput][shared]") {
  arma::arma_rng::set_seed(123);
  const arma::mat X(30, 2, arma::fill::randu);
  const arma::vec y = arma::sin(9 * X.col(0)) + arma::cos(11 * X.col(1)) + 3 * X.col(0);
  const arma::mat Xt(15, 2, arma::fill::randu);

  for (bool normalize : {false, true}) {
    CAPTURE(normalize);
    MultiOutputKriging mo(arma::mat(y), X, "matern5_2", "shared", Trend::RegressionModel::Constant, normalize);
    Kriging kr(y, X, "matern5_2", Trend::RegressionModel::Constant, normalize);

    // same starting point and optimizer: same optimum up to rounding
    CHECK(arma::abs(mo.theta() - kr.theta()).max() < 1e-6 * arma::abs(kr.theta()).max());
    CHECK(std::abs(mo.sigma2()(0) - kr.sigma2()) < 1e-6 * kr.sigma2());
    CHECK(arma::abs(mo.beta().col(0) - kr.beta()).max() < 1e-4 * std::max(1.0, arma::abs(kr.beta()).max()));
    CHECK(std::abs(mo.logLikelihood() - kr.logLikelihood()) < 1e-8 * std::abs(kr.logLikelihood()));

    auto [m1, s1, c1, d1] = mo.predict(Xt, true, true, true);
    auto [m2, s2, c2, dm2, ds2] = kr.predict(Xt, true, true, true);
    CHECK(arma::abs(m1.col(0) - m2).max() < 1e-6);
    CHECK(arma::abs(s1.col(0) - s2).max() < 1e-6);
    CHECK(arma::abs(c1 - c2).max() < 1e-6);
    CHECK(arma::abs(d1.slice(0) - dm2).max() < 1e-5);

    auto [l1, ls1] = mo.leaveOneOutMat();
    auto [l2, ls2] = kr.leaveOneOutVec(kr.theta());
    // Kriging::leaveOneOutVec is on the internal (normalized) scale
    CHECK(arma::abs(l1.col(0) - (l2 * kr.scaleY() + kr.centerY())).max() < 1e-6);
    CHECK(arma::abs(ls1.col(0) - ls2 * kr.scaleY()).max() < 1e-6);

    const arma::cube sim = mo.simulate(10, 42, Xt);
    const arma::mat simk = kr.simulate(10, 42, Xt);
    REQUIRE(sim.n_rows == Xt.n_rows);
    REQUIRE(sim.n_cols == 1);
    REQUIRE(sim.n_slices == 10);
    for (arma::uword s = 0; s < 10; ++s)
      CHECK(arma::abs(sim.slice(s).col(0) - simk.col(s)).max() < 1e-6);
  }
}

TEST_CASE("MultiOutputKriging shared at fixed theta equals one Kriging per output", "[multioutput][shared]") {
  arma::arma_rng::set_seed(7);
  const arma::mat X(25, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0, 10, 12);
  const arma::mat Y = curves(X, t);
  const arma::mat Xt(10, 2, arma::fill::randu);
  const arma::mat theta0 = {{0.4, 0.6}};

  MultiOutputKriging::Parameters mp;
  mp.theta = theta0;
  MultiOutputKriging mo(Y, X, "gauss", "shared", Trend::RegressionModel::Linear, false, "none", "LL", mp);
  CHECK(arma::abs(mo.theta() - theta0.t()).max() < 1e-14);

  Kriging::Parameters kp;
  kp.theta = theta0;
  auto [m1, s1, c1, d1] = mo.predict(Xt, true, true, true);
  REQUIRE(c1.n_rows == Xt.n_rows * Y.n_cols);
  double ll = 0;
  for (arma::uword j = 0; j < Y.n_cols; ++j) {
    CAPTURE(j);
    Kriging kr(Y.col(j), X, "gauss", Trend::RegressionModel::Linear, false, "none", "LL", kp);
    if (j > 0)  // output 0 (t = 0) is constant: infinite likelihood for Kriging
      ll += kr.logLikelihood();
    CHECK(std::abs(mo.sigma2()(j) - kr.sigma2()) <= 1e-8 * kr.sigma2() + 1e-20);
    CHECK(arma::abs(mo.beta().col(j) - kr.beta()).max() < 1e-8);
    auto [m2, s2, c2, dm2, ds2] = kr.predict(Xt, true, true, true);
    CHECK(arma::abs(m1.col(j) - m2).max() < 1e-8);
    CHECK(arma::abs(s1.col(j) - s2).max() < 1e-8);
    const arma::uword m = Xt.n_rows;
    CHECK(arma::abs(c1.submat(j * m, j * m, (j + 1) * m - 1, (j + 1) * m - 1) - c2).max() < 1e-8);
    CHECK(arma::abs(d1.slice(j) - dm2).max() < 1e-6);
  }
  CHECK(std::abs(mo.logLikelihood() - ll) < 1e-8 * std::abs(ll));
  // t = 0 is a constant output: left out of the likelihood, predicted exactly
  CHECK(mo.sigma2()(0) == 0);
  CHECK(arma::abs(m1.col(0) - 1.0).max() < 1e-10);
  CHECK(arma::abs(s1.col(0)).max() == 0);
  // outputs are independent given theta: off-diagonal blocks are zero
  CHECK(arma::abs(c1.submat(0, Xt.n_rows, Xt.n_rows - 1, 2 * Xt.n_rows - 1)).max() == 0);
}

TEST_CASE("MultiOutputKriging shared log-likelihood gradient and optimum", "[multioutput][shared]") {
  arma::arma_rng::set_seed(11);
  const arma::mat X(30, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0, 10, 20);
  const arma::mat Y = curves(X, t);

  MultiOutputKriging mo(Y, X, "matern5_2", "shared");
  const arma::vec th = {0.3, 0.7};
  auto [ll, g] = mo.logLikelihoodFun(th, true);
  for (arma::uword k = 0; k < 2; ++k) {
    const double h = 1e-6;
    arma::vec tp = th, tm = th;
    tp(k) += h;
    tm(k) -= h;
    const double fd = (std::get<0>(mo.logLikelihoodFun(tp)) - std::get<0>(mo.logLikelihoodFun(tm))) / (2 * h);
    CHECK(std::abs(g(k) - fd) < 1e-4 * std::max(1.0, std::abs(fd)));
  }

  // the fitted theta is a local maximum of the summed likelihood
  const double ll_hat = mo.logLikelihood();
  for (arma::uword k = 0; k < 2; ++k)
    for (double f : {0.9, 1.1}) {
      arma::vec tk = mo.theta();
      tk(k) *= f;
      CHECK(std::get<0>(mo.logLikelihoodFun(tk)) <= ll_hat + 1e-6 * std::abs(ll_hat));
    }

  // predictions are sensible on the temporal curves
  const arma::mat Xt(20, 2, arma::fill::randu);
  auto [mean, sd, cov, der] = mo.predict(Xt, true, true, true);
  const arma::mat Yt = curves(Xt, t);
  const double rmse = std::sqrt(arma::accu(arma::square(mean - Yt)) / Yt.n_elem);
  // one theta for all outputs: close to (not better than) one Kriging per output
  arma::mat mean_ind(Xt.n_rows, t.n_elem, arma::fill::value(1.0));  // output 0 is constant (= 1)
  for (arma::uword j = 1; j < t.n_elem; ++j)
    mean_ind.col(j) = std::get<0>(Kriging(Y.col(j), X, "matern5_2").predict(Xt, false, false, false));
  const double rmse_ind = std::sqrt(arma::accu(arma::square(mean_ind - Yt)) / Yt.n_elem);
  CHECK(rmse < 1.5 * rmse_ind);
  CHECK(rmse < 0.3 * arma::stddev(arma::vectorise(Yt)));
  CHECK(arma::abs(arma::vectorise(sd) - arma::sqrt(cov.diag())).max() < 1e-8);
  const arma::uvec act = arma::find(arma::vectorise(sd) > 0);
  const arma::vec z = arma::vectorise(mean - Yt).eval().elem(act) / arma::vectorise(sd).eval().elem(act);
  CHECK(arma::mean(arma::square(z)) < 10.0);

  // derivative vs finite differences of the mean
  const double h = 1e-5;
  for (arma::uword j = 0; j < 2; ++j) {
    arma::mat Xp = Xt, Xm = Xt;
    Xp.col(j) += h;
    Xm.col(j) -= h;
    const arma::mat fd = (std::get<0>(mo.predict(Xp, false)) - std::get<0>(mo.predict(Xm, false))) / (2 * h);
    arma::mat dj(Xt.n_rows, t.n_elem);
    for (arma::uword o = 0; o < t.n_elem; ++o)
      dj.col(o) = der.slice(o).col(j);
    CHECK(arma::abs(dj - fd).max() < 1e-3 * std::max(1.0, arma::abs(fd).max()));
  }
  CHECK(mo.leaveOneOut() < arma::var(arma::vectorise(Y)));
}

TEST_CASE("MultiOutputKriging shared update", "[multioutput][shared]") {
  arma::arma_rng::set_seed(5);
  const arma::mat X(20, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0, 10, 8);
  const arma::mat Y = curves(X, t);
  const arma::mat X_u(4, 2, arma::fill::randu);
  const arma::mat Y_u = curves(X_u, t);

  SECTION("update without refit keeps theta and interpolates the new data") {
    MultiOutputKriging mo(Y, X, "matern5_2", "shared");
    const arma::vec theta = mo.theta();
    mo.update(Y_u, X_u, false);
    CHECK(arma::abs(mo.theta() - theta).max() == 0);
    CHECK(mo.X().n_rows == 24);
    auto [mean, sd, cov, der] = mo.predict(X_u);
    CHECK(arma::abs(mean - Y_u).max() < 1e-6);
    CHECK(sd.max() < 1e-4);
  }
  SECTION("update with refit equals a fit on all data") {
    MultiOutputKriging a(Y, X, "matern5_2", "shared");
    a.update(Y_u, X_u, true);
    MultiOutputKriging b(arma::join_cols(Y, Y_u), arma::join_cols(X, X_u), "matern5_2", "shared");
    CHECK(arma::abs(a.theta() - b.theta()).max() < 1e-10);
    CHECK(arma::abs(a.sigma2() - b.sigma2()).max() < 1e-10);
  }
}

TEST_CASE("MultiOutputKriging shared restrictions", "[multioutput][shared]") {
  arma::arma_rng::set_seed(3);
  const arma::mat X(15, 2, arma::fill::randu);
  const arma::mat Y(15, 3, arma::fill::randu);
  MultiOutputKriging mo("gauss", "shared");
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LOO"), std::invalid_argument);
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "none"), std::invalid_argument);
  mo.fit(Y, X);
  CHECK(mo.sigma2().n_elem == 3);
  CHECK(mo.beta().n_cols == 3);
  CHECK_THROWS_AS(mo.simulate(5, 1, X, true), std::runtime_error);
  CHECK(mo.nb_components() == 0);

  MultiOutputKriging pc(Y, X, "gauss", "pca");
  CHECK_THROWS_AS(pc.theta(), std::runtime_error);
}
