// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/MultiOutputKriging.hpp"
// clang-format on

#include <cstdio>

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
  MultiOutputKriging sep("gauss", "separable(matern5_2)");
  CHECK_THROWS_AS(sep.fit(Y, X), std::invalid_argument);  // no output coordinates
  CHECK_THROWS(MultiOutputKriging("gauss", "separable(foo)"));
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
    MultiOutputKriging mo(Y, X, "matern5_2", "pca(0.999)");
    CHECK_THROWS_AS(mo.update_simulate(Yu, Xu), std::runtime_error);
    const int nsim = 2000;
    const arma::cube S0 = mo.simulate(nsim, 5, Xt, true);
    const arma::cube S1 = mo.update_simulate(Yu, Xu);
    REQUIRE(S1.n_rows == 5);
    REQUIRE(S1.n_cols == 40);
    REQUIRE(S1.n_slices == static_cast<arma::uword>(nsim));

    // reference: the same model actually updated (no refit), then predicted
    // (a second fit may land on another local optimum for some component)
    mo.update(Yu, Xu, false);
    auto [m, s, c, d] = mo.predict(Xt, true);
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
    CHECK(std::abs(mo.sigma2()(0) - kr.sigma2()) < 1e-4 * kr.sigma2());  // flat optimum
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
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LMP"), std::invalid_argument);
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "none"), std::invalid_argument);
  mo.fit(Y, X);
  CHECK(mo.sigma2().n_elem == 3);
  CHECK(mo.beta().n_cols == 3);
  CHECK_THROWS_AS(mo.update_simulate(Y, X), std::runtime_error);  // no simulate(..., true) yet
  CHECK(mo.nb_components() == 0);

  MultiOutputKriging pc(Y, X, "gauss", "pca");
  CHECK_THROWS_AS(pc.theta(), std::runtime_error);
}

TEST_CASE("MultiOutputKriging shared LOO objective", "[multioutput][shared]") {
  arma::arma_rng::set_seed(21);
  const arma::mat X(30, 2, arma::fill::randu);

  SECTION("q = 1 reduces to Kriging") {
    const arma::vec y = arma::sin(9 * X.col(0)) + arma::cos(11 * X.col(1)) + 3 * X.col(0);
    MultiOutputKriging mo(
        arma::mat(y), X, "matern5_2", "shared", Trend::RegressionModel::Constant, false, "BFGS", "LOO");
    Kriging kr(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LOO");
    CHECK(mo.objective() == "LOO");
    const arma::vec th = {0.3, 0.2};
    auto [l1, g1] = mo.leaveOneOutFun(th, true);
    auto [l2, g2] = kr.leaveOneOutFun(th, true, false);
    CHECK(std::abs(l1 - l2) < 1e-10 * l2);
    CHECK(arma::abs(g1 - g2).max() < 1e-8 * arma::abs(g2).max());
    CHECK(arma::abs(mo.theta() - kr.theta()).max() < 1e-4 * arma::abs(kr.theta()).max());
    CHECK(std::abs(mo.sigma2()(0) - kr.sigma2()) < 1e-3 * kr.sigma2());
  }

  SECTION("gradient and optimum on curves") {
    const arma::vec t = arma::linspace(0, 10, 15);
    const arma::mat Y = curves(X, t);
    MultiOutputKriging mo(Y, X, "matern5_2", "shared", Trend::RegressionModel::Constant, false, "BFGS", "LOO");
    const arma::vec th = {0.3, 0.7};
    auto [loo, g] = mo.leaveOneOutFun(th, true);
    for (arma::uword k = 0; k < 2; ++k) {
      const double h = 1e-6;
      arma::vec tp = th, tm = th;
      tp(k) += h;
      tm(k) -= h;
      const double fd = (std::get<0>(mo.leaveOneOutFun(tp)) - std::get<0>(mo.leaveOneOutFun(tm))) / (2 * h);
      CHECK(std::abs(g(k) - fd) < 1e-4 * std::max(std::abs(fd), 1e-3 * loo));
    }
    const double loo_hat = std::get<0>(mo.leaveOneOutFun(mo.theta()));
    for (arma::uword k = 0; k < 2; ++k)
      for (double f : {0.9, 1.1}) {
        arma::vec tk = mo.theta();
        tk(k) *= f;
        CHECK(std::get<0>(mo.leaveOneOutFun(tk)) >= loo_hat * (1 - 1e-6));
      }
    // the LOO fit has (summed) LOO error no worse than the LL fit
    MultiOutputKriging ml(Y, X, "matern5_2", "shared");
    CHECK(loo_hat <= std::get<0>(mo.leaveOneOutFun(ml.theta())) * (1 + 1e-6));
  }
}

TEST_CASE("MultiOutputKriging shared update_simulate", "[multioutput][shared]") {
  arma::arma_rng::set_seed(8);
  const arma::mat X(20, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0.5, 10, 6);
  const arma::mat Y = curves(X, t);
  const arma::mat Xt(8, 2, arma::fill::randu);
  const arma::mat X_u(3, 2, arma::fill::randu);
  const arma::mat Y_u = curves(X_u, t);

  MultiOutputKriging mo(Y, X, "matern5_2", "shared");
  const int nsim = 4000;
  const arma::cube sim = mo.simulate(nsim, 17, Xt, true);
  const arma::cube up = mo.update_simulate(Y_u, X_u);
  REQUIRE(up.n_rows == 8);
  REQUIRE(up.n_cols == 6);
  REQUIRE(up.n_slices == nsim);
  // the stored draws are not modified
  CHECK(arma::approx_equal(mo.simulate(nsim, 17, Xt, true), sim, "absdiff", 0));

  // reference: same theta, data appended; sigma2 is kept by update_simulate
  MultiOutputKriging ref(Y, X, "matern5_2", "shared");
  ref.update(Y_u, X_u, false);
  auto [m_ref, s_ref, c_ref, d_ref] = ref.predict(Xt);
  arma::mat s_exp = s_ref;
  s_exp.each_row() %= arma::trans(arma::sqrt(mo.sigma2() / ref.sigma2()));

  const arma::mat m_mc = arma::mean(up, 2);
  arma::mat s_mc(8, 6);
  for (arma::uword j = 0; j < 6; ++j) {
    arma::mat yj(8, nsim);
    for (int s = 0; s < nsim; ++s)
      yj.col(s) = up.slice(s).col(j);
    s_mc.col(j) = arma::stddev(yj, 0, 1);
  }
  const arma::uvec act = arma::find(s_exp > 1e-3 * s_exp.max());
  REQUIRE(act.n_elem > 0);
  CHECK(arma::abs(m_mc - m_ref).max() < 0.1 * s_exp.max());
  CHECK(arma::abs(s_mc.elem(act) / s_exp.elem(act) - 1).max() < 0.1);

  // conditioning on data at already simulated points pins the draws there
  const arma::cube pin = mo.update_simulate(curves(Xt.rows(0, 1), t), Xt.rows(0, 1));
  for (int s = 0; s < 10; ++s)
    CHECK(arma::abs(pin.slice(s).rows(0, 1) - curves(Xt.rows(0, 1), t)).max() < 1e-4);
}

// =============================================================================
// "separable" output model (ICM, free Σ)
// =============================================================================

// A few correlated, not linearly dependent outputs
static arma::mat few_outputs(const arma::mat& X) {
  arma::mat Y(X.n_rows, 4);
  Y.col(0) = arma::sin(6 * X.col(0)) + X.col(1);
  Y.col(1) = arma::sin(6 * X.col(0)) - 2 * arma::cos(5 * X.col(1));
  Y.col(2) = X.col(0) % X.col(1) + 0.5 * arma::cos(7 * X.col(1));
  Y.col(3) = 3 + arma::cos(4 * X.col(0) + 2 * X.col(1));
  return Y;
}

TEST_CASE("MultiOutputKriging separable with q = 1 reduces to Kriging", "[multioutput][separable]") {
  arma::arma_rng::set_seed(123);
  const arma::mat X(30, 2, arma::fill::randu);
  const arma::vec y = arma::sin(9 * X.col(0)) + arma::cos(11 * X.col(1)) + 3 * X.col(0);
  const arma::mat Xt(15, 2, arma::fill::randu);
  MultiOutputKriging mo(arma::mat(y), X, "matern5_2", "separable");
  Kriging kr(y, X, "matern5_2");
  CHECK(arma::abs(mo.theta() - kr.theta()).max() < 1e-6 * arma::abs(kr.theta()).max());
  CHECK(std::abs(mo.output_cov()(0, 0) - kr.sigma2()) < 1e-4 * kr.sigma2());  // flat optimum
  CHECK(std::abs(mo.logLikelihood() - kr.logLikelihood()) < 1e-8 * std::abs(kr.logLikelihood()));
  auto [m1, s1, c1, d1] = mo.predict(Xt, true, true, false);
  auto [m2, s2, c2, dm2, ds2] = kr.predict(Xt, true, true, false);
  CHECK(arma::abs(m1.col(0) - m2).max() < 1e-6);
  CHECK(arma::abs(s1.col(0) - s2).max() < 1e-6);
  CHECK(arma::abs(c1 - c2).max() < 1e-6);
}

TEST_CASE("MultiOutputKriging separable at fixed theta", "[multioutput][separable]") {
  arma::arma_rng::set_seed(31);
  const arma::mat X(25, 2, arma::fill::randu);
  const arma::mat Y = few_outputs(X);
  const arma::mat Xt(6, 2, arma::fill::randu);
  const arma::uword n = X.n_rows, q = Y.n_cols, m = Xt.n_rows;
  MultiOutputKriging::Parameters mp;
  mp.theta = arma::mat{{0.4, 0.5}};
  MultiOutputKriging sep(Y, X, "matern5_2", "separable", Trend::RegressionModel::Linear, false, "none", "LL", mp);
  MultiOutputKriging sh(Y, X, "matern5_2", "shared", Trend::RegressionModel::Linear, false, "none", "LL", mp);

  // closed-form profiled LL equals the dense Gaussian log-density of vec(Y)
  // under Σ̂ ⊗ R at the GLS trend
  {
    const arma::mat Xn = X.t();
    arma::mat R(n, n);
    for (arma::uword i = 0; i < n; ++i)
      for (arma::uword j = 0; j < n; ++j) {
        const arma::vec h = arma::abs(Xn.col(i) - Xn.col(j)) / arma::vec{0.4, 0.5};
        double r = 1;
        for (double hk : h)
          r *= (1 + std::sqrt(5.0) * hk + 5.0 / 3 * hk * hk) * std::exp(-std::sqrt(5.0) * hk);
        R(i, j) = r;
      }
    arma::mat F = arma::join_rows(arma::ones(n), X);
    const arma::mat Ri = arma::inv_sympd(R);
    const arma::mat Bh = arma::solve(F.t() * Ri * F, F.t() * Ri * Y);
    const arma::mat E = Y - F * Bh;
    const arma::mat Sh = E.t() * Ri * E / n;
    CHECK(arma::abs(Sh - sep.output_cov()).max() < 1e-8 * arma::abs(Sh).max());
    CHECK(arma::abs(Bh - sep.beta()).max() < 1e-8 * arma::abs(Bh).max());
    const arma::mat K = arma::kron(Sh, R);
    const arma::vec e = arma::vectorise(E);
    double ld, sg;
    arma::log_det(ld, sg, K);
    const double ll = -0.5 * (n * q * std::log(2 * arma::datum::pi) + ld + arma::as_scalar(e.t() * arma::solve(K, e)));
    CHECK(std::abs(sep.logLikelihood() - ll) < 1e-8 * std::abs(ll));
    // more parameters than "shared" at the same theta
    CHECK(sep.logLikelihood() >= sh.logLikelihood());
    CHECK(arma::abs(sep.output_cov().diag() - sh.sigma2()).max() < 1e-12 * sh.sigma2().max());
  }

  auto [m1, s1, c1, d1] = sep.predict(Xt, true, true, true);
  auto [m2, s2, c2, d2] = sh.predict(Xt, true, true, true);
  CHECK(arma::abs(m1 - m2).max() < 1e-12);  // autokrigeability
  CHECK(arma::abs(s1 - s2).max() < 1e-12);
  auto [Cx, S] = sep.predictCovFactors(Xt);
  CHECK(arma::abs(c1 - arma::kron(S, Cx)).max() < 1e-10);
  CHECK(arma::abs(arma::vectorise(s1) - arma::sqrt(c1.diag())).max() < 1e-8);
  CHECK(arma::abs(c1.submat(0, m, m - 1, 2 * m - 1)).max() > 0);  // outputs correlated
  CHECK(arma::abs(c1 - c1.t()).max() < 1e-12);

  // joint simulations reproduce the cross-output covariance at one point
  const arma::cube sim = sep.simulate(20000, 4, Xt.rows(0, 0));
  arma::mat Z(20000, q);
  for (arma::uword s = 0; s < 20000; ++s)
    Z.row(s) = sim.slice(s).row(0);
  const arma::mat Cemp = arma::cov(Z);
  const arma::mat Cth = S * Cx(0, 0);
  const arma::vec sd = arma::sqrt(Cth.diag());
  CHECK(arma::abs((Cemp - Cth) / (sd * sd.t())).max() < 0.05);
}

TEST_CASE("MultiOutputKriging separable fit, gradient and update_simulate", "[multioutput][separable]") {
  arma::arma_rng::set_seed(41);
  const arma::mat X(30, 2, arma::fill::randu);
  const arma::mat Y = few_outputs(X);
  MultiOutputKriging sep(Y, X, "matern5_2", "separable");
  INFO(sep.summary());

  const arma::vec th = {0.35, 0.6};
  auto [ll, g] = sep.logLikelihoodFun(th, true);
  for (arma::uword k = 0; k < 2; ++k) {
    const double h = 1e-5;  // smaller steps are dominated by rounding (|LL| ~ 1e2)
    arma::vec tp = th, tm = th;
    tp(k) += h;
    tm(k) -= h;
    const double fd = (std::get<0>(sep.logLikelihoodFun(tp)) - std::get<0>(sep.logLikelihoodFun(tm))) / (2 * h);
    CHECK(std::abs(g(k) - fd) < 1e-4 * std::max(1.0, std::abs(fd)));
  }
  const double ll_hat = sep.logLikelihood();
  for (arma::uword k = 0; k < 2; ++k)
    for (double f : {0.9, 1.1}) {
      arma::vec tk = sep.theta();
      tk(k) *= f;
      CHECK(std::get<0>(sep.logLikelihoodFun(tk)) <= ll_hat + 1e-6 * std::abs(ll_hat));
    }

  // update_simulate vs the model updated with the new data (theta kept)
  const arma::mat Xt(5, 2, arma::fill::randu);
  const arma::mat X_u(3, 2, arma::fill::randu);
  const arma::mat Y_u = few_outputs(X_u);
  const int nsim = 4000;
  sep.simulate(nsim, 2, Xt, true);
  const arma::cube up = sep.update_simulate(Y_u, X_u);
  MultiOutputKriging ref(Y, X, "matern5_2", "separable");
  ref.update(Y_u, X_u, false);
  auto [m_ref, s_ref, c_ref, d_ref] = ref.predict(Xt);
  arma::mat s_exp = s_ref;
  s_exp.each_row() %= arma::trans(arma::sqrt(sep.output_cov().diag() / ref.output_cov().diag()));
  const arma::mat m_mc = arma::mean(up, 2);
  arma::mat s_mc(5, 4);
  for (arma::uword j = 0; j < 4; ++j) {
    arma::mat yj(5, nsim);
    for (int s = 0; s < nsim; ++s)
      yj.col(s) = up.slice(s).col(j);
    s_mc.col(j) = arma::stddev(yj, 0, 1);
  }
  CHECK(arma::abs(m_mc - m_ref).max() < 0.1 * s_exp.max());
  CHECK(arma::abs(s_mc / s_exp - 1).max() < 0.1);

  // update without refit keeps theta, re-estimates Σ, interpolates
  sep.update(Y_u, X_u, false);
  CHECK(arma::abs(std::get<0>(sep.predict(X_u)) - Y_u).max() < 1e-6);
}

TEST_CASE("MultiOutputKriging separable restrictions", "[multioutput][separable]") {
  arma::arma_rng::set_seed(51);
  const arma::mat X(8, 2, arma::fill::randu);
  const arma::mat Y(8, 10, arma::fill::randu);
  MultiOutputKriging sep("matern5_2", "separable");
  CHECK_THROWS_AS(sep.fit(Y, X), std::invalid_argument);  // n - p = 7 < q = 10

  // duplicated output: singular Σ̂
  const arma::mat X2(20, 2, arma::fill::randu);
  arma::mat Y2 = few_outputs(X2);
  Y2 = arma::join_rows(Y2, 2 * Y2.col(0));
  CHECK_THROWS_AS(sep.fit(Y2, X2), std::runtime_error);

  // constant output: left out of Σ̂, predicted exactly
  arma::mat Y3 = few_outputs(X2);
  Y3.col(3).fill(2.5);
  sep.fit(Y3, X2);
  CHECK(sep.output_cov().row(3).max() == 0);
  CHECK(arma::abs(std::get<0>(sep.predict(X2.rows(0, 2))).col(3) - 2.5).max() < 1e-10);

  MultiOutputKriging pc(Y3, X2, "matern5_2", "pca");
  CHECK_THROWS_AS(pc.output_cov(), std::runtime_error);
}

// -----------------------------------------------------------------------------
// "separable(<kernel>)"
// -----------------------------------------------------------------------------

// matern 5/2 correlation, product over dimensions (rows of A, B are points)
static arma::mat matern52(const arma::mat& A, const arma::mat& B, const arma::vec& theta) {
  arma::mat R(A.n_rows, B.n_rows, arma::fill::ones);
  for (arma::uword i = 0; i < A.n_rows; ++i)
    for (arma::uword j = 0; j < B.n_rows; ++j)
      for (arma::uword k = 0; k < theta.n_elem; ++k) {
        const double h = std::sqrt(5.0) * std::abs(A(i, k) - B(j, k)) / theta(k);
        R(i, j) *= (1 + h + h * h / 3) * std::exp(-h);
      }
  return R;
}

TEST_CASE("MultiOutputKriging separable(kernel) likelihood equals the dense Gaussian density",
          "[multioutput][separable_kernel]") {
  arma::arma_rng::set_seed(61);
  const arma::mat X(8, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0.2, 10, 5);
  const arma::mat Y = curves(X, t);
  const arma::vec theta = {0.4, 0.7}, phi = {2.5};

  MultiOutputKriging mo("matern5_2", "separable(matern5_2)");
  mo.set_output_coordinates(t);
  MultiOutputKriging::Parameters p;
  p.theta = arma::mat(theta.t());
  p.output_theta = arma::mat(phi.t());
  p.is_theta_estim = false;
  mo.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", p);

  // dense: vec(Y - 1 b') ~ N(0, s2 R_t ⊗ R_x), b the per-output GLS, s2 its MLE
  const arma::uword n = X.n_rows, q = t.n_elem;
  const arma::mat Rx = matern52(X, X, theta), Rt = matern52(t, t, phi);
  const arma::mat Rxi = arma::inv_sympd(Rx);
  const arma::vec one(n, arma::fill::ones);
  const arma::rowvec b = (one.t() * Rxi * Y) / arma::as_scalar(one.t() * Rxi * one);
  const arma::mat E = Y - one * b;
  const arma::vec e = arma::vectorise(E);
  const arma::mat Om1 = arma::kron(Rt, Rx);
  const double s2 = arma::as_scalar(e.t() * arma::solve(Om1, e)) / (n * q);
  double logdet, sign;
  arma::log_det(logdet, sign, s2 * Om1);
  const double ll_dense = -0.5 * (n * q * std::log(2 * arma::datum::pi) + logdet + n * q);

  CHECK(std::abs(mo.logLikelihood() - ll_dense) < 1e-8 * std::abs(ll_dense));
  CHECK(arma::abs(mo.beta().row(0) - b).max() < 1e-10);
  CHECK(arma::abs(mo.output_cov() - s2 * Rt).max() < 1e-10 * s2);
  CHECK(arma::abs(mo.theta() - theta).max() == 0);
  CHECK(arma::abs(mo.output_theta() - phi).max() == 0);

  arma::vec tp = arma::join_cols(theta, phi);
  CHECK(std::abs(std::get<0>(mo.logLikelihoodFun(tp)) - ll_dense) < 1e-8 * std::abs(ll_dense));
  CHECK_THROWS_AS(mo.logLikelihoodFun(theta), std::invalid_argument);  // θ and φ expected

  // predictive covariance: kron(σ² R_t, C_x), C_x the kriging correlation
  const arma::mat Xn(3, 2, arma::fill::randu);
  auto [mean, sd, cov, dm] = mo.predict(Xn, true, true, false);
  auto [Cx, Sig] = mo.predictCovFactors(Xn);
  CHECK(arma::abs(cov - arma::kron(Sig, Cx)).max() < 1e-12);
  CHECK(arma::abs(Sig - s2 * Rt).max() < 1e-10 * s2);
}

TEST_CASE("MultiOutputKriging separable(kernel) fit with q > n", "[multioutput][separable_kernel]") {
  arma::arma_rng::set_seed(62);
  const arma::mat X(25, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0.2, 10, 40);  // q = 40 > n - p = 24
  const arma::mat Y = curves(X, t);
  MultiOutputKriging mo("matern5_2", "separable(matern5_2)");
  mo.set_output_coordinates(t);
  mo.fit(Y, X);
  INFO(mo.summary());
  REQUIRE(mo.output_theta().n_elem == 1);

  // gradient in (θ, φ)
  const arma::vec tp = {0.4, 0.8, 1.5};
  auto [ll, g] = mo.logLikelihoodFun(tp, true);
  REQUIRE(g.n_elem == 3);
  for (arma::uword k = 0; k < 3; ++k) {
    const double h = 1e-5;
    arma::vec a = tp, b = tp;
    a(k) += h;
    b(k) -= h;
    const double fd = (std::get<0>(mo.logLikelihoodFun(a)) - std::get<0>(mo.logLikelihoodFun(b))) / (2 * h);
    CHECK(std::abs(g(k) - fd) < 1e-4 * std::max(1.0, std::abs(fd)));
  }

  // optimum
  const arma::vec best = arma::join_cols(mo.theta(), mo.output_theta());
  const double ll_hat = mo.logLikelihood();
  CHECK(std::abs(std::get<0>(mo.logLikelihoodFun(best)) - ll_hat) < 1e-10 * std::abs(ll_hat));
  for (arma::uword k = 0; k < 3; ++k)
    for (double f : {0.9, 1.1}) {
      arma::vec tk = best;
      tk(k) *= f;
      CHECK(std::get<0>(mo.logLikelihoodFun(tk)) <= ll_hat + 1e-6 * std::abs(ll_hat));
    }

  // same mean as "shared" at the same θ, coherent covariance
  const arma::mat Xt(50, 2, arma::fill::randu);
  const arma::mat Yt = curves(Xt, t);
  auto [mean, sd, cov, dm] = mo.predict(Xt);
  const arma::vec z = arma::vectorise((mean - Yt) / sd);
  INFO("RMSE " << std::sqrt(arma::accu(arma::square(mean - Yt)) / Yt.n_elem) << ", sd(z) " << arma::stddev(z));
  CHECK(std::sqrt(arma::accu(arma::square(mean - Yt)) / Yt.n_elem) < 0.2 * arma::stddev(arma::vectorise(Yt)));
  CHECK(arma::stddev(z) < 2);
  CHECK(arma::stddev(z) > 0.2);  // conservative: one σ² for decaying curves
  MultiOutputKriging::Parameters p;
  p.theta = arma::mat(mo.theta().t());
  p.is_theta_estim = false;
  MultiOutputKriging sh(Y, X, "matern5_2", "shared", Trend::RegressionModel::Constant, false, "BFGS", "LL", p);
  CHECK(arma::abs(std::get<0>(sh.predict(Xt)) - mean).max() < 1e-10);

  // simulate / update_simulate / update
  const arma::cube sims = mo.simulate(10, 3, Xt.rows(0, 3), true);
  CHECK(sims.n_rows == 4);
  CHECK(sims.n_cols == 40);
  CHECK(sims.n_slices == 10);
  const arma::mat X_u(2, 2, arma::fill::randu);
  const arma::cube up = mo.update_simulate(curves(X_u, t), X_u);
  CHECK(up.n_slices == 10);
  mo.update(curves(X_u, t), X_u, false);
  CHECK(arma::abs(std::get<0>(mo.predict(X_u)) - curves(X_u, t)).max() < 1e-6);
  CHECK(arma::abs(mo.output_theta() - best.tail(1)).max() == 0);
}

TEST_CASE("MultiOutputKriging separable(kernel) restrictions", "[multioutput][separable_kernel]") {
  arma::arma_rng::set_seed(63);
  const arma::mat X(12, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0, 1, 6);
  const arma::mat Y = curves(X, t);
  MultiOutputKriging mo("matern5_2", "separable(gauss)");
  mo.set_output_coordinates(t);
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LOO"), std::invalid_argument);
  MultiOutputKriging::Parameters p;
  p.theta = arma::mat{{0.5, 0.5}};
  p.is_theta_estim = false;
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", p),
                  std::invalid_argument);  // output_theta missing
  p.output_theta = arma::mat{{0.5, 0.5}};
  CHECK_THROWS_AS(mo.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", p),
                  std::invalid_argument);  // d_t = 1
  mo.set_output_coordinates(arma::vec(6, arma::fill::ones));
  CHECK_THROWS_AS(mo.fit(Y, X), std::invalid_argument);  // constant coordinates

  MultiOutputKriging sh("matern5_2", "shared");
  p.output_theta = arma::mat(1, 1, arma::fill::value(0.5));
  CHECK_THROWS_AS(sh.fit(Y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", p), std::invalid_argument);
  sh.fit(Y, X);
  CHECK_THROWS_AS(sh.output_theta(), std::runtime_error);
}

// -----------------------------------------------------------------------------
// save / load
// -----------------------------------------------------------------------------

static void check_same(MultiOutputKriging& a, MultiOutputKriging& b, const arma::mat& Xn) {
  CHECK(a.output_model_string() == b.output_model_string());
  CHECK(a.summary() == b.summary());
  auto [ma, sa, ca, da] = a.predict(Xn, true, true, true);
  auto [mb, sb, cb, db] = b.predict(Xn, true, true, true);
  CHECK(arma::abs(ma - mb).max() <= 1e-12 * std::max(1.0, arma::abs(ma).max()));
  CHECK(arma::abs(sa - sb).max() <= 1e-12 * std::max(1.0, sa.max()));
  CHECK(arma::abs(ca - cb).max() <= 1e-12 * std::max(1.0, arma::abs(ca).max()));
  CHECK(arma::abs(da - db).max() <= 1e-10 * std::max(1.0, arma::abs(da).max()));
  CHECK(arma::abs(a.simulate(5, 9, Xn) - b.simulate(5, 9, Xn)).max() <= 1e-10);
}

TEST_CASE("MultiOutputKriging save and load", "[multioutput][save]") {
  arma::arma_rng::set_seed(71);
  const arma::mat X(25, 2, arma::fill::randu);
  const arma::vec t = arma::linspace(0.2, 10, 20);
  const arma::mat Xn(4, 2, arma::fill::randu);
  const arma::mat X_u(3, 2, arma::fill::randu);
  const std::string file = "MultiOutputKrigingTest_save.json";

  for (const std::string model : {"pca(0.999)", "shared", "separable", "separable(matern5_2)"}) {
    for (bool normalize : {false, true}) {
      INFO(model << (normalize ? ", normalize" : ""));
      const bool few = model == "separable";
      const arma::mat Y = few ? few_outputs(X) : curves(X, t);
      const arma::mat Y_u = few ? few_outputs(X_u) : curves(X_u, t);
      MultiOutputKriging mo("matern5_2", model);
      if (!few)
        mo.set_output_coordinates(t);
      mo.fit(Y, X, Trend::RegressionModel::Linear, normalize);
      mo.save(file);
      MultiOutputKriging lo = MultiOutputKriging::load(file);
      CHECK(arma::abs(lo.Y() - mo.Y()).max() == 0);
      CHECK(arma::approx_equal(lo.output_coordinates(), mo.output_coordinates(), "absdiff", 0));
      CHECK(lo.regmodel() == mo.regmodel());
      CHECK(lo.normalize() == normalize);
      check_same(mo, lo, Xn);

      // update without refit, then save / load again (basis and θ kept)
      mo.update(Y_u, X_u, false);
      lo.update(Y_u, X_u, false);
      check_same(mo, lo, Xn);
      mo.save(file);
      MultiOutputKriging lo2 = MultiOutputKriging::load(file);
      check_same(mo, lo2, Xn);

      // refit uses the saved options
      mo.update(Y_u, X_u, true);
      lo2.update(Y_u, X_u, true);
      check_same(mo, lo2, Xn);
    }
  }

  MultiOutputKriging empty("gauss", "pca(3)");
  empty.save(file);
  MultiOutputKriging le = MultiOutputKriging::load(file);
  CHECK(le.output_model_string() == "pca(3)");
  CHECK_THROWS_AS(le.predict(Xn), std::runtime_error);

  Kriging k(arma::vec(X.col(0)), X, "gauss");
  k.save(file);
  CHECK_THROWS_AS(MultiOutputKriging::load(file), std::runtime_error);
  std::remove(file.c_str());
}
