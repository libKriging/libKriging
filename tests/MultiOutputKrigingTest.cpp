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
  MultiOutputKriging sh("gauss", "shared");
  CHECK_THROWS_AS(sh.fit(Y, X), std::runtime_error);
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
