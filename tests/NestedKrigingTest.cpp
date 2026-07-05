// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/NestedKriging.hpp"
// clang-format on

static double f2d(double x1, double x2) {
  return std::sin(3.0 * x1) + std::cos(5.0 * x2) + x1 * x2;
}

static void make_data(arma::uword n, arma::uword d, arma::mat& X, arma::vec& y, unsigned seed = 123) {
  arma::arma_rng::set_seed(seed);
  X = arma::mat(n, d, arma::fill::randu);
  y = arma::vec(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f2d(X(i, 0), X(i, d > 1 ? 1 : 0));
}

// -----------------------------------------------------------------------------

TEST_CASE("NestedKriging with one group reduces to plain Kriging", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(40, 2, X, y);

  NestedKriging nk(y, X, "gauss", /*nb_groups=*/1, NestedKriging::Aggregation::NK);
  const Kriging& sub = nk.submodel(0);  // p=1: submodel *is* the full model (common prior = its own fit)

  arma::mat Xt(30, 2, arma::fill::randu);
  auto [m_nk, s_nk] = nk.predict(Xt, true);
  auto [m_kr, s_kr, cov, dm, ds] = const_cast<Kriging&>(sub).predict(Xt, true, false, false);

  // means must coincide (NK aggregation of a single expert is that expert)
  CHECK(arma::abs(m_nk - m_kr).max() < 1e-6);
  // stdev: NK uses the simple-kriging variance (beta fixed) => equal up to
  // the trend-uncertainty term, which is zero here since beta is fixed
  CHECK(arma::abs(s_nk - s_kr).max() < 1e-2);

  // PoE family with a single expert must also reduce to the expert
  for (auto agg : {NestedKriging::Aggregation::PoE,
                   NestedKriging::Aggregation::gPoE,
                   NestedKriging::Aggregation::BCM,
                   NestedKriging::Aggregation::rBCM}) {
    NestedKriging nk1(y, X, "gauss", 1, agg);
    auto [m1, s1] = nk1.predict(Xt, true);
    auto [mk, sk, c1, d1, e1] = const_cast<Kriging&>(nk1.submodel(0)).predict(Xt, true, false, false);
    INFO("aggregation = " << NestedKriging::aggregationToString(agg));
    CHECK(arma::abs(m1 - mk).max() < 1e-8);
    CHECK(arma::abs(s1 - sk).max() < 1e-6);
  }
}

TEST_CASE("NestedKriging/NK interpolates the design points", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(100, 2, X, y);

  NestedKriging nk(y, X, "matern5_2", /*nb_groups=*/4, NestedKriging::Aggregation::NK);
  auto [mean, stdev] = nk.predict(X, true);

  // the optimal aggregation is an interpolant (Rullière et al. 2018, Prop. 3)
  CHECK(arma::abs(mean - y).max() < 1e-3);
  CHECK(stdev.max() < 1e-2);
}

TEST_CASE("NestedKriging shares a common prior across submodels", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(120, 2, X, y);

  NestedKriging nk(y, X, "gauss", 4, NestedKriging::Aggregation::NK);
  for (arma::uword g = 0; g < nk.nb_groups(); ++g) {
    CHECK(arma::abs(nk.submodel(g).theta() - nk.theta()).max() < 1e-12);
    CHECK(std::abs(nk.submodel(g).sigma2() - nk.sigma2()) < 1e-12);
    CHECK(std::abs(nk.submodel(g).beta()(0) - nk.beta0()) < 1e-12);
  }
  // groups form a partition of 0..n-1
  arma::uvec all;
  for (arma::uword g = 0; g < nk.nb_groups(); ++g)
    all = arma::join_cols(all, nk.groups()[g]);
  all = arma::sort(all);
  CHECK(all.n_elem == X.n_rows);
  CHECK(all(0) == 0);
  CHECK(all(all.n_elem - 1) == X.n_rows - 1);
}

TEST_CASE("NestedKriging is close to full Kriging on moderate n", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(400, 2, X, y);

  arma::mat Xt;
  arma::vec yt;
  make_data(200, 2, Xt, yt, 456);

  Kriging full(y, X, "matern5_2");
  auto [m_full, s_full, c, d, e] = full.predict(Xt, true, false, false);
  const double rmse_full = std::sqrt(arma::mean(arma::square(m_full - yt)));

  const double sd_y = arma::stddev(y);

  for (auto agg : {NestedKriging::Aggregation::NK, NestedKriging::Aggregation::rBCM}) {
    NestedKriging nk(y, X, "matern5_2", /*nb_groups=*/8, agg);
    auto [m_nk, s_nk] = nk.predict(Xt, true);
    const double rmse_nk = std::sqrt(arma::mean(arma::square(m_nk - yt)));

    INFO("aggregation = " << NestedKriging::aggregationToString(agg));
    INFO("rmse full = " << rmse_full << ", rmse nested = " << rmse_nk << ", sd(y) = " << sd_y);
    // nested predictor should stay accurate...
    CHECK(rmse_nk < 0.05 * sd_y + 2.0 * rmse_full);
    // ...and consistent with the full model
    CHECK(arma::mean(arma::abs(m_nk - m_full)) < 0.05 * sd_y);
    // valid uncertainties
    CHECK(s_nk.is_finite());
    CHECK(s_nk.min() >= 0.0);
  }
}

TEST_CASE("NestedKriging aggregation variances are sane", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(200, 2, X, y);

  arma::mat Xt(50, 2, arma::fill::randu);

  // PoE is known to be overconfident: its variance is below each expert's
  NestedKriging poe(y, X, "gauss", 4, NestedKriging::Aggregation::PoE);
  auto [m_poe, s_poe] = poe.predict(Xt, true);
  for (arma::uword g = 0; g < poe.nb_groups(); ++g) {
    auto [mg, sg, c, d, e] = const_cast<Kriging&>(poe.submodel(g)).predict(Xt, true, false, false);
    CHECK(arma::all(s_poe <= sg + 1e-10));
  }

  // NK variance is bounded by the prior variance
  NestedKriging nk(y, X, "gauss", 4, NestedKriging::Aggregation::NK);
  auto [m_nk, s_nk] = nk.predict(Xt, true);
  CHECK(arma::all(arma::square(s_nk) <= nk.sigma2() * (1.0 + 1e-6)));

  // far from the data, NK variance should revert towards the prior
  arma::mat Xfar(5, 2);
  Xfar.fill(10.0);
  auto [m_far, s_far] = nk.predict(Xfar, true);
  CHECK(arma::square(s_far).min() > 0.5 * nk.sigma2());
  CHECK(arma::abs(m_far - nk.beta0()).max() < 1e-3 * std::max(1.0, std::abs(nk.beta0())) + 1e-6);
}

TEST_CASE("NestedKriging is reproducible given a seed", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(150, 2, X, y);

  arma::mat Xt(20, 2, arma::fill::randu);
  arma::mat Xt_copy = Xt;

  NestedKriging nk1(y, X, "gauss", 5, NestedKriging::Aggregation::NK, NestedKriging::Partition::Random, 42);
  NestedKriging nk2(y, X, "gauss", 5, NestedKriging::Aggregation::NK, NestedKriging::Partition::Random, 42);
  auto [m1, s1] = nk1.predict(Xt, true);
  auto [m2, s2] = nk2.predict(Xt_copy, true);
  CHECK(arma::abs(m1 - m2).max() == 0.0);
  CHECK(arma::abs(s1 - s2).max() == 0.0);
}

TEST_CASE("NestedKriging input validation", "[nested][kriging]") {
  arma::mat X;
  arma::vec y;
  make_data(50, 2, X, y);

  // NK + non-constant trend is not supported (v1)
  CHECK_THROWS_AS(NestedKriging(y,
                                X,
                                "gauss",
                                2,
                                NestedKriging::Aggregation::NK,
                                NestedKriging::Partition::KMeans,
                                123,
                                Trend::RegressionModel::Linear),
                  std::invalid_argument);
  // too many groups
  CHECK_THROWS_AS(NestedKriging(y, X, "gauss", 30), std::invalid_argument);
  // inconsistent dimensions
  NestedKriging nk(y, X, "gauss", 2);
  arma::mat Xbad(10, 3, arma::fill::randu);
  CHECK_THROWS_AS(nk.predict(Xbad, true), std::invalid_argument);
  // unknown aggregation name
  CHECK_THROWS_AS(NestedKriging::aggregationFromString("median"), std::invalid_argument);
}

TEST_CASE("NestedKriging large-n smoke test", "[nested][kriging][intensive]") {
  const arma::uword n = 4000;
  const arma::uword d = 3;
  arma::arma_rng::set_seed(123);
  arma::mat X(n, d, arma::fill::randu);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = std::sin(3.0 * X(i, 0)) + std::cos(5.0 * X(i, 1)) + X(i, 2) * X(i, 2);

  // fixed theta => optim="none": submodel fits are closed-form (fast smoke test)
  Kriging::Parameters params;
  params.theta = arma::mat(1, d);
  params.theta->fill(0.3);
  params.is_theta_estim = false;

  NestedKriging nk("matern5_2");
  nk.fit(y, X, /*nb_groups=*/20, Trend::RegressionModel::Constant, "none", "LL", params);

  arma::mat Xt(100, d, arma::fill::randu);
  arma::vec yt(100);
  for (arma::uword i = 0; i < 100; ++i)
    yt(i) = std::sin(3.0 * Xt(i, 0)) + std::cos(5.0 * Xt(i, 1)) + Xt(i, 2) * Xt(i, 2);

  auto [mean, stdev] = nk.predict(Xt, true);
  CHECK(mean.is_finite());
  CHECK(stdev.is_finite());
  const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
  INFO("rmse = " << rmse << " vs sd(y) = " << arma::stddev(y));
  CHECK(rmse < 0.1 * arma::stddev(y));
}

#include "libKriging/WarpKriging.hpp"

TEST_CASE("NestedKriging supports WarpKriging submodels", "[nested][warp]") {
  arma::mat X;
  arma::vec y;
  make_data(120, 2, X, y);
  const std::vector<std::string> warping{"kumaraswamy", "kumaraswamy"};

  NestedKriging nk(y,
                   X,
                   "gauss",
                   /*nb_groups=*/3,
                   NestedKriging::Aggregation::NK,
                   NestedKriging::Partition::KMeans,
                   123,
                   Trend::RegressionModel::Constant,
                   "BFGS",
                   "LL",
                   {},
                   warping);
  CHECK(nk.warped());
  CHECK(nk.warping() == warping);
  CHECK_THROWS_AS(nk.submodel(0), std::runtime_error);  // warped: use wsubmodel

  SECTION("all submodels share the common warped prior") {
    for (arma::uword g = 0; g < nk.nb_groups(); ++g) {
      CHECK(arma::abs(nk.wsubmodel(g).theta() - nk.theta()).max() < 1e-12);
      CHECK(arma::abs(nk.wsubmodel(g).warp_params() - nk.wsubmodel(0).warp_params()).max() < 1e-12);
    }
  }

  SECTION("NK interpolates the design points under warping") {
    auto [mean, stdev] = nk.predict(X, true);
    CHECK(arma::abs(mean - y).max() < 1e-3);
    CHECK(stdev.max() < 1e-2);
  }

  SECTION("prediction is finite and accurate") {
    arma::mat Xt;
    arma::vec yt;
    make_data(80, 2, Xt, yt, 456);
    auto [mean, stdev] = nk.predict(Xt, true);
    CHECK(mean.is_finite());
    CHECK(stdev.is_finite());
    CHECK(stdev.min() >= 0.0);
    const double rmse = std::sqrt(arma::mean(arma::square(mean - yt)));
    CHECK(rmse < 0.5 * arma::stddev(y));
  }

  SECTION("PoE family also works with warped submodels") {
    arma::mat Xt(30, 2, arma::fill::randu);
    for (auto agg : {NestedKriging::Aggregation::gPoE, NestedKriging::Aggregation::rBCM}) {
      NestedKriging nkp(y,
                        X,
                        "gauss",
                        3,
                        agg,
                        NestedKriging::Partition::KMeans,
                        123,
                        Trend::RegressionModel::Constant,
                        "BFGS",
                        "LL",
                        {},
                        warping);
      auto [m, s] = nkp.predict(Xt, true);
      CHECK(m.is_finite());
      CHECK(s.min() >= 0.0);
    }
  }
}

TEST_CASE("NestedKriging warped with one group reduces to full WarpKriging", "[nested][warp]") {
  arma::mat X;
  arma::vec y;
  make_data(60, 2, X, y);
  const std::vector<std::string> warping{"kumaraswamy", "kumaraswamy"};

  NestedKriging nk(y,
                   X,
                   "gauss",
                   /*nb_groups=*/1,
                   NestedKriging::Aggregation::NK,
                   NestedKriging::Partition::KMeans,
                   123,
                   Trend::RegressionModel::Constant,
                   "BFGS",
                   "LL",
                   {},
                   warping);

  arma::mat Xt(30, 2, arma::fill::randu);
  auto [m_nk, s_nk] = nk.predict(Xt, true);
  auto [m_wk, s_wk, cov, dm, ds] = nk.wsubmodel(0).predict(Xt, true, false, false);

  CHECK(arma::abs(m_nk - m_wk).max() < 1e-6);
  CHECK(arma::abs(s_nk - s_wk).max() < 1e-2);

  // covMat consistency: corr(x,x) == 1 => covMat diag == sigma2
  arma::mat C = nk.wsubmodel(0).covMat(Xt, Xt);
  CHECK(arma::abs(C.diag() / nk.wsubmodel(0).sigma2() - 1.0).max() < 1e-10);
}

TEST_CASE("NestedKriging warped hyperparameters come from a subsample fit", "[nested][warp]") {
  arma::mat X;
  arma::vec y;
  make_data(150, 2, X, y);
  const std::vector<std::string> warping{"kumaraswamy", "kumaraswamy"};

  NestedKriging nk("gauss");
  nk.set_warp_subsample(60);  // reference fit on 60 of 150 points
  CHECK(nk.warp_subsample() == 60);
  nk.fit(y, X, /*nb_groups=*/3, Trend::RegressionModel::Constant, "BFGS", "LL", {}, warping);

  // all submodels carry the same seeded (theta, warp) prior
  for (arma::uword g = 0; g < nk.nb_groups(); ++g) {
    CHECK(arma::abs(nk.wsubmodel(g).theta() - nk.theta()).max() < 1e-12);
    CHECK(arma::abs(nk.wsubmodel(g).warp_params() - nk.wsubmodel(0).warp_params()).max() < 1e-12);
  }

  // NK interpolation is a property of the aggregation, not of the
  // hyperparameters: it must hold even with subsampled hyperparameters
  auto [mean, stdev] = nk.predict(X, true);
  CHECK(arma::abs(mean - y).max() < 1e-3);
  CHECK(stdev.max() < 1e-2);

  // subsample larger than n must degrade gracefully to a full-data fit
  NestedKriging nk_full("gauss");
  nk_full.set_warp_subsample(100000);
  nk_full.fit(y, X, 3, Trend::RegressionModel::Constant, "BFGS", "LL", {}, warping);
  auto [m2, s2] = nk_full.predict(X, true);
  CHECK(arma::abs(m2 - y).max() < 1e-3);
}
