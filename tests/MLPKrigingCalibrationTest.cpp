// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/MLPKriging.hpp"
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
// clang-format on

/**
 * @file MLPKrigingCalibrationTest.cpp
 * @brief Regression test for #407: the predictive stdev of MLPKriging must be
 *        calibrated on held-out points, not only its mean.
 *
 * f(x) = Branin(W x) on [0,1]^d, W a fixed 2 x d projection: a joint MLP can
 * learn the 2D structure (good Q2) but, with more weights than observations,
 * maximising the likelihood also learns features in which the training
 * outputs are almost exactly smooth, with a collapsed predictive variance
 * (95% intervals covering ~1% of test points). The K-fold calibration of the
 * predictive variance must bring the coverage and the NLPD back to sensible
 * values.
 */

using namespace libKriging;

namespace {

constexpr arma::uword kDim = 5;

arma::vec branin_proj(const arma::mat& X) {
  // Fixed projection, rows normalised to |w|_1 = 1, then mapped to [0, 1].
  arma::mat W = {{0.9, -0.4, 0.7, 0.2, -0.5}, {-0.3, 0.8, 0.1, -0.6, 0.4}};
  W.each_col() /= arma::sum(arma::abs(W), 1);
  arma::mat U = X * W.t();
  U.each_row() -= arma::sum(arma::clamp(W, -arma::datum::inf, 0.0), 1).t();
  arma::vec x1 = 15.0 * U.col(0) - 5.0, x2 = 15.0 * U.col(1);
  const double pi = arma::datum::pi;
  return arma::square(x2 - 5.1 / (4.0 * pi * pi) * arma::square(x1) + 5.0 / pi * x1 - 6.0)
         + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * arma::cos(x1) + 10.0;
}

arma::mat lhs(arma::uword n, arma::uword d) {
  arma::mat X(n, d);
  for (arma::uword j = 0; j < d; ++j)
    X.col(j) = (arma::conv_to<arma::vec>::from(arma::randperm(n)) + arma::randu<arma::vec>(n)) / n;
  return X;
}

struct Scores {
  double q2, nlpd, coverage;
};

template <typename Model>
Scores score(Model& model, const arma::mat& Xt, const arma::vec& yt) {
  auto [m, s, _c, _md, _sd] = model.predict(Xt, true, false, false);
  arma::vec mean = m, sd = s;
  const double q2 = 1.0 - arma::accu(arma::square(yt - mean)) / arma::accu(arma::square(yt - arma::mean(yt)));
  const double nlpd
      = arma::mean(0.5 * arma::log(2.0 * arma::datum::pi * arma::square(sd)) + 0.5 * arma::square((yt - mean) / sd));
  const double cov = arma::mean(arma::conv_to<arma::vec>::from(arma::abs(yt - mean) <= 1.96 * sd));
  return {q2, nlpd, cov};
}

}  // namespace

TEST_CASE("MLPKriging predictive variance is calibrated (#407)", "[mlp][calibration]") {
  arma::arma_rng::set_seed(407);
  const arma::mat X = lhs(60, kDim);
  const arma::vec y = branin_proj(X);
  const arma::mat Xt = arma::randu<arma::mat>(1000, kDim);
  const arma::vec yt = branin_proj(Xt);

  Kriging kr(y, X, "matern5_2");
  const Scores sk = score(kr, Xt, yt);

  const std::string optim = GENERATE(as<std::string>{}, "BFGS", "BFGS4");

  MLPKriging raw({8}, 2, "tanh", "matern5_2");
  raw.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", {{"calibration_folds", "0"}});
  MLPKriging cal({8}, 2, "tanh", "matern5_2");
  cal.fit(y, X, Trend::RegressionModel::Constant, false, optim);

  const Scores sr = score(raw, Xt, yt);
  const Scores sc = score(cal, Xt, yt);
  INFO("optim=" << optim << " | Kriging Q2 " << sk.q2 << " NLPD " << sk.nlpd << " cov " << sk.coverage
                << " | uncalibrated Q2 " << sr.q2 << " NLPD " << sr.nlpd << " cov " << sr.coverage
                << " | calibrated NLPD " << sc.nlpd << " cov " << sc.coverage << " var_scale " << cal.var_scale());

  // Same fit, only the predictive variance changes.
  CHECK(raw.var_scale() == 1.0);
  CHECK(cal.logLikelihood() == Approx(raw.logLikelihood()).epsilon(1e-10));
  CHECK(sc.q2 == Approx(sr.q2).epsilon(1e-10));
  CHECK(cal.var_scale() > 1.0);

  CHECK(sc.coverage > 0.8);
  CHECK(sc.nlpd < 0.5 * sr.nlpd);
  // The single start can stop at a poor mean (Q2 < 0 here, #407 point 4), which
  // no predictive variance compensates: compare with Kriging after restarts.
  if (optim != "BFGS")
    CHECK(sc.nlpd < sk.nlpd + 2.0);

  SECTION("save / load keeps the calibration") {
    const std::string path = "MLPKrigingCalibrationTest_" + optim + ".json";
    cal.save(path);
    MLPKriging loaded = MLPKriging::load(path);
    std::remove(path.c_str());
    CHECK(loaded.var_scale() == cal.var_scale());
    const Scores sl = score(loaded, Xt, yt);
    CHECK(sl.nlpd == Approx(sc.nlpd).epsilon(1e-8));
  }
}
