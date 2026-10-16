// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/WarpKriging.hpp"
#include <cmath>
#include <string>
#include <vector>
// clang-format on

/**
 * @file WarpKrigingCollapseTest.cpp
 * @brief Regression test for #406: monotone input warps must not collapse
 *        the training inputs.
 *
 * Gramacy & Lee (2008), f(x) = x1 exp(-x1^2 - x2^2) on [-2, 6]^2: f is
 * numerically 0 on most of the domain, so a warp squeezing that region onto
 * a single point makes the noise-free likelihood unbounded (near-duplicate
 * rows of R with equal outputs). The likelihood alone rewards that
 * degenerate fit, whose predictions are worse than plain Kriging with a
 * collapsed predictive stdev. Checked against plain Kriging on the same
 * design: Q2 not much lower, NLPD not exploding.
 */

using namespace libKriging;

namespace {

constexpr double kLo = -2.0, kUp = 6.0;

arma::vec gl2008(const arma::mat& U) {
  arma::vec x1 = kLo + U.col(0) * (kUp - kLo);
  arma::vec x2 = kLo + U.col(1) * (kUp - kLo);
  return x1 % arma::exp(-x1 % x1 - x2 % x2);
}

// Latin hypercube in [0.01, 0.99]^d.
arma::mat lhs(arma::uword n, arma::uword d) {
  arma::mat X(n, d);
  for (arma::uword j = 0; j < d; ++j) {
    arma::uvec perm = arma::randperm(n);
    for (arma::uword i = 0; i < n; ++i)
      X(i, j) = (static_cast<double>(perm(i)) + arma::randu()) / static_cast<double>(n);
  }
  return 0.01 + 0.98 * X;
}

struct Scores {
  double q2;
  double nlpd;
  double min_sd;
};

template <typename Model>
Scores score(Model& model, const arma::mat& Xt, const arma::vec& yt) {
  auto [m, s, _c, _md, _sd] = model.predict(Xt, true, false, false);
  arma::vec mean = m, sd = s;
  double q2 = 1.0 - arma::accu(arma::square(yt - mean)) / arma::accu(arma::square(yt - arma::mean(yt)));
  double nlpd = arma::mean(0.5 * arma::log(2.0 * arma::datum::pi * arma::square(sd)) + 0.5 * arma::square((yt - mean) / sd));
  return {q2, nlpd, sd.min()};
}

}  // namespace

TEST_CASE("WarpKriging monotone warps do not collapse the inputs (#406)", "[warp][collapse]") {
  // knots(K) is left out: the guard removes its collapse too (Q2 ≈ 0 → 0.66 on
  // this benchmark) but it stays overconfident (NLPD above plain Kriging).
  const std::string warp = GENERATE(as<std::string>{}, "kumaraswamy", "boxcox");

  arma::arma_rng::set_seed(10033);
  const arma::mat Xt = 0.01 + 0.98 * arma::randu<arma::mat>(1000, 2);
  const arma::vec yt = gl2008(Xt);

  for (arma::uword rep = 0; rep < 3; ++rep) {
    arma::arma_rng::set_seed(33 + rep);
    const arma::mat X = lhs(40, 2);
    const arma::vec y = gl2008(X);

    Kriging kr(y, X, "matern5_2");
    WarpKriging wk(y, X, {warp, warp}, "matern5_2");
    const Scores sk = score(kr, Xt, yt);
    const Scores sw = score(wk, Xt, yt);

    INFO("warp=" << warp << " rep=" << rep << " Q2: Kriging " << sk.q2 << ", WarpKriging " << sw.q2
                 << " | NLPD: Kriging " << sk.nlpd << ", WarpKriging " << sw.nlpd << " | min stdev " << sw.min_sd
                 << " | LL " << wk.logLikelihood() << " | " << wk.summary());
    CHECK(sw.q2 > sk.q2 - 0.3);
    CHECK(sw.nlpd < sk.nlpd + 1.0);
  }
}
