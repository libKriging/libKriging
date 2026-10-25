// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/WarpKriging.hpp"
// clang-format on

using libKriging::WarpKriging;

// The Hessian of logLikelihoodFun (central differences of the analytic
// gradient) against second differences of the log-likelihood values alone.
static arma::mat hessian_from_values(const std::function<double(const arma::vec&)>& f, const arma::vec& x) {
  const arma::uword p = x.n_elem;
  arma::mat H(p, p);
  for (arma::uword i = 0; i < p; ++i)
    for (arma::uword j = 0; j < p; ++j) {
      const double hi = 1e-3 * x[i], hj = 1e-3 * x[j];
      arma::vec pp = x, pm = x, mp = x, mm = x;
      pp[i] += hi, pp[j] += hj;
      pm[i] += hi, pm[j] -= hj;
      mp[i] -= hi, mp[j] += hj;
      mm[i] -= hi, mm[j] -= hj;
      H(i, j) = (f(pp) - f(pm) - f(mp) + f(mm)) / (4 * hi * hj);
    }
  return H;
}

static void check_hessian(const arma::mat& H, const arma::mat& ref) {
  REQUIRE(H.n_rows == ref.n_rows);
  REQUIRE(H.n_cols == ref.n_cols);
  CHECK(arma::approx_equal(H, H.t(), "absdiff", 1e-12));
  CHECK(arma::norm(H - ref, "fro") <= 1e-3 * arma::norm(ref, "fro"));
}

static void data(arma::mat& X, arma::vec& y) {
  arma::arma_rng::set_seed(42);
  X = arma::randu<arma::mat>(15, 2);
  y = arma::sin(4 * X.col(0)) + X.col(1) % X.col(1);
}

TEST_CASE("Kriging logLikelihoodFun Hessian", "[hessian][kriging]") {
  arma::mat X;
  arma::vec y;
  data(X, y);
  for (bool normalize : {false, true}) {
    for (std::string noise : {"none", "nugget"}) {
      Kriging k("matern5_2", noise == "nugget" ? Kriging::NoiseModel::Nugget : Kriging::NoiseModel::None);
      k.fit(y, X, Trend::RegressionModel::Constant, normalize, "BFGS", "LL", Kriging::Parameters{});
      arma::vec theta = arma::vec{0.4, 0.7};
      if (noise == "nugget")
        theta = arma::join_cols(theta, arma::vec{0.9});  // + alpha
      auto [ll, grad, H] = k.logLikelihoodFun(theta, true, true, false);
      CHECK(ll == std::get<0>(k.logLikelihoodFun(theta, false, false)));
      CHECK(arma::approx_equal(grad, std::get<1>(k.logLikelihoodFun(theta, true, false)), "absdiff", 0));
      check_hessian(H,
                    hessian_from_values(
                        [&](const arma::vec& t) { return std::get<0>(k.logLikelihoodFun(t, false, false)); }, theta));
      CHECK(std::get<2>(k.logLikelihoodFun(theta, true, false, false)).is_empty());
    }
  }
}

TEST_CASE("WarpKriging logLikelihoodFun Hessian", "[hessian][warp]") {
  arma::mat X;
  arma::vec y;
  data(X, y);
  WarpKriging wk(std::vector<std::string>{"none", "kumaraswamy"}, "matern5_2");
  wk.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
  const arma::vec theta = wk.theta();
  auto [ll, grad, H] = wk.logLikelihoodFun(theta, true, true);
  CHECK(ll == std::get<0>(wk.logLikelihoodFun(theta, false, false)));
  check_hessian(H,
                hessian_from_values(
                    [&](const arma::vec& t) { return std::get<0>(wk.logLikelihoodFun(t, false, false)); }, theta));
  CHECK(std::get<2>(wk.logLikelihoodFun(theta, true, false)).is_empty());
}
