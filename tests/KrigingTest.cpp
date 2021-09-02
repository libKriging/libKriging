#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <cmath>
#include <fstream>
#include <libKriging/Kriging.hpp>

auto f = [](const arma::rowvec& row) {
  double sum = 0;
  for (auto&& x : row) {
    // sum += ((x - .5) * (x - .5));  // cas 1
    sum += (x * x);  // cas 2
  }
  return sum;
};

auto prepare_and_run_bench = [](auto&& bench) {
  const int count = 11;
  const auto i = GENERATE_COPY(range(0, count));

  arma::arma_rng::seed_type seed_val = 123;  // populate somehow (fixed value => reproducible)

  const double logn = 1 + 0.1 * i;
  const arma::uword n = floor(std::pow(10., logn));
  const arma::uword d = floor(2 + i / 3);

  INFO("dimensions are n=" << n << " x d=" << d);

  arma::arma_rng::set_seed(seed_val);
  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  for (arma::uword k = 0; k < n; ++k)
    y(k) = f(X.row(k));

  bench(y, X, i);

  CHECK(true);
};

TEST_CASE("workflow") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    Kriging ok = Kriging("gauss");
    Kriging::Parameters parameters{0, false, arma::vec(), false};
    ok.fit(y, X, Kriging::RegressionModel::Constant, false, "BFGS", "LL", parameters);  // FIXME no move
    const double theta = 0.5;
    arma::vec theta_vec(X.n_cols);
    theta_vec.fill(theta);
    return std::get<1>(ok.logLikelihoodEval(theta_vec, true, false));
  });
}

TEST_CASE("fit benchmark", "[.benchmark]") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    Kriging ok = Kriging("gauss");
    BENCHMARK("Kriging::fit#" + std::to_string(i)) {
      Kriging::Parameters parameters{0, false, arma::vec(), false};
      return ok.fit(y, X, Kriging::RegressionModel::Constant, false, "BFGS", "LL", parameters);  // FIXME no move
    };
  });
}

TEST_CASE("logLikelihoodFun benchmark", "[.benchmark]") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    Kriging ok = Kriging("gauss");
    ok.fit(y, X);  // FIXME no move

    const double theta = 0.5;
    arma::vec theta_vec(X.n_cols);
    theta_vec.fill(theta);

    BENCHMARK("Kriging::logLikelihoodFun#" + std::to_string(i)) {
      return std::get<0>(ok.logLikelihoodEval(theta_vec, false, false));  //
    };
  });
}

TEST_CASE("logLikelihoodGrad benchmark", "[.benchmark]") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    Kriging ok = Kriging("gauss");
    ok.fit(y, X);  // FIXME no move

    const double theta = 0.5;
    arma::vec theta_vec(X.n_cols);
    theta_vec.fill(theta);

    BENCHMARK("Kriging::logLikelihoodGrad#" + std::to_string(i)) {
      return std::get<1>(ok.logLikelihoodEval(theta_vec, true, false));  //
    };
  });
}