#define CATCH_CONFIG_MAIN
#include <armadillo>
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <cmath>
#include <fstream>
#include <libKriging/OrdinaryKriging.hpp>

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
    OrdinaryKriging ok = OrdinaryKriging("gauss");
    ok.fit(y, X);  // FIXME no move
    const double theta = 0.5;
    arma::vec theta_vec(X.n_cols);
    theta_vec.fill(theta);
    return ok.logLikelihoodGrad(theta_vec);
  });
}

TEST_CASE("fit benchmark", "[.benchmark]") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    OrdinaryKriging ok = OrdinaryKriging("gauss");
    BENCHMARK("OrdinaryKriging::fit#" + std::to_string(i)) {
      return ok.fit(y, X);  // FIXME no move
    };
  });
}

TEST_CASE("logLikelihood benchmark", "[.benchmark]") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    OrdinaryKriging ok = OrdinaryKriging("gauss");
    ok.fit(y, X);  // FIXME no move

    const double theta = 0.5;
    arma::vec theta_vec(X.n_cols);
    theta_vec.fill(theta);

    BENCHMARK("OrdinaryKriging::logLikelihood#" + std::to_string(i)) {
      return ok.logLikelihood(theta_vec);  //
    };
  });
}

TEST_CASE("logLikelihoodGrad benchmark", "[.benchmark]") {
  prepare_and_run_bench([](const arma::colvec& y, const arma::mat& X, int i) {
    OrdinaryKriging ok = OrdinaryKriging("gauss");
    ok.fit(y, X);  // FIXME no move

    const double theta = 0.5;
    arma::vec theta_vec(X.n_cols);
    theta_vec.fill(theta);

    BENCHMARK("OrdinaryKriging::logLikelihoodGrad#" + std::to_string(i)) {
      return ok.logLikelihoodGrad(theta_vec);  //
    };
  });
}