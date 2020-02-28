#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// doc : https://github.com/catchorg/Catch2/blob/master/docs/Readme.mda
// More example in https://github.com/catchorg/Catch2/tree/master/examples

#include <armadillo>
#include <libKriging/LinearRegression.hpp>
#include <random>

#include "test_utils.hpp"

SCENARIO("a linear regression reveals randomly generated seed parameters", "[regression]") {
  uint32_t seed_val = 0;          // populate somehow (fixed value => reproducible)
  std::mt19937 engine{seed_val};  // the Mersenne Twister with a popular choice of parameters
  arma::arma_rng::set_seed(seed_val);

  // Sizes of this tests : 3*2 generated combinations
  const int n = GENERATE(40, 100, 1000);
  const int m = GENERATE(3, 6);

  GIVEN("A matrix and generated seed parameters") {
    INFO("dimensions are n=" << n << " m=" << m);

    arma::vec sol(m, arma::fill::randn);
    arma::mat X(n, m);
    std::normal_distribution<double> dist(1, 10);
    X.col(0).fill(1);
    X.cols(1, m - 1).imbue([&]() { return dist(engine); });

    WHEN("value is perfectly computed") {
      arma::vec y = X * sol;

      LinearRegression rl;  // linear regression object
      rl.fit(y, X);

      THEN("the prediction are exact (for X at least)") {
        std::tuple<arma::colvec, arma::colvec> ans = rl.predict(X);
        const double eps = 1000 * std::numeric_limits<double>::epsilon();
        // INFO("y= " << y << " y_pred=" << std::get<0>(ans));
        INFO("absolute diff=" << arma::norm(y - std::get<0>(ans), "inf"));
        INFO("relative diff=" << relative_error(y, std::get<0>(ans)));
        INFO("eps=" << eps);
        REQUIRE(relative_error(y, std::get<0>(ans)) == Approx(0).margin(10 * eps));
      }
    }
    WHEN("value is computed with noise") {
      arma::vec y = X * sol;
      const double noise_amplitude = 1e-8;

      // Add noise
      std::normal_distribution<double> noise(1, noise_amplitude);
      y.for_each([&noise, &engine](arma::vec::elem_type& val) { val *= noise(engine); });

      LinearRegression rl;  // linear regression object
      rl.fit(y, X);

      THEN("the prediction are almost exact (for X at least)") {
        std::tuple<arma::colvec, arma::colvec> ans = rl.predict(X);
        const double eps = 1000 * std::numeric_limits<double>::epsilon();
        // INFO("y= " << y << " y_pred=" << std::get<0>(ans));
        INFO("absolute diff=" << arma::norm(y - std::get<0>(ans), "inf"));
        INFO("relative diff=" << relative_error(y, std::get<0>(ans)));
        INFO("eps=" << eps << " noise_amplitude=" << noise_amplitude);
        REQUIRE(relative_error(y, std::get<0>(ans)) == Approx(0).margin(10 * eps + 10 * noise_amplitude));
      }
    }
  }
}