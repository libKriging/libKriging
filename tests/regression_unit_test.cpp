#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

// doc : https://github.com/catchorg/Catch2/blob/master/docs/Readme.mda
// More example in https://github.com/catchorg/Catch2/tree/master/examples

#include "libKriging/utils/lk_armadillo.hpp"

#include <libKriging/LinearRegression.hpp>
#include <random>

SCENARIO("a linear regression reveals randomly generated seed parameters", "[regression]") {
  std::mt19937 engine;    // the Mersenne Twister with a popular choice of parameters
  uint32_t seed_val = 0;  // populate somehow (fixed value => reproducible)
  engine.seed(seed_val);

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

      // THEN("the results are accurate") {
      //   INFO("dimensions are n=" << n << " m=" << m);
      //   const double eps = 10 * std::numeric_limits<double>::epsilon();
      //   REQUIRE(arma::norm(sol - rl.coef, "inf") == Approx(0).margin(10 * eps));
      //   REQUIRE(arma::norm(rl.stderrest, "inf") == Approx(0).margin(10 * eps));
      // }

      THEN("the prediction are exact (for X at least)") {
        std::tuple<arma::colvec, arma::colvec> ans = rl.predict(X);
        const double eps = 1e-5;  // 1000 * std::numeric_limits<double>::epsilon();
        INFO("y= " << y << " y_pred=" << std::get<0>(ans) << " diff=" << arma::norm(y - std::get<0>(ans), "inf")
                   << " eps=" << eps);
        REQUIRE(arma::norm(y - std::get<0>(ans), "inf") == Approx(0).margin(10 * eps));
      }
    }
    WHEN("value is computed with noise") {
      arma::vec y = X * sol;
      const double e = 1e-8;

      // Add noise
      std::normal_distribution<double> noise(1, e);
      y.for_each([&noise, &engine](arma::vec::elem_type& val) { val *= noise(engine); });

      LinearRegression rl;  // linear regression object
      rl.fit(y, X);

      // THEN("the results are almost accurate") {
      //   INFO("dimensions are n=" << n << " m=" << m);
      //   REQUIRE(arma::norm(sol - rl.coef, "inf") == Approx(0).margin(10 * eps));
      //   REQUIRE(arma::norm(rl.stderrest, "inf") == Approx(0).margin(10 * eps));
      // }

      THEN("the prediction are almost exact (for X at least)") {
        std::tuple<arma::colvec, arma::colvec> ans = rl.predict(X);
        const double eps = 1e-5;  // 1000 * std::numeric_limits<double>::epsilon();
        INFO("y= " << y << " y_pred=" << std::get<0>(ans) << " diff=" << arma::norm(y - std::get<0>(ans), "inf")
                   << " eps=" << eps << " e=" << e);
        REQUIRE(arma::norm(y - std::get<0>(ans), "inf") == Approx(0).margin(10 * eps + 10 * e));
      }
    }
  }
}