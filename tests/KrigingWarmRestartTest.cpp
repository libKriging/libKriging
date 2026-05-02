// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include <chrono>
#include "libKriging/Kriging.hpp"
// clang-format on

// Test function: sum of squares
static double test_function(const arma::rowvec& x) {
  double result = 0;
  for (arma::uword k = 0; k < x.n_elem; ++k)
    result += x(k) * x(k);
  return result;
}

TEST_CASE("KrigingWarmRestartTest - Warm restart is faster than cold fit", "[update][kriging][warm]") {
  arma::arma_rng::set_seed(42);

  const arma::uword n_old = 30;
  const arma::uword n_new = 3;
  const arma::uword d = 2;

  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);
  for (arma::uword i = 0; i < n_old; ++i)
    y_old(i) = test_function(X_old.row(i));

  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::colvec y_new(n_new);
  for (arma::uword i = 0; i < n_new; ++i)
    y_new(i) = test_function(X_new.row(i));

  arma::mat X_combined = arma::join_cols(X_old, X_new);
  arma::colvec y_combined = arma::join_cols(y_old, y_new);

  SECTION("Warm restart produces correct predictions") {
    Kriging kr("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    // Warm restart update (refit=true uses single-start BFGS from current theta)
    kr.update(y_new, X_new, true);

    // Fresh combined fit
    Kriging kr_ref("gauss");
    kr_ref.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    arma::mat X_test(10, d, arma::fill::randu);
    auto pred_warm = kr.predict(X_test, true, false, false);
    auto pred_ref = kr_ref.predict(X_test, true, false, false);

    INFO("Warm theta: " << kr.theta().t());
    INFO("Ref  theta: " << kr_ref.theta().t());
    INFO("Max pred diff: " << arma::max(arma::abs(std::get<0>(pred_warm) - std::get<0>(pred_ref))));

    CHECK(arma::approx_equal(std::get<0>(pred_warm), std::get<0>(pred_ref), "reldiff", 0.1));
    // stdev may differ since warm restart may find a different local optimum for theta
  }

  SECTION("Warm restart is faster than cold multistart fit") {
    // Initial fit with multi-start BFGS
    Kriging kr("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS4", "LL", params);

    // Measure warm restart update (forces single-start BFGS)
    auto t0 = std::chrono::high_resolution_clock::now();
    kr.update(y_new, X_new, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double warm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Measure cold multi-start fit on combined data
    Kriging kr_cold("gauss");
    auto t2 = std::chrono::high_resolution_clock::now();
    kr_cold.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS4", "LL", params);
    auto t3 = std::chrono::high_resolution_clock::now();
    double cold_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    INFO("Warm restart: " << warm_ms << " ms");
    INFO("Cold fit:     " << cold_ms << " ms");

    // Warm restart should be noticeably faster (at most 80% of cold fit time)
    CHECK(warm_ms < cold_ms * 0.8);

    // And predictions should still be close
    arma::mat X_test(10, d, arma::fill::randu);
    auto pred_warm = kr.predict(X_test, true, false, false);
    auto pred_cold = kr_cold.predict(X_test, true, false, false);
    CHECK(arma::approx_equal(std::get<0>(pred_warm), std::get<0>(pred_cold), "reldiff", 0.15));
  }

  SECTION("Multiple warm restart updates accumulate correctly") {
    Kriging kr("gauss");
    Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    // Apply updates one point at a time
    for (arma::uword i = 0; i < n_new; ++i) {
      kr.update(arma::vec{y_new(i)}, X_new.row(i), true);
    }

    // Fresh combined fit
    Kriging kr_ref("gauss");
    kr_ref.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    arma::mat X_test(10, d, arma::fill::randu);
    auto pred_warm = kr.predict(X_test, true, false, false);
    auto pred_ref = kr_ref.predict(X_test, true, false, false);

    CHECK(arma::approx_equal(std::get<0>(pred_warm), std::get<0>(pred_ref), "reldiff", 0.15));
  }
}

TEST_CASE("NuggetKrigingWarmRestartTest - Warm restart passes current params", "[update][nugget][warm]") {
  arma::arma_rng::set_seed(42);

  const arma::uword n_old = 30;
  const arma::uword n_new = 3;
  const arma::uword d = 2;

  arma::mat X_old(n_old, d, arma::fill::randu);
  arma::colvec y_old(n_old);
  for (arma::uword i = 0; i < n_old; ++i)
    y_old(i) = test_function(X_old.row(i)) + 0.01 * arma::randn();

  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::colvec y_new(n_new);
  for (arma::uword i = 0; i < n_new; ++i)
    y_new(i) = test_function(X_new.row(i)) + 0.01 * arma::randn();

  arma::mat X_combined = arma::join_cols(X_old, X_new);
  arma::colvec y_combined = arma::join_cols(y_old, y_new);

  SECTION("Warm restart produces close predictions") {
    Kriging kr("gauss", Kriging::NoiseModel::Nugget);
    KrigingParameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true, std::nullopt, true};
    kr.fit(y_old, X_old, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    kr.update(y_new, X_new, true);

    Kriging kr_ref("gauss", Kriging::NoiseModel::Nugget);
    kr_ref.fit(y_combined, X_combined, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

    arma::mat X_test(10, d, arma::fill::randu);
    auto pred_warm = kr.predict(X_test, true, false, false);
    auto pred_ref = kr_ref.predict(X_test, true, false, false);

    INFO("Warm theta: " << kr.theta().t());
    INFO("Ref  theta: " << kr_ref.theta().t());
    INFO("Warm nugget: " << kr.nugget());
    INFO("Ref  nugget: " << kr_ref.nugget());

    CHECK(arma::approx_equal(std::get<0>(pred_warm), std::get<0>(pred_ref), "reldiff", 0.15));
  }
}
