#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>
#include <libKriging/NoiseKriging.hpp>
#include <libKriging/KrigingLoader.hpp>
#include <libKriging/Trend.hpp>
#include <libKriging/Optim.hpp>

// Cross-platform environment variable functions
#ifdef _WIN32
#include <cstdlib>
inline int setenv_portable(const char* name, const char* value, int overwrite) {
  if (!overwrite && std::getenv(name) != nullptr) {
    return 0;
  }
  return _putenv_s(name, value);
}
inline int unsetenv_portable(const char* name) {
  return _putenv_s(name, "");
}
#else
inline int setenv_portable(const char* name, const char* value, int overwrite) {
  return setenv(name, value, overwrite);
}
inline int unsetenv_portable(const char* name) {
  return unsetenv(name);
}
#endif

// Simple 1D test function
auto simple_f = [](double x) {
  return std::sin(3.0 * x) + 0.5 * std::cos(7.0 * x);
};

TEST_CASE("NoiseKriging multistart and parallel tests", "[multistart]") {
  arma::arma_rng::set_seed(123);
  const arma::uword n = 20;
  const arma::uword d = 1;

  arma::mat X(n, d);
  X.col(0) = arma::linspace(0, 1, n);
  arma::colvec y(n);
  for (arma::uword k = 0; k < n; ++k)
    y(k) = simple_f(X(k, 0));
  y += 0.05 * arma::randn(n);  // Add noise
  arma::colvec noise(n);
  noise.fill(0.01);  // Known noise variance

  SECTION("Basic fit") {
    NoiseKriging nk = NoiseKriging("gauss");
    NoiseKriging::Parameters parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", parameters);

    CHECK(nk.theta().n_elem == d);
    CHECK(nk.sigma2() > 0);

    arma::mat X_new(1, d);
    X_new(0, 0) = 0.5;
    auto pred = nk.predict(X_new, true, false, false);
    CHECK(std::get<0>(pred).n_elem == 1);
    CHECK(std::get<1>(pred).n_elem == 1);
  }

  SECTION("BFGS20 multistart") {
    NoiseKriging nk = NoiseKriging("gauss");
    NoiseKriging::Parameters parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS20", "LL", parameters);

    CHECK(nk.theta().n_elem == d);
    CHECK(nk.sigma2() > 0);

    arma::mat X_new(1, d);
    X_new(0, 0) = 0.5;
    auto pred = nk.predict(X_new, true, false, false);
    CHECK(std::get<0>(pred).n_elem == 1);
    CHECK(std::get<1>(pred).n_elem == 1);
  }

  SECTION("BFGS vs BFGS20 comparison") {
    NoiseKriging nk_single = NoiseKriging("gauss");
    NoiseKriging::Parameters parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true};
    nk_single.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", parameters);
    double ll_single = nk_single.logLikelihood();

    NoiseKriging nk_multi = NoiseKriging("gauss");
    nk_multi.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS20", "LL", parameters);
    double ll_multi = nk_multi.logLikelihood();

    INFO("Single-start log-likelihood: " << ll_single);
    INFO("Multi-start log-likelihood: " << ll_multi);

#ifdef _OPENMP
    INFO("OpenMP is ENABLED - parallel execution active");
#else
    INFO("OpenMP is DISABLED - sequential execution only");
    WARN("Enable OpenMP for parallel speedup: configure with -DCMAKE_CXX_FLAGS=\"-fopenmp\"");
#endif

    CHECK(ll_multi >= ll_single - std::abs(ll_single) * 1e-6);
  }

  SECTION("BFGS10 vs BFGS20 timing (parallel efficiency)") {
    using namespace std::chrono;

    // Limit OpenBLAS/OpenMP threads to avoid contention with multistart parallelism
    const char* old_openblas = std::getenv("OPENBLAS_NUM_THREADS");
    const char* old_blas = std::getenv("OMP_NUM_THREADS");
    setenv_portable("OPENBLAS_NUM_THREADS", "1", 1);
    setenv_portable("OMP_NUM_THREADS", "1", 1);

    const int n_runs = 3;
    std::vector<double> times10, times20;

    INFO("Running " << n_runs << " iterations of each test...");

    for (int run = 0; run < n_runs; ++run) {
      auto start10 = high_resolution_clock::now();
      NoiseKriging nk10 = NoiseKriging("gauss");
      NoiseKriging::Parameters parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true};
      nk10.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS10", "LL", parameters);
      auto end10 = high_resolution_clock::now();
      double time10 = duration_cast<milliseconds>(end10 - start10).count();
      times10.push_back(time10);

      auto start20 = high_resolution_clock::now();
      NoiseKriging nk20 = NoiseKriging("gauss");
      nk20.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS20", "LL", parameters);
      auto end20 = high_resolution_clock::now();
      double time20 = duration_cast<milliseconds>(end20 - start20).count();
      times20.push_back(time20);

      INFO("Run " << (run + 1) << ": BFGS10=" << time10 << "ms, BFGS20=" << time20 << "ms");
    }

    // Restore environment
    if (old_openblas) {
      setenv_portable("OPENBLAS_NUM_THREADS", old_openblas, 1);
    } else {
      unsetenv_portable("OPENBLAS_NUM_THREADS");
    }
    if (old_blas) {
      setenv_portable("OMP_NUM_THREADS", old_blas, 1);
    } else {
      unsetenv_portable("OMP_NUM_THREADS");
    }

    double avg10 = 0.0, avg20 = 0.0;
    for (int i = 0; i < n_runs; ++i) {
      avg10 += times10[i];
      avg20 += times20[i];
    }
    avg10 /= n_runs;
    avg20 /= n_runs;

    double ratio = (avg10 > 0) ? (avg20 / avg10) : 0.0;

    INFO("Average BFGS10 time: " << avg10 << " ms");
    INFO("Average BFGS20 time: " << avg20 << " ms");
    INFO("Average ratio (time20/time10): " << ratio);

    if (ratio < 1.3) {
      INFO("Ratio < 1.3: Parallelization is working effectively!");
    } else if (ratio < 2.0) {
      INFO("Ratio < 2.0: Parallelization is working but could be better");
    } else {
      WARN("Ratio >= 2.0: Parallelization may not be working correctly (expected with very fast optimizations)");
    }

    CHECK(ratio < 4.0);
  }

  SECTION("Thread pool configuration") {
    NoiseKriging nk = NoiseKriging("gauss");
    NoiseKriging::Parameters parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true};

    // Test with explicit thread pool size
    INFO("Testing with thread_pool_size=2");
    Optim::set_thread_pool_size(2);
    CHECK(Optim::get_thread_pool_size() == 2);
    
    nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS5", "LL", parameters);
    CHECK(nk.theta().n_elem == d);
    CHECK(nk.sigma2() > 0);

    // Test with auto thread pool size
    INFO("Testing with thread_pool_size=0 (auto)");
    Optim::set_thread_pool_size(0);
    CHECK(Optim::get_thread_pool_size() == 0);
    
    nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS5", "LL", parameters);
    CHECK(nk.theta().n_elem == d);
    CHECK(nk.sigma2() > 0);
  }

  SECTION("Thread startup delay") {
    NoiseKriging nk = NoiseKriging("gauss");
    NoiseKriging::Parameters parameters{std::nullopt, true, std::nullopt, true, std::nullopt, true};

    // Test with different delays
    INFO("Testing with thread_start_delay_ms=1");
    Optim::set_thread_start_delay_ms(1);
    CHECK(Optim::get_thread_start_delay_ms() == 1);
    
    nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS5", "LL", parameters);
    CHECK(nk.theta().n_elem == d);

    INFO("Testing with thread_start_delay_ms=20");
    Optim::set_thread_start_delay_ms(20);
    CHECK(Optim::get_thread_start_delay_ms() == 20);
    
    nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS5", "LL", parameters);
    CHECK(nk.theta().n_elem == d);

    // Reset to default
    Optim::set_thread_start_delay_ms(10);
  }
}

TEST_CASE("NoiseKriging fit with given parameters - BFGS1") {
  arma::arma_rng::set_seed(123);
  const arma::uword n = 20;
  const arma::uword d = 2;

  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  for (arma::uword k = 0; k < n; ++k)
    y(k) = simple_f(X(k, 0)) * simple_f(X(k, 1));
  y += 0.05 * arma::randn(n);
  
  arma::colvec noise(n);
  noise.fill(0.01);

  // Define specific starting parameters
  arma::mat theta_start(1, d);
  theta_start(0, 0) = 0.5;
  theta_start(0, 1) = 0.3;
  arma::vec sigma2_start(1);
  sigma2_start(0) = 0.1;

  NoiseKriging nk = NoiseKriging("gauss");
  // Provide starting values, optimize all
  NoiseKriging::Parameters parameters{sigma2_start, true, theta_start, true, std::nullopt, true};
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", parameters);

  // Check that optimization ran
  CHECK(nk.sigma2() > 0);
  CHECK(nk.theta().n_elem == d);

  // Verify predictions work
  arma::mat X_new(1, d);
  X_new.fill(0.5);
  auto pred = nk.predict(X_new, true, false, false);
  CHECK(std::get<0>(pred).n_elem == 1);
  CHECK(std::get<1>(pred).n_elem == 1);
}

TEST_CASE("NoiseKriging fit with given parameters - BFGS20") {
  arma::arma_rng::set_seed(123);
  const arma::uword n = 20;
  const arma::uword d = 2;

  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  for (arma::uword k = 0; k < n; ++k)
    y(k) = simple_f(X(k, 0)) * simple_f(X(k, 1));
  y += 0.05 * arma::randn(n);
  
  arma::colvec noise(n);
  noise.fill(0.01);

  // Define specific theta starting points for multistart
  arma::mat theta_starts(20, d);
  for (arma::uword i = 0; i < 20; i++) {
    theta_starts.row(i) = arma::randu<arma::rowvec>(d) * 2.0 + 0.1;
  }
  arma::vec sigma2_start(1);
  sigma2_start(0) = 0.1;

  NoiseKriging nk = NoiseKriging("gauss");
  // Provide starting values, optimize all
  NoiseKriging::Parameters parameters{sigma2_start, true, theta_starts, true, std::nullopt, true};
  nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS20", "LL", parameters);

  // Check that optimization ran
  CHECK(nk.sigma2() > 0);
  CHECK(nk.theta().n_elem == d);

  // Verify predictions work
  arma::mat X_new(1, d);
  X_new.fill(0.5);
  auto pred = nk.predict(X_new, true, false, false);
  CHECK(std::get<0>(pred).n_elem == 1);
  CHECK(std::get<1>(pred).n_elem == 1);
}



TEST_CASE("NoiseKriging all parameter combinations") {
  arma::arma_rng::set_seed(123);
  const arma::uword n = 20;
  const arma::uword d = 2;

  arma::mat X(n, d, arma::fill::randu);
  arma::colvec y(n);
  for (arma::uword k = 0; k < n; ++k)
    y(k) = simple_f(X(k, 0)) * simple_f(X(k, 1));
  y += 0.05 * arma::randn(n);
  
  arma::colvec noise(n);
  noise.fill(0.01);

  // Pre-defined parameter values
  arma::vec sigma2_val(1);
  sigma2_val(0) = 0.1;
  arma::mat theta_val(1, d);
  theta_val.fill(0.5);
  arma::mat theta_starts(20, d);
  for (arma::uword i = 0; i < 20; i++) {
    theta_starts.row(i) = arma::randu<arma::rowvec>(d) * 2.0 + 0.1;
  }

  // Test all combinations with different optimizers
  std::vector<std::string> optims = {"none", "BFGS", "BFGS20"};
  
  for (const auto& optim : optims) {
    DYNAMIC_SECTION("Optim: " << optim) {
      // Combination 1: Estimate all parameters (not valid for "none")
      if (optim != "none") {
        SECTION("Estimate all (sigma2, theta, beta)") {
          NoiseKriging nk("gauss");
          NoiseKriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
          nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, optim, "LL", params);
          CHECK(nk.sigma2() > 0);
          CHECK(nk.theta().n_elem == d);
        }
      }

      // Combination 2: Fix sigma2, estimate theta and beta (not valid for "none")
      if (optim != "none") {
        SECTION("Fix sigma2, estimate theta and beta") {
          NoiseKriging nk("gauss");
          NoiseKriging::Parameters params{sigma2_val, false, std::nullopt, true, std::nullopt, true};
          nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, optim, "LL", params);
          CHECK(nk.sigma2() == sigma2_val(0));
          CHECK(nk.theta().n_elem == d);
        }
      }

      // Combination 3: Estimate sigma2, fix theta, estimate beta (not valid for "none")
      if (optim != "none") {
        SECTION("Estimate sigma2, fix theta, estimate beta") {
          NoiseKriging nk("gauss");
          NoiseKriging::Parameters params{std::nullopt, true, theta_val, false, std::nullopt, true};
          nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, optim, "LL", params);
          CHECK(nk.sigma2() > 0);
          CHECK(nk.theta().n_elem == d);
        }
      }

      // Combination 4: Fix both sigma2 and theta, estimate beta
      SECTION("Fix sigma2 and theta, estimate beta") {
        NoiseKriging nk("gauss");
        NoiseKriging::Parameters params{sigma2_val, false, theta_val, false, std::nullopt, true};
        nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, optim, "LL", params);
        CHECK(nk.sigma2() == sigma2_val(0));
        if (optim == "none") {
          CHECK(arma::approx_equal(nk.theta(), theta_val.t(), "absdiff", 1e-10));
        } else {
          CHECK(nk.theta().n_elem == d);
        }
      }

      // Combination 5: Multistart with theta starting points (BFGS20 only)
      if (optim == "BFGS20") {
        SECTION("Multistart with theta starting points") {
          NoiseKriging nk("gauss");
          NoiseKriging::Parameters params{std::nullopt, true, theta_starts, true, std::nullopt, true};
          nk.fit(y, noise, X, Trend::RegressionModel::Constant, false, optim, "LL", params);
          CHECK(nk.sigma2() > 0);
          CHECK(nk.theta().n_elem == d);
        }
      }
    }
  }

  // Verify predictions work for a representative case
  NoiseKriging nk_final("gauss");
  NoiseKriging::Parameters params_final{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  nk_final.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params_final);
  
  arma::mat X_new(1, d);
  X_new.fill(0.5);
  auto pred = nk_final.predict(X_new, true, false, false);
  CHECK(std::get<0>(pred).n_elem == 1);
  CHECK(std::get<1>(pred).n_elem == 1);
}
