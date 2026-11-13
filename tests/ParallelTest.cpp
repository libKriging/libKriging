// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Parallel.hpp"
#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
// clang-format on

// Parallel (Parallel.hpp): the thread settings of the library in one place, the
// same settings as Optim's (kept for compatibility), the OpenMP threshold at run
// time, and the read-only state of each level.

namespace {
struct Restore {
  double percent = Parallel::get_num_threads_percent();
  int pool = Parallel::get_thread_pool_size();
  int delay = Parallel::get_thread_start_delay_ms();
  int blas_min_n = Parallel::get_blas_threads_min_n();
  int omp_min_rows = Parallel::get_omp_min_rows();
  ~Restore() {
    Parallel::set_num_threads_percent(percent);
    Parallel::set_thread_pool_size(pool);
    Parallel::set_thread_start_delay_ms(delay);
    Parallel::set_blas_threads_min_n(blas_min_n);
    Parallel::set_omp_min_rows(omp_min_rows);
  }
};
}  // namespace

TEST_CASE("Parallel and Optim share the thread settings", "[parallel]") {
  Restore restore;

  Parallel::set_num_threads_percent(0.5);
  CHECK(Optim::get_num_threads_percent() == 0.5);
  CHECK(Parallel::get_num_threads() == Optim::get_num_threads());
  CHECK(Parallel::get_num_threads() >= 1);
  Optim::set_num_threads_percent(1e-6);  // at least 1 thread
  CHECK(Parallel::get_num_threads_percent() == 1e-6);
  CHECK(Parallel::get_num_threads() == 1);
  CHECK_THROWS_AS(Parallel::set_num_threads_percent(-1), std::invalid_argument);

  Parallel::set_thread_pool_size(3);
  CHECK(Optim::get_thread_pool_size() == 3);
  Parallel::set_thread_start_delay_ms(7);
  CHECK(Optim::get_thread_start_delay_ms() == 7);
  Parallel::set_blas_threads_min_n(123);
  CHECK(Optim::get_blas_threads_min_n() == 123);
  Optim::set_blas_threads_min_n(0);
  CHECK(Parallel::get_blas_threads_min_n() == 0);
}

TEST_CASE("OpenMP threshold: set at run time, results do not depend on it", "[parallel]") {
  Restore restore;
  const char* env = std::getenv("LK_OMP_MIN_ROWS");
  if (env == nullptr || *env == '\0')
    CHECK(Parallel::get_omp_min_rows() > 0);  // the platform value of the build
  CHECK_THROWS_AS(Parallel::set_omp_min_rows(-1), std::invalid_argument);

  arma::arma_rng::set_seed(7);
  const arma::uword n = 300;
  arma::mat X(n, 3, arma::fill::randu);
  arma::vec y = arma::sin(4 * X.col(0)) + X.col(1) % X.col(2);
  Kriging::Parameters par;
  par.theta = arma::mat{{0.3, 0.5, 0.7}};
  par.is_theta_estim = false;
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", par);
  const arma::vec theta{0.3, 0.5, 0.7};

  Parallel::set_omp_min_rows(0);  // parallel from the first row
  CHECK(Parallel::get_omp_min_rows() == 0);
  const auto [ll_par, g_par] = k.logLikelihoodFun(theta, true, false);
  Parallel::set_omp_min_rows(1000000);  // always serial
  CHECK(Parallel::get_omp_min_rows() == 1000000);
  const auto [ll_ser, g_ser] = k.logLikelihoodFun(theta, true, false);

  // covariance entries written once, gradient sums in a fixed order (#441)
  CHECK(std::abs(ll_par - ll_ser) <= 1e-13 * std::abs(ll_ser));
  CHECK(arma::abs(g_par - g_ser).max() <= 1e-12 * arma::abs(g_ser).max());
}

TEST_CASE("Parallel reports the BLAS and OpenMP levels", "[parallel]") {
  const std::string backend = Parallel::blas_backend();
  INFO("backend " << backend << ", BLAS threads " << Parallel::blas_num_threads() << ", OpenMP loops "
                  << Parallel::omp_enabled());
  CHECK(std::set<std::string>{"mkl", "openblas", "accelerate", "unknown"}.count(backend) == 1);
  CHECK(Parallel::blas_num_threads() >= 0);
  if (backend == "mkl" || backend == "openblas")
    CHECK(Parallel::blas_num_threads() >= 1);
  const char* expected = std::getenv("LK_EXPECT_BLAS");  // set by the BLAS backend CI jobs (T9)
  if (expected != nullptr && *expected != '\0')
    CHECK(backend == expected);
#if defined(_OPENMP)
  CHECK(Parallel::omp_enabled() == Parallel::omp_enabled());  // compiled-in state, read only
#else
  CHECK_FALSE(Parallel::omp_enabled());
#endif
}
