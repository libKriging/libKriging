// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

// A caller's OpenBLAS thread count must survive fit / predict / update, with
// or without a parallel multistart (BlasThreadGuard used to reset it to 1),
// and a fork (the pthread_atfork parent handler used to leave it at 1).

#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#if defined(__linux__)
// Weak: null when the BLAS backend is not OpenBLAS (the tests are skipped).
extern "C" {
void openblas_set_num_threads(int) __attribute__((weak));
int openblas_get_num_threads(void) __attribute__((weak));
}
static bool has_openblas() {
  return openblas_set_num_threads != nullptr && openblas_get_num_threads != nullptr;
}
#else
static bool has_openblas() {
  return false;
}
static void openblas_set_num_threads(int) {}
static int openblas_get_num_threads() {
  return 0;
}
#endif

namespace {

struct Data {
  arma::mat X;
  arma::vec y;
};

Data make_data(arma::uword n, arma::uword d) {
  arma::arma_rng::set_seed(42);
  Data data{arma::mat(n, d, arma::fill::randu), arma::vec(n)};
  for (arma::uword i = 0; i < n; ++i)
    data.y(i) = std::sin(6.0 * data.X(i, 0)) + (d > 1 ? data.X(i, 1) * data.X(i, 1) : 0.0);
  return data;
}

// Sets the caller's thread count to 4 and returns the value OpenBLAS kept.
int set_caller_threads() {
  openblas_set_num_threads(4);
  return openblas_get_num_threads();
}

}  // namespace

TEST_CASE("Kriging fit keeps the caller's OpenBLAS thread count", "[BlasThreads]") {
  if (!has_openblas()) {
    WARN("BLAS backend is not OpenBLAS: skipped");
    return;
  }
  const Data data = make_data(60, 2);
  for (const std::string optim : {"BFGS", "BFGS4"}) {
    for (const int min_n : {400, 0, -1}) {  // policy: 1 thread / cap only / disabled
      Optim::set_blas_threads_min_n(min_n);
      const int caller = set_caller_threads();
      Kriging k(data.y, data.X, "matern5_2", Trend::RegressionModel::Constant, false, optim, "LL");
      INFO("optim=" << optim << " min_n=" << min_n);
      CHECK(openblas_get_num_threads() == caller);

      k.predict(data.X.rows(0, 9), true, false, false);
      CHECK(openblas_get_num_threads() == caller);

      const Data more = make_data(5, 2);
      k.update(more.y, more.X, true);
      CHECK(openblas_get_num_threads() == caller);
    }
  }
  Optim::set_blas_threads_min_n(400);
}

TEST_CASE("Kriging with nugget fit keeps the caller's OpenBLAS thread count", "[BlasThreads]") {
  if (!has_openblas()) {
    WARN("BLAS backend is not OpenBLAS: skipped");
    return;
  }
  const Data data = make_data(60, 2);
  const int caller = set_caller_threads();
  Kriging k("matern5_2", Kriging::NoiseModel::Nugget);
  k.fit(data.y, data.X, Trend::RegressionModel::Constant, false, "BFGS4", "LL");
  CHECK(openblas_get_num_threads() == caller);
}

TEST_CASE("WarpKriging fit keeps the caller's OpenBLAS thread count", "[BlasThreads]") {
  if (!has_openblas()) {
    WARN("BLAS backend is not OpenBLAS: skipped");
    return;
  }
  const Data data = make_data(30, 2);
  for (const std::string optim : {"BFGS", "BFGS2"}) {  // BFGS2: run_task_pool
    const int caller = set_caller_threads();
    libKriging::WarpKriging wk(data.y,
                               data.X,
                               std::vector<std::string>{"kumaraswamy", "kumaraswamy"},
                               "matern5_2",
                               Trend::RegressionModel::Constant,
                               false,
                               optim,
                               "LL",
                               {{"calibration_folds", "0"}, {"max_iter_bfgs", "20"}});
    INFO("optim=" << optim);
    CHECK(openblas_get_num_threads() == caller);
    wk.predict(data.X.rows(0, 4), true, false, false);
    CHECK(openblas_get_num_threads() == caller);
  }
}

#if defined(__linux__)
TEST_CASE("fork keeps the parent's OpenBLAS thread count", "[BlasThreads]") {
  if (!has_openblas()) {
    WARN("BLAS backend is not OpenBLAS: skipped");
    return;
  }
  // Load libKriging (and its pthread_atfork handlers) before forking.
  Kriging k("gauss");
  const int caller = set_caller_threads();
  const pid_t pid = fork();
  if (pid == 0) {
    // child: single-threaded BLAS (exit code 0 if so)
    _exit(openblas_get_num_threads() == 1 ? 0 : 1);
  }
  REQUIRE(pid > 0);
  int status = 0;
  waitpid(pid, &status, 0);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  CHECK(openblas_get_num_threads() == caller);
}
#endif

TEST_CASE("Optim BLAS thread policy settings", "[BlasThreads]") {
  const int min_n = Optim::get_blas_threads_min_n();
  const int max_t = Optim::get_blas_threads_max();
  Optim::set_blas_threads_min_n(123);
  Optim::set_blas_threads_max(2);
  CHECK(Optim::get_blas_threads_min_n() == 123);
  CHECK(Optim::get_blas_threads_max() == 2);
  Optim::set_blas_threads_min_n(min_n);
  Optim::set_blas_threads_max(max_t);
}
