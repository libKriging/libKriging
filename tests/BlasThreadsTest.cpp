// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/WarpKriging.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>
#include <thread>
// clang-format on

#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(_WIN32) && defined(_POSIX_VERSION)
#include <sys/wait.h>
#include <unistd.h>
#define LK_TEST_FORK 1
#endif

// A fit must leave the caller's BLAS / OpenMP thread counts as it found them:
// a multistart that resets them to 1 thread makes every later predict,
// simulate or update single-threaded. A fork() quiesces them to 1 thread in
// the child only; the parent gets its counts back once fork() returns.

#if !defined(_MSC_VER) && (!defined(__APPLE__) || !defined(__arm64__))
extern "C" {
void openblas_set_num_threads(int num_threads) __attribute__((weak));
int openblas_get_num_threads() __attribute__((weak));
}
#define LK_TEST_OPENBLAS 1
#endif

namespace {

struct Threads {
  int blas = 0;
  int omp = 0;
};

Threads get_threads() {
  Threads t;
#ifdef LK_TEST_OPENBLAS
  if (openblas_get_num_threads)
    t.blas = openblas_get_num_threads();
#endif
#ifdef _OPENMP
  t.omp = omp_get_max_threads();
#endif
  return t;
}

void set_threads(int n) {
#ifdef LK_TEST_OPENBLAS
  if (openblas_set_num_threads)
    openblas_set_num_threads(n);
#endif
#ifdef _OPENMP
  omp_set_num_threads(n);
#endif
}

// Sets 3 threads (read back: OpenBLAS may cap it) for the test scope.
struct ThreeThreads {
  Threads initial = get_threads();
  Threads set;
  ThreeThreads() {
    set_threads(3);
    set = get_threads();
  }
  ~ThreeThreads() { set_threads(initial.omp > 0 ? initial.omp : (initial.blas > 0 ? initial.blas : 1)); }
};

bool readable(const Threads& t) {
  return t.blas > 0 || t.omp > 0;
}

void data(arma::vec& y, arma::mat& X, arma::uword n = 40) {
  arma::arma_rng::set_seed(123);
  X = arma::randu(n, 2);
  y = arma::sum(arma::sin(3 * X), 1);
}

// Largest OpenBLAS thread count seen by a monitoring thread while `run` runs
// (0 when OpenBLAS is absent): the library sets it to the budget for a call,
// to each worker's share in a multistart pool.
template <typename F>
int max_blas_threads_during(F run) {
#ifdef LK_TEST_OPENBLAS
  if (!openblas_get_num_threads)
    return (run(), 0);
  std::atomic<bool> done{false};
  std::atomic<int> seen{0};
  std::thread monitor([&] {
    while (!done.load()) {
      const int b = openblas_get_num_threads();
      if (b > seen.load())
        seen.store(b);
      std::this_thread::yield();
    }
  });
  run();
  done.store(true);
  monitor.join();
  return seen.load();
#else
  return (run(), 0);
#endif
}

// Smallest OpenBLAS thread count seen while `run` runs (0 without OpenBLAS).
template <typename F>
int min_blas_threads_during(F run) {
#ifdef LK_TEST_OPENBLAS
  if (!openblas_get_num_threads)
    return (run(), 0);
  std::atomic<bool> done{false};
  std::atomic<int> seen{openblas_get_num_threads()};
  std::thread monitor([&] {
    while (!done.load()) {
      const int b = openblas_get_num_threads();
      if (b < seen.load())
        seen.store(b);
      std::this_thread::yield();
    }
  });
  run();
  done.store(true);
  monitor.join();
  return seen.load();
#else
  return (run(), 0);
#endif
}

// Restores the thread budget setting at the end of a test.
struct BudgetGuard {
  double saved = Optim::get_num_threads_percent();
  ~BudgetGuard() { Optim::set_num_threads_percent(saved); }
};

}  // namespace

TEST_CASE("Kriging multistart fit restores the BLAS / OpenMP thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  arma::vec y;
  arma::mat X;
  data(y, X);
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4");
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}

TEST_CASE("WarpKriging parallel multistart fit restores the BLAS / OpenMP thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  const int pool_size = Optim::get_thread_pool_size();
  Optim::set_thread_pool_size(2);
  arma::vec y;
  arma::mat X;
  data(y, X, 20);
  libKriging::WarpKriging w(y, X, {"affine", "affine"}, "gauss", Trend::RegressionModel::Constant, false, "BFGS4");
  Optim::set_thread_pool_size(pool_size);
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}

#ifdef LK_TEST_FORK
TEST_CASE("fork() quiesces the child only and restores the parent's thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  const pid_t pid = fork();
  if (pid == 0) {
    const Threads c = get_threads();
    _exit((c.blas <= 1 && c.omp <= 1) ? 0 : 1);
  }
  REQUIRE(pid > 0);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}
#endif

#ifdef _OPENMP
// The OpenMP loops (covariance matrix, log-likelihood gradient) write
// independent entries or reduce in a fixed order: the objective, its gradient
// and an updated model do not depend on the OpenMP thread count (the BLAS
// thread count is left unchanged).
TEST_CASE("Objective, gradient and update do not depend on the OpenMP thread count", "[threads]") {
  const int initial = omp_get_max_threads();
  arma::vec y;
  arma::mat X;
  data(y, X, 1200);
  const arma::vec theta{0.3, 0.5};
  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;

  auto run = [&](int threads) {
    omp_set_num_threads(threads);
    Kriging k(y.head(900), X.head_rows(900), "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", par);
    auto [ll, grad] = k.logLikelihoodFun(theta, true, false);
    k.update(y.tail(300), X.tail_rows(300), false);
    return std::make_tuple(ll, arma::vec(grad), k.logLikelihood());
  };
  const auto [ll1, grad1, llu1] = run(1);
  const auto [ll4, grad4, llu4] = run(4);
  omp_set_num_threads(initial);

  CHECK(ll1 == ll4);
  CHECK(arma::all(grad1 == grad4));
  CHECK(llu1 == llu4);
}
#endif

TEST_CASE("The BLAS thread count never exceeds the thread budget during a fit", "[threads]") {
#ifdef LK_TEST_OPENBLAS
  if (!openblas_get_num_threads || !openblas_set_num_threads) {
    WARN("no OpenBLAS in this build");
    return;
  }
  BudgetGuard guard;
  // a budget of 2 threads, below the count OpenBLAS was loaded with (by
  // default the logical CPUs)
  Optim::set_num_threads_percent(100);
  const int full = Optim::get_num_threads();
  if (full < 2 || openblas_get_num_threads() <= 2) {
    WARN("fewer than 2 physical cores or OpenBLAS loaded with <= 2 threads: nothing to bound");
    return;
  }
  Optim::set_num_threads_percent(2.0 / full);
  const int budget = Optim::get_num_threads();
  REQUIRE(budget == 2);
  arma::vec y;
  arma::mat X;
  data(y, X, 60);
  // the library reads the count OpenBLAS was loaded with at its first use:
  // use it once before changing the caller's count
  Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS");
  // the caller's own count at 1: any larger value seen comes from the library
  const int initial = openblas_get_num_threads();
  openblas_set_num_threads(1);
  const int kriging_max = max_blas_threads_during(
      [&] { Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4"); });
  const int warp_max = max_blas_threads_during([&] {
    libKriging::WarpKriging w(y, X, {"affine", "affine"}, "gauss", Trend::RegressionModel::Constant, false, "BFGS4");
  });
  // single start: run inline with the whole budget (also shows that the
  // monitor sees the count set by the library)
  arma::vec y2;
  arma::mat X2;
  data(y2, X2, 300);
  const int single_max = max_blas_threads_during(
      [&] { Kriging k(y2, X2, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS"); });
  const int after = openblas_get_num_threads();
  openblas_set_num_threads(initial);
  INFO("budget " << budget << ", OpenBLAS max seen: Kriging BFGS4 " << kriging_max << ", WarpKriging BFGS4 " << warp_max
                 << ", Kriging BFGS " << single_max);
  CHECK(kriging_max <= budget);
  CHECK(warp_max <= budget);
  CHECK(single_max == budget);
  CHECK(after == 1);
#else
  WARN("no OpenBLAS in this build");
#endif
}

TEST_CASE("Below Optim::blas_threads_min_n observations a fit runs OpenBLAS on 1 thread", "[threads]") {
#ifdef LK_TEST_OPENBLAS
  if (!openblas_get_num_threads || !openblas_set_num_threads) {
    WARN("no OpenBLAS in this build");
    return;
  }
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  const int budget = Optim::get_num_threads();
  const int min_n = Optim::get_blas_threads_min_n();
  arma::vec y;
  arma::mat X;
  data(y, X, 120);
  // first use: the library reads the count OpenBLAS was loaded with
  Kriging(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS");
  const int initial = openblas_get_num_threads();
  const int expected_large = std::min(budget, initial);
  if (expected_large < 2) {
    WARN("fewer than 2 threads: nothing to lower");
    return;
  }
  Optim::set_blas_threads_min_n(100);
  const int small_min = min_blas_threads_during([&] {
    Kriging k(y.head(60), X.head_rows(60), "matern5_2", Trend::RegressionModel::Constant, false, "BFGS");
    k.predict(X, true, false, false);
  });
  const int after = openblas_get_num_threads();
  const int large_min
      = min_blas_threads_during([&] { Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS"); });
  Optim::set_blas_threads_min_n(min_n);
  INFO("budget " << budget << ", initial " << initial << ", OpenBLAS min seen: n = 60 " << small_min << ", n = 120 "
                 << large_min);
  CHECK(small_min == 1);
  CHECK(after == initial);
  CHECK(large_min == expected_large);
#else
  WARN("no OpenBLAS in this build");
#endif
}

#ifdef _OPENMP
TEST_CASE("An explicit OpenMP limit of the calling thread caps the thread budget", "[threads]") {
  BudgetGuard guard;
  const int initial = omp_get_max_threads();
  Optim::set_num_threads_percent(200);
  omp_set_num_threads(2);
  CHECK(Optim::get_num_threads() <= 2);
  omp_set_num_threads(1);
  CHECK(Optim::get_num_threads() == 1);
  omp_set_num_threads(initial);
}
#endif

TEST_CASE("predict, simulate and update leave the caller's BLAS / OpenMP thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  arma::vec y;
  arma::mat X;
  data(y, X, 60);
  Kriging k(y.head(50), X.head_rows(50), "matern5_2", Trend::RegressionModel::Constant, false, "BFGS");
  const arma::mat Xn = arma::randu(30, 2);
  k.predict(Xn, true, true, false);
  k.simulate(10, 123, Xn, false);
  k.update(y.tail(10), X.tail_rows(10), false);
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}

// The budget changes only how many threads run the computations: the
// objective, its gradient and an updated model agree to rounding for budgets
// of 1 to 2x the cores (bit-identical with OpenBLAS here; another BLAS may
// split its sums by thread).
TEST_CASE("Objective, gradient and update do not depend on the thread budget", "[threads]") {
  BudgetGuard guard;
  arma::vec y;
  arma::mat X;
  data(y, X, 1200);
  const arma::vec theta{0.3, 0.5};
  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;

  auto run = [&](double percent) {
    Optim::set_num_threads_percent(percent);
    Kriging k(y.head(900), X.head_rows(900), "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", par);
    auto [ll, grad] = k.logLikelihoodFun(theta, true, false);
    k.update(y.tail(300), X.tail_rows(300), false);
    return std::make_tuple(ll, arma::vec(grad), k.logLikelihood());
  };
  const auto [ll0, grad0, llu0] = run(1e-6);  // 1 thread
  for (const double percent : {20.0, 50.0, 100.0, 200.0}) {
    const auto [ll, grad, llu] = run(percent);
    INFO("budget " << Optim::get_num_threads() << " threads");
    CHECK(std::abs(ll - ll0) <= 1e-12 * std::abs(ll0));
    CHECK(arma::norm(grad - grad0) <= 1e-12 * arma::norm(grad0));
    CHECK(std::abs(llu - llu0) <= 1e-12 * std::abs(llu0));
  }
}

// Hidden: run by the BlasThreadsTest/OPENBLAS_NUM_THREADS test with
// OPENBLAS_NUM_THREADS=1 in its environment (tests/CMakeLists.txt): an OpenBLAS
// loaded with fewer threads than the budget is kept as it is, the budget
// itself (pool, OpenMP) is not reduced.
TEST_CASE("An OpenBLAS loaded with fewer threads than the budget keeps its count", "[.openblas_env]") {
  const char* v = std::getenv("OPENBLAS_NUM_THREADS");
  REQUIRE(v != nullptr);
  const int loaded = std::stoi(v);
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  arma::vec y;
  arma::mat X;
  data(y, X, 60);
  const int seen = max_blas_threads_during(
      [&] { Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4"); });
  INFO("OPENBLAS_NUM_THREADS " << loaded << ", budget " << Optim::get_num_threads() << ", OpenBLAS max seen " << seen);
  CHECK(seen <= loaded);
#ifndef _OPENMP
  CHECK(Optim::get_num_threads() >= 1);
#endif
}
