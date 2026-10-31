// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
// clang-format on

#if !defined(_WIN32) && defined(_POSIX_VERSION)
#include <sys/wait.h>
#include <unistd.h>
#define LK_TEST_FORK 1
#endif

#if defined(__SANITIZE_THREAD__)
#define LK_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define LK_TEST_TSAN 1
#endif
#endif

#if !defined(_MSC_VER) && (!defined(__APPLE__) || !defined(__arm64__))
extern "C" {
void openblas_set_num_threads(int num_threads) __attribute__((weak));
int openblas_get_num_threads() __attribute__((weak));
}
#define LK_TEST_OPENBLAS 1
#endif

// Public fit / predict / simulate calls from several user threads at once
// (Python ThreadPoolExecutor, joblib threads, std::thread): no crash, the
// results of the same calls run one after the other, and the caller's
// OpenBLAS count back at the end (docs/dev/rationalization/T6-thread-safety.md).

namespace {

int blas_count() {
#ifdef LK_TEST_OPENBLAS
  if (openblas_get_num_threads)
    return openblas_get_num_threads();
#endif
  return 0;
}

void set_blas_count(int n) {
#ifdef LK_TEST_OPENBLAS
  if (openblas_set_num_threads && n > 0)
    openblas_set_num_threads(n);
#endif
}

struct Job {
  arma::vec y;
  arma::mat X;
  std::string kernel;
  Kriging::NoiseModel noise;
  std::string optim;
  std::string objective;
};

struct Result {
  arma::vec theta;
  double sigma2 = 0;
  double objective = 0;
  arma::vec mean;
  arma::vec stdev;
  arma::mat sims;
};

// Drawn in the main thread before any call
std::vector<Job> make_jobs(const arma::mat& Xn) {
  std::vector<Job> jobs;
  const struct {
    const char* kernel;
    Kriging::NoiseModel noise;
    const char* optim;
    const char* objective;
  } configs[] = {
      {"gauss", Kriging::NoiseModel::None, "BFGS", "LL"},
      {"matern5_2", Kriging::NoiseModel::None, "BFGS4", "LL"},
      {"matern3_2", Kriging::NoiseModel::None, "BFGS", "LOO"},
      {"matern5_2", Kriging::NoiseModel::Nugget, "BFGS4", "LL"},
      {"gauss", Kriging::NoiseModel::None, "BFGS2", "LMP"},
  };
  arma::arma_rng::set_seed(1234);
  for (const auto& c : configs) {
    Job j;
    j.X = arma::randu(50, Xn.n_cols);
    j.y = arma::sum(arma::sin(3 * j.X), 1);
    if (c.noise == Kriging::NoiseModel::Nugget)
      j.y += 0.05 * arma::randn(j.y.n_elem);
    j.kernel = c.kernel;
    j.noise = c.noise;
    j.optim = c.optim;
    j.objective = c.objective;
    jobs.push_back(j);
  }
  return jobs;
}

Result run(const Job& j, const arma::mat& Xn, int seed) {
  // simulate() reseeds the thread's engine, which then draws the next fit's
  // starts: reset it so that a job does not depend on the one run before
  Random::reset_seed(123);
  Kriging k(j.kernel, j.noise);
  k.fit(j.y, j.X, Trend::RegressionModel::Constant, false, j.optim, j.objective);
  Result r;
  r.theta = k.theta();
  r.sigma2 = k.sigma2();
  r.objective = j.objective == "LL" ? k.logLikelihood() : (j.objective == "LOO" ? k.leaveOneOut() : k.logMargPost());
  auto [mean, stdev, cov, dmean, dstdev] = k.predict(Xn, true, false, false);
  r.mean = mean;
  r.stdev = stdev;
  r.sims = k.simulate(3, seed, Xn.head_rows(20));
  return r;
}

// Largest relative difference between two results
double rel_diff(const Result& a, const Result& b) {
  auto rd = [](const arma::mat& u, const arma::mat& v) {
    if (u.n_elem != v.n_elem)
      return 1.0;
    if (u.n_elem == 0)
      return 0.0;
    return arma::abs(u - v).max() / std::max(1e-300, arma::abs(v).max());
  };
  double d = 0;
  d = std::max(d, rd(a.theta, b.theta));
  d = std::max(d, std::abs(a.sigma2 - b.sigma2) / std::abs(b.sigma2));
  d = std::max(d, std::abs(a.objective - b.objective) / std::max(1.0, std::abs(b.objective)));
  d = std::max(d, rd(a.mean, b.mean));
  d = std::max(d, rd(a.stdev, b.stdev));
  d = std::max(d, rd(a.sims, b.sims));
  return d;
}

}  // namespace

TEST_CASE("Concurrent fit / predict / simulate from user threads", "[threads][concurrent]") {
  const int initial = blas_count();
  // the caller's own count: a known value, restored by the library at the end
  const int caller = std::max(1, std::min(3, initial));
  set_blas_count(caller);

  arma::arma_rng::set_seed(42);
  const arma::mat Xn = arma::randu(200, 2);
  const std::vector<Job> jobs = make_jobs(Xn);

  // sequential reference
  std::vector<Result> ref;
  for (size_t i = 0; i < jobs.size(); ++i)
    ref.push_back(run(jobs[i], Xn, static_cast<int>(i)));
  REQUIRE(blas_count() == (initial > 0 ? caller : 0));

#ifdef LK_TEST_TSAN
  const int n_threads = 3, n_rounds = 1;
#else
  const int n_threads = 6, n_rounds = 3;
#endif
  // every thread runs every job, each from a different one
  std::vector<std::vector<Result>> got(n_threads, std::vector<Result>(jobs.size() * n_rounds));
  std::vector<std::string> errors(n_threads);
  std::atomic<bool> done{false};
  std::atomic<int> max_seen{0};
  std::thread monitor([&] {
    while (!done.load()) {
      const int b = blas_count();
      if (b > max_seen.load())
        max_seen.store(b);
      std::this_thread::yield();
    }
  });
  std::vector<std::thread> threads;
  for (int t = 0; t < n_threads; ++t)
    threads.emplace_back([&, t] {
      try {
        for (int r = 0; r < n_rounds; ++r)
          for (size_t k = 0; k < jobs.size(); ++k) {
            const size_t i = (k + t) % jobs.size();
            got[t][r * jobs.size() + i] = run(jobs[i], Xn, static_cast<int>(i));
          }
      } catch (const std::exception& e) {
        errors[t] = e.what();
      }
    });
  for (auto& th : threads)
    th.join();
  done.store(true);
  monitor.join();

  for (int t = 0; t < n_threads; ++t)
    CHECK(errors[t].empty());
  // Concurrent calls share the budget, so their OpenBLAS count is lower than
  // the one of the sequential calls, which changes the rounding of the BLAS
  // sums (FAQ, reproducibility): bit-identical results when the count cannot
  // change (no OpenBLAS, or loaded with 1 thread: the ctest variant
  // ConcurrentCallsTest/OPENBLAS_NUM_THREADS=1), else up to rounding (4e-14
  // measured).
  const double tol = initial > 1 ? 1e-10 : 0.0;
  double worst = 0;
  for (int t = 0; t < n_threads; ++t)
    for (int r = 0; r < n_rounds; ++r)
      for (size_t i = 0; i < jobs.size(); ++i) {
        const double d = rel_diff(got[t][r * jobs.size() + i], ref[i]);
        INFO("thread " << t << ", round " << r << ", job " << i << " (" << jobs[i].kernel << " " << jobs[i].optim << " "
                       << jobs[i].objective << "): relative difference " << d);
        CHECK(d <= tol);
        worst = std::max(worst, d);
      }
  WARN("largest relative difference to the sequential results: " << worst << " (tolerance " << tol << ")");
  // the count the library sets is a share of the budget, never more than the
  // count OpenBLAS had when the library first used it
  INFO("OpenBLAS count: caller " << caller << ", max seen " << max_seen.load() << ", initial " << initial);
  CHECK(max_seen.load() <= std::max(initial, caller));
  CHECK(blas_count() == (initial > 0 ? caller : 0));
  set_blas_count(initial);
}

TEST_CASE("Optim::get_num_threads is not divided between concurrent calls", "[threads][concurrent]") {
  const int full = Optim::get_num_threads();
  std::atomic<bool> stop{false};
  std::atomic<int> seen_min{full};
  arma::arma_rng::set_seed(7);
  const arma::mat X = arma::randu(80, 2);
  const arma::vec y = arma::sum(arma::cos(2 * X), 1);
  std::thread worker([&] {
    while (!stop.load())
      Kriging k(y, X, "gauss", Trend::RegressionModel::Constant, false, "BFGS");
  });
  for (int i = 0; i < 50; ++i) {
    seen_min.store(std::min(seen_min.load(), Optim::get_num_threads()));
    std::this_thread::yield();
  }
  stop.store(true);
  worker.join();
  CHECK(seen_min.load() == full);
}

#if defined(LK_TEST_FORK) && !defined(LK_TEST_TSAN)
// fork() from one thread while others fit: the child runs a fit of its own
// (no lock held by a thread that does not exist in the child, no scope of
// those threads left in effect), the parent gets its count back.
TEST_CASE("fork() during concurrent fits", "[threads][concurrent]") {
  const int initial = blas_count();
  const int caller = std::max(1, std::min(3, initial));
  set_blas_count(caller);
  arma::arma_rng::set_seed(5);
  const arma::mat X = arma::randu(60, 2);
  const arma::vec y = arma::sum(arma::sin(3 * X), 1);
  std::atomic<bool> stop{false};
  std::vector<std::thread> workers;
  for (int t = 0; t < 3; ++t)
    workers.emplace_back([&] {
      while (!stop.load())
        Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4");
    });
  int failures = 0;
  for (int f = 0; f < 5; ++f) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const pid_t pid = fork();
    if (pid == 0) {
      // alarm: a deadlock in the child fails the test instead of hanging it
      alarm(60);
      if (blas_count() > 1)
        _exit(2);
      Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS2");
      if (!k.theta().is_finite())
        _exit(3);
      _exit(blas_count() <= 1 ? 0 : 4);
    }
    REQUIRE(pid > 0);
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++failures;
      WARN("child " << f << ": " << (WIFEXITED(status) ? "exit " : "signal ")
                    << (WIFEXITED(status) ? WEXITSTATUS(status) : WTERMSIG(status)));
    }
  }
  stop.store(true);
  for (auto& w : workers)
    w.join();
  CHECK(failures == 0);
  CHECK(blas_count() == (initial > 0 ? caller : 0));
  set_blas_count(initial);
}
#endif
