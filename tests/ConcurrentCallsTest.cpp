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
#include <ostream>
#include <string>
#include <thread>
#include <vector>
// clang-format on

#if !defined(_WIN32) && defined(_POSIX_VERSION)
#include <sys/wait.h>
#include <unistd.h>
#include "test_fork_exit.hpp"
#define LK_TEST_FORK 1
#endif

#if defined(__SANITIZE_THREAD__)
#define LK_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define LK_TEST_TSAN 1
#endif
#endif

// OpenBLAS found the way the library finds it (Kriging.cpp): weak symbols with
// GCC / Clang, GetProcAddress on libopenblas.dll / openblas.dll with MSVC. The
// full-budget case relies on it: a count the library can change but the test
// cannot read would make it require bit-identical results (CI, MSVC with
// conda's openblas.dll: count read as 0 while the library set it; T6 log).
#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace {
FARPROC openblas_proc(const char* name) {
  HMODULE m = GetModuleHandleA("libopenblas.dll");
  if (!m)
    m = GetModuleHandleA("openblas.dll");
  return m ? GetProcAddress(m, name) : nullptr;
}
using openblas_get_t = int (*)();
using openblas_set_t = void (*)(int);
// resolved at the first call, as the library does
openblas_get_t openblas_get() {
  static const auto f = reinterpret_cast<openblas_get_t>(openblas_proc("openblas_get_num_threads"));
  return f;
}
openblas_set_t openblas_set() {
  static const auto f = reinterpret_cast<openblas_set_t>(openblas_proc("openblas_set_num_threads"));
  return f;
}
}  // namespace
#define LK_TEST_OPENBLAS 1
#elif !defined(__APPLE__) || !defined(__arm64__)
extern "C" {
void openblas_set_num_threads(int num_threads) __attribute__((weak));
int openblas_get_num_threads() __attribute__((weak));
}
namespace {
using openblas_get_t = int (*)();
using openblas_set_t = void (*)(int);
openblas_get_t openblas_get() {
  return openblas_get_num_threads;
}
openblas_set_t openblas_set() {
  return openblas_set_num_threads;
}
}  // namespace
#define LK_TEST_OPENBLAS 1
#endif

// MKL found the way the library finds it (Kriging.cpp): weak symbol on ELF,
// GetProcAddress on mkl_rt.2.dll / mkl_rt.dll / mkl_rt.1.dll on Windows (MSVC
// and MinGW), dlsym elsewhere on POSIX. The library sets the MKL count of each
// call / worker wherever it finds MKL, so the test must see MKL there too, or
// it would take the count for fixed and require bit-identical results (T9).
#if defined(__ELF__)
extern "C" {
int MKL_Get_Max_Threads() __attribute__((weak));
}
namespace {
using mkl_get_t = int (*)();
mkl_get_t mkl_get() {
  return MKL_Get_Max_Threads;
}
}  // namespace
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace {
using mkl_get_t = int (*)();
mkl_get_t mkl_get() {
  static const mkl_get_t f = [] {
    for (const char* dll : {"mkl_rt.2.dll", "mkl_rt.dll", "mkl_rt.1.dll"})
      if (HMODULE m = GetModuleHandleA(dll))
        return reinterpret_cast<mkl_get_t>(GetProcAddress(m, "MKL_Get_Max_Threads"));
    return mkl_get_t{nullptr};
  }();
  return f;
}
}  // namespace
#elif defined(__APPLE__) || defined(__unix__)
#include <dlfcn.h>
namespace {
using mkl_get_t = int (*)();
mkl_get_t mkl_get() {
  static const mkl_get_t f = reinterpret_cast<mkl_get_t>(dlsym(RTLD_DEFAULT, "MKL_Get_Max_Threads"));
  return f;
}
}  // namespace
#else
namespace {
using mkl_get_t = int (*)();
mkl_get_t mkl_get() {
  return nullptr;
}
}  // namespace
#endif

// Public fit / predict / simulate calls from several user threads at once
// (Python ThreadPoolExecutor, joblib threads, std::thread): no crash, the
// results of the same calls run one after the other, and the caller's
// OpenBLAS count back at the end (docs/dev/rationalization/T6-thread-safety.md).

namespace {

int blas_count() {
#ifdef LK_TEST_OPENBLAS
  if (const auto get = openblas_get())
    return get();
#endif
  return 0;
}

// MKL count of the calling thread, 0 without MKL
int mkl_count() {
  if (const auto get = mkl_get())
    return get();
  return 0;
}

void set_blas_count(int n) {
#ifdef LK_TEST_OPENBLAS
  const auto set = openblas_set();
  if (set && n > 0)
    set(n);
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

// Relative differences between two results, per compared quantity
struct Gaps {
  double theta = 0, sigma2 = 0, objective = 0, mean = 0, stdev = 0, sims = 0;
  double max() const { return std::max({theta, sigma2, objective, mean, stdev, sims}); }
  void keep_worst(const Gaps& g) {
    theta = std::max(theta, g.theta);
    sigma2 = std::max(sigma2, g.sigma2);
    objective = std::max(objective, g.objective);
    mean = std::max(mean, g.mean);
    stdev = std::max(stdev, g.stdev);
    sims = std::max(sims, g.sims);
  }
};

std::ostream& operator<<(std::ostream& os, const Gaps& g) {
  return os << "theta " << g.theta << ", sigma2 " << g.sigma2 << ", objective " << g.objective << ", mean " << g.mean
            << ", stdev " << g.stdev << ", sims " << g.sims;
}

Gaps rel_diff(const Result& a, const Result& b) {
  auto rd = [](const arma::mat& u, const arma::mat& v) {
    if (u.n_elem != v.n_elem)
      return 1.0;
    if (u.n_elem == 0)
      return 0.0;
    return arma::abs(u - v).max() / std::max(1e-300, arma::abs(v).max());
  };
  Gaps g;
  g.theta = rd(a.theta, b.theta);
  g.sigma2 = std::abs(a.sigma2 - b.sigma2) / std::abs(b.sigma2);
  g.objective = std::abs(a.objective - b.objective) / std::max(1.0, std::abs(b.objective));
  g.mean = rd(a.mean, b.mean);
  g.stdev = rd(a.stdev, b.stdev);
  g.sims = rd(a.sims, b.sims);
  return g;
}

}  // namespace

namespace {

// Runs every job sequentially (reference), then from 6 user threads x 3 rounds,
// at the given thread budget; checks every compared quantity against `tol`
// and returns the largest relative differences to the reference.
Gaps concurrent_vs_sequential(double budget_percent, double tol) {
  const double saved_percent = Optim::get_num_threads_percent();
  Optim::set_num_threads_percent(budget_percent);
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
  Gaps worst;
  for (int t = 0; t < n_threads; ++t)
    for (int r = 0; r < n_rounds; ++r)
      for (size_t i = 0; i < jobs.size(); ++i) {
        const Gaps g = rel_diff(got[t][r * jobs.size() + i], ref[i]);
        INFO("budget " << budget_percent << " %, thread " << t << ", round " << r << ", job " << i << " ("
                       << jobs[i].kernel << " " << jobs[i].optim << " " << jobs[i].objective
                       << "): relative differences " << g);
        CHECK(g.theta <= tol);
        CHECK(g.sigma2 <= tol);
        CHECK(g.objective <= tol);
        CHECK(g.mean <= tol);
        CHECK(g.stdev <= tol);
        CHECK(g.sims <= tol);
        worst.keep_worst(g);
      }
  WARN("budget " << budget_percent << " %: largest relative differences to the sequential results: " << worst
                 << " (tolerance " << tol << ")");
  // the count the library sets is a share of the budget, never more than the
  // count OpenBLAS had when the library first used it
  INFO("OpenBLAS count: caller " << caller << ", max seen " << max_seen.load() << ", initial " << initial);
  CHECK(max_seen.load() <= std::max(initial, caller));
  CHECK(blas_count() == (initial > 0 ? caller : 0));
  set_blas_count(initial);
  Optim::set_num_threads_percent(saved_percent);
  return worst;
}

}  // namespace

// Budget of 1 thread: the sequential and the concurrent calls all run their
// BLAS (OpenBLAS, MKL; Accelerate is never changed) and OpenMP loops at 1
// thread, so the results must be bit-identical on every backend: what remains
// is the concurrency itself (shared state, RNG, count save / restore).
TEST_CASE("Concurrent calls at a budget of 1 thread give the sequential results", "[threads][concurrent]") {
  concurrent_vs_sequential(1e-6, 0.0);
}

TEST_CASE("Concurrent fit / predict / simulate from user threads", "[threads][concurrent]") {
  // Concurrent calls share the budget, so their BLAS count is lower than the
  // one of the sequential calls, which changes the rounding of the BLAS sums
  // (FAQ, reproducibility); a fit amplifies it through the optimizer's
  // stopping point in the flat part of the likelihood. Bit-identical results
  // when no count can change (no OpenBLAS / MKL, or loaded with 1 thread: the
  // ctest variant ConcurrentCallsTest/OPENBLAS_NUM_THREADS=1), else 2e-3
  // relative on every compared quantity (decision D2, docs/dev/Rationalization.md
  // section 3; 1e-4 until #473). Largest difference measured over all of them
  // (theta, sigma2, objective, predictions, simulations): 4e-14 (Intel Xeon 4
  // cores), 4.6e-7 to 7.0e-7 (CI runners: Linux Coverage, Octave Linux, Windows),
  // 6.6e-7 (AMD EPYC 9V74 2 cores, gauss single start), 7.2e-5 (i9-10900K 10
  // cores, Linux, matern5_2 BFGS4); 8.1e-4 (same i9-10900K, Windows MSVC,
  // OpenBLAS 0.3.33 loaded with 20 threads, matern5_2 BFGS4: sigma2 8.1e-4,
  // stdev 2.1e-4, theta 1.9e-4, objective 2.5e-9, deterministic; #473); T6 / T9
  // logs. Races (RNG, shared state) give differences
  // of order 1 (other starts), and are checked bit for bit by the budget-1 case
  // above.
  const bool count_can_change = blas_count() > 1 || mkl_count() > 1;
  WARN("BLAS counts read by the test: OpenBLAS "
       << blas_count() << ", MKL " << mkl_count()
       << (count_can_change ? ": tolerance 2e-3" : ": bit-identical required"));
  concurrent_vs_sequential(100, count_can_change ? 2e-3 : 0.0);
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
// Tagged [fork]: excluded from the ASan / LSan builds by tests/CMakeLists.txt
// (their allocator is not fork-safe in a multithreaded process: T6 log).
TEST_CASE("fork() during concurrent fits", "[threads][concurrent][fork]") {
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
    // The child's verdict comes back through a pipe, not its exit status:
    // under valgrind the exit status is valgrind's (--error-exitcode), whose
    // leak check in the child counts the objects of the parent's threads,
    // which do not exist there.
    int fds[2];
    REQUIRE(pipe(fds) == 0);
    const pid_t pid = fork();
    if (pid == 0) {
      close(fds[0]);
      // alarm: a deadlock in the child fails the test instead of hanging it
      alarm(60);
      char verdict = blas_count() > 1 ? 2 : 0;
      if (verdict == 0) {
        Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS2");
        verdict = !k.theta().is_finite() ? 3 : (blas_count() > 1 ? 4 : 0);
      }
      (void)!write(fds[1], &verdict, 1);
      lk_test_fork_child_exit(verdict);
    }
    REQUIRE(pid > 0);
    close(fds[1]);
    char verdict = -1;
    const bool got = read(fds[0], &verdict, 1) == 1;  // 0 bytes: the child died first
    close(fds[0]);
    int status = 0;
    REQUIRE(waitpid(pid, &status, 0) == pid);
    if (!got || verdict != 0) {
      ++failures;
      WARN("child " << f << ": " << (got ? "verdict " + std::to_string(verdict) : std::string("no verdict")) << ", "
                    << (WIFEXITED(status) ? "exit " : "signal ")
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
