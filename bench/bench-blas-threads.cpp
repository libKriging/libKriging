// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

// Fixed-parameter micro-benchmark for the BLAS thread-count study
// (REPORT-blas-threads.md). Unlike a fit, whose time depends on the
// optimizer path (itself sensitive to the rounding differences that the BLAS
// thread count induces), each operation here is evaluated at fixed
// hyperparameters, so timings at different thread counts are comparable.
//
// Usage: bench-blas-threads <op> <n> <d> [min_seconds=1.5]
//   op: kriging_ll | mlp_ll | warp_ll | kriging_predict | mlp_predict | set_threads
//    or: bench-blas-threads fit <class> <optim> <n> <d> <seed>  (one timed fit, see run_fit)
// Prints one CSV line: op,n,d,blas_threads,omp_threads,calls,ms_per_call,cpu_ms_per_call
// The BLAS thread count is the one OpenBLAS picked up at start-up (set
// OPENBLAS_NUM_THREADS in the environment of each run).

#include "libKriging/Kriging.hpp"
#include "libKriging/MLPKriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
// Weak: resolved to null when the BLAS backend is not OpenBLAS.
extern "C" {
int openblas_get_num_threads(void) __attribute__((weak));
void openblas_set_num_threads(int) __attribute__((weak));
}
#else
static int (*const openblas_get_num_threads)(void) = nullptr;
static void (*const openblas_set_num_threads)(int) = nullptr;
#endif

using libKriging::MLPKriging;
using libKriging::WarpKriging;

namespace {

using Clock = std::chrono::steady_clock;

// Process CPU time (user + system, all threads), in seconds; 0 if unknown.
double cpu_seconds() {
#if defined(__unix__) || defined(__APPLE__)
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + 1e-6 * ru.ru_utime.tv_usec + ru.ru_stime.tv_sec + 1e-6 * ru.ru_stime.tv_usec;
#else
  return 0.0;
#endif
}

struct Timing {
  double ms_per_call;      // median over 3 blocks of the wall time per call
  double cpu_ms_per_call;  // CPU (user + sys) time per call, over all blocks
  long calls;
};

// Median, over 3 blocks of >= min_seconds/3 each, of the mean time per call.
Timing time_op(const std::function<void()>& op, double min_seconds) {
  op();  // warm-up (allocations, caches)
  std::vector<double> block_ms;
  long total = 0;
  const double cpu0 = cpu_seconds();
  for (int b = 0; b < 3; ++b) {
    long calls = 0;
    auto t0 = Clock::now();
    double elapsed = 0.0;
    do {
      op();
      ++calls;
      elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    } while (elapsed < min_seconds / 3.0 || calls < 2);
    block_ms.push_back(1e3 * elapsed / calls);
    total += calls;
  }
  const double cpu_ms = 1e3 * (cpu_seconds() - cpu0) / total;
  std::sort(block_ms.begin(), block_ms.end());
  return {block_ms[1], cpu_ms, total};
}

// LK_BENCH_WARMUP_MS (default 300): busy-wait before measuring, as in
// bench-threads: OpenBLAS pthreads threads spin for ~100 ms after the library
// is loaded (OPENBLAS_THREAD_TIMEOUT), and a measure taken inside that window
// sees them as contention (bench/thread-budget/RESULTS.md, section 9).
void bench_warmup() {
  const char* v = std::getenv("LK_BENCH_WARMUP_MS");
  const int ms = v ? std::atoi(v) : 300;
  const auto end = Clock::now() + std::chrono::milliseconds(ms);
  volatile double x = 0;
  while (Clock::now() < end)
    x = x + 1;
}

double f(const arma::rowvec& x) {
  double s = 0.0;
  for (arma::uword i = 0; i < x.n_elem; ++i)
    s += std::sin(2.0 * M_PI * x(i)) + 0.5 * x(i) * x(i);
  return s;
}

}  // namespace

// Fit mode: one fit on seeded data, prints
// fit,class,optim,n,d,seed,blas_threads_before,blas_threads_after,seconds,cpu_seconds,ll
int run_fit(int argc, char** argv) {
  if (argc < 7) {
    std::cerr << "usage: " << argv[0] << " fit <kriging|nugget|warp_kuma|warp_mlp|mlp> <optim> <n> <d> <seed>"
              << std::endl;
    return 1;
  }
  const std::string cls = argv[2], optim = argv[3];
  const arma::uword n = std::stoul(argv[4]), d = std::stoul(argv[5]);
  const unsigned int seed = std::stoul(argv[6]);

  arma::arma_rng::set_seed(seed);
  arma::mat X(n, d, arma::fill::randu);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f(X.row(i));
  if (cls == "nugget")
    y += 0.05 * arma::randn<arma::vec>(n);
  Random::reset_seed(seed);

  const std::map<std::string, std::string> no_calibration = {{"calibration_folds", "0"}};
  const int before = openblas_get_num_threads ? openblas_get_num_threads() : -1;
  double ll = 0.0;
  bench_warmup();
  const double cpu0 = cpu_seconds();
  auto t0 = Clock::now();
  if (cls == "kriging") {
    Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, optim, "LL");
    ll = k.logLikelihood();
  } else if (cls == "nugget") {
    Kriging k("matern5_2", Kriging::NoiseModel::Nugget);
    k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL");
    ll = k.logLikelihood();
  } else if (cls == "warp_kuma" || cls == "warp_mlp") {
    WarpKriging w(y,
                  X,
                  std::vector<std::string>(d, cls == "warp_kuma" ? "kumaraswamy" : "mlp"),
                  "matern5_2",
                  Trend::RegressionModel::Constant,
                  false,
                  optim,
                  "LL",
                  no_calibration);
    ll = w.logLikelihood();
  } else if (cls == "mlp") {
    MLPKriging m(
        y, X, {16, 8}, 2, "selu", "matern5_2", Trend::RegressionModel::Constant, false, optim, "LL", no_calibration);
    ll = m.logLikelihood();
  } else {
    std::cerr << "unknown class: " << cls << std::endl;
    return 1;
  }
  const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  const double cpu = cpu_seconds() - cpu0;
  const int after = openblas_get_num_threads ? openblas_get_num_threads() : -1;
  std::printf("fit,%s,%s,%llu,%llu,%u,%d,%d,%.4f,%.4f,%.6g\n",
              cls.c_str(),
              optim.c_str(),
              (unsigned long long)n,
              (unsigned long long)d,
              seed,
              before,
              after,
              seconds,
              cpu,
              ll);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "fit")
    return run_fit(argc, argv);
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <op> <n> <d> [min_seconds=1.5]" << std::endl;
    return 1;
  }
  const std::string op = argv[1];
  const arma::uword n = std::stoul(argv[2]);
  const arma::uword d = std::stoul(argv[3]);
  const double min_seconds = argc > 4 ? std::stod(argv[4]) : 1.5;

  arma::arma_rng::set_seed(123);
  arma::mat X(n, d, arma::fill::randu);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = f(X.row(i));
  arma::mat Xp(1000, d, arma::fill::randu);

  const int blas_threads = openblas_get_num_threads ? openblas_get_num_threads() : -1;
  int omp_threads = -1;
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif

  const std::map<std::string, std::string> no_calibration = {{"calibration_folds", "0"}};
  std::function<void()> body;

  // The models must outlive `body`.
  Kriging kr("matern5_2");
  MLPKriging mlp({16, 8}, 2, "selu", "matern5_2");
  WarpKriging wk(std::vector<std::string>(d, "kumaraswamy"), "matern5_2");
  arma::vec theta;

  if (op == "kriging_ll" || op == "kriging_predict") {
    Kriging::Parameters p;
    p.theta = arma::mat(1, d, arma::fill::value(0.3));
    p.is_theta_estim = false;
    kr.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", p);
    theta = arma::vec(d, arma::fill::value(0.3));
    if (op == "kriging_ll")
      body = [&] { kr.logLikelihoodFun(theta, true, false); };
    else
      body = [&] { kr.predict(Xp, true, false, false); };
  } else if (op == "mlp_ll" || op == "mlp_predict") {
    mlp.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", no_calibration);
    theta = mlp.theta();
    if (op == "mlp_ll")
      body = [&] { mlp.logLikelihoodFun(theta, true); };
    else
      body = [&] { mlp.predict(Xp, true, false, false); };
  } else if (op == "warp_ll") {
    wk.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", no_calibration);
    theta = wk.theta();
    body = [&] { wk.logLikelihoodFun(theta, true); };
  } else if (op == "set_threads") {
    // Cost of switching the OpenBLAS thread count back and forth (the pool
    // is resized lazily: this times the call itself, then one tiny dgemm
    // that forces the pool to be used at the new size).
    if (!openblas_set_num_threads) {
      std::cerr << "openblas_set_num_threads not available" << std::endl;
      return 2;
    }
    const int nt = std::max(1, blas_threads);
    arma::mat A(n, n, arma::fill::randu), C;
    bool toggle = false;
    body = [&] {
      toggle = !toggle;
      openblas_set_num_threads(toggle ? 1 : nt);
      C = A * A;
    };
  } else {
    std::cerr << "unknown op: " << op << std::endl;
    return 1;
  }

  bench_warmup();
  const Timing t = time_op(body, min_seconds);
  std::printf("%s,%llu,%llu,%d,%d,%ld,%.6g,%.6g\n",
              op.c_str(),
              (unsigned long long)n,
              (unsigned long long)d,
              blas_threads,
              omp_threads,
              t.calls,
              t.ms_per_call,
              t.cpu_ms_per_call);
  return 0;
}
