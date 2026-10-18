// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

// Thread-budget benchmark of the methods other than the objective
// (docs/dev/ThreadBudget-TODO.md, step 1): times one predict / simulate /
// update call of a Kriging model with fixed theta under the BLAS / OpenMP
// thread counts of the environment, and prints one CSV row (median over the reps).
//
//   bench-threads-ops <op> <n> <d> <n_new> [nsim=1] [reps=7] [--header]
//
// op: predict (mean + stdev), predict_cov (+ covariance), simulate,
// update (no refit, n_new points added; the model copy is not timed).

#include "libKriging/Kriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(_MSC_VER) && (!defined(__APPLE__) || !defined(__arm64__))
extern "C" {
int openblas_get_num_threads() __attribute__((weak));
int openblas_get_parallel() __attribute__((weak));
}
#define LK_BENCH_OPENBLAS 1
#endif

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double median(std::vector<double> t) {
  std::nth_element(t.begin(), t.begin() + t.size() / 2, t.end());
  return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <predict|predict_cov|simulate|update> <n> <d> <n_new> [nsim=1] [reps=7] [--header]\n",
                 argv[0]);
    return 2;
  }
  const std::string op = argv[1];
  const arma::uword n = std::strtoul(argv[2], nullptr, 10);
  const arma::uword d = std::strtoul(argv[3], nullptr, 10);
  const arma::uword n_new = std::strtoul(argv[4], nullptr, 10);
  const int nsim = argc > 5 && argv[5][0] != '-' ? std::atoi(argv[5]) : 1;
  const int reps = argc > 6 && argv[6][0] != '-' ? std::atoi(argv[6]) : 7;
  bool header = false;
  for (int i = 1; i < argc; ++i)
    header = header || std::strcmp(argv[i], "--header") == 0;
  if (header) {
    std::printf("op,n,d,n_new,nsim,blas_kind,blas_threads,omp_threads,reps,t_ms\n");
    return 0;
  }

  int blas_threads = -1, blas_kind = -1, omp_threads = -1;
#ifdef LK_BENCH_OPENBLAS
  if (openblas_get_num_threads)
    blas_threads = openblas_get_num_threads();
  if (openblas_get_parallel)
    blas_kind = openblas_get_parallel();  // 0 sequential, 1 pthreads, 2 OpenMP
#endif
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif

  arma::arma_rng::set_seed(123);
  arma::mat X(n, d, arma::fill::randu);
  arma::vec y = arma::sum(arma::sin(3 * X), 1);
  arma::mat X_new(n_new, d, arma::fill::randu);
  arma::vec y_new = arma::sum(arma::sin(3 * X_new), 1);
  arma::vec theta(d);
  for (arma::uword k = 0; k < d; ++k)
    theta(k) = 0.3 + 0.6 * k / std::max<arma::uword>(1, d - 1);

  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", par);

  std::vector<double> t(reps);
  for (int r = 0; r < reps; ++r) {
    if (op == "predict") {
      auto t0 = std::chrono::steady_clock::now();
      k.predict(X_new, true, false, false);
      t[r] = elapsed_ms(t0);
    } else if (op == "predict_cov") {
      auto t0 = std::chrono::steady_clock::now();
      k.predict(X_new, true, true, false);
      t[r] = elapsed_ms(t0);
    } else if (op == "simulate") {
      auto t0 = std::chrono::steady_clock::now();
      k.simulate(nsim, 123 + r, X_new);
      t[r] = elapsed_ms(t0);
    } else if (op == "update") {
      Kriging ku(k, ExplicitCopySpecifier{});
      auto t0 = std::chrono::steady_clock::now();
      ku.update(y_new, X_new, false);
      t[r] = elapsed_ms(t0);
    } else {
      std::fprintf(stderr, "unknown op %s\n", op.c_str());
      return 2;
    }
  }

  std::printf("%s,%llu,%llu,%llu,%d,%d,%d,%d,%d,%.4f\n",
              op.c_str(),
              static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(d),
              static_cast<unsigned long long>(n_new),
              nsim,
              blas_kind,
              blas_threads,
              omp_threads,
              reps,
              median(t));
  return 0;
}
