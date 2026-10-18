// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

// Thread-budget benchmark (docs/dev/ThreadBudget-TODO.md, step 1): times the
// phases of one exact log-likelihood evaluation under the BLAS / OpenMP thread
// counts of the environment, and prints one CSV row (medians over the reps).
//
//   bench-threads <n> <d> [reps=7] [kernel=matern5_2] [--header]
//
// Thread counts are read back from the libraries (OPENBLAS_NUM_THREADS /
// OMP_NUM_THREADS are set by bench-threads.sh), so that a row describes the
// configuration it was measured in.

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
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

template <typename F>
double median_ms(int reps, F&& f) {
  std::vector<double> t(reps);
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    t[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::nth_element(t.begin(), t.begin() + reps / 2, t.end());
  return t[reps / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <n> <d> [reps=7] [kernel=matern5_2] [--header]\n", argv[0]);
    return 2;
  }
  const arma::uword n = std::strtoul(argv[1], nullptr, 10);
  const arma::uword d = std::strtoul(argv[2], nullptr, 10);
  const int reps = argc > 3 ? std::atoi(argv[3]) : 7;
  const std::string kernel = argc > 4 ? argv[4] : "matern5_2";
  bool header = false;
  for (int i = 1; i < argc; ++i)
    header = header || std::strcmp(argv[i], "--header") == 0;

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
  arma::vec theta(d);
  for (arma::uword k = 0; k < d; ++k)
    theta(k) = 0.3 + 0.6 * k / std::max<arma::uword>(1, d - 1);

  const auto cov = Covariance::resolve(kernel).Cov;
  arma::mat dX = LinearAlgebra::compute_dX(X);
  arma::mat R(n, n, arma::fill::none);

  const double t_dX = median_ms(reps, [&] { dX = LinearAlgebra::compute_dX(X); });
  const double t_cholcov = median_ms(reps, [&] { LinearAlgebra::cholCov(&R, dX, theta, cov, 1.0, arma::vec()); });
  const double t_chol = median_ms(reps, [&] { arma::mat L = LinearAlgebra::safe_chol_lower(R); });

  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;
  Kriging k(y, X, kernel, Trend::RegressionModel::Constant, false, "none", "LL", par);
  const double t_ll = median_ms(reps, [&] { k.logLikelihoodFun(theta, false, false); });
  const double t_llgrad = median_ms(reps, [&] { k.logLikelihoodFun(theta, true, false); });

  if (header)
    std::printf(
        "n,d,kernel,blas_kind,blas_threads,omp_threads,reps,t_dX_ms,t_cholCov_ms,t_chol_ms,t_cov_ms,t_ll_ms,t_llgrad_"
        "ms\n");
  std::printf("%llu,%llu,%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              static_cast<unsigned long long>(n),
              static_cast<unsigned long long>(d),
              kernel.c_str(),
              blas_kind,
              blas_threads,
              omp_threads,
              reps,
              t_dX,
              t_cholcov,
              t_chol,
              std::max(0.0, t_cholcov - t_chol),
              t_ll,
              t_llgrad);
  return 0;
}
