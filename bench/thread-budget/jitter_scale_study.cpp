// Study driver (not built by CMake, see fit_repro_study.cpp for the build):
// the fixed Cholesky jitter (current default: kernel jitter for the
// objective, <= num_nugget for the committed model) against no jitter, on a
// wider grid. Each fit is run with Optim::set_num_threads_percent 10 / 100 / 200.
// Usage: jitter_scale_study <family> <kernel> <n> <d> <sets> <optim> [noise] [jitter]
//   family: sin (sum_j sin(2 pi f_j x_j), f_j ~ U[0.5, 3]), branin (d = 2),
//           hartmann6 (d = 6)
//   noise: observation noise sd relative to sd(y) (default 0), the fit is
//          still noise-free (interpolating slightly noisy data)
//   jitter: set_chol_jitter value of the "on" fits (default -1: per kernel)
// Output CSV: set,jitter,pct,theta0,ll,rmse,ms
// See docs/dev/CholJitter.md, values per kernel.
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

static arma::vec f(const std::string& family, const arma::mat& X, const arma::rowvec& freq) {
  arma::vec y(X.n_rows, arma::fill::zeros);
  static const double A[4][6]
      = {{10, 3, 17, 3.5, 1.7, 8}, {0.05, 10, 17, 0.1, 8, 14}, {3, 3.5, 1.7, 10, 17, 8}, {17, 8, 0.05, 10, 0.1, 14}};
  static const double P[4][6] = {{0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
                                 {0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
                                 {0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
                                 {0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}};
  static const double alpha[4] = {1.0, 1.2, 3.0, 3.2};
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    if (family == "branin") {
      const double x1 = 15 * X(i, 0) - 5, x2 = 15 * X(i, 1);
      y(i) = std::pow(x2 - 5.1 / (4 * M_PI * M_PI) * x1 * x1 + 5 / M_PI * x1 - 6, 2)
             + 10 * (1 - 1 / (8 * M_PI)) * std::cos(x1) + 10;
    } else if (family == "hartmann6") {
      double s = 0;
      for (int k = 0; k < 4; ++k) {
        double e = 0;
        for (int j = 0; j < 6; ++j)
          e += A[k][j] * std::pow(X(i, j) - P[k][j], 2);
        s += alpha[k] * std::exp(-e);
      }
      y(i) = -s;
    } else {
      for (arma::uword j = 0; j < X.n_cols; ++j)
        y(i) += std::sin(2 * M_PI * freq(j) * X(i, j));
    }
  }
  return y;
}

int main(int argc, char** argv) {
  const std::string family = argv[1], kernel = argv[2];
  const arma::uword n = std::strtoul(argv[3], nullptr, 10), d = std::strtoul(argv[4], nullptr, 10);
  const int nsets = std::atoi(argv[5]);
  const std::string optim = argv[6];
  const double noise = argc > 7 ? std::atof(argv[7]) : 0.0;
  const double jitter_on = argc > 8 ? std::atof(argv[8]) : -1.0;
  {
    auto e = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    volatile double x = 0;
    while (std::chrono::steady_clock::now() < e)
      x = x + 1;
  }
  std::printf("set,jitter,pct,theta0,ll,rmse,ms\n");
  for (int s = 0; s < nsets; ++s) {
    arma::arma_rng::set_seed(9000 + s);
    const arma::mat X(n, d, arma::fill::randu);
    const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
    arma::vec y = f(family, X, freq);
    if (noise > 0)
      y += noise * arma::stddev(y) * arma::randn<arma::vec>(n);
    arma::arma_rng::set_seed(777);
    const arma::mat Xt(500, d, arma::fill::randu);
    const arma::vec yt = f(family, Xt, freq);
    for (double jitter : {0.0, jitter_on}) {
      LinearAlgebra::set_chol_jitter(jitter);
      for (double pct : {10.0, 100.0, 200.0}) {
        Optim::set_num_threads_percent(pct);
        try {
          Kriging k(kernel);
          const auto t0 = std::chrono::steady_clock::now();
          k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
          const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          const arma::vec m = std::get<0>(k.predict(Xt, false, false, false));
          const double rmse = std::sqrt(arma::mean(arma::square(m - yt))) / arma::stddev(yt);
          std::printf("%d,%s,%g,%.17g,%.17g,%.6g,%.1f\n",
                      s,
                      jitter != 0.0 ? "on" : "off",
                      pct,
                      k.theta()(0),
                      k.logLikelihood(),
                      rmse,
                      ms);
        } catch (const std::exception& e) {
          std::printf("%d,%s,%g,nan,nan,nan,nan\n", s, jitter != 0.0 ? "on" : "off", pct);
        }
        std::fflush(stdout);
      }
    }
  }
  LinearAlgebra::set_chol_jitter(-1);
  Optim::set_num_threads_percent(0);
}
