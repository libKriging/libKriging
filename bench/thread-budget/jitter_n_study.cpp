// Study driver (not built by CMake, see README.md for the build): the fixed
// Cholesky jitter as a function of n (#426). For one (family, kernel, n, d),
// fits every data set with each jitter value of a list, each at
// Optim::set_num_threads_percent 10 / 100 / 200. A rule J(n) is a path through
// the (n, J) table: run_jitter_n.sh passes, per n, the constants and the values
// of the rules at that n (summarize_jitter_n.py reassembles the rules).
// Usage: jitter_n_study <family> <kernel> <n> <d> <sets> <optim> <J1,J2,...>
//   family: sin (sum_j sin(2 pi f_j x_j), f_j ~ U[0.5, 3]), sinhalf (the same on
//           the first ceil(d / 2) inputs, the others inactive), branin (inputs 1-2,
//           the others inactive), hartmann6 (inputs 1-6, the others inactive);
//           same data and test sets as jitter_scale_study.cpp
// Output CSV: set,J,pct,theta (';'-separated),ll,rmse,ms,lmax,lmin
//   lmax / lmin: extreme eigenvalues of the exact correlation matrix at the
//   fitted theta (gauss only, pct 100, n <= 1000; nan otherwise)
// See docs/dev/CholJitter.md, jitter growing with n.
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
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
    } else if (family == "sinhalf") {
      for (arma::uword j = 0; j < (X.n_cols + 1) / 2; ++j)
        y(i) += std::sin(2 * M_PI * freq(j) * X(i, j));
    } else {
      for (arma::uword j = 0; j < X.n_cols; ++j)
        y(i) += std::sin(2 * M_PI * freq(j) * X(i, j));
    }
  }
  return y;
}

// extreme eigenvalues of the gauss correlation matrix (no jitter) at theta
static void gauss_eig(const arma::mat& X, const arma::vec& theta, double& lmax, double& lmin) {
  const arma::uword n = X.n_rows;
  arma::mat R(n, n);
  for (arma::uword i = 0; i < n; ++i)
    for (arma::uword j = 0; j <= i; ++j) {
      double s = 0;
      for (arma::uword k = 0; k < X.n_cols; ++k) {
        const double u = (X(i, k) - X(j, k)) / theta(k);
        s += u * u;
      }
      R(i, j) = R(j, i) = std::exp(-0.5 * s);
    }
  const arma::vec ev = arma::eig_sym(R);
  lmax = ev.max();
  lmin = ev.min();
}

int main(int argc, char** argv) {
  const std::string family = argv[1], kernel = argv[2];
  const arma::uword n = std::strtoul(argv[3], nullptr, 10), d = std::strtoul(argv[4], nullptr, 10);
  const int nsets = std::atoi(argv[5]);
  const std::string optim = argv[6];
  std::vector<double> jitters;
  {
    std::stringstream ss(argv[7]);
    std::string tok;
    while (std::getline(ss, tok, ','))
      jitters.push_back(std::atof(tok.c_str()));
  }
  {
    auto e = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    volatile double x = 0;
    while (std::chrono::steady_clock::now() < e)
      x = x + 1;
  }
  std::printf("set,J,pct,theta,ll,rmse,ms,lmax,lmin\n");
  for (int s = 0; s < nsets; ++s) {
    arma::arma_rng::set_seed(9000 + s);
    const arma::mat X(n, d, arma::fill::randu);
    const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
    const arma::vec y = f(family, X, freq);
    arma::arma_rng::set_seed(777);
    const arma::mat Xt(500, d, arma::fill::randu);
    const arma::vec yt = f(family, Xt, freq);
    for (double jitter : jitters) {
      LinearAlgebra::set_chol_jitter(jitter);
      for (double pct : {10.0, 100.0, 200.0}) {
        Optim::set_num_threads_percent(pct);
        try {
          Kriging k(kernel);
          std::fprintf(stderr, "FIT %d %g %g\n", s, jitter, pct);
          const auto t0 = std::chrono::steady_clock::now();
          k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
          const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
          const arma::vec m = std::get<0>(k.predict(Xt, false, false, false));
          const double rmse = std::sqrt(arma::mean(arma::square(m - yt))) / arma::stddev(yt);
          double lmax = NAN, lmin = NAN;
          if (kernel == "gauss" && pct == 100.0 && n <= 1000)
            gauss_eig(X, k.theta(), lmax, lmin);
          std::string th;
          for (arma::uword j = 0; j < d; ++j) {
            char b[32];
            std::snprintf(b, sizeof b, "%s%.17g", j ? ";" : "", k.theta()(j));
            th += b;
          }
          std::printf("%d,%g,%g,%s,%.17g,%.6g,%.1f,%.6g,%.6g\n",
                      s, jitter, pct, th.c_str(), k.logLikelihood(), rmse, ms, lmax, lmin);
        } catch (const std::exception& e) {
          std::printf("%d,%g,%g,nan,nan,nan,nan,nan,nan\n", s, jitter, pct);
        }
        std::fflush(stdout);
      }
    }
  }
  LinearAlgebra::set_chol_jitter(-1);
  Optim::set_num_threads_percent(0);
}
