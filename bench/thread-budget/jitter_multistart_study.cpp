// Study driver (not built by CMake, see fit_repro_study.cpp for the build):
// does the fixed Cholesky jitter make a single-start BFGS fit miss the
// multistart (BFGS10) optimum more often? For each data set, fit BFGS and
// BFGS10 with the jitter off (set_chol_jitter(0)) and on (per-kernel
// default), default LinearAlgebra settings otherwise.
// Usage: jitter_multistart_study <family> <kernel> <n> <d> <sets>
//   family: unstable (f of tests/unstableLLTest.cpp, random design + 0, 1),
//           sin (sum_j sin(2 pi f_j x_j), f_j ~ U[0.5, 3]), branin (d = 2)
// Output CSV: set,jitter,optim,theta0,ll,rmse
// See docs/dev/CholJitter-TODO.md 3.4.
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

static arma::vec f(const std::string& family, const arma::mat& X, const arma::rowvec& freq) {
  arma::vec y(X.n_rows, arma::fill::zeros);
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    if (family == "unstable") {
      const double x = X(i, 0);
      y(i) = 1.0 - 0.5 * (std::sin(4.0 * x) / (1.0 + x) + 2.0 * std::cos(12.0 * x) * std::pow(x, 6.0) + 0.7);
    } else if (family == "branin") {
      const double x1 = 15 * X(i, 0) - 5, x2 = 15 * X(i, 1);
      y(i) = std::pow(x2 - 5.1 / (4 * M_PI * M_PI) * x1 * x1 + 5 / M_PI * x1 - 6, 2)
             + 10 * (1 - 1 / (8 * M_PI)) * std::cos(x1) + 10;
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
  std::printf("set,jitter,optim,theta0,ll,rmse\n");
  for (int s = 0; s < nsets; ++s) {
    arma::arma_rng::set_seed(5000 + s);
    arma::mat X(n, d, arma::fill::randu);
    if (family == "unstable") {
      X(0, 0) = 0.0;
      X(1, 0) = 1.0;
    }
    const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
    const arma::vec y = f(family, X, freq);
    arma::arma_rng::set_seed(777);
    const arma::mat Xt(500, d, arma::fill::randu);
    const arma::vec yt = f(family, Xt, freq);
    for (double jitter : {0.0, -1.0}) {
      LinearAlgebra::set_chol_jitter(jitter);
      for (const char* optim : {"BFGS", "BFGS10"}) {
        Kriging k(kernel);
        Kriging::Parameters p;
        try {
          k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", p);
          const arma::vec m = std::get<0>(k.predict(Xt, false, false, false));
          const double rmse = std::sqrt(arma::mean(arma::square(m - yt))) / arma::stddev(yt);
          std::printf(
              "%d,%s,%s,%.6g,%.10g,%.6g\n", s, jitter < 0 ? "on" : "off", optim, k.theta()(0), k.logLikelihood(), rmse);
        } catch (const std::exception& e) {
          std::printf("%d,%s,%s,nan,nan,nan\n", s, jitter < 0 ? "on" : "off", optim);
        }
      }
    }
  }
}
