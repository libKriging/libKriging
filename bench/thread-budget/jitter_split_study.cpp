// Study driver (not built by CMake, see fit_repro_study.cpp for the build):
// where does the accuracy cost of the fixed Cholesky jitter come from, the
// theta it selects or its smoothing of the final predictor? Variants:
//   off   : fit with jitter 0 (former behaviour)
//   on    : fit with the per-kernel jitter (current default)
//   on>0  : theta of "on", final model rebuilt at that theta with jitter 0
//   on>10 : same, final jitter 1e-10
//   on~0  : continuation, a last BFGS with jitter 0 started from the theta of "on"
// Each variant is run with Optim::set_num_threads_percent 10 / 100 / 200 (column pct).
// Usage: jitter_split_study <family> <kernel> <n> <d> <sets> <optim>
//   family: sin (random freq in [0.5, 3]), sin1 (freq 1), branin, unstable
// Output CSV: set,pct,variant,theta0,ll,rmse,pred0 (prediction at the first test point)
// See docs/dev/CholJitter-TODO.md 4c.
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
        y(i) += std::sin(2 * M_PI * (family == "sin1" ? 1.0 : freq(j)) * X(i, j));
    }
  }
  return y;
}

int main(int argc, char** argv) {
  const std::string family = argv[1], kernel = argv[2];
  const arma::uword n = std::strtoul(argv[3], nullptr, 10), d = std::strtoul(argv[4], nullptr, 10);
  const int nsets = std::atoi(argv[5]);
  const std::string optim = argv[6];
  std::printf("set,pct,variant,theta0,ll,rmse,pred0\n");
  {
    auto e = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    volatile double x = 0;
    while (std::chrono::steady_clock::now() < e)
      x = x + 1;
  }
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
    for (double pct : {10.0, 100.0, 200.0}) {
      Optim::set_num_threads_percent(pct);
      auto report = [&](const char* variant, Kriging& k) {
        const arma::vec m = std::get<0>(k.predict(Xt, false, false, false));
        const double rmse = std::sqrt(arma::mean(arma::square(m - yt))) / arma::stddev(yt);
        std::printf("%d,%g,%s,%.17g,%.17g,%.6g,%.17g\n", s, pct, variant, k.theta()(0), k.logLikelihood(), rmse, m(0));
      };
      try {
        LinearAlgebra::set_chol_jitter(0);
        Kriging k_off(kernel);
        k_off.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
        report("off", k_off);
        LinearAlgebra::set_chol_jitter(-1);
        Kriging k_on(kernel);
        k_on.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
        report("on", k_on);
        for (double final_jitter : {0.0, 1e-10}) {
          LinearAlgebra::set_chol_jitter(final_jitter);
          Kriging k(kernel);
          Kriging::Parameters p;
          p.theta = arma::mat(k_on.theta().t());
          p.is_theta_estim = false;
          k.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", p);
          report(final_jitter == 0.0 ? "on>0" : "on>10", k);
        }
        LinearAlgebra::set_chol_jitter(0);
        Kriging k_cont(kernel);
        Kriging::Parameters pc;
        pc.theta = arma::mat(k_on.theta().t());
        k_cont.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", pc);
        report("on~0", k_cont);
      } catch (const std::exception& e) {
        std::printf("%d,%g,error,nan,nan,nan,nan\n", s, pct);
      }
    }
  }
  LinearAlgebra::set_chol_jitter(-1);
}
