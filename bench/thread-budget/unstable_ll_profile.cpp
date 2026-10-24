// Study driver (not built by CMake, see fit_repro_study.cpp for the build):
// the data of tests/unstableLLTest.cpp ("BFGS vs BFGS10" section). For each
// (num_nugget/rcond settings) x (fixed jitter 0 / default), print the LL
// profile over theta and the BFGS / BFGS10 fits. See docs/dev/CholJitter.md, starting points.
#include <cmath>
#include <cstdio>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
int main() {
  arma::mat X(7, 1);
  X << 0.0 << arma::endr << 0.25 << arma::endr << 0.33 << arma::endr << 0.45 << arma::endr << 0.5 << arma::endr << 0.75
    << arma::endr << 1.0 << arma::endr;
  arma::vec y(7);
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    double x = X(i, 0);
    y(i) = 1.0 - 0.5 * (std::sin(4.0 * x) / (1.0 + x) + 2.0 * std::cos(12.0 * x) * std::pow(x, 6.0) + 0.7);
  }
  const double default_nugget = LinearAlgebra::get_num_nugget();
  const bool default_rcond = LinearAlgebra::chol_rcond_checked();
  for (int test_settings : {0, 1}) {
    LinearAlgebra::check_chol_rcond(test_settings ? true : default_rcond);
    LinearAlgebra::set_num_nugget(test_settings ? 1e-15 : default_nugget);
    for (double jitter : {0.0, -1.0}) {
      LinearAlgebra::set_chol_jitter(jitter);
      std::printf("== settings=%s jitter=%s\n",
                  test_settings ? "test(nugget 1e-15, rcond)" : "default",
                  jitter < 0 ? "default(1e-8)" : "0");
      for (const char* optim : {"BFGS", "BFGS10", "BFGS20"}) {
        Kriging k("gauss");
        Kriging::Parameters p;
        k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", p);
        std::printf("  %-6s theta=%.6g LL=%.6g sigma2=%.4g\n", optim, k.theta()(0), k.logLikelihood(), k.sigma2());
      }
      Kriging k("gauss");
      Kriging::Parameters p;
      p.theta = arma::mat(1, 1, arma::fill::value(0.5));
      p.is_theta_estim = false;
      k.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", p);
      std::printf("  profile theta:LL ");
      for (double lt = -2; lt <= 1.001; lt += 0.1) {
        const double th = std::pow(10.0, lt);
        const double ll = std::get<0>(k.logLikelihoodFun(arma::vec{th}, false, false));
        std::printf(" %.3g:%.4g", th, ll);
      }
      std::printf("\n");
    }
  }
}
