// Probe (not built by CMake): predict mean derivative vs central finite differences
// for several steps h, on the data of tests/KrigingPredictTest.cpp. Usage:
// predict_fd_probe [chol_jitter] (0 = off, -1 = automatic). Built like
// fit_repro_study.cpp. See docs/dev/CholJitter-TODO.md, failing test 1.
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
int main(int argc, char** argv) {
  if (argc > 1)
    LinearAlgebra::set_chol_jitter(std::atof(argv[1]));
  arma::arma_rng::set_seed(123);
  const arma::uword n = 20, d = 2;
  arma::mat X(n, d, arma::fill::randu);
  arma::vec y(n);
  for (arma::uword i = 0; i < n; ++i)
    y(i) = std::sin(3.0 * X(i, 0)) + std::cos(5.0 * X(i, 1));
  Kriging kr("gauss");
  Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  kr.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
  arma::mat Xn(5, d, arma::fill::randu);
  auto r = kr.predict(Xn, true, false, true);
  arma::mat md = std::get<3>(r);
  std::printf("jitter=%g theta=%g %g\n", LinearAlgebra::kernel_chol_jitter("gauss"), kr.theta()(0), kr.theta()(1));
  for (double h : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    arma::mat Xp = Xn, Xm = Xn;
    Xp(2, 0) += h;
    Xm(2, 0) -= h;
    double fd
        = (std::get<0>(kr.predict(Xp, false, false, false))(2) - std::get<0>(kr.predict(Xm, false, false, false))(2))
          / (2 * h);
    const arma::mat sdd = std::get<4>(r);
    double fds
        = (std::get<1>(kr.predict(Xp, true, false, false))(2) - std::get<1>(kr.predict(Xm, true, false, false))(2))
          / (2 * h);
    std::printf(
        "h=%g mean: fd=%.8f analytical=%.8f | stdev: fd=%.8f analytical=%.8f\n", h, fd, md(2, 0), fds, sdd(2, 0));
  }
}
