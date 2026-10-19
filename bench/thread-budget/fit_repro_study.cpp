// Study driver (not built by CMake): a single-start fit of the same data with 1, 10
// and 20 BLAS threads (Optim::set_num_threads_percent 10 / 100 / 200), then the
// prediction RMSE on 1000 test points. Built against build-tb with the flags of
// bench-fit-profile (compile_commands.json). The fixed-jitter variants used a
// temporary TMP_JITTER hook in LinearAlgebra::cholCov (diag += jitter * factor),
// not kept. Usage: fit_repro_study <sets> <n> <d> [kernel]. Results:
// bench/thread-budget/fit-repro-<kernel>-<n>-<d>-<jitter>.csv. See docs/dev/parallel.md 8.6.
// Scratch study: does a single-start fit depend on the BLAS thread count?
#define _USE_MATH_DEFINES
#include <cmath>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
int main(int argc, char** argv) {
  const int nsets = std::atoi(argv[1]);
  const arma::uword n = std::strtoul(argv[2], nullptr, 10), d = std::strtoul(argv[3], nullptr, 10);
  const std::string kernel = argc > 4 ? argv[4] : "gauss";
  { auto e = std::chrono::steady_clock::now() + std::chrono::milliseconds(300); volatile double x = 0; while (std::chrono::steady_clock::now() < e) x = x + 1; }
  if (std::getenv("WARN")) LinearAlgebra::set_chol_warning(true);
  std::printf("set,pct,threads,ll,theta0,ms,rmse,sdratio\n");
  for (int s = 0; s < nsets; ++s) {
    arma::arma_rng::set_seed(1000 + s);
    arma::mat X(n, d, arma::fill::randu);
    arma::vec y(n, arma::fill::zeros);
    for (arma::uword j = 0; j < d; ++j) y += arma::sin(2 * M_PI * X.col(j));
    for (double pct : {10.0, 100.0, 200.0}) {
      Optim::set_num_threads_percent(pct);
      Kriging k(kernel);
      Kriging::Parameters p{std::nullopt, true, std::nullopt, true};
      arma::arma_rng::set_seed(1);
      auto t0 = std::chrono::steady_clock::now();
      k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", p);
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      arma::arma_rng::set_seed(777);
      arma::mat Xt(1000, d, arma::fill::randu);
      arma::vec yt(1000, arma::fill::zeros);
      for (arma::uword j = 0; j < d; ++j) yt += arma::sin(2 * M_PI * Xt.col(j));
      auto pr = k.predict(Xt, true, false, false);
      arma::vec m = std::get<0>(pr), sd = std::get<1>(pr);
      double rmse = std::sqrt(arma::mean(arma::square(m - yt)));
      double sdr = arma::median(sd) / std::max(1e-300, rmse);
      std::printf("%d,%g,%d,%.10g,%.6g,%.2f,%.6g,%.4g\n", s, pct, Optim::get_num_threads(), k.logLikelihood(), k.theta()(0), ms, rmse, sdr);
    }
  }
}
