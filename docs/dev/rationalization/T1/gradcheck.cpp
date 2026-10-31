// T1 harness: gradcheck <n> <d> <kernel> check|time [reps]
//  check: prints LL and LMP theta gradients (%.17g), one value per line
//  time : median ms of logLikelihoodFun / logMargPostFun with and without gradient
#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
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
int main(int argc, char** argv) {
  const arma::uword n = std::strtoul(argv[1], nullptr, 10);
  const arma::uword d = std::strtoul(argv[2], nullptr, 10);
  const std::string kernel = argv[3];
  const std::string mode = argv[4];
  const int reps = argc > 5 ? std::atoi(argv[5]) : 5;
  arma::arma_rng::set_seed(123);
  arma::mat X(n, d, arma::fill::randu);
  arma::vec y = arma::sum(arma::sin(3 * X), 1);
  arma::vec theta(d);
  // short ranges: well-conditioned R
  for (arma::uword k = 0; k < d; ++k)
    theta(k) = (0.05 + 0.1 * k / std::max<arma::uword>(1, d - 1)) * std::sqrt((double)d);
  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;
  Kriging k(y, X, kernel, Trend::RegressionModel::Linear, false, "none", "LL", par);
  if (mode == "check") {
    auto [ll, gll] = k.logLikelihoodFun(theta, true, false);
    auto [lmp, glmp] = k.logMargPostFun(theta, true, false);
    if (std::getenv("GC_COND")) {
      auto cov = Covariance::resolve(kernel).Cov;
      arma::mat R(n, n);
      for (arma::uword i = 0; i < n; i++)
        for (arma::uword j = 0; j < n; j++)
          R(i, j) = cov(arma::vec((X.row(i) - X.row(j)).t()), theta);
      std::printf("COND %.6g\n", arma::cond(R));
    }
    for (auto v : gll) std::printf("LL %.17g\n", v);
    for (auto v : glmp) std::printf("LMP %.17g\n", v);
  } else if (mode == "nugget") {
    // Nugget model, LMP and LL gradients at gamma = (theta, alpha = 0.8), plus a central FD check
    Kriging kn(kernel, Kriging::NoiseModel::Nugget);
    kn.fit(y + 0.05 * arma::randn<arma::vec>(n), X, Trend::RegressionModel::Constant, false, "BFGS", "LMP");
    arma::vec g = arma::join_cols(theta, arma::vec{0.8});
    for (int which = 0; which < 2; which++) {
      auto f = [&](const arma::vec& x, bool gr) {
        return which == 0 ? kn.logMargPostFun(x, gr, false) : kn.logLikelihoodFun(x, gr, false);
      };
      auto [v, gr] = f(g, true);
      for (arma::uword i = 0; i < g.n_elem; i++) {
        arma::vec gp = g, gm = g;
        const double h = 1e-5 * g(i);
        gp(i) += h;
        gm(i) -= h;
        const double fd = (std::get<0>(f(gp, false)) - std::get<0>(f(gm, false))) / (2 * h);
        std::printf("%s %.17g FD %.8g\n", which == 0 ? "NLMP" : "NLL", gr(i), fd);
      }
    }
  } else if (mode == "bench") {
    k.logLikelihoodFun(theta, true, true);
    k.logMargPostFun(theta, true, true);
  } else {
    const double a = median_ms(reps, [&] { k.logLikelihoodFun(theta, false, false); });
    const double b = median_ms(reps, [&] { k.logLikelihoodFun(theta, true, false); });
    const double c = median_ms(reps, [&] { k.logMargPostFun(theta, false, false); });
    const double e = median_ms(reps, [&] { k.logMargPostFun(theta, true, false); });
    std::printf("%lu,%lu,%s,%.2f,%.2f,%.2f,%.2f\n", (unsigned long)n, (unsigned long)d, kernel.c_str(), a, b, c, e);
  }
  return 0;
}
