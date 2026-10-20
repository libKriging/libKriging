// Study driver (not built by CMake, see fit_repro_study.cpp for the build):
// WarpKriging / MLPKriging multistart starting points, log-uniform (default)
// against uniform in the theta bounds. The log draw was a temporary hook
// (LK_TMP_WARP_UNIFORM=1 selected uniform), not kept: WarpKriging draws uniform.
// Usage: warp_start_study <model> <family> <kernel> <n> <d> <sets> <optim>
//   model: mlp (MLPKriging({8}, 2, tanh)), kuma (kumaraswamy per variable),
//          none (WarpKriging with "none" warps, i.e. plain Kriging)
//   family: sin, branin (d = 2), proj (Branin of a fixed 2 x 5 projection, d = 5)
// Output CSV: set,ll,q2,nlpd,cov,ms
// See docs/dev/CholJitter-TODO.md 4f.
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "libKriging/MLPKriging.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

using namespace libKriging;

static double branin(double u, double v) {
  const double x1 = 15 * u - 5, x2 = 15 * v;
  return std::pow(x2 - 5.1 / (4 * M_PI * M_PI) * x1 * x1 + 5 / M_PI * x1 - 6, 2)
         + 10 * (1 - 1 / (8 * M_PI)) * std::cos(x1) + 10;
}

static arma::vec f(const std::string& family, const arma::mat& X, const arma::rowvec& freq) {
  arma::vec y(X.n_rows, arma::fill::zeros);
  if (family == "proj") {
    arma::mat W = {{0.9, -0.4, 0.7, 0.2, -0.5}, {-0.3, 0.8, 0.1, -0.6, 0.4}};
    W.each_col() /= arma::sum(arma::abs(W), 1);
    arma::mat U = X * W.t();
    U.each_row() -= arma::sum(arma::clamp(W, -arma::datum::inf, 0.0), 1).t();
    for (arma::uword i = 0; i < X.n_rows; ++i)
      y(i) = branin(U(i, 0), U(i, 1));
    return y;
  }
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    if (family == "branin")
      y(i) = branin(X(i, 0), X(i, 1));
    else
      for (arma::uword j = 0; j < X.n_cols; ++j)
        y(i) += std::sin(2 * M_PI * freq(j) * X(i, j));
  }
  return y;
}

static arma::mat lhs(arma::uword n, arma::uword d) {
  arma::mat X(n, d);
  for (arma::uword j = 0; j < d; ++j)
    X.col(j) = (arma::conv_to<arma::vec>::from(arma::randperm(n)) + arma::randu<arma::vec>(n)) / n;
  return X;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: %s <model> <family> <kernel> <n> <d> <sets> <optim>\n", argv[0]);
    return 1;
  }
  const std::string model = argv[1], family = argv[2], kernel = argv[3], optim = argv[7];
  const arma::uword n = std::stoul(argv[4]), d = std::stoul(argv[5]);
  const int sets = std::stoi(argv[6]);
  std::printf("set,ll,q2,nlpd,cov,ms\n");
  for (int s = 0; s < sets; ++s) {
    arma::arma_rng::set_seed(1000 + s);
    const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
    const arma::mat X = lhs(n, d);
    const arma::vec y = f(family, X, freq);
    const arma::mat Xt = arma::randu<arma::mat>(1000, d);
    const arma::vec yt = f(family, Xt, freq);
    arma::vec m, sd;
    double ll = NAN;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      if (model == "mlp") {
        MLPKriging k({8}, 2, "tanh", kernel);
        k.fit(y, X, Trend::RegressionModel::Constant, false, optim);
        auto p = k.predict(Xt, true, false, false);
        m = std::get<0>(p), sd = std::get<1>(p), ll = k.logLikelihood();
      } else {
        WarpKriging k(std::vector<std::string>(d, model == "kuma" ? "kumaraswamy" : "none"), kernel);
        k.fit(y, X, Trend::RegressionModel::Constant, false, optim);
        auto p = k.predict(Xt, true, false, false);
        m = std::get<0>(p), sd = std::get<1>(p), ll = k.logLikelihood();
      }
    } catch (const std::exception& e) {
      std::printf("%d,nan,nan,nan,nan,0\n", s);
      continue;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double q2 = 1 - arma::accu(arma::square(yt - m)) / arma::accu(arma::square(yt - arma::mean(yt)));
    const double nlpd = arma::mean(0.5 * arma::log(2 * M_PI * arma::square(sd)) + 0.5 * arma::square((yt - m) / sd));
    const double cov = arma::mean(arma::conv_to<arma::vec>::from(arma::abs(yt - m) <= 1.96 * sd));
    std::printf("%d,%.10g,%.6g,%.6g,%.4g,%.1f\n", s, ll, q2, nlpd, cov, ms);
    std::fflush(stdout);
  }
  return 0;
}
