// Study driver of T3 (docs/dev/rationalization/T3-default-multistart.md), not
// built by CMake (build: see README.md): single start (BFGS, with / without the
// long-range restart) against parallel multistarts (BFGS4, BFGS10), on the
// data families of bench/thread-budget/jitter_n_study.cpp (same data and test
// sets for the noise-free case).
// Usage: t3_multistart_study <family> <kernel> <noise> <objective> <n> <d>
//                            <set0> <set1> <pct> <variants>
//   family:    sin, sinhalf (first ceil(d / 2) inputs active), branin (inputs
//              1-2 active), hartmann6 (inputs 1-6 active)
//   noise:     none (Kriging noise model None, exact data) or nugget (Nugget
//              noise model, y + N(0, (0.1 sd(y))^2), seed 5000 + set)
//   objective: LL, LOO or LMP
//   sets:      data sets set0 .. set1 - 1 (seed 9000 + set, as jitter_n_study)
//   pct:       Optim::set_num_threads_percent
//   variants:  comma list of BFGS (long-range restart as configured, default
//              c = 2), BFGS-nolr (Optim::set_long_range_restart(0)), BFGSk
// Output CSV (header + one row per fit):
//   family,kernel,noise,objective,n,d,set,variant,pct,threads,obj,ll,rmse,
//   wall_ms,cpu_ms,theta
//   obj: value of the optimized objective (logLikelihood / leaveOneOut /
//   logMargPost), ll: logLikelihood(), rmse: test RMSE (500 points, against the
//   noise-free function) / sd(test values), cpu_ms: process CPU time (all
//   threads) of the fit (after a 200 ms pause), theta ';'-separated.
#define _USE_MATH_DEFINES
#include <sys/resource.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "libKriging/Kriging.hpp"
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

static double cpu_ms() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  return 1e3 * (u.ru_utime.tv_sec + u.ru_stime.tv_sec) + 1e-3 * (u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}

int main(int argc, char** argv) {
  if (argc != 11) {
    std::fprintf(stderr, "usage: %s family kernel noise objective n d set0 set1 pct variants\n", argv[0]);
    return 1;
  }
  const std::string family = argv[1], kernel = argv[2], noise = argv[3], objective = argv[4];
  const arma::uword n = std::strtoul(argv[5], nullptr, 10), d = std::strtoul(argv[6], nullptr, 10);
  const int set0 = std::atoi(argv[7]), set1 = std::atoi(argv[8]);
  const double pct = std::atof(argv[9]);
  std::vector<std::string> variants;
  {
    std::stringstream ss(argv[10]);
    std::string tok;
    while (std::getline(ss, tok, ','))
      variants.push_back(tok);
  }
  const double lr_default = Optim::get_long_range_restart();
  Optim::set_num_threads_percent(pct);
  {
    // OpenBLAS pthreads threads spin ~100 ms after loading (bench/thread-budget/README.md)
    auto e = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    volatile double x = 0;
    while (std::chrono::steady_clock::now() < e)
      x = x + 1;
  }
  std::printf("family,kernel,noise,objective,n,d,set,variant,pct,threads,obj,ll,rmse,wall_ms,cpu_ms,theta\n");
  for (int s = set0; s < set1; ++s) {
    arma::arma_rng::set_seed(9000 + s);
    const arma::mat X(n, d, arma::fill::randu);
    const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
    arma::vec y = f(family, X, freq);
    arma::arma_rng::set_seed(777);
    const arma::mat Xt(500, d, arma::fill::randu);
    const arma::vec yt = f(family, Xt, freq);
    if (noise == "nugget") {
      arma::arma_rng::set_seed(5000 + s);
      y += 0.1 * arma::stddev(y) * arma::randn<arma::vec>(n);
    }
    for (const std::string& v : variants) {
      const bool nolr = v == "BFGS-nolr";
      Optim::set_long_range_restart(nolr ? 0.0 : lr_default);
      const std::string optim = nolr ? "BFGS" : v;
      std::string row;
      try {
        Kriging k = noise == "nugget" ? Kriging(kernel, Kriging::NoiseModel::Nugget) : Kriging(kernel);
        std::fprintf(stderr, "FIT %s %s %s %s n%llu d%llu set %d %s\n", family.c_str(), kernel.c_str(),
                     noise.c_str(), objective.c_str(), (unsigned long long)n, (unsigned long long)d, s, v.c_str());
        // the OpenBLAS threads of the previous predict spin ~100 ms: not this fit's CPU time
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const double c0 = cpu_ms();
        const auto t0 = std::chrono::steady_clock::now();
        k.fit(y, X, Trend::RegressionModel::Constant, false, optim, objective, Kriging::Parameters{});
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double cms = cpu_ms() - c0;
        const arma::vec m = std::get<0>(k.predict(Xt, false, false, false));
        const double rmse = std::sqrt(arma::mean(arma::square(m - yt))) / arma::stddev(yt);
        const double obj = objective == "LOO" ? k.leaveOneOut() : objective == "LMP" ? k.logMargPost() : k.logLikelihood();
        std::string th;
        for (arma::uword j = 0; j < d; ++j) {
          char b[32];
          std::snprintf(b, sizeof b, "%s%.10g", j ? ";" : "", k.theta()(j));
          th += b;
        }
        char b[512];
        std::snprintf(b, sizeof b, "%.17g,%.17g,%.6g,%.1f,%.1f,%s", obj, k.logLikelihood(), rmse, ms, cms, th.c_str());
        row = b;
      } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        row = "nan,nan,nan,nan,nan,nan";
      }
      std::printf("%s,%s,%s,%s,%llu,%llu,%d,%s,%g,%d,%s\n", family.c_str(), kernel.c_str(), noise.c_str(),
                  objective.c_str(), (unsigned long long)n, (unsigned long long)d, s, v.c_str(), pct,
                  Optim::get_num_threads(), row.c_str());
      std::fflush(stdout);
    }
  }
  Optim::set_long_range_restart(lr_default);
  Optim::set_num_threads_percent(0);
}
