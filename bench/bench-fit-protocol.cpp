// Fit protocol of the statistical benchmark base (T7, docs/dev/Rationalization.md;
// docs/dev/rationalization/T7-benchmark-base.md). For every configuration
// (kernel x noise x objective x function x n) and every seeded data set, fits
// one Kriging model per optimizer of --optim and prints one CSV row per fit.
// The settings that are not arguments come from the environment, read by the
// library (LK_LONG_RANGE_RESTART, LK_CHOL_JITTER, LK_NUM_THREADS_PERCENT, ...)
// and written in the CSV, so that the rows of several runs (one per setting)
// can be merged and compared on the same data sets (bench/fit-protocol/).
//
// Usage: bench-fit-protocol [--key value ...]
//   --optim      BFGS,BFGS4,BFGS10   optimizers (one fit each per data set)
//   --kernel     gauss,matern5_2
//   --noise      none,nugget         nugget: y + N(0, (0.05 sd(y))^2), nugget estimated
//   --objective  LL,LOO,LMP          LOO with a nugget is not supported: skipped
//   --function   sin4,hartmann6,branin4,sinhalf8
//   --n          100,200,500,1000
//   --sets       30                  data sets per configuration
//   --first-set  0                   index of the first data set
//   --ntest      500                 test points (RMSE)
//   --noise-sd   0.05                nugget data: noise sd relative to sd(y)
//   --variant    ""                  label of the settings (default: built from them)
//   --header     1                   0: no CSV header (to append)
// Functions (on [0, 1]^d; "<family><d>" or "<family>_<d>", d defaults to the
// active dimension):
//   sin<d>     sum_j sin(2 pi f_j x_j), f_j ~ U[0.5, 3] (all inputs active)
//   sinhalf<d> the same on the first ceil(d / 2) inputs, the others inactive
//   branin<d>  Branin on inputs 1-2, the others inactive (d >= 2)
//   hartmann6[_<d>] Hartmann6 on inputs 1-6, the others inactive (d >= 6)
// The data of (family, d, n, set) are those of bench/thread-budget/jitter_n_study.cpp
// (X ~ U[0, 1]^{n x d} seeded 9000 + set, test set seeded 777).
// Output CSV columns: see `header` below. objective is the optimized
// criterion at the fitted parameters (LL / LMP: to maximize, LOO: mean
// squared LOO error, to minimize), ll the log-likelihood there; rmse is the
// test RMSE over sd(y_test), against the noise-free function; wall_s / cpu_s
// the wall and process CPU time of the fit (all threads).
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

namespace {

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty())
      v.push_back(tok);
  return v;
}

double cpu_seconds() {
#ifndef _WIN32
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
#else
  return static_cast<double>(std::clock()) / CLOCKS_PER_SEC;  // wall time with MSVC
#endif
}

struct Function {
  std::string family;
  arma::uword d;
};

Function parse_function(const std::string& name) {
  for (const std::string fam : {"sinhalf", "sin", "branin", "hartmann6"}) {
    if (name.rfind(fam, 0) == 0) {
      std::string rest = name.substr(fam.size());
      if (!rest.empty() && rest[0] == '_')
        rest = rest.substr(1);
      const arma::uword dmin = fam == "hartmann6" ? 6 : fam == "branin" ? 2 : 1;
      const arma::uword d = rest.empty() ? dmin : std::strtoul(rest.c_str(), nullptr, 10);
      if (d < dmin)
        break;
      return {fam, d};
    }
  }
  throw std::invalid_argument("unknown function: " + name + " (sin<d>, sinhalf<d>, branin<d>, hartmann6[_<d>])");
}

// same families and formulas as bench/thread-budget/jitter_n_study.cpp
arma::vec f(const std::string& family, const arma::mat& X, const arma::rowvec& freq) {
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

std::string settings_label() {
  char b[160];
  const double c = LinearAlgebra::get_chol_jitter();
  char jit[32];
  if (c < 0)
    std::snprintf(jit, sizeof jit, "auto");
  else
    std::snprintf(jit, sizeof jit, "%g", c);
  std::snprintf(
      b, sizeof b, "lrr=%g,jitter=%s,pct=%g", Optim::get_long_range_restart(), jit, Optim::get_num_threads_percent());
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> opt = {{"optim", "BFGS,BFGS4,BFGS10"},
                                            {"kernel", "gauss,matern5_2"},
                                            {"noise", "none,nugget"},
                                            {"objective", "LL,LOO,LMP"},
                                            {"function", "sin4,hartmann6,branin4,sinhalf8"},
                                            {"n", "100,200,500,1000"},
                                            {"sets", "30"},
                                            {"first-set", "0"},
                                            {"ntest", "500"},
                                            {"noise-sd", "0.05"},
                                            {"variant", ""},
                                            {"header", "1"}};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help" || a.rfind("--", 0) != 0 || i + 1 >= argc || !opt.count(a.substr(2))) {
      std::fprintf(stderr, "usage: %s [--key value ...], keys (default):\n", argv[0]);
      for (const auto& kv : opt)
        std::fprintf(stderr, "  --%s (%s)\n", kv.first.c_str(), kv.second.c_str());
      return a == "-h" || a == "--help" ? 0 : 1;
    }
    opt[a.substr(2)] = argv[++i];
  }
  const auto optims = split(opt["optim"]), kernels = split(opt["kernel"]), noises = split(opt["noise"]),
             objectives = split(opt["objective"]), functions = split(opt["function"]), ns = split(opt["n"]);
  const int nsets = std::atoi(opt["sets"].c_str()), first = std::atoi(opt["first-set"].c_str());
  const arma::uword ntest = std::strtoul(opt["ntest"].c_str(), nullptr, 10);
  const double noise_sd = std::atof(opt["noise-sd"].c_str());
  const std::string settings = settings_label();
  const std::string variant_base = opt["variant"].empty() ? settings : opt["variant"];

  {  // warm-up: OpenBLAS pthreads threads spin ~100 ms after loading
    auto e = std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
    volatile double x = 0;
    while (std::chrono::steady_clock::now() < e)
      x = x + 1;
  }
  if (opt["header"] != "0")
    std::printf(
        "variant,optim,settings,num_threads,kernel,noise,objective,function,d,n,set,status,objective_value,ll,"
        "theta,nugget_ratio,rmse,wall_s,cpu_s\n");
  for (const auto& fname : functions) {
    const Function fun = parse_function(fname);
    for (const auto& ns_ : ns) {
      const arma::uword n = std::strtoul(ns_.c_str(), nullptr, 10);
      for (int s = first; s < first + nsets; ++s) {
        arma::arma_rng::set_seed(9000 + s);
        const arma::mat X(n, fun.d, arma::fill::randu);
        const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(fun.d);
        const arma::vec y0 = f(fun.family, X, freq);
        arma::arma_rng::set_seed(5000 + s);
        const arma::vec eps = arma::randn<arma::vec>(n);
        arma::arma_rng::set_seed(777);
        const arma::mat Xt(ntest, fun.d, arma::fill::randu);
        const arma::vec yt = f(fun.family, Xt, freq);
        for (const auto& noise : noises) {
          const bool nugget = noise == "nugget";
          if (!nugget && noise != "none")
            throw std::invalid_argument("unknown noise: " + noise + " (none, nugget)");
          const arma::vec y = nugget ? arma::vec(y0 + noise_sd * arma::stddev(y0) * eps) : y0;
          for (const auto& kernel : kernels)
            for (const auto& objective : objectives) {
              if (nugget && objective == "LOO")
                continue;  // not supported by Kriging
              for (const auto& optim : optims) {
                std::fprintf(stderr,
                             "FIT %s %s %s %s %s n=%llu set=%d\n",
                             optim.c_str(),
                             kernel.c_str(),
                             noise.c_str(),
                             objective.c_str(),
                             fname.c_str(),
                             (unsigned long long)n,
                             s);
                std::string status = "ok", theta = "";
                double obj = NAN, ll = NAN, nug = NAN, rmse = NAN, wall = NAN, cpu = NAN;
                try {
                  Kriging k(kernel, nugget ? Kriging::NoiseModel::Nugget : Kriging::NoiseModel::None);
                  const double c0 = cpu_seconds();
                  const auto t0 = std::chrono::steady_clock::now();
                  k.fit(y, X, Trend::RegressionModel::Constant, false, optim, objective, Kriging::Parameters{});
                  wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                  cpu = cpu_seconds() - c0;
                  ll = k.logLikelihood();
                  obj = objective == "LL" ? ll : objective == "LOO" ? k.leaveOneOut() : k.logMargPost();
                  if (nugget)
                    nug = k.nugget() / (k.nugget() + k.sigma2());
                  const arma::vec m = std::get<0>(k.predict(Xt, false, false, false));
                  rmse = std::sqrt(arma::mean(arma::square(m - yt))) / arma::stddev(yt);
                  for (arma::uword j = 0; j < fun.d; ++j) {
                    char b[32];
                    std::snprintf(b, sizeof b, "%s%.6g", j ? ";" : "", k.theta()(j));
                    theta += b;
                  }
                } catch (const std::exception& e) {
                  status = "error";
                  std::fprintf(stderr, "  error: %s\n", e.what());
                }
                std::printf("\"%s|%s\",%s,\"%s\",%d,%s,%s,%s,%s,%llu,%llu,%d,%s,%.10g,%.10g,%s,%.6g,%.6g,%.4g,%.4g\n",
                            optim.c_str(),
                            variant_base.c_str(),
                            optim.c_str(),
                            settings.c_str(),
                            Optim::get_num_threads(),
                            kernel.c_str(),
                            noise.c_str(),
                            objective.c_str(),
                            fname.c_str(),
                            (unsigned long long)fun.d,
                            (unsigned long long)n,
                            s,
                            status.c_str(),
                            obj,
                            ll,
                            theta.c_str(),
                            nug,
                            rmse,
                            wall,
                            cpu);
                std::fflush(stdout);
              }
            }
        }
      }
    }
  }
  return 0;
}
