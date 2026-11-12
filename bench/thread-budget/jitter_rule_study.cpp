// Study driver (not built by CMake, see README.md for the build): jitter rule
// of the classes other than Kriging (#465, docs/dev/CholJitter.md, "One rule
// for every class"). Kriging is measured by bench-fit-protocol --candidates 1.
// For every model x kernel x noise x function x n x data set (the T7 data of
// bench/bench-fit-protocol.cpp), one fit, then:
//   ll        logLikelihood()
//   llfun     the objective re-evaluated at the committed parameters
//             (logLikelihoodFun(theta()), with sigma2 for Kriging + noise)
//   llk       Kriging's objective at the same parameters (model "none" only:
//             a WarpKriging with "none" warps is a Kriging model)
//   ll_upd / llfun_upd  the same after update(refit = false) with 10 points
//   ll_i / ll_i_upd     candidate (i), logLikelihood() of the committed model, with
//             a library carrying the study hook (LK_JR_HOOK=1; see ll_committed)
// Variant (environment LK_JR=ii, same hook): commit with the search jitter.
//   rmse / nlpd / cov95 test scores (500 points, noise-free truth), wall_s
// Models: kriging, none (WarpKriging, "none" warps), kuma (kumaraswamy warps),
// mlp (MLPKriging({8}, 2, tanh), noise-free only), nested (NestedKriging, 4
// groups, NK aggregation, noise-free only: scores only).
// Usage: jitter_rule_study <models> <kernels> <noises> <functions> <ns> <sets> <optim>
//   comma-separated lists; noise: none, hetero (known noise variance
//   (0.05 sd(y))^2, y perturbed accordingly)
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include "libKriging/Kriging.hpp"
#include "libKriging/MLPKriging.hpp"
#include "libKriging/NestedKriging.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

using namespace libKriging;

static std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty())
      v.push_back(tok);
  return v;
}

// bench/bench-fit-protocol.cpp: the same families, formulas and seeds
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

static void parse(const std::string& name, std::string& fam, arma::uword& d) {
  for (const std::string f0 : {"sinhalf", "sin", "branin", "hartmann6"})
    if (name.rfind(f0, 0) == 0) {
      fam = f0;
      std::string rest = name.substr(f0.size());
      if (!rest.empty() && rest[0] == '_')
        rest = rest.substr(1);
      d = rest.empty() ? (f0 == "hartmann6" ? 6 : f0 == "branin" ? 2 : 1) : std::stoul(rest);
      return;
    }
  throw std::invalid_argument("unknown function " + name);
}


// Candidate (i), logLikelihood() of the committed model: read through a study
// hook (LK_JR=i, a temporary patch of the library, not committed) toggled
// around the call; NaN with the committed library (no hook, same value as ll).
template <class M>
static double ll_committed(M& k, bool active) {
  if (!active)
    return NAN;
  setenv("LK_JR", "i", 1);
  const double v = k.logLikelihood();
  unsetenv("LK_JR");
  return v;
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(stderr, "usage: %s <models> <kernels> <noises> <functions> <ns> <sets> <optim>\n", argv[0]);
    return 1;
  }
  const auto models = split(argv[1]), kernels = split(argv[2]), noises = split(argv[3]), functions = split(argv[4]),
             ns = split(argv[5]);
  const int sets = std::atoi(argv[6]);
  const std::string optim = argv[7];
  const char* jr = std::getenv("LK_JR");
  const std::string variant = jr ? jr : "current";
  const bool hook = std::getenv("LK_JR_HOOK") != nullptr;  // library built with the study hook
  std::printf(
      "variant,model,kernel,noise,function,d,n,set,status,ll,llfun,llk,ll_upd,llfun_upd,ll_i,ll_i_upd,theta,rmse,nlpd,cov95,wall_s\n");
  for (const auto& fname : functions) {
    std::string fam;
    arma::uword d;
    parse(fname, fam, d);
    for (const auto& ns_ : ns) {
      const arma::uword n = std::stoul(ns_);
      for (int s = 0; s < sets; ++s) {
        arma::arma_rng::set_seed(9000 + s);
        const arma::mat X(n, d, arma::fill::randu);
        const arma::rowvec freq = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
        const arma::vec y0 = f(fam, X, freq);
        arma::arma_rng::set_seed(5000 + s);
        const arma::vec eps = arma::randn<arma::vec>(n);
        arma::arma_rng::set_seed(777);
        const arma::mat Xt(500, d, arma::fill::randu);
        const arma::vec yt = f(fam, Xt, freq);
        arma::arma_rng::set_seed(3000 + s);
        const arma::mat Xu(10, d, arma::fill::randu);
        const arma::vec yu0 = f(fam, Xu, freq);
        const arma::vec epsu = arma::randn<arma::vec>(10);
        for (const auto& noise : noises) {
          const bool het = noise == "hetero";
          const double nsd = 0.05 * arma::stddev(y0);
          const arma::vec y = het ? arma::vec(y0 + nsd * eps) : y0;
          const arma::vec yu = het ? arma::vec(yu0 + nsd * epsu) : yu0;
          const arma::vec nv(n, arma::fill::value(nsd * nsd)), nvu(10, arma::fill::value(nsd * nsd));
          for (const auto& kernel : kernels)
            for (const auto& model : models) {
              if (het && (model == "nested" || model == "mlp"))
                continue;  // no noise channel
              std::fprintf(stderr, "FIT %s %s %s %s n=%llu set=%d\n", model.c_str(), kernel.c_str(), noise.c_str(),
                           fname.c_str(), (unsigned long long)n, s);
              double ll = NAN, llfun = NAN, llk = NAN, llu = NAN, llfu = NAN, wall = NAN, lli = NAN, lliu = NAN;
              arma::vec th, m, sd;
              std::string status = "ok";
              const auto t0 = std::chrono::steady_clock::now();
              try {
                if (model == "kriging") {
                  Kriging k(kernel, het ? Kriging::NoiseModel::Heterogeneous : Kriging::NoiseModel::None);
                  if (het)
                    k.fit(y, nv, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
                  else
                    k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
                  wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                  auto p = k.predict(Xt, true, false, false);
                  m = std::get<0>(p), sd = std::get<1>(p), th = k.theta();
                  auto gam = [&]() { return het ? arma::vec(arma::join_cols(k.theta(), arma::vec{k.sigma2()})) : k.theta(); };
                  ll = k.logLikelihood();
                  lli = ll_committed(k, hook && !jr);
                  llfun = std::get<0>(k.logLikelihoodFun(gam(), false, false));
                  if (het)
                    k.update(yu, nvu, Xu, false);
                  else
                    k.update(yu, Xu, false);
                  llu = k.logLikelihood();
                  lliu = ll_committed(k, hook && !jr);
                  llfu = std::get<0>(k.logLikelihoodFun(gam(), false, false));
                } else if (model == "nested") {
                  NestedKriging k(kernel);
                  k.fit(y, X, 4, Trend::RegressionModel::Constant, optim, "LL");
                  wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                  auto p = k.predict(Xt, true);
                  m = std::get<0>(p), sd = std::get<1>(p);
                } else {
                  auto run = [&](auto& k, auto&& do_fit) {
                    do_fit();
                    wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    auto p = k.predict(Xt, true, false, false);
                    m = std::get<0>(p), sd = std::get<1>(p), th = k.theta();
                    ll = k.logLikelihood();
                    lli = ll_committed(k, hook && !jr);
                    llfun = std::get<0>(k.logLikelihoodFun(k.theta(), false));
                    if (model == "none") {
                      Kriging kr(kernel, het ? Kriging::NoiseModel::Heterogeneous : Kriging::NoiseModel::None);
                      Kriging::Parameters fx;
                      fx.theta = arma::mat(th.t());
                      fx.is_theta_estim = false;
                      if (het) {
                        fx.sigma2 = k.sigma2();
                        fx.is_sigma2_estim = false;
                      }
                      if (het)
                        kr.fit(y, nv, X, Trend::RegressionModel::Constant, false, "none", "LL", fx);
                      else
                        kr.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", fx);
                      const arma::vec g = het ? arma::vec(arma::join_cols(th, arma::vec{k.sigma2()})) : th;
                      llk = std::get<0>(kr.logLikelihoodFun(g, false, false));
                    }
                  };
                  if (model == "mlp") {
                    MLPKriging k({8}, 2, "tanh", kernel);
                    run(k, [&] { k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL"); });
                    k.update(yu, Xu, false);
                    llu = k.logLikelihood();
                    lliu = ll_committed(k, hook && !jr);
                    llfu = std::get<0>(k.logLikelihoodFun(k.theta(), false));
                  } else {
                    WarpKriging k(std::vector<std::string>(d, model == "kuma" ? "kumaraswamy" : "none"), kernel);
                    WarpKriging::Parameters pa;
                    if (het)
                      pa.noise = nv;
                    run(k, [&] { k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", pa); });
                    k.update(yu, Xu, false, het ? nvu : arma::vec());
                    llu = k.logLikelihood();
                    lliu = ll_committed(k, hook && !jr);
                    llfu = std::get<0>(k.logLikelihoodFun(k.theta(), false));
                  }
                }
              } catch (const std::exception& e) {
                status = "error";
                std::fprintf(stderr, "  error: %s\n", e.what());
              }
              std::string ths;
              for (arma::uword j = 0; j < th.n_elem; ++j) {
                char b[32];
                std::snprintf(b, sizeof b, "%s%.6g", j ? ";" : "", th(j));
                ths += b;
              }
              double rmse = NAN, nlpd = NAN, cov = NAN;
              if (m.n_elem == yt.n_elem) {
                const arma::vec e = m - yt;
                rmse = std::sqrt(arma::mean(arma::square(e))) / arma::stddev(yt);
                if (sd.n_elem == yt.n_elem) {
                  nlpd = arma::mean(0.5 * arma::log(2 * M_PI * arma::square(sd)) + 0.5 * arma::square(e / sd));
                  cov = arma::mean(arma::conv_to<arma::vec>::from(arma::abs(e) <= 1.959964 * sd));
                }
              }
              std::printf("%s,%s,%s,%s,%s,%llu,%llu,%d,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s,%.6g,%.6g,%.4g,%.4g\n",
                          variant.c_str(), model.c_str(), kernel.c_str(), noise.c_str(), fname.c_str(),
                          (unsigned long long)d, (unsigned long long)n, s, status.c_str(), ll, llfun, llk, llu, llfu, lli, lliu,
                          ths.c_str(), rmse, nlpd, cov, wall);
              std::fflush(stdout);
            }
        }
      }
    }
  }
  return 0;
}
