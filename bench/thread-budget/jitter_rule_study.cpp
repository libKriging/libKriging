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
//   rmse / nlpd / cov95 test scores (500 points, noise-free truth), wall_s
// Variant (environment LK_JR=ii, same hook): commit with the search jitter.
// The study hook (measured on master 6e008b2, not committed) reads LK_JR:
//   LinearAlgebra::model_chol_jitter(double kj): return kj if LK_JR == "ii";
//   Kriging::logLikelihood(): if LK_JR == "i", evaluate with m_chol_jitter
//     temporarily set to m_model_chol_jitter (the committed, exact diagonal);
//   WarpKriging::logLikelihood(): if LK_JR == "i", return concentrated_ll()
//     (the committed exact cache).
// Models: kriging, none (WarpKriging, "none" warps), kuma (kumaraswamy warps),
// mlp (MLPKriging({8}, 2, tanh), noise-free only), nested (NestedKriging, 4
// groups, NK aggregation, noise-free only: scores only), cat / ord (WarpKriging,
// "none" warps plus one categorical(3,2) / ordinal(4) column whose levels add an
// effect to y), disc (discrete inputs only, categorical(4,2) x ordinal(5):
// duplicated rows, R exactly singular: the case of the num_nugget search floor).
// Variant LK_JR=nofloor (same hook): WarpKriging searches with Kriging's jitter
// (kernel_chol_jitter noise-free, 0 with a noise vector), without the floor.
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
      "variant,model,kernel,noise,function,d,n,set,status,ll,llfun,llk,ll_upd,llfun_upd,ll_i,ll_i_upd,theta,rmse,nlpd,"
      "cov95,wall_s\n");
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
        // discrete column of the models cat / ord: level in 0..L-1, effect added to y
        arma::arma_rng::set_seed(4000 + s);
        const arma::vec ulev = arma::randu<arma::vec>(n), ulev_t = arma::randu<arma::vec>(500),
                        ulev_u = arma::randu<arma::vec>(10);
        const arma::vec ulev2 = arma::randu<arma::vec>(n), ulev2_t = arma::randu<arma::vec>(500),
                        ulev2_u = arma::randu<arma::vec>(10);
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
              std::fprintf(stderr,
                           "FIT %s %s %s %s n=%llu set=%d\n",
                           model.c_str(),
                           kernel.c_str(),
                           noise.c_str(),
                           fname.c_str(),
                           (unsigned long long)n,
                           s);
              double ll = NAN, llfun = NAN, llk = NAN, llu = NAN, llfu = NAN, wall = NAN, lli = NAN, lliu = NAN;
              arma::vec th, m, sd;
              std::string status = "ok";
              arma::vec yt_used = yt;
              arma::mat Xt_used = Xt;
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
                  auto gam = [&]() {
                    return het ? arma::vec(arma::join_cols(k.theta(), arma::vec{k.sigma2()})) : k.theta();
                  };
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
                    auto p = k.predict(Xt_used, true, false, false);
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
                    if (model == "disc") {
                      // discrete inputs only: categorical(4,2) x ordinal(5), 20 combinations,
                      // duplicated rows (R exactly singular), y a function of the levels
                      const double sdy = arma::stddev(y0);
                      auto mk = [&](const arma::vec& u1, const arma::vec& u2) {
                        return arma::mat(arma::join_rows(arma::floor(4 * u1), arma::floor(5 * u2)));
                      };
                      auto g = [&](const arma::mat& Z) {
                        arma::vec v(Z.n_rows);
                        for (arma::uword i = 0; i < Z.n_rows; ++i)
                          v(i) = sdy * (std::sin(1.7 * Z(i, 0) + 0.3) + 0.25 * Z(i, 1) * (1 + 0.5 * Z(i, 0)));
                        return v;
                      };
                      const arma::mat Z = mk(ulev, ulev2), Zt = mk(ulev_t, ulev2_t), Zu = mk(ulev_u, ulev2_u);
                      const arma::vec yz = g(Z) + (het ? arma::vec(nsd * eps) : arma::vec(n, arma::fill::zeros));
                      const arma::vec yzu = g(Zu) + (het ? arma::vec(nsd * epsu) : arma::vec(10, arma::fill::zeros));
                      yt_used = g(Zt);
                      Xt_used = Zt;
                      WarpKriging k({"categorical(4,2)", "ordinal(5)"}, kernel);
                      WarpKriging::Parameters pa;
                      if (het)
                        pa.noise = nv;
                      run(k, [&] { k.fit(yz, Z, Trend::RegressionModel::Constant, false, optim, "LL", pa); });
                      k.update(yzu, Zu, false, het ? nvu : arma::vec());
                      llu = k.logLikelihood();
                      lliu = ll_committed(k, hook && !jr);
                      llfu = std::get<0>(k.logLikelihoodFun(k.theta(), false));
                    } else {
                      const bool disc = model == "cat" || model == "ord";
                      std::vector<std::string> warps(d, model == "kuma" ? "kumaraswamy" : "none");
                      if (disc)
                        warps.push_back(model == "cat" ? "categorical(3,2)" : "ordinal(4)");
                      WarpKriging k(warps, kernel);
                      WarpKriging::Parameters pa;
                      if (het)
                        pa.noise = nv;
                      if (disc) {
                        // 3 unordered levels (effects 0, 0.8, -0.5 sd(y0)) or 4 ordered ones (0.4 level sd(y0))
                        const double L = model == "cat" ? 3 : 4, sdy = arma::stddev(y0);
                        auto lev = [&](const arma::vec& u) { return arma::vec(arma::floor(L * u)); };
                        auto eff = [&](const arma::vec& l) {
                          arma::vec e = l;
                          e.transform([&](double v) {
                            return model == "cat" ? (v == 1 ? 0.8 : v == 2 ? -0.5 : 0.0) * sdy : 0.4 * v * sdy;
                          });
                          return e;
                        };
                        const arma::vec l = lev(ulev), lt = lev(ulev_t), lu = lev(ulev_u);
                        const arma::mat Xd = arma::join_rows(X, l), Xtd = arma::join_rows(Xt, lt),
                                        Xud = arma::join_rows(Xu, lu);
                        const arma::vec yd = y + eff(l), yud = yu + eff(lu);
                        yt_used = yt + eff(lt);
                        Xt_used = Xtd;
                        run(k, [&] { k.fit(yd, Xd, Trend::RegressionModel::Constant, false, optim, "LL", pa); });
                        k.update(yud, Xud, false, het ? nvu : arma::vec());
                      } else {
                        run(k, [&] { k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", pa); });
                        k.update(yu, Xu, false, het ? nvu : arma::vec());
                      }
                      llu = k.logLikelihood();
                      lliu = ll_committed(k, hook && !jr);
                      llfu = std::get<0>(k.logLikelihoodFun(k.theta(), false));
                    }
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
              if (m.n_elem == yt_used.n_elem) {
                const arma::vec e = m - yt_used;
                rmse = std::sqrt(arma::mean(arma::square(e))) / arma::stddev(yt_used);
                if (sd.n_elem == yt_used.n_elem) {
                  nlpd = arma::mean(0.5 * arma::log(2 * M_PI * arma::square(sd)) + 0.5 * arma::square(e / sd));
                  cov = arma::mean(arma::conv_to<arma::vec>::from(arma::abs(e) <= 1.959964 * sd));
                }
              }
              std::printf(
                  "%s,%s,%s,%s,%s,%llu,%llu,%d,%s,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%s,%.6g,%.6g,%.4g,%.4g\n",
                  variant.c_str(),
                  model.c_str(),
                  kernel.c_str(),
                  noise.c_str(),
                  fname.c_str(),
                  (unsigned long long)d,
                  (unsigned long long)n,
                  s,
                  status.c_str(),
                  ll,
                  llfun,
                  llk,
                  llu,
                  llfu,
                  lli,
                  lliu,
                  ths.c_str(),
                  rmse,
                  nlpd,
                  cov,
                  wall);
              std::fflush(stdout);
            }
        }
      }
    }
  }
  return 0;
}
