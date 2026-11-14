// Study driver (not built by CMake, see README.md for the build): the rcond
// test of the Cholesky retry (#465, docs/dev/CholJitter.md, "The rcond test of
// the Cholesky retry"), on every class, kernel and noise model, and on the
// paths after the fit (predict, update, simulate).
// Variant: environment LK_RCOND
//   on    current: retry when the Cholesky fails or rcond(L)^2 < min_rcond
//   off   LinearAlgebra::check_chol_rcond(false): retry on a failing Cholesky only
//   nf    set_chol_rcond_bound(3): no rcond test on a noise-free matrix (the
//         search with J and the exact committed model), kept with a noise model
//   J     set_chol_rcond_bound(1): test skipped when the search jitter bounds rcond
//   diag  set_chol_rcond_bound(2): test skipped when any diagonal shift bounds rcond
//   diagc set_chol_rcond_bound(4): diag, and no test on the noise-free committed model
// For every model x kernel x noise x function x n x data set (the T7 data of
// bench/bench-fit-protocol.cpp), one fit, then:
//   ll, theta, rmse / nlpd / cov95 (500 test points, noise-free truth), wall_s
//   rf_fit / rr_fit  retries of the Cholesky during the fit and predict, on a
//                    failing Cholesky / on the rcond test
//   interp           noise-free: max |predict(X) - y| / sd(y) (the committed
//                    model interpolates when exact)
//   rmse_upd / nlpd_upd / rr_upd  after update(refit = false) with 10 points
//   sim_z / sim_sdr / rr_sim  Kriging: 200 conditional simulations at 200 test
//                    points: RMS of (mean - predict mean) / (sd / sqrt(nsim))
//                    (~1), median sd(simulations) / predict sd (~1); status
//                    "sim_error" if simulate throws
// Models: kriging (noise none, nugget: nugget estimated on y + noise,
// hetero: known noise variance), none (WarpKriging, "none" warps), kuma
// (kumaraswamy warps), nested (NestedKriging, 4 groups, noise-free), mlp
// (MLPKriging({8}, 2, tanh), noise-free). Noise sd: 0.05 sd(y).
// Usage: rcond_study <models> <kernels> <noises> <functions> <ns> <sets> <optim> [first_set]
#define _USE_MATH_DEFINES
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
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

static unsigned long long rcond_retries() {
  unsigned long long a, b;
  LinearAlgebra::chol_retry_counts(a, b);
  return b;
}
static unsigned long long fail_retries() {
  unsigned long long a, b;
  LinearAlgebra::chol_retry_counts(a, b);
  return a;
}

static std::string vec_str(const arma::vec& th) {
  std::string s;
  for (arma::uword j = 0; j < th.n_elem; ++j) {
    char b[32];
    std::snprintf(b, sizeof b, "%s%.6g", j ? ";" : "", th(j));
    s += b;
  }
  return s;
}

static void scores(const arma::vec& m,
                   const arma::vec& sd,
                   const arma::vec& yt,
                   double& rmse,
                   double& nlpd,
                   double& cov) {
  rmse = nlpd = cov = NAN;
  if (m.n_elem != yt.n_elem)
    return;
  const arma::vec e = m - yt;
  rmse = std::sqrt(arma::mean(arma::square(e))) / arma::stddev(yt);
  if (sd.n_elem == yt.n_elem) {
    nlpd = arma::mean(0.5 * arma::log(2 * M_PI * arma::square(sd)) + 0.5 * arma::square(e / sd));
    cov = arma::mean(arma::conv_to<arma::vec>::from(arma::abs(e) <= 1.959964 * sd));
  }
}

int main(int argc, char** argv) {
  if (argc < 8) {
    std::fprintf(
        stderr, "usage: %s <models> <kernels> <noises> <functions> <ns> <sets> <optim> [first_set]\n", argv[0]);
    return 1;
  }
  const auto models = split(argv[1]), kernels = split(argv[2]), noises = split(argv[3]), functions = split(argv[4]),
             ns = split(argv[5]);
  const int sets = std::atoi(argv[6]);
  const std::string optim = argv[7];
  const int first = argc > 8 ? std::atoi(argv[8]) : 0;
  const char* v = std::getenv("LK_RCOND");
  const std::string variant = v ? v : "on";
  if (variant == "off")
    LinearAlgebra::check_chol_rcond(false);
  else if (variant == "nf")
    LinearAlgebra::set_chol_rcond_bound(3);
  else if (variant == "J")
    LinearAlgebra::set_chol_rcond_bound(1);
  else if (variant == "diag")
    LinearAlgebra::set_chol_rcond_bound(2);
  else if (variant == "diagc")
    LinearAlgebra::set_chol_rcond_bound(4);
  else if (variant != "on") {
    std::fprintf(stderr, "unknown LK_RCOND %s\n", variant.c_str());
    return 1;
  }
  std::printf(
      "variant,model,kernel,noise,function,d,n,set,status,ll,theta,rmse,nlpd,cov95,wall_s,rf_fit,rr_fit,interp,"
      "rmse_upd,nlpd_upd,rr_upd,sim_z,sim_sdr,rr_sim\n");
  for (const auto& fname : functions) {
    std::string fam;
    arma::uword d;
    parse(fname, fam, d);
    for (const auto& ns_ : ns) {
      const arma::uword n = std::stoul(ns_);
      for (int s = first; s < first + sets; ++s) {
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
        const arma::mat Xs = Xt.rows(0, 199);
        for (const auto& noise : noises) {
          const bool noisy = noise != "none";
          const double nsd = 0.05 * arma::stddev(y0);
          const arma::vec y = noisy ? arma::vec(y0 + nsd * eps) : y0;
          const arma::vec yu = noisy ? arma::vec(yu0 + nsd * epsu) : yu0;
          const arma::vec nv(n, arma::fill::value(nsd * nsd)), nvu(10, arma::fill::value(nsd * nsd));
          for (const auto& kernel : kernels)
            for (const auto& model : models) {
              if (noisy && (model == "nested" || model == "mlp"))
                continue;  // no noise channel
              if (noise == "nugget" && model != "kriging")
                continue;  // nugget model: Kriging only
              std::fprintf(stderr,
                           "FIT %s %s %s %s %s n=%llu set=%d\n",
                           variant.c_str(),
                           model.c_str(),
                           kernel.c_str(),
                           noise.c_str(),
                           fname.c_str(),
                           (unsigned long long)n,
                           s);
              double ll = NAN, wall = NAN, interp = NAN, rmse = NAN, nlpd = NAN, cov = NAN, rmseu = NAN, nlpdu = NAN,
                     simz = NAN, simsdr = NAN;
              unsigned long long rf = 0, rr = 0, rru = 0, rrs = 0;
              unsigned long long base_f = 0, base_r = 0;  // retries of the fit, kriging (counted before simulate)
              arma::vec th;
              std::string status = "ok";
              LinearAlgebra::reset_chol_retry_counts();
              const auto t0 = std::chrono::steady_clock::now();
              try {
                auto after_fit = [&](auto& k, auto&& pred, auto&& upd) {
                  wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                  auto [m, sd] = pred(Xt);
                  scores(m, sd, yt, rmse, nlpd, cov);
                  if (!noisy) {
                    auto [mx, sdx] = pred(X);
                    interp = arma::abs(mx - y).max() / arma::stddev(y);
                  }
                  const unsigned long long r1 = rcond_retries();
                  rf = base_f + fail_retries();
                  rr = base_r + r1;
                  upd();
                  auto [mu, sdu] = pred(Xt);
                  double c;
                  scores(mu, sdu, yt, rmseu, nlpdu, c);
                  rru = rcond_retries() - r1;
                  (void)k;
                };
                if (model == "kriging") {
                  const auto nm = noise == "nugget"   ? Kriging::NoiseModel::Nugget
                                  : noise == "hetero" ? Kriging::NoiseModel::Heterogeneous
                                                      : Kriging::NoiseModel::None;
                  Kriging k(kernel, nm);
                  if (noise == "hetero")
                    k.fit(y, nv, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
                  else
                    k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", Kriging::Parameters{});
                  ll = k.logLikelihood();
                  th = k.theta();
                  // simulate on the fitted model, before the update
                  base_f = fail_retries();
                  base_r = rcond_retries();
                  const unsigned long long r0 = base_r;
                  try {
                    const int nsim = 200;
                    const arma::mat S = k.simulate(nsim, 123, Xs);
                    auto p = k.predict(Xs, true, false, false);
                    const arma::vec ms = arma::mean(S, 1), ss = arma::stddev(S, 0, 1);
                    const arma::vec m = std::get<0>(p), sd = std::get<1>(p);
                    const arma::uvec ok = arma::find(sd > 1e-12 * arma::stddev(y));
                    if (!ok.is_empty()) {
                      simz = std::sqrt(
                          arma::mean(arma::square((ms(ok) - m(ok)) / (sd(ok) / std::sqrt(static_cast<double>(nsim))))));
                      simsdr = arma::median(ss(ok) / sd(ok));
                    }
                  } catch (const std::exception& e) {
                    status = "sim_error";
                    std::fprintf(stderr, "  sim error: %s\n", e.what());
                  }
                  rrs = rcond_retries() - r0;
                  LinearAlgebra::reset_chol_retry_counts();  // rr_fit: fit (wall) + predict, not simulate
                  after_fit(
                      k,
                      [&](const arma::mat& Z) {
                        auto p = k.predict(Z, true, false, false);
                        return std::make_pair(arma::vec(std::get<0>(p)), arma::vec(std::get<1>(p)));
                      },
                      [&] {
                        if (noise == "hetero")
                          k.update(yu, nvu, Xu, false);
                        else
                          k.update(yu, Xu, false);
                      });
                } else if (model == "nested") {
                  NestedKriging k(kernel);
                  k.fit(y, X, 4, Trend::RegressionModel::Constant, optim, "LL");
                  after_fit(
                      k,
                      [&](const arma::mat& Z) {
                        auto p = k.predict(Z, true);
                        return std::make_pair(arma::vec(std::get<0>(p)), arma::vec(std::get<1>(p)));
                      },
                      [&] {});
                } else {
                  auto pred_of = [](auto& k) {
                    return [&k](const arma::mat& Z) {
                      auto p = k.predict(Z, true, false, false);
                      return std::make_pair(arma::vec(std::get<0>(p)), arma::vec(std::get<1>(p)));
                    };
                  };
                  if (model == "mlp") {
                    MLPKriging k({8}, 2, "tanh", kernel);
                    k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL");
                    ll = k.logLikelihood();
                    th = k.theta();
                    after_fit(k, pred_of(k), [&] { k.update(yu, Xu, false); });
                  } else {
                    std::vector<std::string> warps(d, model == "kuma" ? "kumaraswamy" : "none");
                    WarpKriging k(warps, kernel);
                    WarpKriging::Parameters pa;
                    if (noisy)
                      pa.noise = nv;
                    k.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", pa);
                    ll = k.logLikelihood();
                    th = k.theta();
                    after_fit(k, pred_of(k), [&] { k.update(yu, Xu, false, noisy ? nvu : arma::vec()); });
                  }
                }
              } catch (const std::exception& e) {
                status = "error";
                std::fprintf(stderr, "  error: %s\n", e.what());
              }
              std::printf(
                  "%s,%s,%s,%s,%s,%llu,%llu,%d,%s,%.10g,%s,%.6g,%.6g,%.4g,%.4g,%llu,%llu,%.3g,%.6g,%.6g,%llu,%.4g,%.4g,"
                  "%"
                  "llu\n",
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
                  vec_str(th).c_str(),
                  rmse,
                  nlpd,
                  cov,
                  wall,
                  rf,
                  rr,
                  interp,
                  rmseu,
                  nlpdu,
                  rru,
                  simz,
                  simsdr,
                  rrs);
              std::fflush(stdout);
            }
        }
      }
    }
  }
  return 0;
}
