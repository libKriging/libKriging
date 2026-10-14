// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/KrigingImpl.hpp"
#include "libKriging/Trend.hpp"
// clang-format on

// With normalize=true, Kriging reads its Parameters in the original scale of
// X and y and must give every public output (accessors, objective functions
// and their gradients) in that same original scale: input and output
// round-trip, and a normalize=true model agrees with a normalize=false one
// fitted on the same data with the same (original-scale) parameters.

namespace {

using RM = Trend::RegressionModel;
using NM = Kriging::NoiseModel;

// anisotropic, far-from-unit-scale design and output
struct Data {
  arma::mat X;
  arma::vec y;
  arma::vec noise;
};

Data make_data(arma::uword n = 25) {
  arma::arma_rng::set_seed(4321);
  Data D;
  D.X = arma::mat(n, 2, arma::fill::randu);
  D.X.col(0) = 10.0 + 20.0 * D.X.col(0);  // [10, 30]
  D.X.col(1) = -2.0 + 5.0 * D.X.col(1);   // [-2, 3]
  D.y = arma::vec(n);
  for (arma::uword i = 0; i < n; i++) {
    const double u = (D.X(i, 0) - 10.0) / 20.0, v = (D.X(i, 1) + 2.0) / 5.0;
    D.y(i) = 100.0 + 50.0 * (std::sin(5.0 * u) * std::cos(3.0 * v) + 0.5 * u * v);
  }
  D.noise = 0.5 + 0.5 * arma::vec(n, arma::fill::randu);  // original-scale noise variances
  return D;
}

arma::uword trend_size(RM rm, arma::uword d) {
  switch (rm) {
    case RM::None:
      return 0;
    case RM::Constant:
      return 1;
    case RM::Linear:
      return 1 + d;
    case RM::Interactive:
      return 1 + d + d * (d - 1) / 2;
    case RM::Quadratic:
      return 1 + 2 * d + d * (d - 1) / 2;
  }
  return 0;
}

Kriging make_model(NM nm, const std::string& kernel = "matern5_2") {
  return Kriging(kernel, nm);
}

void fit(Kriging& k,
         const Data& D,
         RM rm,
         bool normalize,
         const std::string& optim,
         const std::string& objective,
         const Kriging::Parameters& p) {
  if (k.noise_model() == NM::Heterogeneous)
    k.fit(D.y, D.noise, D.X, rm, normalize, optim, objective, p);
  else
    k.fit(D.y, D.X, rm, normalize, optim, objective, p);
}

void require_close(const arma::vec& a, const arma::vec& b, double rel) {
  REQUIRE(a.n_elem == b.n_elem);
  for (arma::uword i = 0; i < a.n_elem; i++) {
    CAPTURE(i, a(i), b(i));
    CHECK(a(i) == Approx(b(i)).epsilon(rel).margin(rel));
  }
}

// gamma = [theta] / [theta, alpha] / [theta, sigma2] of a fitted model, original scale
arma::vec public_gamma(const Kriging& k) {
  if (k.noise_model() == NM::None)
    return k.theta();
  const double extra = (k.noise_model() == NM::Nugget) ? k.sigma2() / (k.sigma2() + k.nugget()) : k.sigma2();
  return arma::join_cols(k.theta(), arma::vec{extra});
}

// central finite-difference gradient of f at x (relative steps)
template <typename F>
arma::vec fd_grad(F&& f, const arma::vec& x, double rel_step = 1e-5) {
  arma::vec g(x.n_elem);
  for (arma::uword i = 0; i < x.n_elem; i++) {
    const double h = rel_step * std::max(std::abs(x(i)), 1e-3);
    arma::vec xp = x, xm = x;
    xp(i) += h;
    xm(i) -= h;
    g(i) = (f(xp) - f(xm)) / (2 * h);
  }
  return g;
}

}  // namespace

TEST_CASE("normalize=true: fixed parameters round-trip through the accessors", "[normalize][kriging]") {
  const Data D = make_data();
  const arma::uword d = D.X.n_cols;
  const RM rm = GENERATE(RM::None, RM::Constant, RM::Linear, RM::Interactive, RM::Quadratic);
  const NM nm = GENERATE(NM::None, NM::Nugget, NM::Heterogeneous);
  CAPTURE(Trend::toString(rm), static_cast<int>(nm));

  Kriging::Parameters p;
  p.theta = arma::mat{{7.0, 1.3}};
  p.is_theta_estim = false;
  p.sigma2 = 321.0;
  p.is_sigma2_estim = false;
  const arma::uword np = trend_size(rm, d);
  arma::vec beta_given = arma::linspace(-3.0, 5.0, np) + 0.37;
  if (np > 0) {
    p.beta = beta_given;
    p.is_beta_estim = false;
  }
  if (nm == NM::Nugget) {
    p.nugget = 12.5;
    p.is_nugget_estim = false;
  }

  Kriging k = make_model(nm);
  fit(k, D, rm, /*normalize=*/true, "none", "LL", p);

  require_close(k.theta(), arma::vec{7.0, 1.3}, 1e-12);
  CHECK(k.sigma2() == Approx(321.0).epsilon(1e-12));
  if (np > 0)
    require_close(k.beta(), beta_given, 1e-12);
  else
    CHECK(k.beta().is_empty());
  if (nm == NM::Nugget)
    CHECK(k.nugget() == Approx(12.5).epsilon(1e-12));
  if (nm == NM::Heterogeneous)
    require_close(k.noise(), D.noise, 1e-12);

  // internal storage is the normalized one
  const KrigingImpl& impl = k;
  require_close(impl.theta() % k.scaleX().t(), k.theta(), 1e-14);
  CHECK(impl.sigma2() * k.scaleY() * k.scaleY() == Approx(k.sigma2()).epsilon(1e-14));

  // the fixed trend is the given one, in the original scale: F(x) * beta
  // (None trend: implicit centerY)
  if (np > 0) {
    // predict far from the data so that the kriging correction vanishes
    const arma::mat Xfar{{1e4, -1e4}, {-5e3, 2e3}};
    const arma::vec mean = std::get<0>(k.predict(Xfar, false, false, false));
    const arma::vec trend = Trend::regressionModelMatrix(rm, Xfar) * beta_given;
    require_close(mean, trend, 1e-9);
  }
}

TEST_CASE("normalize=true agrees with normalize=false at the same original-scale parameters", "[normalize][kriging]") {
  const Data D = make_data();
  // None trend excluded: with normalize=true its (implicit) mean is centerY,
  // a different model from the zero mean of normalize=false.
  const RM rm = GENERATE(RM::Constant, RM::Linear, RM::Interactive, RM::Quadratic);
  const NM nm = GENERATE(NM::None, NM::Nugget, NM::Heterogeneous);
  const bool fixed_beta = GENERATE(false, true);
  CAPTURE(Trend::toString(rm), static_cast<int>(nm), fixed_beta);

  Kriging::Parameters p;
  p.theta = arma::mat{{9.0, 2.1}};
  p.is_theta_estim = false;
  if (nm != NM::None) {  // sigma2 must be given for heterogeneous noise / both for nugget
    p.sigma2 = 450.0;
    p.is_sigma2_estim = false;
  }
  if (nm == NM::Nugget) {
    p.nugget = 3.0;
    p.is_nugget_estim = false;
  }
  if (fixed_beta) {
    p.beta = arma::linspace(1.0, 2.0, trend_size(rm, D.X.n_cols)) * 10.0;
    p.is_beta_estim = false;
  }

  Kriging kn = make_model(nm), ko = make_model(nm);
  fit(kn, D, rm, /*normalize=*/true, "none", "LL", p);
  fit(ko, D, rm, /*normalize=*/false, "none", "LL", p);

  const double tol = 1e-7;
  require_close(kn.theta(), ko.theta(), tol);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  require_close(kn.beta(), ko.beta(), tol);
  if (nm == NM::Nugget)
    CHECK(kn.nugget() == Approx(ko.nugget()).epsilon(tol));
  if (nm == NM::Heterogeneous)
    require_close(kn.noise(), ko.noise(), tol);

  arma::mat Xn = D.X.rows(0, 4) + 0.3;
  auto [mn, sn, cn, dmn, dsn] = kn.predict(Xn, true, true, false);
  auto [mo, so, co, dmo, dso] = ko.predict(Xn, true, true, false);
  require_close(mn, mo, tol);
  require_close(sn, so, tol);
  require_close(arma::vectorise(cn), arma::vectorise(co), tol);
  require_close(arma::vectorise(kn.covMat(Xn, D.X)), arma::vectorise(ko.covMat(Xn, D.X)), tol);

  // objective values and gradients, original scale
  CHECK(kn.logLikelihood() == Approx(ko.logLikelihood()).epsilon(tol));
  const arma::vec g = public_gamma(ko);
  auto [lln, gln] = kn.logLikelihoodFun(g, true, false);
  auto [llo, glo] = ko.logLikelihoodFun(g, true, false);
  CHECK(lln == Approx(llo).epsilon(tol));
  require_close(gln, glo, 1e-6);

  CHECK(kn.leaveOneOut() == Approx(ko.leaveOneOut()).epsilon(tol));
  auto [loon, gloon] = kn.leaveOneOutFun(ko.theta(), true, false);
  auto [looo, glooo] = ko.leaveOneOutFun(ko.theta(), true, false);
  CHECK(loon == Approx(looo).epsilon(tol));
  require_close(gloon, glooo, 1e-6);
  auto [loovm_n, loovs_n] = kn.leaveOneOutVec(ko.theta());
  auto [loovm_o, loovs_o] = ko.leaveOneOutVec(ko.theta());
  require_close(loovm_n, loovm_o, tol);
  require_close(loovs_n, loovs_o, tol);

  if (nm != NM::Heterogeneous) {  // LMP is not defined with a heterogeneous noise
    CHECK(kn.logMargPost() == Approx(ko.logMargPost()).epsilon(tol));
    auto [lmpn, glmpn] = kn.logMargPostFun(ko.theta(), true, false);
    auto [lmpo, glmpo] = ko.logMargPostFun(ko.theta(), true, false);
    CHECK(lmpn == Approx(lmpo).epsilon(tol));
    if (nm == NM::None)  // (Nugget: alpha component from finite differences)
      require_close(glmpn, glmpo, 1e-6);
    else  // fixed sigma2/nugget: finite-difference theta gradient (internal step)
      require_close(glmpn.head(D.X.n_cols), glmpo.head(D.X.n_cols), 1e-3);
  }
}

TEST_CASE("normalize=true BFGS fit agrees with normalize=false", "[normalize][kriging]") {
  const Data D = make_data(30);
  const RM rm = GENERATE(RM::Constant, RM::Linear);
  const NM nm = GENERATE(NM::None, NM::Nugget);
  CAPTURE(Trend::toString(rm), static_cast<int>(nm));

  Kriging kn = make_model(nm), ko = make_model(nm);
  fit(kn, D, rm, /*normalize=*/true, "BFGS10", "LL", {});
  fit(ko, D, rm, /*normalize=*/false, "BFGS10", "LL", {});

  // same optimum up to the optimizer tolerance
  CHECK(kn.logLikelihood() == Approx(ko.logLikelihood()).epsilon(1e-5));
  const double tol = 2e-2;
  require_close(kn.theta(), ko.theta(), tol);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  require_close(kn.beta(), ko.beta(), tol);
  if (nm == NM::Nugget)
    CHECK(kn.nugget() == Approx(ko.nugget()).epsilon(0.1).margin(1e-3 * ko.sigma2()));

  arma::mat Xn = D.X.rows(0, 9) + 0.5;
  auto pn = kn.predict(Xn, true, false, false);
  auto po = ko.predict(Xn, true, false, false);
  require_close(std::get<0>(pn), std::get<0>(po), 1e-3);
  require_close(std::get<1>(pn), std::get<1>(po), 2e-2);
}

TEST_CASE("normalize=true: objectives at theta() are the committed ones, gradients are original-scale",
          "[normalize][kriging]") {
  const Data D = make_data();
  const RM rm = GENERATE(RM::Constant, RM::Linear);
  const NM nm = GENERATE(NM::None, NM::Nugget, NM::Heterogeneous);
  CAPTURE(Trend::toString(rm), static_cast<int>(nm));

  Kriging k = make_model(nm);
  fit(k, D, rm, /*normalize=*/true, "BFGS", "LL", {});

  const arma::vec g = public_gamma(k);
  CHECK(std::get<0>(k.logLikelihoodFun(g, false, false)) == Approx(k.logLikelihood()).epsilon(1e-10));
  CHECK(std::get<0>(k.leaveOneOutFun(k.theta(), false, false)) == Approx(k.leaveOneOut()).epsilon(1e-10));
  if (nm != NM::Heterogeneous)
    CHECK(std::get<0>(k.logMargPostFun(k.theta(), false, false)) == Approx(k.logMargPost()).epsilon(1e-10));

  // analytic gradients vs finite differences, in the original scale, at a
  // point away from the optimum (where the gradient is ~0)
  arma::vec g1 = g;
  g1.head(D.X.n_cols) %= arma::vec{1.3, 0.8};
  if (nm == NM::Nugget)
    g1(D.X.n_cols) = 0.9;  // alpha
  {
    auto [v, grad] = k.logLikelihoodFun(g1, true, false);
    const arma::vec fd
        = fd_grad([&](const arma::vec& x) { return std::get<0>(k.logLikelihoodFun(x, false, false)); }, g1);
    CAPTURE(grad, fd);
    require_close(grad, fd, 1e-4);
  }
  {
    const arma::vec t1 = g1.head(D.X.n_cols);
    auto [v, grad] = k.leaveOneOutFun(t1, true, false);
    const arma::vec fd
        = fd_grad([&](const arma::vec& x) { return std::get<0>(k.leaveOneOutFun(x, false, false)); }, t1);
    CAPTURE(grad, fd);
    require_close(grad, fd, 1e-4);
  }
  if (nm == NM::None) {
    const arma::vec t1 = g1.head(D.X.n_cols);
    auto [v, grad] = k.logMargPostFun(t1, true, false);
    const arma::vec fd
        = fd_grad([&](const arma::vec& x) { return std::get<0>(k.logMargPostFun(x, false, false)); }, t1);
    CAPTURE(grad, fd);
    require_close(grad, fd, 1e-4);
  }
}

TEST_CASE("normalize=true: Vecchia / Nystrom log-likelihoods in the original scale", "[normalize][kriging]") {
  const Data D = make_data(40);
  const std::string method = GENERATE(std::string("vecchia(10)"), std::string("nystrom(15)"));
  const bool vecchia = method.rfind("vecchia", 0) == 0;
  CAPTURE(method);

  Kriging k("matern5_2");
  k.fit(D.y, D.X, RM::Constant, /*normalize=*/true, "BFGS", "LL", {}, method);
  auto fun = [&](const arma::vec& t, bool grad) {
    return vecchia ? k.logLikelihoodVecchiaFun(t, grad) : k.logLikelihoodNystromFun(t, grad);
  };
  const std::string m = vecchia ? "vecchia" : "nystrom";
  // The Nystrom likelihood is only smooth up to ~1e-7 relative (its jittered
  // low-rank factorization; same with normalize=false): looser tolerances.
  const double val_tol = vecchia ? 1e-10 : 1e-6;
  CHECK(std::get<0>(fun(k.theta(), false)) == Approx(k.logLikelihood(m)).epsilon(val_tol));

  const arma::vec t1 = k.theta() % arma::vec{1.2, 0.7};
  auto [v, grad] = fun(t1, true);
  const arma::vec fd
      = fd_grad([&](const arma::vec& x) { return std::get<0>(fun(x, false)); }, t1, vecchia ? 1e-5 : 1e-2);
  CAPTURE(grad, fd);
  require_close(grad, fd, vecchia ? 1e-4 : 5e-2);
}

TEST_CASE("normalize=true: approximate LOO / LMP in the original scale", "[normalize][kriging][vecchia][nystrom]") {
  const Data D = make_data(30);
  // Vecchia with m = n-1 (all predecessors: the maxmin ordering, built in the
  // normalized inputs, then only changes roundoff; and the criterion is the
  // exact one) and Nystrom (landmarks ranked with a reference range scaled to
  // the data extent: the same rows with normalize=true and false), so
  // normalize=true and false coincide at the same original-scale theta
  const std::string method = GENERATE(std::string("vecchia(29)"), std::string("nystrom(15)"));
  const std::string objective = GENERATE(std::string("LOO"), std::string("LMP"));
  const RM rm = GENERATE(RM::Constant, RM::Linear);
  const bool fixed_sigma2 = GENERATE(false, true);
  const bool vecchia = method.rfind("vecchia", 0) == 0;
  const std::string m = vecchia ? "vecchia" : "nystrom";
  CAPTURE(method, objective, Trend::toString(rm), fixed_sigma2);

  Kriging::Parameters p;
  p.theta = arma::mat{{9.0, 2.1}};
  p.is_theta_estim = false;
  if (fixed_sigma2) {
    p.sigma2 = 450.0;
    p.is_sigma2_estim = false;
  }
  Kriging kn("matern5_2"), ko("matern5_2"), ke("matern5_2");
  kn.fit(D.y, D.X, rm, /*normalize=*/true, "none", objective, p, method);
  ko.fit(D.y, D.X, rm, /*normalize=*/false, "none", objective, p, method);
  ke.fit(D.y, D.X, rm, /*normalize=*/false, "none", objective, p);  // exact reference
  REQUIRE(kn.method() == method);

  // roundoff of the different Vecchia orderings / of the jittered Nystrom
  // factorization (smooth to ~1e-7 relative only)
  const double tol = vecchia ? 1e-6 : 1e-5;
  require_close(kn.theta(), ko.theta(), 1e-12);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  // (the Nystrom GLS beta of a linear trend is less stable: ~1e-4, as for LL)
  require_close(kn.beta(), ko.beta(), vecchia ? tol : 1e-3);

  auto loo_fun = [&](Kriging& k, const arma::vec& t, bool g) {
    return vecchia ? k.leaveOneOutVecchiaFun(t, g) : k.leaveOneOutNystromFun(t, g);
  };
  auto lmp_fun = [&](Kriging& k, const arma::vec& t, bool g) {
    return vecchia ? k.logMargPostVecchiaFun(t, g) : k.logMargPostNystromFun(t, g);
  };

  // committed values: no double conversion, == *Fun(theta())
  CHECK(kn.leaveOneOut(m) == Approx(std::get<0>(loo_fun(kn, kn.theta(), false))).epsilon(1e-12));
  CHECK(kn.logMargPost(m) == Approx(std::get<0>(lmp_fun(kn, kn.theta(), false))).epsilon(1e-12));
  CHECK(kn.leaveOneOut(kn.method()) == Approx(ko.leaveOneOut(m)).epsilon(tol));
  CHECK(kn.logMargPost(kn.method()) == Approx(ko.logMargPost(m)).epsilon(tol));
  // Vecchia m = n-1 is exact, in the original scale
  if (vecchia) {
    CHECK(kn.leaveOneOut(m) == Approx(ke.leaveOneOut()).epsilon(1e-6));
    CHECK(kn.logMargPost(m) == Approx(ke.logMargPost()).epsilon(1e-6));
  }

  // values and gradients at an original-scale theta away from the fit
  const arma::vec t1{12.0, 1.4};
  auto [loon, gloon] = loo_fun(kn, t1, true);
  auto [looo, glooo] = loo_fun(ko, t1, true);
  CHECK(loon == Approx(looo).epsilon(tol));
  require_close(gloon, glooo, vecchia ? 1e-5 : 1e-3);
  auto [lmpn, glmpn] = lmp_fun(kn, t1, true);
  auto [lmpo, glmpo] = lmp_fun(ko, t1, true);
  CHECK(lmpn == Approx(lmpo).epsilon(tol));
  require_close(glmpn, glmpo, vecchia ? 1e-5 : 1e-3);
  // the gradients are those of the original-scale theta (the Nystrom
  // criteria are only smooth to ~1e-7 relative: larger steps, as above)
  const double step = vecchia ? 1e-5 : 1e-2;
  const arma::vec fd_loo = fd_grad([&](const arma::vec& x) { return std::get<0>(loo_fun(kn, x, false)); }, t1, step);
  const arma::vec fd_lmp = fd_grad([&](const arma::vec& x) { return std::get<0>(lmp_fun(kn, x, false)); }, t1, step);
  CAPTURE(gloon, fd_loo, glmpn, fd_lmp);
  require_close(gloon, fd_loo, vecchia ? 1e-4 : 1e-2);
  require_close(glmpn, fd_lmp, vecchia ? 1e-4 : 1e-2);
  if (vecchia) {  // exact reference values and gradients at t1
    auto [looe, glooe] = ke.leaveOneOutFun(t1, true, false);
    CHECK(loon == Approx(looe).epsilon(1e-6));
    require_close(gloon, glooe, 1e-5);
    auto [lmpe, glmpe] = ke.logMargPostFun(t1, true, false);
    CHECK(lmpn == Approx(lmpe).epsilon(1e-6));
    require_close(glmpn, glmpe, 1e-5);
  }

  // LOO predictions, original scale of y
  auto [mn, sn] = kn.leaveOneOutVec(t1, m);
  auto [mo, so] = ko.leaveOneOutVec(t1, m);
  require_close(mn, mo, tol);
  require_close(sn, so, tol);
  if (vecchia) {
    auto [me, se] = ke.leaveOneOutVec(t1);
    require_close(mn, me, 1e-6);
    require_close(sn, se, 1e-6);
  }
}

TEST_CASE("normalize=true: approximate LMP fit commits sigma2 = SSE/(n-p) and predict rescales once",
          "[normalize][kriging][vecchia][nystrom]") {
  const Data D = make_data(30);
  const double n = 30.0;
  const double p = 3.0;  // linear trend in d = 2
  const std::string method = GENERATE(std::string("vecchia(29)"), std::string("nystrom(15)"));
  const bool vecchia = method.rfind("vecchia", 0) == 0;
  // (prediction conditions on every observation: m = n for Vecchia)
  const std::string mp = vecchia ? "vecchia(30)" : "nystrom";
  CAPTURE(method);
  const arma::mat Xn = D.X.rows(0, 4) + 0.3;

  // BFGS LMP fit: sigma2 = SSE / (n-p), vs SSE / n for the optim="none"
  // commit at the same theta (same normalization: same numerics)
  Kriging kb("matern5_2");
  kb.fit(D.y, D.X, RM::Linear, /*normalize=*/true, "BFGS", "LMP", {}, method);
  Kriging::Parameters par;
  par.theta = arma::mat(kb.theta().t());
  par.is_theta_estim = false;
  Kriging kf("matern5_2");
  kf.fit(D.y, D.X, RM::Linear, /*normalize=*/true, "none", "LMP", par, method);
  require_close(kb.theta(), kf.theta(), 1e-12);
  CHECK(kb.sigma2() == Approx(kf.sigma2() * n / (n - p)).epsilon(1e-10));
  require_close(kb.beta(), kf.beta(), 1e-10);
  {
    auto [mb, sb, cb, dmb, dsb] = kb.predict(Xn, true, false, false, mp);
    auto [mf, sf, cf, dmf, dsf] = kf.predict(Xn, true, false, false, mp);
    require_close(mb, mf, 1e-10);
    require_close(sb, sf * std::sqrt(n / (n - p)), 1e-8);
  }

  // fixed parameters: the approximate LMP predictor of a normalize=true model
  // applies the LMP variance factor (n-p)/(n-p-2) once (vs the same model
  // with objective LL), and agrees with normalize=false
  Kriging::Parameters pf;
  pf.theta = arma::mat{{4.0, 1.0}};
  pf.is_theta_estim = false;
  pf.sigma2 = 450.0;
  pf.is_sigma2_estim = false;
  Kriging kn("matern5_2"), kl("matern5_2"), ko("matern5_2");
  kn.fit(D.y, D.X, RM::Linear, /*normalize=*/true, "none", "LMP", pf, method);
  kl.fit(D.y, D.X, RM::Linear, /*normalize=*/true, "none", "LL", pf, method);
  ko.fit(D.y, D.X, RM::Linear, /*normalize=*/false, "none", "LMP", pf, method);
  CHECK(kn.sigma2() == Approx(450.0).epsilon(1e-12));
  auto [mn, sn, cn, dmn, dsn] = kn.predict(Xn, true, false, false, mp);
  auto [ml, sl, cl, dml, dsl] = kl.predict(Xn, true, false, false, mp);
  auto [mo, so, co, dmo, dso] = ko.predict(Xn, true, false, false, mp);
  require_close(mn, ml, 1e-12);
  require_close(sn, sl * std::sqrt((n - p) / (n - p - 2.0)), 1e-10);
  // (the Nystrom GLS beta / factorization are stable to ~1e-5 only)
  require_close(mn, mo, vecchia ? 1e-6 : 1e-4);
  require_close(sn, so, vecchia ? 1e-5 : 1e-3);
}

TEST_CASE("normalize=true: Vecchia noise models agree with normalize=false", "[normalize][kriging][vecchia]") {
  const Data D = make_data();
  const arma::uword n = D.X.n_rows;
  // m >= n - 1 (even after the update below, which adds nu points): every
  // observation conditions on all its predecessors, so the Vecchia likelihood
  // is the exact one. With fewer neighbors the two models differ by design:
  // neighbors are searched in the internal (normalized) input space.
  const arma::uword nu = 6;
  const std::string method = "vecchia(" + std::to_string(n + nu - 1) + ")";
  const RM rm = GENERATE(RM::Constant, RM::Linear);
  const NM nm = GENERATE(NM::Nugget, NM::Heterogeneous);
  const bool est_sigma2 = GENERATE(false, true);
  CAPTURE(Trend::toString(rm), static_cast<int>(nm), est_sigma2);

  Kriging::Parameters p;
  p.theta = arma::mat{{9.0, 2.1}};
  p.is_theta_estim = false;
  p.sigma2 = 450.0;
  p.is_sigma2_estim = est_sigma2;
  if (nm == NM::Nugget) {
    p.nugget = 3.0;
    p.is_nugget_estim = est_sigma2;
  }

  auto vfit = [&](bool normalize) {
    Kriging k = make_model(nm);
    if (nm == NM::Heterogeneous)
      k.fit(D.y, D.noise, D.X, rm, normalize, "none", "LL", p, method);
    else
      k.fit(D.y, D.X, rm, normalize, "none", "LL", p, method);
    return k;
  };
  Kriging kn = vfit(true), ko = vfit(false);
  REQUIRE(kn.is_vecchia_light());
  REQUIRE(ko.is_vecchia_light());

  const double tol = 1e-7;
  require_close(kn.theta(), ko.theta(), tol);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  require_close(kn.beta(), ko.beta(), tol);
  if (nm == NM::Nugget) {
    CHECK(kn.nugget() == Approx(ko.nugget()).epsilon(tol));
    CHECK(kn.alpha() == Approx(ko.alpha()).epsilon(tol));  // scale-free
    CHECK(kn.alpha() == Approx(kn.sigma2() / (kn.sigma2() + kn.nugget())).epsilon(tol));
  } else {
    require_close(kn.noise(), ko.noise(), tol);
    require_close(kn.noise(), D.noise, 1e-12);  // round-trip of the given noise
  }

  // likelihood value and gradient at original-scale gamma
  CHECK(kn.logLikelihood("vecchia") == Approx(ko.logLikelihood("vecchia")).epsilon(tol));
  const arma::vec g0 = public_gamma(ko);
  CHECK(std::get<0>(kn.logLikelihoodVecchiaFun(g0, false)) == Approx(kn.logLikelihood("vecchia")).epsilon(1e-10));
  for (const arma::vec& g : {g0, arma::vec(g0 % arma::vec{1.3, 0.8, nm == NM::Nugget ? 0.97 : 1.5})}) {
    CAPTURE(g);
    auto [vn, gn] = kn.logLikelihoodVecchiaFun(g, true);
    auto [vo, go] = ko.logLikelihoodVecchiaFun(g, true);
    CHECK(vn == Approx(vo).epsilon(tol));
    require_close(gn, go, 1e-6);
    // theta-only argument: the committed alpha / sigma2
    CHECK(std::get<0>(kn.logLikelihoodVecchiaFun(g.head(2), false))
          == Approx(std::get<0>(ko.logLikelihoodVecchiaFun(g.head(2), false))).epsilon(tol));
    // finite differences of the original-scale objective
    const arma::vec fd
        = fd_grad([&](const arma::vec& x) { return std::get<0>(kn.logLikelihoodVecchiaFun(x, false)); }, g);
    require_close(gn, fd, 1e-4);
  }

  // exact model at the same parameters: Vecchia(n-1) is exact
  Kriging ke = make_model(nm);
  fit(ke, D, rm, /*normalize=*/true, "none", "LL", p);
  CHECK(std::get<0>(kn.logLikelihoodVecchiaFun(g0, false))
        == Approx(std::get<0>(ke.logLikelihoodFun(g0, false, false))).epsilon(1e-8));

  // predictions (all neighbors) and simulations (all predecessors)
  const arma::mat Xn = arma::join_cols(D.X.rows(0, 3) + 0.3, D.X.rows(5, 6));
  auto [mn, sn] = kn.predictVecchia(Xn, true, n);
  auto [mo, so] = ko.predictVecchia(Xn, true, n);
  require_close(mn, mo, tol);
  require_close(sn, so, tol);
  const arma::uword m_all = n + Xn.n_rows;
  const arma::vec sdn = arma::linspace(0.5, 2.0, Xn.n_rows);  // original-scale standard deviations
  for (bool with_nugget : {false, true}) {
    if (with_nugget && nm != NM::Nugget)
      continue;
    const arma::mat sim_n = kn.simulateVecchia(7, 123, Xn, m_all, with_nugget, sdn);
    const arma::mat sim_o = ko.simulateVecchia(7, 123, Xn, m_all, with_nugget, sdn);
    CAPTURE(with_nugget);
    require_close(arma::vectorise(sim_n), arma::vectorise(sim_o), 1e-6);
  }
  // with_noise is in the original scale, as for the exact simulate(): the
  // added noise does not depend on normalize
  const arma::mat s0 = kn.simulateVecchia(4, 5, Xn, m_all);
  const arma::mat s1 = kn.simulateVecchia(4, 5, Xn, m_all, false, sdn);
  arma::mat eps = s1 - s0;
  eps.each_col() /= sdn;
  Kriging ko2 = vfit(false);
  arma::mat eps_o = ko2.simulateVecchia(4, 5, Xn, m_all, false, sdn) - ko2.simulateVecchia(4, 5, Xn, m_all);
  eps_o.each_col() /= sdn;
  require_close(arma::vectorise(eps), arma::vectorise(eps_o), 1e-6);

  // update without refit (new data in the original scale): both stay equal
  const Data U = make_data(nu);
  const arma::mat Xu = U.X + 0.7;
  if (nm == NM::Heterogeneous) {
    kn.update(U.y, U.noise, Xu, false);
    ko.update(U.y, U.noise, Xu, false);
    require_close(kn.noise(), ko.noise(), tol);
    require_close(kn.noise(), arma::join_cols(D.noise, U.noise), 1e-12);
  } else {
    kn.update(U.y, Xu, false);
    ko.update(U.y, Xu, false);
    CHECK(kn.nugget() == Approx(ko.nugget()).epsilon(tol));
    CHECK(kn.alpha() == Approx(ko.alpha()).epsilon(tol));
  }
  REQUIRE(kn.is_vecchia_light());
  require_close(kn.theta(), ko.theta(), tol);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  require_close(kn.beta(), ko.beta(), tol);
  CHECK(kn.logLikelihood("vecchia") == Approx(ko.logLikelihood("vecchia")).epsilon(tol));
  auto [mun, sun] = kn.predictVecchia(Xn, true, n + nu);
  auto [muo, suo] = ko.predictVecchia(Xn, true, n + nu);
  require_close(mun, muo, tol);
  require_close(sun, suo, tol);
}

TEST_CASE("normalize=true: Vecchia noise-model BFGS fit, objective at the accessors", "[normalize][kriging][vecchia]") {
  const Data D = make_data(40);
  const NM nm = GENERATE(NM::Nugget, NM::Heterogeneous);
  CAPTURE(static_cast<int>(nm));
  Kriging k = make_model(nm);
  if (nm == NM::Heterogeneous)
    k.fit(D.y, D.noise, D.X, RM::Constant, /*normalize=*/true, "BFGS", "LL", {}, "vecchia(10)");
  else
    k.fit(D.y, D.X, RM::Constant, /*normalize=*/true, "BFGS", "LL", {}, "vecchia(10)");
  const arma::vec g = public_gamma(k);
  CHECK(std::get<0>(k.logLikelihoodVecchiaFun(g, false)) == Approx(k.logLikelihood("vecchia")).epsilon(1e-9));
  // the original-scale optimum: (near) zero gradient in theta, and no
  // better value nearby
  const arma::vec g1 = g % arma::vec{1.2, 0.8, nm == NM::Nugget ? 0.99 : 1.3};
  auto [v1, grad1] = k.logLikelihoodVecchiaFun(g1, true);
  const arma::vec fd
      = fd_grad([&](const arma::vec& x) { return std::get<0>(k.logLikelihoodVecchiaFun(x, false)); }, g1);
  CAPTURE(grad1, fd);
  require_close(grad1, fd, 1e-4);
  CHECK(v1 <= k.logLikelihood("vecchia") + 1e-6);

  // save / load keeps the original-scale accessors and objective
  const std::string f = "KrigingNormalizeScaleTest_vecchia_saved.json";
  k.save(f);
  Kriging kl = Kriging::load(f);
  require_close(kl.theta(), k.theta(), 1e-12);
  CHECK(kl.sigma2() == Approx(k.sigma2()).epsilon(1e-12));
  if (nm == NM::Heterogeneous)
    require_close(kl.noise(), D.noise, 1e-12);
  else
    CHECK(kl.nugget() == Approx(k.nugget()).epsilon(1e-12));
  CHECK(std::get<0>(kl.logLikelihoodVecchiaFun(g, false)) == Approx(k.logLikelihood("vecchia")).epsilon(1e-9));
}

TEST_CASE("normalize=true: Nystrom noise models agree with normalize=false", "[normalize][kriging][nystrom]") {
  const Data D = make_data();
  const arma::uword n = D.X.n_rows;
  // k = n + nu (even after the update below, which adds nu points): every
  // observation is a landmark, so the Nystrom correlation is the exact one
  // (up to the num_nugget jitter), and the landmark ranking (at a reference
  // kernel scaled to the data extent) does not depend on normalize.
  const arma::uword nu = 6;
  const std::string method = "nystrom(" + std::to_string(n + nu) + ")";
  const RM rm = GENERATE(RM::Constant, RM::Linear);
  const NM nm = GENERATE(NM::Nugget, NM::Heterogeneous);
  const bool est_sigma2 = GENERATE(false, true);
  CAPTURE(Trend::toString(rm), static_cast<int>(nm), est_sigma2);

  Kriging::Parameters p;
  p.theta = arma::mat{{9.0, 2.1}};
  p.is_theta_estim = false;
  p.sigma2 = 450.0;
  p.is_sigma2_estim = est_sigma2;
  if (nm == NM::Nugget) {
    p.nugget = 3.0;
    p.is_nugget_estim = est_sigma2;
  }

  auto nfit = [&](bool normalize) {
    Kriging k = make_model(nm);
    if (nm == NM::Heterogeneous)
      k.fit(D.y, D.noise, D.X, rm, normalize, "none", "LL", p, method);
    else
      k.fit(D.y, D.X, rm, normalize, "none", "LL", p, method);
    return k;
  };
  Kriging kn = nfit(true), ko = nfit(false);
  REQUIRE(kn.is_nystrom_light());
  REQUIRE(ko.is_nystrom_light());

  const double tol = 1e-6;
  require_close(kn.theta(), ko.theta(), tol);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  require_close(kn.beta(), ko.beta(), tol);
  if (nm == NM::Nugget) {
    CHECK(kn.nugget() == Approx(ko.nugget()).epsilon(tol));
    CHECK(kn.alpha() == Approx(ko.alpha()).epsilon(tol));  // scale-free
  } else {
    require_close(kn.noise(), ko.noise(), tol);
    require_close(kn.noise(), D.noise, 1e-12);  // round-trip of the given noise
  }

  // likelihood value and gradient at original-scale gamma
  CHECK(kn.logLikelihood("nystrom") == Approx(ko.logLikelihood("nystrom")).epsilon(tol));
  const arma::vec g0 = public_gamma(ko);
  CHECK(std::get<0>(kn.logLikelihoodNystromFun(g0, false)) == Approx(kn.logLikelihood("nystrom")).epsilon(1e-10));
  for (const arma::vec& g : {g0, arma::vec(g0 % arma::vec{1.3, 0.8, nm == NM::Nugget ? 0.97 : 1.5})}) {
    CAPTURE(g);
    auto [vn, gn] = kn.logLikelihoodNystromFun(g, true);
    auto [vo, go] = ko.logLikelihoodNystromFun(g, true);
    CHECK(vn == Approx(vo).epsilon(tol));
    require_close(gn, go, 1e-5);
    // theta-only argument: the committed alpha / sigma2
    CHECK(std::get<0>(kn.logLikelihoodNystromFun(g.head(2), false))
          == Approx(std::get<0>(ko.logLikelihoodNystromFun(g.head(2), false))).epsilon(tol));
    // finite differences of the original-scale objective (the Nystrom
    // likelihood is only smooth to ~1e-7 relative: larger steps/tolerance)
    const arma::vec fd
        = fd_grad([&](const arma::vec& x) { return std::get<0>(kn.logLikelihoodNystromFun(x, false)); }, g, 1e-3);
    require_close(gn, fd, 2e-2);
  }

  // exact model at the same parameters: full-rank Nystrom is (near) exact
  Kriging ke = make_model(nm);
  fit(ke, D, rm, /*normalize=*/true, "none", "LL", p);
  CHECK(std::get<0>(kn.logLikelihoodNystromFun(g0, false))
        == Approx(std::get<0>(ke.logLikelihoodFun(g0, false, false))).epsilon(1e-5));

  // predictions and simulations (latent process)
  const arma::mat Xn = arma::join_cols(D.X.rows(0, 3) + 0.3, D.X.rows(5, 6));
  auto [mn, sn] = kn.predictNystrom(Xn, true);
  auto [mo, so] = ko.predictNystrom(Xn, true);
  require_close(mn, mo, tol);
  require_close(sn, so, 1e-5);
  require_close(arma::vectorise(kn.simulateNystrom(7, 123, Xn)), arma::vectorise(ko.simulateNystrom(7, 123, Xn)), 1e-5);
  require_close(
      arma::vectorise(kn.simulate(7, 123, Xn, "nystrom")), arma::vectorise(kn.simulateNystrom(7, 123, Xn)), 0.0);

  // update without refit (new data in the original scale): both stay equal
  const Data U = make_data(nu);
  const arma::mat Xu = U.X + 0.7;
  if (nm == NM::Heterogeneous) {
    kn.update(U.y, U.noise, Xu, false);
    ko.update(U.y, U.noise, Xu, false);
    require_close(kn.noise(), ko.noise(), tol);
    require_close(kn.noise(), arma::join_cols(D.noise, U.noise), 1e-12);
  } else {
    kn.update(U.y, Xu, false);
    ko.update(U.y, Xu, false);
    CHECK(kn.nugget() == Approx(ko.nugget()).epsilon(tol));
    CHECK(kn.alpha() == Approx(ko.alpha()).epsilon(tol));
  }
  REQUIRE(kn.is_nystrom_light());
  require_close(kn.theta(), ko.theta(), tol);
  CHECK(kn.sigma2() == Approx(ko.sigma2()).epsilon(tol));
  require_close(kn.beta(), ko.beta(), tol);
  CHECK(kn.logLikelihood("nystrom") == Approx(ko.logLikelihood("nystrom")).epsilon(tol));
  auto [mun, sun] = kn.predictNystrom(Xn, true);
  auto [muo, suo] = ko.predictNystrom(Xn, true);
  require_close(mun, muo, tol);
  require_close(sun, suo, 1e-5);
}

TEST_CASE("normalize=true: Nystrom noise-model BFGS fit, objective at the accessors", "[normalize][kriging][nystrom]") {
  const Data D = make_data(40);
  const NM nm = GENERATE(NM::Nugget, NM::Heterogeneous);
  CAPTURE(static_cast<int>(nm));
  Kriging k = make_model(nm);
  if (nm == NM::Heterogeneous)
    k.fit(D.y, D.noise, D.X, RM::Constant, /*normalize=*/true, "BFGS", "LL", {}, "nystrom(20)");
  else
    k.fit(D.y, D.X, RM::Constant, /*normalize=*/true, "BFGS", "LL", {}, "nystrom(20)");
  REQUIRE(k.is_nystrom_light());
  const arma::vec g = public_gamma(k);
  CHECK(std::get<0>(k.logLikelihoodNystromFun(g, false)) == Approx(k.logLikelihood("nystrom")).epsilon(1e-9));
  // the original-scale optimum: no better value nearby, gradient = finite differences
  const arma::vec g1 = g % arma::vec{1.2, 0.8, nm == NM::Nugget ? 0.99 : 1.3};
  auto [v1, grad1] = k.logLikelihoodNystromFun(g1, true);
  const arma::vec fd
      = fd_grad([&](const arma::vec& x) { return std::get<0>(k.logLikelihoodNystromFun(x, false)); }, g1, 1e-3);
  CAPTURE(grad1, fd);
  require_close(grad1, fd, 2e-2);
  CHECK(v1 <= k.logLikelihood("nystrom") + 1e-6);

  // save / load keeps the original-scale accessors, objective and predictor
  const std::string f = "KrigingNormalizeScaleTest_nystrom_saved.json";
  k.save(f);
  Kriging kl = Kriging::load(f);
  require_close(kl.theta(), k.theta(), 1e-12);
  CHECK(kl.sigma2() == Approx(k.sigma2()).epsilon(1e-12));
  if (nm == NM::Heterogeneous)
    require_close(kl.noise(), D.noise, 1e-12);
  else
    CHECK(kl.nugget() == Approx(k.nugget()).epsilon(1e-12));
  CHECK(std::get<0>(kl.logLikelihoodNystromFun(g, false)) == Approx(k.logLikelihood("nystrom")).epsilon(1e-9));
  const arma::mat Xn = D.X.rows(0, 4) + 0.3;
  require_close(std::get<0>(kl.predictNystrom(Xn, false)), std::get<0>(k.predictNystrom(Xn, false)), 1e-12);
}

TEST_CASE("normalize=false: accessors are the internal members, bit for bit", "[normalize][kriging]") {
  const Data D = make_data();
  const NM nm = GENERATE(NM::None, NM::Nugget, NM::Heterogeneous);
  CAPTURE(static_cast<int>(nm));
  Kriging k = make_model(nm);
  fit(k, D, RM::Linear, /*normalize=*/false, "BFGS", "LL", {});
  const KrigingImpl& impl = k;
  CHECK(arma::approx_equal(k.theta(), impl.theta(), "absdiff", 0.0));
  CHECK(k.sigma2() == impl.sigma2());
  CHECK(arma::approx_equal(k.beta(), impl.beta(), "absdiff", 0.0));
  CHECK(std::get<0>(k.logLikelihoodFun(public_gamma(k), false, false)) == k.logLikelihood());
  if (nm == NM::Heterogeneous)
    CHECK(arma::approx_equal(k.noise(), D.noise, "absdiff", 0.0));

  // fixed parameters are kept exactly
  Kriging::Parameters p;
  p.theta = arma::mat{{7.0, 1.3}};
  p.sigma2 = 321.0;
  p.is_sigma2_estim = false;
  p.beta = arma::vec{1.0, 2.0, 3.0};
  p.is_beta_estim = false;
  if (nm == NM::Nugget) {
    p.nugget = 12.5;
    p.is_nugget_estim = false;
  }
  Kriging kf = make_model(nm);
  fit(kf, D, RM::Linear, /*normalize=*/false, "none", "LL", p);
  CHECK(arma::approx_equal(kf.theta(), arma::vec{7.0, 1.3}, "absdiff", 0.0));
  CHECK(kf.sigma2() == 321.0);
  CHECK(arma::approx_equal(kf.beta(), arma::vec{1.0, 2.0, 3.0}, "absdiff", 0.0));
  if (nm == NM::Nugget)
    CHECK(kf.nugget() == 12.5);
}

TEST_CASE("normalize=true: a fixed beta of the wrong length is rejected", "[normalize][kriging]") {
  const Data D = make_data();
  Kriging::Parameters p;
  p.theta = arma::mat{{7.0, 1.3}};
  p.beta = arma::vec{1.0, 2.0};  // linear trend in d=2 needs 3
  p.is_beta_estim = false;
  Kriging k("matern5_2");
  CHECK_THROWS_AS(k.fit(D.y, D.X, RM::Linear, true, "none", "LL", p), std::invalid_argument);
}

TEST_CASE("normalize=true: save/load keeps the original-scale accessors", "[normalize][kriging]") {
  const Data D = make_data();
  Kriging k("matern5_2", NM::Heterogeneous);
  k.fit(D.y, D.noise, D.X, RM::Linear, true, "BFGS", "LL", {});
  const std::string f = "KrigingNormalizeScaleTest_saved.json";
  k.save(f);
  Kriging l = Kriging::load(f);
  require_close(l.theta(), k.theta(), 1e-14);
  CHECK(l.sigma2() == Approx(k.sigma2()).epsilon(1e-14));
  require_close(l.beta(), k.beta(), 1e-14);
  require_close(l.noise(), D.noise, 1e-12);
  CHECK(l.logLikelihood() == Approx(k.logLikelihood()).epsilon(1e-12));
  std::remove(f.c_str());
}

TEST_CASE("normalize=true: summary prints original-scale parameters", "[normalize][kriging]") {
  const Data D = make_data();
  Kriging::Parameters p;
  p.theta = arma::mat{{7.25, 1.5}};
  p.sigma2 = 321.0;
  p.is_sigma2_estim = false;
  Kriging k("matern5_2");
  k.fit(D.y, D.X, RM::Constant, true, "none", "LL", p);
  const std::string s = k.summary();
  CAPTURE(s);
  CHECK(s.find("7.25") != std::string::npos);
  CHECK(s.find("321") != std::string::npos);
}
