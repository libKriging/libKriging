// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/MLPKriging.hpp"
#include "libKriging/NestedKriging.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"
#include <cmath>
#include <cstdio>
#include <string>
#include <tuple>
#include <vector>
// clang-format on

/**
 * @file JitterRuleTest.cpp
 * @brief One jitter rule for Kriging, WarpKriging / MLPKriging and
 *        NestedKriging (#465, docs/dev/CholJitter.md "One rule for every
 *        class"): the search and the objective accessors (logLikelihood(),
 *        leaveOneOut(), logMargPost()) use the search diagonal 1 + J, the
 *        committed model (predict, simulate, update) is exact, and an
 *        accessor always equals its objective function re-evaluated at the
 *        committed parameters: after fit, update(refit = false / true) and
 *        save / load, for every noise model.
 */

namespace {

// sum_j sin(2 pi f_j x_j) on n random points of [0,1]^d
void sin_data(int seed, arma::uword n, arma::uword d, arma::mat& X, arma::vec& y) {
  arma::arma_rng::set_seed(seed);
  X = arma::randu(n, d);
  const arma::rowvec f = 0.5 + 2.5 * arma::randu<arma::rowvec>(d);
  y.zeros(n);
  for (arma::uword j = 0; j < d; ++j)
    y += arma::sin(2 * arma::datum::pi * f(j) * X.col(j));
}

bool close(double a, double b, double rel) {
  return std::abs(a - b) <= rel * std::max(1.0, std::abs(b));
}

// gamma of Kriging::logLikelihoodFun at the committed parameters
arma::vec gamma_of(const Kriging& k, Kriging::NoiseModel noise) {
  if (noise == Kriging::NoiseModel::Nugget)
    return arma::join_cols(k.theta(), arma::vec{k.sigma2() / (k.sigma2() + k.nugget())});
  if (noise == Kriging::NoiseModel::Heterogeneous)
    return arma::join_cols(k.theta(), arma::vec{k.sigma2()});
  return k.theta();
}

// Every objective accessor of k equals its *Fun at the committed parameters
void check_kriging(Kriging& k, Kriging::NoiseModel noise, const std::string& when) {
  INFO(when);
  const double ll = k.logLikelihood();
  CHECK(close(ll, std::get<0>(k.logLikelihoodFun(gamma_of(k, noise), false, false)), 1e-10));
  if (noise == Kriging::NoiseModel::None) {
    CHECK(close(k.leaveOneOut(), std::get<0>(k.leaveOneOutFun(k.theta(), false, false)), 1e-10));
    CHECK(close(k.logMargPost(), std::get<0>(k.logMargPostFun(k.theta(), false, false)), 1e-10));
  }
}

std::string tmp_file(const std::string& name) {
  return "jitter_rule_" + name + ".json";
}

}  // namespace

TEST_CASE("jitter rule: Kriging accessors are the objective at the committed parameters", "[jitter][rule]") {
  arma::mat X, Xu;
  arma::vec y, yu;
  sin_data(11, 40, 2, X, y);
  sin_data(12, 5, 2, Xu, yu);
  const arma::vec noise(X.n_rows, arma::fill::value(1e-3)), noise_u(Xu.n_rows, arma::fill::value(1e-3));
  for (const std::string kernel : {"gauss", "matern5_2", "matern3_2"})
    for (const auto nm : {Kriging::NoiseModel::None, Kriging::NoiseModel::Nugget, Kriging::NoiseModel::Heterogeneous})
      for (const std::string objective : {"LL", "LOO", "LMP"}) {
        if (nm != Kriging::NoiseModel::None && objective != "LL")
          continue;  // LOO / LMP accessors: noise-free (LOO is not supported with a noise model)
        INFO(kernel << " noise " << static_cast<int>(nm) << " " << objective);
        Kriging k(kernel, nm);
        if (nm == Kriging::NoiseModel::Heterogeneous)
          k.fit(y, noise, X, Trend::RegressionModel::Constant, false, "BFGS", objective, Kriging::Parameters{});
        else
          k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", objective, Kriging::Parameters{});
        // search jitter: the kernel's, noise-free only; committed model exact
        CHECK(k.chol_jitter() == (nm == Kriging::NoiseModel::None ? LinearAlgebra::kernel_chol_jitter(kernel) : 0.0));
        CHECK(k.model_chol_jitter() == 0.0);
        check_kriging(k, nm, "fit");

        const std::string f = tmp_file("k");
        k.save(f);
        Kriging kl = Kriging::load(f);
        std::remove(f.c_str());
        CHECK(kl.chol_jitter() == k.chol_jitter());
        CHECK(close(kl.logLikelihood(), k.logLikelihood(), 1e-12));
        check_kriging(kl, nm, "load");

        Kriging k2(k, ExplicitCopySpecifier{});
        if (nm == Kriging::NoiseModel::Heterogeneous) {
          k.update(yu, noise_u, Xu, false);
          k2.update(yu, noise_u, Xu, true);
        } else {
          k.update(yu, Xu, false);
          k2.update(yu, Xu, true);
        }
        check_kriging(k, nm, "update(refit = false)");
        check_kriging(k2, nm, "update(refit = true)");
      }
}

TEST_CASE("jitter rule: Kriging committed model is exact, its objective has the search jitter", "[jitter][rule]") {
  arma::mat X;
  arma::vec y;
  sin_data(13, 30, 2, X, y);
  for (const std::string kernel : {"gauss", "matern5_2"}) {
    Kriging k(kernel);
    k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
    INFO(kernel << " theta " << k.theta().t());
    // the committed model interpolates the data (exact factorization)
    const arma::vec yp = std::get<0>(k.predict(X, false, false, false));
    CHECK(arma::abs(yp - y).max() < 1e-6 * arma::abs(y).max());
    // the objective of the fit is the jittered one: with the jitter off, the
    // same parameters give another value (the exact model's LL)
    const double ll = k.logLikelihood();
    const double saved = LinearAlgebra::get_chol_jitter();
    LinearAlgebra::set_chol_jitter(0.0);
    Kriging k0(kernel);
    Kriging::Parameters fx;
    fx.theta = arma::mat(k.theta().t());
    fx.is_theta_estim = false;
    k0.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", fx);
    LinearAlgebra::set_chol_jitter(saved);
    CHECK(k0.chol_jitter() == 0.0);
    CHECK(close(k.logLikelihood(), ll, 1e-12));  // a later setting does not change a fitted model
    INFO("LL(J) " << ll << " LL(exact) " << k0.logLikelihood());
    CHECK(ll != k0.logLikelihood());
  }
}

TEST_CASE("jitter rule: WarpKriging accessors are the objective at the committed parameters", "[jitter][rule][warp]") {
  arma::mat X, Xu;
  arma::vec y, yu;
  sin_data(21, 40, 2, X, y);
  sin_data(22, 5, 2, Xu, yu);
  const arma::vec noise(X.n_rows, arma::fill::value(1e-3)), noise_u(Xu.n_rows, arma::fill::value(1e-3));
  for (const std::string warp : {"none", "kumaraswamy"})
    for (const std::string kernel : {"gauss", "matern3_2"})
      for (const bool het : {false, true}) {
        INFO(warp << " " << kernel << (het ? " heterogeneous noise" : " noise-free"));
        libKriging::WarpKriging k({warp, warp}, kernel);
        libKriging::WarpKriging::Parameters pa;
        if (het)
          pa.noise = noise;
        k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", pa);
        CHECK(k.model_chol_jitter() == 0.0);
        auto check = [&](const libKriging::WarpKriging& m, const std::string& when) {
          INFO(when);
          CHECK(close(m.logLikelihood(), std::get<0>(m.logLikelihoodFun(m.theta(), false)), 1e-6));
        };
        check(k, "fit");
        const std::string f = tmp_file("w");
        k.save(f);
        libKriging::WarpKriging kl = libKriging::WarpKriging::load(f);
        std::remove(f.c_str());
        check(kl, "load");
        // refit = false: no search, the objective is re-evaluated (it was the
        // committed exact model's LL before #465, i.e. another quantity)
        k.update(yu, Xu, false, het ? noise_u : arma::vec());
        check(k, "update(refit = false)");
        kl.update(yu, Xu, true, het ? noise_u : arma::vec());
        check(kl, "update(refit = true)");
      }
}

TEST_CASE("jitter rule: WarpKriging none and Kriging share the objective", "[jitter][rule][warp]") {
  // "none" warps: a Kriging model. Same search diagonal for the kernels with a
  // jitter, so the same logLikelihoodFun at the same theta.
  arma::mat X;
  arma::vec y;
  sin_data(31, 40, 3, X, y);
  for (const std::string kernel : {"gauss", "matern5_2"}) {
    libKriging::WarpKriging w({"none", "none", "none"}, kernel);
    w.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
    Kriging k(kernel);
    Kriging::Parameters fx;
    fx.theta = arma::mat(w.theta().t());
    fx.is_theta_estim = false;
    k.fit(y, X, Trend::RegressionModel::Constant, false, "none", "LL", fx);
    INFO(kernel);
    CHECK(w.chol_jitter() == k.chol_jitter());
    CHECK(close(w.logLikelihood(), k.logLikelihood(), 1e-8));
  }
}

TEST_CASE("jitter rule: MLPKriging accessors are the objective at the committed parameters", "[jitter][rule][mlp]") {
  arma::mat X, Xu;
  arma::vec y, yu;
  sin_data(41, 40, 3, X, y);
  sin_data(42, 5, 3, Xu, yu);
  libKriging::MLPKriging k({6}, 2, "tanh", "gauss");
  k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
  CHECK(k.model_chol_jitter() == 0.0);
  CHECK(close(k.logLikelihood(), std::get<0>(k.logLikelihoodFun(k.theta(), false)), 1e-6));
  k.update(yu, Xu, false);
  CHECK(close(k.logLikelihood(), std::get<0>(k.logLikelihoodFun(k.theta(), false)), 1e-6));
}

TEST_CASE("jitter rule: NestedKriging submodels follow the rule", "[jitter][rule][nested]") {
  arma::mat X;
  arma::vec y;
  sin_data(51, 80, 2, X, y);
  NestedKriging nk("gauss");
  nk.fit(y, X, 4, Trend::RegressionModel::Constant, "BFGS", "LL");
  for (arma::uword g = 0; g < 4; ++g) {
    Kriging s(nk.submodel(g), ExplicitCopySpecifier{});
    INFO("group " << g);
    CHECK(s.chol_jitter() == LinearAlgebra::kernel_chol_jitter("gauss"));
    CHECK(s.model_chol_jitter() == 0.0);
    check_kriging(s, Kriging::NoiseModel::None, "submodel");
  }
  // the aggregate interpolates the design (exact committed submodels)
  const arma::vec yp = std::get<0>(nk.predict(X, false));
  CHECK(arma::abs(yp - y).max() < 1e-5 * arma::abs(y).max());
}

TEST_CASE("rcond test bounded by the diagonal (study mode, #465)", "[jitter][rcond]") {
  // matern5_2 correlation, n = 200 points of [0,1]^2, long range: with a
  // diagonal shift delta the 1-norm rcond is >= delta / (sqrt(n) (max diag +
  // factor (n - 1))), here 3.5e-17 >= min_rcond (1e-18), while the rcond
  // estimate of the Cholesky factor can be lower (6e-19 with OpenBLAS).
  const arma::uword n = 200;
  arma::arma_rng::set_seed(1);
  const arma::mat X = arma::randu(n, 2);
  const arma::mat dX = LinearAlgebra::compute_dX(X);
  auto cov = [](const arma::vec& d, const arma::vec& t) {
    const arma::vec h = arma::abs(d) / t * std::sqrt(5.0);
    double p = 1;
    for (const double v : h)
      p *= (1 + v + v * v / 3) * std::exp(-v);
    return p;
  };
  const arma::vec theta(2, arma::fill::value(3.0));
  const int saved = LinearAlgebra::get_chol_rcond_bound();
  auto chol_with = [&](int mode, double factor, double d) {
    LinearAlgebra::set_chol_rcond_bound(mode);
    arma::mat R(n, n);
    arma::mat L = LinearAlgebra::cholCov(&R, dX, theta, cov, factor, arma::vec(n, arma::fill::value(d)));
    arma::mat Lp;
    const bool ok = arma::chol(Lp, R, "lower");
    return std::make_tuple(L, Lp, ok);
  };
  {  // jitter: constant shift on the unit diagonal, modes 1 and 2 skip the test
    const auto [L0, Lp, ok] = chol_with(0, 1.0, 1.0 + 1e-13);
    REQUIRE(ok);
    INFO("rcond estimate " << LinearAlgebra::rcond_chol(Lp));
    if (LinearAlgebra::rcond_chol(Lp) < 1e-18)
      CHECK(!arma::approx_equal(L0, Lp, "absdiff", 0.0));  // mode 0 retries
    for (const int mode : {1, 2}) {
      const auto [L, Lq, ok2] = chol_with(mode, 1.0, 1.0 + 1e-13);
      CHECK(arma::approx_equal(L, Lq, "absdiff", 0.0));  // plain Cholesky, no retry
    }
  }
  {  // nugget-like: factor < 1, unit diagonal; only mode 2 uses the bound
    const double alpha = 1.0 - 1e-13;
    const auto [L0, Lp, ok] = chol_with(0, alpha, 1.0);
    const auto [L1, Lp1, ok1] = chol_with(1, alpha, 1.0);
    CHECK(arma::approx_equal(L0, L1, "absdiff", 0.0));
    const auto [L2, Lp2, ok2] = chol_with(2, alpha, 1.0);
    if (ok2)
      CHECK(arma::approx_equal(L2, Lp2, "absdiff", 0.0));
  }
  LinearAlgebra::set_chol_rcond_bound(saved);
}
