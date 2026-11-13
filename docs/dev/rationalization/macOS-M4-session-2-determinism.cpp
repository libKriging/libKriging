// Determinism check (session 2, section 2): prints every number of a fixed set of computations with
// %.17g, one per line. The output must be bitwise identical between builds that differ only in how the
// parallel loops run (sequential, libomp, GCD): each loop writes disjoint outputs and reductions stay in a
// fixed order. Not committed to the library.
//   determinism_check <n> > out.txt        (n = 500 and 1500, d = 4)
#include "libKriging/Kriging.hpp"
#include "libKriging/NestedKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void pv(const char* tag, const arma::vec& v) { for (arma::uword i = 0; i < v.n_elem; ++i) std::printf("%s[%u] %.17g\n", tag, (unsigned)i, v(i)); }
static void pm(const char* tag, const arma::mat& m) { for (arma::uword j = 0; j < m.n_cols; ++j) for (arma::uword i = 0; i < m.n_rows; ++i) std::printf("%s[%u,%u] %.17g\n", tag, (unsigned)i, (unsigned)j, m(i, j)); }
static void ps(const char* tag, double x) { std::printf("%s %.17g\n", tag, x); }

static arma::mat design(arma::uword n, arma::uword d, double shift) {
  const double a[6] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898, 0.6457513110645907, 0.3166247903553998, 0.8228756555322953};
  arma::mat X(n, d);
  for (arma::uword i = 0; i < n; ++i) for (arma::uword j = 0; j < d; ++j) X(i, j) = std::fmod(shift + (double)(i + 1) * a[j], 1.0);
  return X;
}
static arma::vec resp(const arma::mat& X) { return arma::sin(6.0 * X.col(0)) + arma::cos(4.0 * X.col(1)) % X.col(2) + 0.5 * arma::sin(5.0 * X.col(3)); }

int main(int argc, char** argv) {
  const arma::uword n = argc > 1 ? atoi(argv[1]) : 500, d = 4, q = 300;
  const arma::mat X = design(n, d, 0.5), Xn = design(q, d, 0.137);
  const arma::vec y = resp(X);
  const arma::vec theta0 = {0.3, 0.5, 0.7, 0.9};

  // exact fit, multistart (the default optimizer path), then everything at the fitted model
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4", "LL");
  pv("fit.theta", k.theta()); ps("fit.sigma2", k.sigma2()); pv("fit.beta", k.beta()); ps("fit.LL", k.logLikelihood());
  // LL / LMP / LOO with gradients at a fixed theta
  { auto [v, g] = k.logLikelihoodFun(theta0, true, false); ps("LL", v); pv("LL.grad", g); }
  { auto [v, g] = k.logMargPostFun(theta0, true, false); ps("LMP", v); pv("LMP.grad", g); }
  { auto [v, g] = k.leaveOneOutFun(theta0, true, false); ps("LOO", v); pv("LOO.grad", g); }
  // predict with stdev and cov, simulate (fixed seed)
  { auto [m, s, c, dm, ds] = k.predict(Xn, true, true, false); pv("pred.mean", m); pv("pred.stdev", s); pm("pred.cov", c); }
  { arma::mat sim = k.simulate(5, 123, Xn); pm("sim", sim); }
  // Vecchia predict (m = 100 neighbours: parallel branch for q > 32 and n m >= 40000) and Vecchia simulate
  { auto [m, s] = k.predictVecchia(Xn, true, 100); pv("vecchia.mean", m); pv("vecchia.stdev", s); }
  { arma::mat sim = k.simulateVecchia(3, 7, Xn, 100); pm("vecchia.sim", sim); }
  // Nystrom fit (method nystrom(60)) and predict
  { Kriging kn(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "nystrom(60)");
    pv("nystrom.theta", kn.theta()); ps("nystrom.LL", kn.logLikelihood("nystrom"));
    auto [m, s, c, dm, ds] = kn.predict(Xn, true, false, false, "nystrom"); pv("nystrom.mean", m); pv("nystrom.stdev", s); }
  // Vecchia fit
  { Kriging kv(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS", "LL", {}, "vecchia(30)");
    pv("vecchiafit.theta", kv.theta()); ps("vecchiafit.LL", kv.logLikelihood("vecchia")); }
  // NestedKriging (4 groups, NK aggregation: the pair loop)
  { NestedKriging nk(y, X, "matern5_2", 4, NestedKriging::Aggregation::NK, NestedKriging::Partition::KMeans, 123);
    auto [m, s] = nk.predict(Xn, true); pv("nested.mean", m); pv("nested.stdev", s); }
  return 0;
}
