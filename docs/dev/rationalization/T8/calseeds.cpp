#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include "libKriging/Kriging.hpp"
#include "libKriging/MLPKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"
using namespace libKriging;
namespace {

constexpr arma::uword kDim = 5;

arma::vec branin_proj(const arma::mat& X) {
  // Fixed projection, rows normalised to |w|_1 = 1, then mapped to [0, 1].
  arma::mat W = {{0.9, -0.4, 0.7, 0.2, -0.5}, {-0.3, 0.8, 0.1, -0.6, 0.4}};
  W.each_col() /= arma::sum(arma::abs(W), 1);
  arma::mat U = X * W.t();
  U.each_row() -= arma::sum(arma::clamp(W, -arma::datum::inf, 0.0), 1).t();
  arma::vec x1 = 15.0 * U.col(0) - 5.0, x2 = 15.0 * U.col(1);
  const double pi = arma::datum::pi;
  return arma::square(x2 - 5.1 / (4.0 * pi * pi) * arma::square(x1) + 5.0 / pi * x1 - 6.0)
         + 10.0 * (1.0 - 1.0 / (8.0 * pi)) * arma::cos(x1) + 10.0;
}

arma::mat lhs(arma::uword n, arma::uword d) {
  arma::mat X(n, d);
  for (arma::uword j = 0; j < d; ++j)
    X.col(j) = (arma::conv_to<arma::vec>::from(arma::randperm(n)) + arma::randu<arma::vec>(n)) / n;
  return X;
}

struct Scores {
  double q2, nlpd, coverage;
};

template <typename Model>
Scores score(Model& model, const arma::mat& Xt, const arma::vec& yt) {
  auto [m, s, _c, _md, _sd] = model.predict(Xt, true, false, false);
  arma::vec mean = m, sd = s;
  const double q2 = 1.0 - arma::accu(arma::square(yt - mean)) / arma::accu(arma::square(yt - arma::mean(yt)));
  const double nlpd
      = arma::mean(0.5 * arma::log(2.0 * arma::datum::pi * arma::square(sd)) + 0.5 * arma::square((yt - mean) / sd));
  const double cov = arma::mean(arma::conv_to<arma::vec>::from(arma::abs(yt - mean) <= 1.96 * sd));
  return {q2, nlpd, cov};
}

}  // namespace
int main(int argc, char** argv) {
  const int nseed = std::atoi(argv[1]);
  for (int s = 0; s < nseed; ++s) {
    const unsigned seed = s == 0 ? 407 : 407 + 1000 * s;
    arma::arma_rng::set_seed(seed);
    const arma::mat X = lhs(60, kDim);
    const arma::vec y = branin_proj(X);
    const arma::mat Xt = arma::randu<arma::mat>(1000, kDim);
    const arma::vec yt = branin_proj(Xt);
    Kriging kr(y, X, "matern5_2");
    const Scores sk = score(kr, Xt, yt);
    for (std::string optim : {"BFGS", "BFGS4"}) {
      MLPKriging raw({8}, 2, "tanh", "matern5_2");
      raw.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", {{"calibration_folds", "0"}});
      const Scores sr = score(raw, Xt, yt);
      for (std::string K : {"5", "3"}) {
        MLPKriging cal({8}, 2, "tanh", "matern5_2");
        cal.fit(y, X, Trend::RegressionModel::Constant, false, optim, "LL", {{"calibration_folds", K}});
        const Scores sc = score(cal, Xt, yt);
        std::printf("%u %s %s %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                    seed,
                    optim.c_str(),
                    K.c_str(),
                    sk.nlpd,
                    sr.q2,
                    sr.nlpd,
                    sr.coverage,
                    sc.nlpd,
                    sc.coverage,
                    cal.var_scale(),
                    sc.q2);
        std::fflush(stdout);
      }
    }
  }
}
