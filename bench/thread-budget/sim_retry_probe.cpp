// Measure for docs/dev/CholJitter-usage.md (inconsistency 2), built like the other
// thread-budget drivers (scratch build script, see CholJitter-TODO.md).
// Conditional simulation: does the safe_chol_lower retry fire, and does it
// depend on the thread budget?
#include <cstdio>
#include <string>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

int main(int argc, char** argv) {
  const int nrep = argc > 1 ? std::stoi(argv[1]) : 10;
  struct Case {
    const char* kernel;
    arma::uword n, d;
  };
  const Case cases[]
      = {{"gauss", 10, 2}, {"gauss", 50, 2}, {"gauss", 200, 3}, {"matern5_2", 100, 2}, {"gauss", 100, 1}};
  for (const auto& c : cases)
    for (int rep = 0; rep < nrep; ++rep) {
      arma::arma_rng::set_seed(1000 * rep + c.n);
      arma::mat X = arma::randu(c.n, c.d);
      arma::vec y = arma::sum(arma::sin(3 * X), 1) + X.col(0);
      Optim::set_num_threads_percent(100);
      Kriging k(c.kernel);
      k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL");
      arma::mat Xr = arma::randu(200, c.d);
      // random points, design points (exactly and 1e-6 away), a tight cluster
      arma::mat Xn = arma::join_vert(arma::join_vert(Xr, X.rows(0, std::min<arma::uword>(19, c.n - 1))),
                                     X.rows(0, std::min<arma::uword>(19, c.n - 1)) + 1e-6);
      Xn = arma::join_vert(Xn, arma::repmat(Xr.row(0), 20, 1) + 1e-7 * arma::randu(20, c.d));
      LinearAlgebra::set_chol_warning(true);
      for (const double pct : {10.0, 100.0, 200.0}) {
        Optim::set_num_threads_percent(pct);
        std::printf(
            "@ %s n=%llu d=%llu rep=%d pct=%g\n", c.kernel, (unsigned long long)c.n, (unsigned long long)c.d, rep, pct);
        std::fflush(stdout);
        arma::mat S = k.simulate(50, 123, Xn);
        std::printf("= sum %.17g\n", arma::accu(S));
        std::fflush(stdout);
      }
      LinearAlgebra::set_chol_warning(false);
    }
  return 0;
}
