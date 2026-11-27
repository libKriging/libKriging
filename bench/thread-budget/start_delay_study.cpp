// Study driver (not built by CMake, see README.md for the build): start delay
// of the multistart workers (Parallel::set_thread_start_delay_ms, #468). Are
// WarpKriging / MLPKriging multistart fits bit-identical whatever the delay
// (0, 1, 2, 5, 10 ms) and equal to the sequential run (pool size 1)? Median
// time per fit. Usage: [POOL=k] start_delay_study [reps]  (POOL: pool size,
// default 0 = the thread budget). Exit code 1 if a fit differs.
// Results: docs/dev/parallel.md ("No start stagger").
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "libKriging/MLPKriging.hpp"
#include "libKriging/Parallel.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/WarpKriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

using namespace libKriging;
struct Res {
  double ll;
  arma::vec theta;
  arma::vec pred;
  double secs;
};

static void data(arma::mat& X, arma::vec& y, arma::mat& Xn, arma::uword n, arma::uword d, int seed) {
  arma::arma_rng::set_seed(seed);
  X = arma::randu<arma::mat>(n, d);
  y = arma::sin(6 * X.col(0)) % arma::exp(X.col(d - 1)) + arma::square(X.col(0) - 0.3);
  Xn = arma::randu<arma::mat>(50, d);
}

template <class F>
static Res timed(F f) {
  auto t0 = std::chrono::steady_clock::now();
  Res r = f();
  r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

static Res warp(const arma::mat& X, const arma::vec& y, const arma::mat& Xn, const std::string& optim) {
  return timed([&] {
    arma::arma_rng::set_seed(123);
    std::vector<std::string> w(X.n_cols, "kumaraswamy");
    WarpKriging m(w, "matern5_2");
    m.fit(y, X, Trend::RegressionModel::Constant, true, optim, "LL");
    auto p = m.predict(Xn, false, false, false);
    return Res{m.logLikelihood(), m.theta(), std::get<0>(p), 0.0};
  });
}

static Res mlp(const arma::mat& X, const arma::vec& y, const arma::mat& Xn, const std::string& optim) {
  return timed([&] {
    arma::arma_rng::set_seed(123);
    MLPKriging m(std::vector<arma::uword>{8}, 2, "selu", "matern5_2");
    m.fit(y, X, Trend::RegressionModel::Constant, true, optim, "LL");
    auto p = m.predict(Xn, false, false, false);
    return Res{m.logLikelihood(), m.theta(), std::get<0>(p), 0.0};
  });
}

static bool same(const Res& a, const Res& b) {
  return a.ll == b.ll && arma::approx_equal(a.theta, b.theta, "absdiff", 0.0)
         && arma::approx_equal(a.pred, b.pred, "absdiff", 0.0);
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
  const std::vector<int> delays = {0, 1, 2, 5, 10};
  struct Case {
    const char* name;
    arma::uword n, d;
    const char* optim;
    bool is_mlp;
  };
  const std::vector<Case> cases = {{"warp n60 d2 BFGS8", 60, 2, "BFGS8", false},
                                   {"warp n150 d3 BFGS8", 150, 3, "BFGS8", false},
                                   {"warp n60 d2 BFGS4+Adam", 60, 2, "BFGS4+Adam", false},
                                   {"mlp n80 d3 BFGS8", 80, 3, "BFGS8", true}};
  std::printf("case\tdelay_ms\tpool\treps\tidentical_to_sequential\tmedian_s\tll\n");
  int failures = 0;
  for (const auto& c : cases) {
    arma::mat X, Xn;
    arma::vec y;
    data(X, y, Xn, c.n, c.d, 7);
    auto run = [&] { return c.is_mlp ? mlp(X, y, Xn, c.optim) : warp(X, y, Xn, c.optim); };
    Parallel::set_thread_pool_size(1);
    Parallel::set_thread_start_delay_ms(0);
    const Res ref = run();
    std::printf("%s\t-\t1\t1\tref\t%.3f\t%.10g\n", c.name, ref.secs, ref.ll);
    const int pool = std::getenv("POOL") ? std::atoi(std::getenv("POOL")) : 0;
    Parallel::set_thread_pool_size(pool);
    for (int dly : delays) {
      Parallel::set_thread_start_delay_ms(dly);
      int ok = 0;
      std::vector<double> t;
      for (int r = 0; r < reps; ++r) {
        Res x = run();
        ok += same(x, ref);
        t.push_back(x.secs);
      }
      std::sort(t.begin(), t.end());
      failures += reps - ok;
      std::printf("%s\t%d\t%d\t%d\t%d/%d\t%.3f\t%.10g\n", c.name, dly, pool, reps, ok, reps, t[t.size() / 2], ref.ll);
      std::fflush(stdout);
    }
  }
  std::printf("budget %d threads, failures %d\n", Parallel::get_num_threads(), failures);
  return failures ? 1 : 0;
}
