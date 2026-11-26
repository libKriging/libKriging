// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/MLPKriging.hpp"
#include "libKriging/Parallel.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/WarpKriging.hpp"
#include <cstdlib>
#include <string>
#include <vector>
// clang-format on

// The multistart fits of WarpKriging and MLPKriging do not depend on the start
// delay of the workers nor on the pool size: bit-identical to the sequential
// run (pool of 1). This is what allows a default start delay of 0 ms (#468,
// docs/dev/parallel.md; bench/thread-budget/start_delay_study.cpp for the
// measurement). Run on every CI platform (BLAS backends, sanitizers).

using namespace libKriging;

namespace {
struct Fit {
  double ll;
  arma::vec theta;
  arma::vec pred;
};

struct Restore {
  int pool = Parallel::get_thread_pool_size();
  int delay = Parallel::get_thread_start_delay_ms();
  ~Restore() {
    Parallel::set_thread_pool_size(pool);
    Parallel::set_thread_start_delay_ms(delay);
  }
};

void data(arma::mat& X, arma::vec& y, arma::mat& Xn, arma::uword n, arma::uword d) {
  arma::arma_rng::set_seed(7);
  X = arma::randu<arma::mat>(n, d);
  y = arma::sin(6 * X.col(0)) % arma::exp(X.col(d - 1)) + arma::square(X.col(0) - 0.3);
  Xn = arma::randu<arma::mat>(20, d);
}

Fit warp_fit(const arma::mat& X, const arma::vec& y, const arma::mat& Xn) {
  arma::arma_rng::set_seed(123);
  WarpKriging m(std::vector<std::string>(X.n_cols, "kumaraswamy"), "matern5_2");
  m.fit(y, X, Trend::RegressionModel::Constant, true, "BFGS8", "LL");
  return {m.logLikelihood(), m.theta(), std::get<0>(m.predict(Xn, false, false, false))};
}

Fit mlp_fit(const arma::mat& X, const arma::vec& y, const arma::mat& Xn) {
  arma::arma_rng::set_seed(123);
  MLPKriging m(std::vector<arma::uword>{8}, 2, "selu", "matern5_2");
  m.fit(y, X, Trend::RegressionModel::Constant, true, "BFGS4", "LL");
  return {m.logLikelihood(), m.theta(), std::get<0>(m.predict(Xn, false, false, false))};
}

void check_same(const Fit& a, const Fit& ref) {
  CHECK(a.ll == ref.ll);
  CHECK(arma::approx_equal(a.theta, ref.theta, "absdiff", 0.0));
  CHECK(arma::approx_equal(a.pred, ref.pred, "absdiff", 0.0));
}

template <class F>
void check_delay_and_pool(F fit) {
  Restore restore;
  Parallel::set_thread_pool_size(1);  // sequential reference
  Parallel::set_thread_start_delay_ms(0);
  const Fit ref = fit();
  // (pool size, start delay in ms): automatic pool, more workers than cores
  const std::vector<std::pair<int, int>> configs = {{0, 0}, {0, 0}, {0, 0}, {8, 0}, {8, 0}, {0, 10}};
  for (const auto& c : configs) {
    INFO("pool " << c.first << ", start delay " << c.second << " ms");
    Parallel::set_thread_pool_size(c.first);
    Parallel::set_thread_start_delay_ms(c.second);
    check_same(fit(), ref);
  }
}
}  // namespace

TEST_CASE("Start delay default is 0 ms", "[multistart]") {
  if (std::getenv("LK_THREAD_START_DELAY_MS") == nullptr)
    CHECK(Parallel::get_thread_start_delay_ms() == 0);
}

TEST_CASE("WarpKriging multistart: same fit whatever the start delay and the pool size", "[multistart]") {
  arma::mat X, Xn;
  arma::vec y;
  data(X, y, Xn, 60, 2);
  check_delay_and_pool([&] { return warp_fit(X, y, Xn); });
}

TEST_CASE("MLPKriging multistart: same fit whatever the start delay and the pool size", "[multistart]") {
  arma::mat X, Xn;
  arma::vec y;
  data(X, y, Xn, 40, 2);
  check_delay_and_pool([&] { return mlp_fit(X, y, Xn); });
}
