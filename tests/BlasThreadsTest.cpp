// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/WarpKriging.hpp"
// clang-format on

#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(_WIN32) && defined(_POSIX_VERSION)
#include <sys/wait.h>
#include <unistd.h>
#define LK_TEST_FORK 1
#endif

// A fit must leave the caller's BLAS / OpenMP thread counts as it found them:
// a multistart that resets them to 1 thread makes every later predict,
// simulate or update single-threaded. A fork() quiesces them to 1 thread in
// the child only; the parent gets its counts back once fork() returns.

#if !defined(_MSC_VER) && (!defined(__APPLE__) || !defined(__arm64__))
extern "C" {
void openblas_set_num_threads(int num_threads) __attribute__((weak));
int openblas_get_num_threads() __attribute__((weak));
}
#define LK_TEST_OPENBLAS 1
#endif

namespace {

struct Threads {
  int blas = 0;
  int omp = 0;
};

Threads get_threads() {
  Threads t;
#ifdef LK_TEST_OPENBLAS
  if (openblas_get_num_threads)
    t.blas = openblas_get_num_threads();
#endif
#ifdef _OPENMP
  t.omp = omp_get_max_threads();
#endif
  return t;
}

void set_threads(int n) {
#ifdef LK_TEST_OPENBLAS
  if (openblas_set_num_threads)
    openblas_set_num_threads(n);
#endif
#ifdef _OPENMP
  omp_set_num_threads(n);
#endif
}

// Sets 3 threads (read back: OpenBLAS may cap it) for the test scope.
struct ThreeThreads {
  Threads initial = get_threads();
  Threads set;
  ThreeThreads() {
    set_threads(3);
    set = get_threads();
  }
  ~ThreeThreads() { set_threads(initial.omp > 0 ? initial.omp : (initial.blas > 0 ? initial.blas : 1)); }
};

bool readable(const Threads& t) {
  return t.blas > 0 || t.omp > 0;
}

void data(arma::vec& y, arma::mat& X, arma::uword n = 40) {
  arma::arma_rng::set_seed(123);
  X = arma::randu(n, 2);
  y = arma::sum(arma::sin(3 * X), 1);
}

}  // namespace

TEST_CASE("Kriging multistart fit restores the BLAS / OpenMP thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  arma::vec y;
  arma::mat X;
  data(y, X);
  Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4");
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}

TEST_CASE("WarpKriging parallel multistart fit restores the BLAS / OpenMP thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  const int pool_size = Optim::get_thread_pool_size();
  Optim::set_thread_pool_size(2);
  arma::vec y;
  arma::mat X;
  data(y, X, 20);
  libKriging::WarpKriging w(y, X, {"affine", "affine"}, "gauss", Trend::RegressionModel::Constant, false, "BFGS4");
  Optim::set_thread_pool_size(pool_size);
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}

#ifdef LK_TEST_FORK
TEST_CASE("fork() quiesces the child only and restores the parent's thread counts", "[threads]") {
  ThreeThreads guard;
  if (!readable(guard.set)) {
    WARN("no readable BLAS / OpenMP thread count in this build");
    return;
  }
  const pid_t pid = fork();
  if (pid == 0) {
    const Threads c = get_threads();
    _exit((c.blas <= 1 && c.omp <= 1) ? 0 : 1);
  }
  REQUIRE(pid > 0);
  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status));
  CHECK(WEXITSTATUS(status) == 0);
  const Threads after = get_threads();
  CHECK(after.blas == guard.set.blas);
  CHECK(after.omp == guard.set.omp);
}
#endif

#ifdef _OPENMP
// The OpenMP loops (covariance matrix, log-likelihood gradient) write
// independent entries or reduce in a fixed order: the objective, its gradient
// and an updated model do not depend on the OpenMP thread count (the BLAS
// thread count is left unchanged).
TEST_CASE("Objective, gradient and update do not depend on the OpenMP thread count", "[threads]") {
  const int initial = omp_get_max_threads();
  arma::vec y;
  arma::mat X;
  data(y, X, 450);
  const arma::vec theta{0.3, 0.5};
  Kriging::Parameters par;
  par.theta = arma::mat(theta.t());
  par.is_theta_estim = false;

  auto run = [&](int threads) {
    omp_set_num_threads(threads);
    Kriging k(y.head(300), X.head_rows(300), "matern5_2", Trend::RegressionModel::Constant, false, "none", "LL", par);
    auto [ll, grad] = k.logLikelihoodFun(theta, true, false);
    k.update(y.tail(150), X.tail_rows(150), false);
    return std::make_tuple(ll, arma::vec(grad), k.logLikelihood());
  };
  const auto [ll1, grad1, llu1] = run(1);
  const auto [ll4, grad4, llu4] = run(4);
  omp_set_num_threads(initial);

  CHECK(ll1 == ll4);
  CHECK(arma::all(grad1 == grad4));
  CHECK(llu1 == llu4);
}
#endif
