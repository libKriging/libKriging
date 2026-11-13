// clang-format off
// Must be first
#define CATCH_CONFIG_MAIN
#include "libKriging/utils/lk_armadillo.hpp"

#include <catch2/catch.hpp>
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>
// clang-format on

// BLAS backends (docs/dev/rationalization/T9-blas-backends-ci.md): what the
// library does with the thread count of the BLAS it runs on, checked on the
// backend actually loaded (OpenBLAS, Intel MKL / oneMKL linked or preloaded,
// Apple Accelerate). A case for another backend warns and returns. The CI jobs
// set LK_EXPECT_BLAS (mkl / openblas / accelerate) so that a backend silently
// not loaded fails instead of skipping every case.

#if !defined(_WIN32)
#include <dlfcn.h>
#define LK_TEST_DLADDR 1
#endif

#if defined(__ELF__)
extern "C" {
int MKL_Get_Max_Threads() __attribute__((weak));
int MKL_Set_Num_Threads_Local(int nth) __attribute__((weak));
void openblas_set_num_threads(int num_threads) __attribute__((weak));
int openblas_get_num_threads() __attribute__((weak));
}
#define LK_TEST_WEAK 1
#endif

#if defined(_WIN32)
// No weak symbols on Windows: looked up in the loaded modules (last include)
#include "blas_symbols_win.hpp"
#endif

#if defined(LK_BACKEND_PROBE)
// tests/backend_probe.c: records every dpotrf_ call of the process (thread,
// MKL and OpenBLAS counts in effect in that thread) before forwarding it.
extern "C" {
int lk_probe_available();
const char* lk_probe_potrf_provider();
void lk_probe_start();
void lk_probe_stop();
int lk_probe_count();
int lk_probe_total_calls();
void lk_probe_get(int i, unsigned long* tid, int* mkl, int* openblas);
unsigned long lk_probe_self();
}
#endif

#if defined(__APPLE__) && defined(__has_include)
#if __has_include(<vecLib/thread_api.h>)
#include <vecLib/thread_api.h>
#define LK_TEST_BLAS_THREADING_API 1
#endif
#endif

namespace {

// Shared object serving dgemm_ (the BLAS every Armadillo product goes to)
std::string blas_provider() {
#ifdef LK_TEST_DLADDR
  void* f = dlsym(RTLD_DEFAULT, "dgemm_");
  Dl_info info;
  if (f != nullptr && dladdr(f, &info) != 0 && info.dli_fname != nullptr)
    return info.dli_fname;
#elif defined(_WIN32)
  return lk_blas_win::module_path("dgemm_");
#endif
  return "";
}

std::string backend() {
  const std::string p = blas_provider();
  auto has = [&](const char* s) { return p.find(s) != std::string::npos; };
  if (has("mkl"))
    return "mkl";
  if (has("openblas"))
    return "openblas";
  if (has("Accelerate") || has("vecLib") || has("libBLAS.dylib"))
    return "accelerate";
#if defined(__APPLE__)
  if (p.empty())
    return "accelerate";  // two-level namespace: dgemm_ may not be global
#endif
  return p.empty() ? "unknown" : "other";
}

int mkl_count() {
#ifdef LK_TEST_WEAK
  if (MKL_Get_Max_Threads)
    return MKL_Get_Max_Threads();
#elif defined(_WIN32)
  static const auto f = lk_blas_win::proc<int (*)()>("MKL_Get_Max_Threads");
  if (f)
    return f();
#endif
  return 0;
}

// Sets the calling thread's MKL count (0: back to the global count), returns
// the previous thread-local count (0: none), -1 without MKL
int mkl_set_local(int n) {
#ifdef LK_TEST_WEAK
  if (MKL_Set_Num_Threads_Local)
    return MKL_Set_Num_Threads_Local(n);
#elif defined(_WIN32)
  static const auto f = lk_blas_win::proc<int (*)(int)>("MKL_Set_Num_Threads_Local");
  if (f)
    return f(n);
#endif
  (void)n;
  return -1;
}

int openblas_count() {
#ifdef LK_TEST_WEAK
  if (openblas_get_num_threads)
    return openblas_get_num_threads();
#elif defined(_WIN32)
  static const auto f = lk_blas_win::proc<int (*)()>("openblas_get_num_threads");
  if (f)
    return f();
#endif
  return 0;
}

// Deterministic design, identical on every platform (no RNG distribution:
// libstdc++ and libc++ differ): Weyl sequence in [0, 1)^3.
void weyl_data(arma::uword n, arma::mat& X, arma::vec& y) {
  const double a[3] = {0.41421356237309515, 0.7320508075688772, 0.2360679774997898};
  X.set_size(n, 3);
  for (arma::uword i = 0; i < n; ++i)
    for (arma::uword j = 0; j < 3; ++j)
      X(i, j) = std::fmod(0.5 + static_cast<double>(i + 1) * a[j], 1.0);
  y = arma::sin(6.0 * X.col(0)) + arma::cos(4.0 * X.col(1)) % X.col(2) + 0.5 * arma::sin(5.0 * X.col(2));
}

struct BudgetGuard {
  double percent = Optim::get_num_threads_percent();
  int pool = Optim::get_thread_pool_size();
  ~BudgetGuard() {
    Optim::set_num_threads_percent(percent);
    Optim::set_thread_pool_size(pool);
  }
};

#if defined(LK_BACKEND_PROBE)
struct Record {
  unsigned long tid;
  int mkl;
  int openblas;
};

// dpotrf_ calls made while `run` runs
template <typename F>
std::vector<Record> probe(F run) {
  lk_probe_start();
  run();
  lk_probe_stop();
  std::vector<Record> r(static_cast<std::size_t>(lk_probe_count()));
  for (int i = 0; i < lk_probe_count(); ++i)
    lk_probe_get(i, &r[i].tid, &r[i].mkl, &r[i].openblas);
  return r;
}
#endif

#if defined(LK_BACKEND_PROBE)
// Pool size and worker share of a multistart fit with `starts` starts at
// budget B (TaskPool.cpp)
void pool_of(int B, int starts, int& pool, int& share) {
  pool = std::max(1, std::min(B, starts));
  share = std::max(1, B / pool);
}

// Count a backend loaded with b0 threads runs at for a share (ThreadBudget.cpp:
// lowered to the share only when loaded with more)
int lowered(int b0, int share) {
  return b0 > share ? share : b0;
}
#endif

}  // namespace

TEST_CASE("BLAS backend in use", "[backend]") {
  const std::string b = backend();
  Optim::set_num_threads_percent(100);
  std::printf("[backend] dgemm_ from: %s\n[backend] backend: %s, budget: %d, MKL count: %d, OpenBLAS count: %d\n",
              blas_provider().c_str(),
              b.c_str(),
              Optim::get_num_threads(),
              mkl_count(),
              openblas_count());
#if defined(LK_BACKEND_PROBE)
  std::printf("[backend] dpotrf_ from: %s\n", lk_probe_potrf_provider());
#endif
  const char* expected = std::getenv("LK_EXPECT_BLAS");
  if (expected != nullptr && *expected != '\0') {
    INFO("LK_EXPECT_BLAS=" << expected << ", dgemm_ from " << blas_provider());
    CHECK(b == expected);
  }
}

#if defined(LK_BACKEND_PROBE)
// T5: each pool worker sets its own MKL count (MKL_Set_Num_Threads_Local) to
// its share, the calling thread to the budget; read in the thread of every
// Cholesky factorization the fit makes.
TEST_CASE("MKL: each multistart worker runs its BLAS at its share", "[backend][mkl]") {
  if (backend() != "mkl" || !lk_probe_available()) {
    WARN("MKL is not the BLAS of this process, or the probe is not in the call path");
    return;
  }
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  Optim::set_thread_pool_size(0);
  const int B = Optim::get_num_threads();
  if (B < 2) {
    WARN("budget " << B << ": no pool");
    return;
  }
  int pool = 0, share = 0;
  pool_of(B, 4, pool, share);
  const int m0 = mkl_count();
  const int worker_expected = lowered(m0, share);
  const int caller_expected = lowered(m0, B);
  arma::mat X;
  arma::vec y;
  weyl_data(200, X, y);
  Random::reset_seed(123);
  const auto rec = probe([&] { Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4"); });
  const unsigned long self = lk_probe_self();
  std::set<unsigned long> workers;
  std::map<int, int> worker_counts, caller_counts;
  for (const auto& r : rec) {
    if (r.tid == self) {
      ++caller_counts[r.mkl];
    } else {
      workers.insert(r.tid);
      ++worker_counts[r.mkl];
    }
  }
  std::string wc, cc;
  for (const auto& kv : worker_counts)
    wc += " NThr " + std::to_string(kv.first) + ": " + std::to_string(kv.second) + " calls;";
  for (const auto& kv : caller_counts)
    cc += " NThr " + std::to_string(kv.first) + ": " + std::to_string(kv.second) + " calls;";
  std::printf("[backend] MKL loaded with %d, budget %d, pool %d x %d: workers%s caller%s\n",
              m0,
              B,
              pool,
              share,
              wc.c_str(),
              cc.c_str());
  INFO("MKL loaded with " << m0 << ", budget " << B << ", pool " << pool << " x " << share << "; workers:" << wc
                          << " caller:" << cc);
  REQUIRE(!rec.empty());
  CHECK(static_cast<int>(workers.size()) == pool);
  REQUIRE(!worker_counts.empty());
  CHECK(worker_counts.size() == 1);
  CHECK(worker_counts.begin()->first == worker_expected);
  for (const auto& kv : caller_counts)
    CHECK(kv.first == caller_expected);
  CHECK(mkl_count() == m0);
}
#endif

#if defined(LK_BACKEND_PROBE)
// T6: the OpenBLAS count is process-global; during a multistart, the smallest
// live request is the pool's share, so every worker's BLAS runs at it.
TEST_CASE("OpenBLAS: each multistart worker runs its BLAS at its share", "[backend][openblas]") {
  if (backend() != "openblas" || !lk_probe_available()) {
    WARN("OpenBLAS is not the BLAS of this process, or the probe is not in the call path");
    return;
  }
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  Optim::set_thread_pool_size(0);
  const int B = Optim::get_num_threads();
  if (B < 2) {
    WARN("budget " << B << ": no pool");
    return;
  }
  int pool = 0, share = 0;
  pool_of(B, 4, pool, share);
  const int b0 = openblas_count();
  arma::mat X;
  arma::vec y;
  weyl_data(200, X, y);
  Random::reset_seed(123);
  const auto rec = probe([&] { Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "BFGS4"); });
  const unsigned long self = lk_probe_self();
  std::set<unsigned long> workers;
  std::set<int> worker_counts;
  for (const auto& r : rec)
    if (r.tid != self) {
      workers.insert(r.tid);
      worker_counts.insert(r.openblas);
    }
  INFO("OpenBLAS loaded with " << b0 << ", budget " << B << ", pool " << pool << " x " << share);
  REQUIRE(!rec.empty());
  CHECK(static_cast<int>(workers.size()) == pool);
  REQUIRE(worker_counts.size() == 1);
  CHECK(*worker_counts.begin() == lowered(b0, share));
  CHECK(openblas_count() == b0);
}
#endif

// T5: the library changes the MKL count of the calling thread only for the
// duration of a call (thread-local), and puts back what the caller had: no
// thread-local count, or the caller's own one.
TEST_CASE("MKL: the caller's count is restored after fit / predict / simulate / update", "[backend][mkl]") {
  if (backend() != "mkl") {
    WARN("MKL is not the BLAS of this process");
    return;
  }
  BudgetGuard guard;
  arma::mat X;
  arma::vec y;
  weyl_data(120, X, y);
  arma::mat Xn;
  arma::vec yn;
  weyl_data(130, Xn, yn);
  const arma::mat Xu = Xn.tail_rows(10);
  const arma::vec yu = yn.tail(10);

  auto all_calls = [&](int expected_count, int expected_local) {
    Random::reset_seed(123);
    Kriging k("matern5_2");
    k.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS4");
    CHECK(mkl_count() == expected_count);
    k.predict(Xu, true, false, false);
    CHECK(mkl_count() == expected_count);
    k.simulate(3, 1, Xu);
    CHECK(mkl_count() == expected_count);
    k.update(yu, Xu, true);
    CHECK(mkl_count() == expected_count);
    // previous thread-local count, then back to it
    const int local = mkl_set_local(0);
    CHECK(local == expected_local);
    mkl_set_local(local);
  };

  SECTION("no thread-local count in the caller, budget 100 %") {
    Optim::set_num_threads_percent(100);
    REQUIRE(mkl_set_local(0) == 0);
    const int m0 = mkl_count();
    all_calls(m0, 0);
  }
  SECTION("no thread-local count in the caller, budget 1 thread") {
    Optim::set_num_threads_percent(1e-6);
    REQUIRE(mkl_set_local(0) == 0);
    const int m0 = mkl_count();
    all_calls(m0, 0);
  }
  SECTION("the caller's own thread-local count, budget 50 %") {
    Optim::set_num_threads_percent(50);
    mkl_set_local(3);
    all_calls(3, 3);
    mkl_set_local(0);
  }
}

// BFGS4 with a pool of one worker (every start in turn, the worker at the
// whole budget) and with the default pool (starts in parallel, each worker at
// its share): same starts, same best start. Only the BLAS thread count differs
// between the two runs, which changes the rounding of the BLAS sums
// (parallel.md, section 4), not the optimum.
TEST_CASE("BFGS4 with 1 or 4 workers reaches the same optimum", "[backend]") {
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  for (const char* kernel : {"matern5_2", "gauss"}) {
    arma::mat X;
    arma::vec y;
    weyl_data(300, X, y);
    auto fit = [&](int pool) {
      Optim::set_thread_pool_size(pool);
      Random::reset_seed(123);
      Kriging k(y, X, kernel, Trend::RegressionModel::Constant, false, "BFGS4");
      return std::make_pair(arma::vec(k.theta()), k.logLikelihood());
    };
    const auto one = fit(1);
    const auto four = fit(4);
    const double dtheta = arma::max(arma::abs(one.first - four.first) / arma::abs(four.first));
    const double dll = std::abs(one.second - four.second) / std::abs(four.second);
    std::printf("[backend] %s BFGS4 pool 1 vs 4: theta rel. gap %.3g, LL rel. gap %.3g (LL %.12g)\n",
                kernel,
                dtheta,
                dll,
                four.second);
    INFO(kernel << ": theta pool 1 " << one.first.t() << "theta pool 4 " << four.first.t() << "LL " << one.second
                << " vs " << four.second);
    // Measured (T9 log): LL within 4.7e-9, theta within 8.8e-5 (MKL, matern5_2)
    // and 2.1e-5 (OpenBLAS, gauss); 0 with the BLAS at 1 thread.
    CHECK(dll <= 1e-7);
    CHECK(dtheta <= 1e-3);
  }
}

// Shared reference values (Linux x86_64, OpenBLAS 0.3.26 at 1 thread; T9 log):
// 200 <= n < 800 is where T5 found the fitted model to depend on the OS
// (OpenMP threshold 200 on Linux, 800 elsewhere at the time; 200 with MSVC since #473). Single start "BFGS", so the
// starting point does not depend on the RNG. Two levels:
// - the log-likelihood and its gradient at a fixed theta (no optimizer): only
//   the rounding of the BLAS / libm differs between platforms;
// - the fitted theta and LL: the optimizer stops anywhere in the flat region of
//   the LL that the rounding allows, so the gaps are those of the BFGS stopping
//   rule (matern5_2 theta within 1.5e-4 between OpenBLAS 1 / 4 threads and MKL
//   1 / 4 threads; gauss within 1.1e-7; LL within 2.2e-8).
TEST_CASE("Fitted theta for 200 <= n < 800 matches the shared reference", "[backend]") {
  struct Ref {
    arma::uword n;
    const char* kernel;
    double theta[3];
    double ll;
    double ll_fixed;
    double grad_fixed[3];
  };
  const Ref refs[] = {
      {250,
       "matern5_2",
       {3.2754932418623013, 3.0690458391583739, 4.4629996899310607},
       1109.5101095997225,
       527.42934773177967,
       {759.51264993399093, 369.46861788876845, 39.778282965721218}},
      {500,
       "matern5_2",
       {2.5189133758854529, 2.5366705249161834, 3.4442774177499254},
       2926.8011182253831,
       1599.2002894858213,
       {1733.5957458480284, 875.72614274220314, 158.81481810717719}},
      {750,
       "matern5_2",
       {2.1553359846126408, 2.2729566353239727, 2.94411724021298},
       4957.8597940146274,
       2871.563468249401,
       {2771.9529253154328, 1483.5499368461465, 297.0681694973897}},
      {250,
       "gauss",
       {0.49935935316285834, 0.75371258920697159, 0.66923927103649961},
       1250.9431745756963,
       966.4135921720931,
       {1785.9329943811229, 773.01539082188026, -1047.8450658937313}},
      {500,
       "gauss",
       {0.43402972826249087, 0.65174666025175432, 0.59144220997045682},
       3385.7541437205359,
       2983.5225015189039,
       {2719.8303117855394, 1230.1040372889831, -2353.118134283292}},
      {750,
       "gauss",
       {0.40474083199417227, 0.61217729082119399, 0.5592214804336717},
       5707.0128160533568,
       5179.014749633574,
       {3353.3164556304741, 1469.4400037397313, -3808.1834318667106}},
  };
  const arma::vec theta_fixed = {0.3, 0.5, 0.7};
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  const bool print = std::getenv("LK_PRINT_REFERENCE") != nullptr;
  for (const auto& r : refs) {
    arma::mat X;
    arma::vec y;
    weyl_data(r.n, X, y);
    Random::reset_seed(123);
    Kriging k(y, X, r.kernel, Trend::RegressionModel::Constant, false, "BFGS");
    const arma::vec theta = k.theta();
    const double ll = k.logLikelihood();
    const auto [ll_fixed, grad_fixed] = k.logLikelihoodFun(theta_fixed, true, false);
    const arma::vec ref = {r.theta[0], r.theta[1], r.theta[2]};
    const arma::vec ref_grad = {r.grad_fixed[0], r.grad_fixed[1], r.grad_fixed[2]};
    const double dtheta = arma::max(arma::abs(theta - ref) / arma::abs(ref));
    const double dll = std::abs(ll - r.ll) / std::abs(r.ll);
    const double dll_fixed = std::abs(ll_fixed - r.ll_fixed) / std::abs(r.ll_fixed);
    const double dgrad_fixed = arma::max(arma::abs(grad_fixed - ref_grad)) / arma::max(arma::abs(ref_grad));
    if (print)
      std::printf("      {%u, \"%s\", {%.17g, %.17g, %.17g}, %.17g, %.17g, {%.17g, %.17g, %.17g}},\n",
                  static_cast<unsigned>(r.n),
                  r.kernel,
                  theta(0),
                  theta(1),
                  theta(2),
                  ll,
                  ll_fixed,
                  grad_fixed(0),
                  grad_fixed(1),
                  grad_fixed(2));
    std::printf(
        "[backend] %s n=%u: relative gaps to the reference: theta %.3g, LL %.3g; at fixed theta: LL %.3g, gradient "
        "%.3g\n",
        r.kernel,
        static_cast<unsigned>(r.n),
        dtheta,
        dll,
        dll_fixed,
        dgrad_fixed);
    INFO(r.kernel << " n=" << r.n << ": theta " << theta.t() << "reference " << ref.t() << "LL " << ll << " vs " << r.ll
                  << "; fixed theta: LL " << ll_fixed << " vs " << r.ll_fixed << ", gradient " << grad_fixed.t()
                  << "reference " << ref_grad.t());
    // Tolerances: about 10-50 times the largest gaps measured between OpenBLAS
    // 1 / 4 threads and MKL 1 / 4 threads (T9 log). gauss: R is ill-conditioned
    // at theta_fixed (LL 1.5e-9, gradient 3e-8 between backends at n = 750).
    const bool gauss = std::string(r.kernel) == "gauss";
    CHECK(dll_fixed <= (gauss ? 1e-8 : 1e-11));
    CHECK(dgrad_fixed <= (gauss ? 1e-6 : 1e-10));
    CHECK(dll <= 1e-7);
    // Fitted theta: the LL is flat along some directions, so theta moves more
    // than the LL. Same 1e-3 for both kernels: gauss n = 750 on Accelerate
    // (macOS-latest arm64, main.yml) gave theta 5.1e-5 with LL 1.5e-9, LL at
    // fixed theta 5.8e-10, gradient 2.9e-8; matern5_2 n = 250 gave theta 1.0e-4.
    // (The first gauss bound, 1e-5, came from a single macos-15 sample, 5.4e-8.)
    CHECK(dtheta <= 1e-3);
  }
}

// Accelerate's dpotri intermittently returned a wrong inverse on 5-core macOS
// runners (gauss n = 750 at theta_fixed: LL gradient 2.7e10 against 3.4e3, LL
// unchanged; T9 log, #452). Calls dpotri directly, repeatedly, and reports the
// results that disagree with two triangular solves (diagnostic, no check);
// checks that LinearAlgebra::inv_sympd, which verifies its potri result, gives
// the solves' inverse every time.
TEST_CASE("inv_sympd is right even when the backend's potri is not", "[backend]") {
  const arma::uword n = 750;
  arma::mat X;
  arma::vec y;
  weyl_data(n, X, y);
  const arma::vec theta = {0.3, 0.5, 0.7};
  arma::mat R(n, n);
  for (arma::uword i = 0; i < n; ++i)
    for (arma::uword j = 0; j < n; ++j)
      R.at(i, j) = std::exp(-0.5 * arma::accu(arma::square((X.row(i) - X.row(j)).t() / theta)));
  R.diag() += 1e-9;  // the gauss jitter of the fits (LinearAlgebra::kernel_chol_jitter)
  const arma::mat L = arma::chol(R, "lower");
  const arma::mat ref = arma::solve(arma::trimatu(L.t()), arma::solve(arma::trimatl(L), arma::eye(n, n)));
  const double scale = arma::abs(ref).max();
  const int reps = 20;
  int faults = 0;
  double worst_potri = 0.0, worst_lib = 0.0;
  for (int r = 0; r < reps; ++r) {
    arma::mat P = L;
    char uplo = 'L';
    arma::blas_int nn = static_cast<arma::blas_int>(n), info = 0;
    arma::lapack::potri(&uplo, &nn, P.memptr(), &nn, &info);
    const double gap = (info == 0) ? arma::abs(arma::symmatl(P) - ref).max() / scale : arma::datum::inf;
    worst_potri = std::max(worst_potri, gap);
    if (!(gap <= 1e-8))
      ++faults;
    const double gap_lib = arma::abs(LinearAlgebra::inv_sympd(L) - ref).max() / scale;
    worst_lib = std::max(worst_lib, gap_lib);
    CHECK(gap_lib <= 1e-8);
  }
  std::printf(
      "[backend] %s potri, n = %u: %d / %d results off the triangular solves (worst relative gap %.3g); "
      "inv_sympd worst %.3g\n",
      backend().c_str(),
      static_cast<unsigned>(n),
      faults,
      reps,
      worst_potri,
      worst_lib);
}

namespace {

double seconds_of(const std::function<void()>& f) {
  const auto t0 = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

// Hidden (run explicitly by the blas-backends workflow): BFGS1 vs BFGS4 wall
// time, and on Accelerate (macOS 15+) the effect of BLASSetThreading
// single-threaded during the multistart. Measures, asserts nothing.
TEST_CASE("Multistart timing per BLAS threading", "[.timing]") {
  BudgetGuard guard;
  Optim::set_num_threads_percent(100);
  const int reps = 3;
  std::printf("[timing] backend %s, budget %d\n", backend().c_str(), Optim::get_num_threads());
  for (arma::uword n : {200, 400, 800}) {
    arma::mat X;
    arma::vec y;
    weyl_data(n, X, y);
    auto run = [&](const char* optim) {
      std::vector<double> t;
      for (int r = 0; r < reps; ++r)
        t.push_back(seconds_of([&] {
          Random::reset_seed(123);
          Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, optim);
        }));
      return median_of(t);
    };
    const double t1 = run("BFGS");
    const double t4 = run("BFGS4");
    std::printf("[timing] n=%u BFGS %.3f s, BFGS4 %.3f s (x%.2f)\n", static_cast<unsigned>(n), t1, t4, t4 / t1);
#if defined(LK_TEST_BLAS_THREADING_API)
    if (__builtin_available(macOS 15.0, *)) {
      const BLAS_THREADING before = BLASGetThreading();
      BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
      const double t1s = run("BFGS");
      const double t4s = run("BFGS4");
      BLASSetThreading(before);
      std::printf("[timing] n=%u BLASSetThreading(single): BFGS %.3f s, BFGS4 %.3f s (BFGS4 multi / single x%.2f)\n",
                  static_cast<unsigned>(n),
                  t1s,
                  t4s,
                  t4 / t4s);
    } else {
      std::printf("[timing] BLASSetThreading: macOS < 15\n");
    }
#endif
  }
}
