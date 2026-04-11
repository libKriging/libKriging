// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/AdamBFGS.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

// ---------------------------------------------------------------------------
//  Test function: Branin on [0,1]^2
// ---------------------------------------------------------------------------
double branin(const arma::rowvec& x) {
  double x1 = x(0) * 15.0 - 5.0;
  double x2 = x(1) * 15.0;
  return std::pow(x2 - 5.0 / (4.0 * M_PI * M_PI) * x1 * x1 + 5.0 / M_PI * x1 - 6.0, 2)
         + 10.0 * (1.0 - 1.0 / (8.0 * M_PI)) * std::cos(x1) + 10.0;
}

// ---------------------------------------------------------------------------
//  Build training data (simple LHS)
// ---------------------------------------------------------------------------
void make_data(arma::uword n, arma::uword d, arma::mat& X, arma::vec& y) {
  arma::arma_rng::set_seed(42);
  X.set_size(n, d);
  y.set_size(n);

  for (arma::uword j = 0; j < d; ++j) {
    arma::uvec perm = arma::randperm(n);
    for (arma::uword i = 0; i < n; ++i) {
      X(i, j) = (static_cast<double>(perm(i)) + arma::randu()) / n;
    }
  }
  for (arma::uword i = 0; i < n; ++i) {
    y(i) = branin(X.row(i));
  }
}

// ---------------------------------------------------------------------------
//  Timer helper
// ---------------------------------------------------------------------------
struct Timer {
  std::chrono::high_resolution_clock::time_point t0;
  Timer() : t0(std::chrono::high_resolution_clock::now()) {}
  double elapsed_ms() const {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
};

// ---------------------------------------------------------------------------
//  Benchmark 1: Plain Kriging.fit (L-BFGS-B)
// ---------------------------------------------------------------------------
void bench_bfgs(const arma::vec& y, const arma::mat& X) {
  std::cout << "\n=== BFGS (Kriging.fit) ===" << std::endl;

  Timer t;
  Kriging kr("matern5_2");
  Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  kr.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);
  double dt = t.elapsed_ms();

  std::cout << "  theta : " << kr.theta().t();
  std::cout << "  sigma2: " << kr.sigma2() << std::endl;
  std::cout << "  LL    : " << kr.logLikelihood() << std::endl;
  std::cout << "  time  : " << std::fixed << std::setprecision(1) << dt << " ms" << std::endl;
}

// ---------------------------------------------------------------------------
//  Benchmark 2: AdamBFGS on the Kriging LL
//
//  We use logLikelihoodFun (public API) as the objective.
//  Split theta into x_outer (Adam) and x_inner (BFGS) to simulate
//  the WarpKriging pattern.
// ---------------------------------------------------------------------------
void bench_adambfgs(const arma::vec& y, const arma::mat& X) {
  std::cout << "\n=== AdamBFGS ===" << std::endl;

  const arma::uword d = X.n_cols;

  // First, fit a reference Kriging model to get baseline LL and the model object
  Kriging kr_ref("matern5_2");
  Kriging::Parameters params{std::nullopt, true, std::nullopt, true, std::nullopt, true};
  kr_ref.fit(y, X, Trend::RegressionModel::Constant, false, "BFGS", "LL", params);

  // Use same theta range as Kriging internally uses
  arma::vec theta_ref = kr_ref.theta();
  arma::vec theta_lower = theta_ref * 0.01;
  arma::vec theta_upper = theta_ref * 100.0;

  // Start from a perturbed version of the optimum (not too far)
  arma::arma_rng::set_seed(42);
  arma::vec theta0 = theta_ref % arma::exp(0.5 * arma::randn<arma::vec>(d));
  theta0 = arma::min(arma::max(theta0, theta_lower), theta_upper);

  std::cout << "  ref theta: " << theta_ref.t();
  std::cout << "  start theta: " << theta0.t();

  // --- Test 1: All theta as inner (BFGS-only mode via AdamBFGS) ---
  {
    std::cout << "\n  -- All theta as inner (BFGS-only path) --" << std::endl;
    AdamBFGS opt(0, d);
    opt.max_iter_adam = 1;  // single iteration = just BFGS
    opt.max_iter_bfgs = 100;
    opt.bfgs_pgtol = 1e-6;
    opt.maximize = true;

    // Work in log-space for better BFGS conditioning
    arma::vec log_theta0 = arma::log(theta0);
    arma::vec log_lower = arma::log(theta_lower);
    arma::vec log_upper = arma::log(theta_upper);

    auto obj_fn = [&kr_ref](const arma::vec& /*x_outer*/,
                            const arma::vec& x_inner,
                            arma::vec* /*grad_outer*/,
                            arma::vec* grad_inner) -> double {
      arma::vec theta = arma::exp(x_inner);
      auto [ll, grad_theta, hess] = kr_ref.logLikelihoodFun(theta, grad_inner != nullptr, false, false);
      if (grad_inner) {
        // Chain rule: ∂LL/∂(log θ) = ∂LL/∂θ · θ
        *grad_inner = grad_theta % theta;
      }
      return ll;
    };

    Timer t;
    arma::vec dummy_outer;
    auto result = opt.optimize(dummy_outer, log_theta0, log_lower, log_upper, obj_fn);
    double dt = t.elapsed_ms();

    arma::vec theta_opt = arma::exp(result.x_inner);
    std::cout << "    theta : " << theta_opt.t();
    std::cout << "    LL    : " << result.f_opt << std::endl;
    std::cout << "    BFGS evals: " << result.n_bfgs_evals << std::endl;
    std::cout << "    time  : " << std::fixed << std::setprecision(1) << dt << " ms" << std::endl;
  }

  // --- Test 2: Split - first dim = Adam, rest = BFGS ---
  {
    std::cout << "\n  -- Split: dim0=Adam, rest=BFGS --" << std::endl;
    arma::uword n_out = 1;
    arma::uword n_in = d - 1;

    AdamBFGS opt(n_out, n_in);
    opt.max_iter_adam = 50;
    opt.adam_lr = 1e-2;
    opt.max_iter_bfgs = 100;
    opt.bfgs_pgtol = 1e-6;
    opt.maximize = true;

    arma::vec log_theta0 = arma::log(theta0);
    arma::vec log_lower = arma::log(theta_lower);
    arma::vec log_upper = arma::log(theta_upper);

    arma::vec outer0 = log_theta0.head(n_out);
    arma::vec inner0 = log_theta0.tail(n_in);
    arma::vec inner_lower = log_lower.tail(n_in);
    arma::vec inner_upper = log_upper.tail(n_in);

    auto obj_fn = [&kr_ref, n_out, n_in](const arma::vec& x_outer,
                                         const arma::vec& x_inner,
                                         arma::vec* grad_outer,
                                         arma::vec* grad_inner) -> double {
      arma::vec log_theta = arma::join_cols(x_outer, x_inner);
      arma::vec theta = arma::exp(log_theta);
      bool need_grad = (grad_outer != nullptr || grad_inner != nullptr);
      auto [ll, grad_theta, hess] = kr_ref.logLikelihoodFun(theta, need_grad, false, false);
      if (need_grad) {
        arma::vec grad_log = grad_theta % theta;
        if (grad_outer)
          *grad_outer = grad_log.head(n_out);
        if (grad_inner)
          *grad_inner = grad_log.tail(n_in);
      }
      return ll;
    };

    Timer t;
    auto result = opt.optimize(outer0, inner0, inner_lower, inner_upper, obj_fn);
    double dt = t.elapsed_ms();

    arma::vec log_theta_opt = arma::join_cols(result.x_outer, result.x_inner);
    arma::vec theta_opt = arma::exp(log_theta_opt);
    std::cout << "    theta : " << theta_opt.t();
    std::cout << "    LL    : " << result.f_opt << std::endl;
    std::cout << "    Adam iters: " << result.n_adam_iters << ", BFGS evals: " << result.n_bfgs_evals << std::endl;
    std::cout << "    time  : " << std::fixed << std::setprecision(1) << dt << " ms" << std::endl;
  }
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  arma::uword n = 20;
  arma::uword d = 2;
  if (argc > 1)
    n = std::atoi(argv[1]);
  if (argc > 2)
    d = std::atoi(argv[2]);

  std::cout << "AdamBFGS Benchmark: n=" << n << " d=" << d << std::endl;

  arma::mat X;
  arma::vec y;
  make_data(n, d, X, y);

  bench_bfgs(y, X);
  bench_adambfgs(y, X);

  return 0;
}
