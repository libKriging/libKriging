// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES
#include <cmath>
// clang-format on

#include "libKriging/Kriging.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include "cuda/CudaLinearAlgebra.cuh"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

// Three-way fit-time comparison, built only when -DENABLE_CUDA_ITERATIVE=ON
// (see src/lib/cuda/CudaLinearAlgebra.cuh):
//   - exact Cholesky (objective="LL"): O(n^3) dense factorization, the
//     baseline every large-n method in this project (Vecchia, Nystrom,
//     Iterative, NestedKriging, subsetOfData -- see docs/math/Scalability.md)
//     is trying to avoid.
//   - objective="LLIterative(m)" on the CPU (LinearAlgebra::conjugateGradient,
//     Rmul-based matvec).
//   - the same LLIterative(m) on CUDA (LinearAlgebraCuda::conjugateGradient,
//     batched+tiled matvec -- see CudaLinearAlgebraKernel.cu).
// LinearAlgebraCuda::set_enabled toggles the CG solve's backend at runtime
// so all three run in the same process, on the same fitted problem, with
// the same fixed theta -- no recompilation, no run-to-run MLE-fit variation
// to average out.

static double test_function(const arma::rowvec& x) {
  double sum = 0.0;
  for (arma::uword i = 0; i < x.n_elem; i++)
    sum += std::sin(2.0 * M_PI * x(i));
  return sum;
}

static double time_fit(const arma::vec& y, const arma::mat& X, const std::string& objective,
                       const arma::mat& theta0, int repeats) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    Kriging::Parameters params;
    params.theta = theta0;
    params.is_theta_estim = false;
    Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", objective, params);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / repeats;
}

// Fits once (outside the timed region), then times repeated predictIterative
// calls with return_stdev=true (the expensive path: one CG solve for the
// mean -- ncols=1 -- one CG solve per prediction point for the stdev --
// ncols=n_n -- and one more for the GLS trend correction -- ncols =
// m_F.n_cols, 1 here for a constant trend). See
// KrigingImpl::predictIterative_impl. use_nystrom_precond/precond_rank are
// forwarded to predictIterative as-is; note the CUDA backend only
// implements the no-preconditioner CG path (see
// LinearAlgebraCuda::conjugateGradient's docstring) -- with
// use_nystrom_precond=true, cgSolve falls back to the CPU
// LinearAlgebra::conjugateGradient regardless of LinearAlgebraCuda::enabled().
static double time_predict(const Kriging& k, const arma::mat& Xt, int repeats, bool use_nystrom_precond = false,
                           arma::uword precond_rank = 50) {
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeats; ++i) {
    auto [mean, stdev] = k.predictIterative(Xt, true, 0, 1e-8, use_nystrom_precond, precond_rank);
    (void)mean;
    (void)stdev;
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count() / repeats;
}

int main(int argc, char* argv[]) {
  std::cout << "Exact Cholesky vs LLIterative(m): CPU vs CUDA matvec/CG backend" << std::endl;

  if (!LinearAlgebraCuda::available()) {
    std::cout << "No CUDA device available at runtime -- nothing to benchmark." << std::endl;
    return 0;
  }

  std::vector<arma::uword> n_values = {500, 1000, 2000, 4000, 8000};
  if (argc > 1) {
    n_values.clear();
    for (int i = 1; i < argc; ++i)
      n_values.push_back(static_cast<arma::uword>(std::atoi(argv[i])));
  }

  const arma::uword d = 4;
  const arma::uword nprobe = 10;
  const std::string iterative_objective = "LLIterative(" + std::to_string(nprobe) + ")";
  // A single fit already takes seconds to minutes at the n where the CPU
  // vs CUDA comparison gets interesting (each fit does O(nprobe) CG solves,
  // each up to max_iter=2n iterations) -- repeats > 1 buys little precision
  // for a lot more wall time. Bump this if benchmarking small n in isolation.
  const int repeats = 1;

  // BENCH_SKIP_FIT=1 skips the fit-comparison loop below (useful when
  // profiling predictIterative in isolation with nsys/ncu -- avoids the
  // fit-loop's own kernel launches cluttering that trace).
  const bool skip_fit = std::getenv("BENCH_SKIP_FIT") != nullptr;

  std::cout << std::setw(8) << "n" << " | " << std::setw(14) << "Cholesky (s)" << " | " << std::setw(14)
            << "Iter CPU (s)" << " | " << std::setw(14) << "Iter CUDA (s)" << " | " << std::setw(12)
            << "CUDA/Chol" << std::endl;

  for (arma::uword n : n_values) {
    if (skip_fit)
      break;
    arma::arma_rng::set_seed(123);
    arma::mat X(n, d, arma::fill::randu);
    arma::vec y(n);
    for (arma::uword i = 0; i < n; ++i)
      y(i) = test_function(X.row(i));
    arma::mat theta0(1, d, arma::fill::value(0.3));

    const double t_chol = time_fit(y, X, "LL", theta0, repeats);

    LinearAlgebraCuda::set_enabled(false);
    const double t_cpu = time_fit(y, X, iterative_objective, theta0, repeats);

    LinearAlgebraCuda::set_enabled(true);
    const double t_cuda = time_fit(y, X, iterative_objective, theta0, repeats);

    std::cout << std::setw(8) << n << " | " << std::setw(14) << std::fixed << std::setprecision(4) << t_chol << " | "
              << std::setw(14) << std::fixed << std::setprecision(4) << t_cpu << " | " << std::setw(14) << std::fixed
              << std::setprecision(4) << t_cuda << " | " << std::setw(11) << std::fixed << std::setprecision(2)
              << (t_chol / t_cuda) << "x" << std::endl;
  }

  std::cout << std::endl
            << "predictIterative(return_stdev=true): CPU vs CUDA (unpreconditioned) vs "
               "Nystrom-preconditioned (CPU-only, no CUDA path yet)"
            << std::endl;
  std::cout << std::setw(8) << "n" << " | " << std::setw(6) << "n_n" << " | " << std::setw(14) << "CPU (s)" << " | "
            << std::setw(14) << "CUDA (s)" << " | " << std::setw(16) << "Nystrom-pc (s)" << " | " << std::setw(11)
            << "CUDA/CPU" << " | " << std::setw(13) << "pc/CUDA" << std::endl;

  for (arma::uword n : n_values) {
    arma::arma_rng::set_seed(123);
    arma::mat X(n, d, arma::fill::randu);
    arma::vec y(n);
    for (arma::uword i = 0; i < n; ++i)
      y(i) = test_function(X.row(i));
    arma::mat theta0(1, d, arma::fill::value(0.3));

    Kriging::Parameters params;
    params.theta = theta0;
    params.is_theta_estim = false;
    Kriging k(y, X, "matern5_2", Trend::RegressionModel::Constant, false, "none", iterative_objective, params);

    const arma::uword n_n = 30;
    arma::mat Xt(n_n, d, arma::fill::randu);

    // BENCH_CUDA_PREDICT_ONLY=1 skips the (slow) unpreconditioned CPU predict
    // timing -- useful when profiling only the CUDA path with nsys/ncu.
    const bool cuda_only = std::getenv("BENCH_CUDA_PREDICT_ONLY") != nullptr;
    double t_cpu = 0.0;
    if (!cuda_only) {
      LinearAlgebraCuda::set_enabled(false);
      t_cpu = time_predict(k, Xt, repeats);
    }

    LinearAlgebraCuda::set_enabled(true);
    const double t_cuda = time_predict(k, Xt, repeats);

    // use_nystrom_precond=true always runs on CPU (see time_predict's
    // docstring) regardless of LinearAlgebraCuda::enabled() -- the point
    // here is whether the preconditioner fixes CG's slow convergence at
    // all, not which backend it runs on.
    const double t_precond = time_predict(k, Xt, repeats, /*use_nystrom_precond=*/true, /*precond_rank=*/50);

    std::cout << std::setw(8) << n << " | " << std::setw(6) << n_n << " | " << std::setw(14) << std::fixed
              << std::setprecision(4) << t_cpu << " | " << std::setw(14) << std::fixed << std::setprecision(4)
              << t_cuda << " | " << std::setw(16) << std::fixed << std::setprecision(4) << t_precond << " | "
              << std::setw(10) << std::fixed << std::setprecision(2) << (t_cpu / t_cuda) << "x" << " | "
              << std::setw(12) << std::fixed << std::setprecision(2) << (t_precond / t_cuda) << "x" << std::endl;
  }

  return 0;
}
