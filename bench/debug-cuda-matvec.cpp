#include "libKriging/Covariance.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/utils/lk_armadillo.hpp"

#include "cuda/CudaLinearAlgebra.cuh"
#include "cuda/CudaLinearAlgebraKernel.cuh"

#include <cuda_runtime.h>

#include <iostream>

static void check(cudaError_t s, const char* what) {
  if (s != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(s) << std::endl;
    std::exit(1);
  }
}

int main() {
  arma::arma_rng::set_seed(42);
  const int n = 25;
  const int d = 2;
  const int ncols = 1;
  arma::mat X(n, d, arma::fill::randu);
  arma::mat Xt = X.t();
  arma::vec theta = {0.3, 0.3};
  arma::mat P(n, ncols, arma::fill::randu);

  auto cov = Covariance::resolve("matern5_2").Cov;
  arma::mat Ap_cpu(n, ncols, arma::fill::zeros);
  for (int c = 0; c < ncols; ++c)
    for (int i = 0; i < n; ++i) {
      double acc = P(i, c);
      for (int j = 0; j < n; ++j) {
        if (j == i)
          continue;
        acc += cov(Xt.col(i) - Xt.col(j), theta) * P(j, c);
      }
      Ap_cpu(i, c) = acc;
    }

  const int scratch_elems = lk_cuda_rmul_batched_scratch_elems(n, ncols);
  std::cout << "scratch_elems = " << scratch_elems << std::endl;

  double *d_Xt, *d_theta, *d_P, *d_Ap, *d_scratch = nullptr;
  check(cudaMalloc(&d_Xt, sizeof(double) * n * d), "malloc Xt");
  check(cudaMalloc(&d_theta, sizeof(double) * d), "malloc theta");
  check(cudaMalloc(&d_P, sizeof(double) * n * ncols), "malloc P");
  check(cudaMalloc(&d_Ap, sizeof(double) * n * ncols), "malloc Ap");
  if (scratch_elems > 0)
    check(cudaMalloc(&d_scratch, sizeof(double) * scratch_elems), "malloc scratch");
  check(cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * n * d, cudaMemcpyHostToDevice), "cpy Xt");
  check(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * d, cudaMemcpyHostToDevice), "cpy theta");
  check(cudaMemcpy(d_P, P.memptr(), sizeof(double) * n * ncols, cudaMemcpyHostToDevice), "cpy P");

  lk_cuda_rmul_batched_launch(d_Xt, n, d, d_theta, /*covKind=matern5_2*/ 3, d_P, ncols, d_Ap, d_scratch);
  check(cudaGetLastError(), "launch");
  check(cudaDeviceSynchronize(), "sync");

  arma::mat Ap_gpu(n, ncols, arma::fill::none);
  check(cudaMemcpy(Ap_gpu.memptr(), d_Ap, sizeof(double) * n * ncols, cudaMemcpyDeviceToHost), "cpy back");

  std::cout << "max abs diff (matvec only) = " << arma::abs(Ap_cpu - Ap_gpu).max() << std::endl;
  std::cout << "CPU col0: " << Ap_cpu.col(0).t();
  std::cout << "GPU col0: " << Ap_gpu.col(0).t();

  // Determinism check: run the SAME matvec call again with identical
  // inputs and require a BIT-IDENTICAL result -- this is what the two
  // Kriging tests actually require (predict() vs predictIterative() /
  // optim=none vs optim=BFGS calling the same CG solve twice).
  arma::mat Ap_gpu2(n, ncols, arma::fill::none);
  lk_cuda_rmul_batched_launch(d_Xt, n, d, d_theta, /*covKind=matern5_2*/ 3, d_P, ncols, d_Ap, d_scratch);
  check(cudaGetLastError(), "launch2");
  check(cudaDeviceSynchronize(), "sync2");
  check(cudaMemcpy(Ap_gpu2.memptr(), d_Ap, sizeof(double) * n * ncols, cudaMemcpyDeviceToHost), "cpy back2");
  std::cout << "determinism: max abs diff between two identical-input calls = " << arma::abs(Ap_gpu - Ap_gpu2).max()
            << std::endl;

  // Now the full CG solve (repeated matvecs), same P used as the RHS B.
  auto Rmul = [&](const arma::vec& v) -> arma::vec {
    arma::vec out(n, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
      double acc = v(i);
      for (int j = 0; j < n; ++j) {
        if (j == i)
          continue;
        acc += cov(Xt.col(i) - Xt.col(j), theta) * v(j);
      }
      out(i) = acc;
    }
    return out;
  };
  arma::mat X_cpu = LinearAlgebra::conjugateGradient(Rmul, P, 2 * n, 1e-10);
  arma::mat X_gpu = LinearAlgebraCuda::conjugateGradient(Xt, theta, "matern5_2", P, 2 * n, 1e-10);
  std::cout << "max abs diff (full CG)     = " << arma::abs(X_cpu - X_gpu).max() << std::endl;
  std::cout << "CG CPU col0: " << X_cpu.col(0).t();
  std::cout << "CG GPU col0: " << X_gpu.col(0).t();

  // Residual check: does A*x_gpu actually equal b (the real correctness bar)?
  for (int c = 0; c < ncols; ++c) {
    arma::vec resid = Rmul(X_gpu.col(c)) - P.col(c);
    std::cout << "col " << c << " ||A*x_gpu - b|| = " << arma::norm(resid) << "  ||b|| = " << arma::norm(P.col(c))
              << std::endl;
  }
  return 0;
}
