// Host-compiler-compiled (NOT nvcc) on purpose: this file touches
// arma::mat/arma::vec (via .memptr()/.n_elem/etc), and must be compiled by
// the SAME compiler as the rest of libKriging to guarantee it agrees with
// Kriging.cpp/KrigingImpl.cpp on those types' memory layout -- see the
// comment in CudaLinearAlgebraKernel.cuh for why crossing that boundary
// with nvcc silently breaks (arma::vec::memptr() came back null on the
// nvcc side despite n_elem reading correctly). The only nvcc-compiled code
// this file talks to is the plain-C, pointer-only lk_cuda_*_launch family.
#include "CudaLinearAlgebra.cuh"

#ifdef LIBKRIGING_USE_CUDA_ITERATIVE

#include "CudaLinearAlgebraKernel.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

#define LK_CUDA_CHECK(expr)                                                            \
  do {                                                                                 \
    cudaError_t lk_cuda_status__ = (expr);                                              \
    if (lk_cuda_status__ != cudaSuccess) {                                              \
      std::ostringstream lk_cuda_oss__;                                                \
      lk_cuda_oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "          \
                    << cudaGetErrorString(lk_cuda_status__);                            \
      throw std::runtime_error(lk_cuda_oss__.str());                                   \
    }                                                                                  \
  } while (0)

enum class CovKind : int { Gauss = 0, Exp = 1, Matern32 = 2, Matern52 = 3 };

bool covKindFromString(const std::string& covType, CovKind* out) {
  if (covType == "gauss") {
    *out = CovKind::Gauss;
    return true;
  }
  if (covType == "exp") {
    *out = CovKind::Exp;
    return true;
  }
  if (covType == "matern3_2") {
    *out = CovKind::Matern32;
    return true;
  }
  if (covType == "matern5_2") {
    *out = CovKind::Matern52;
    return true;
  }
  return false;
}

}  // namespace

namespace LinearAlgebraCuda {

bool available() {
  static const bool cached = [] {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0;
  }();
  return cached;
}

namespace {
bool g_enabled_initialized = false;
bool g_enabled = false;
std::mutex g_enabled_mutex;
}  // namespace

bool enabled() {
  std::lock_guard<std::mutex> lock(g_enabled_mutex);
  if (!g_enabled_initialized) {
    g_enabled = available();
    g_enabled_initialized = true;
  }
  return g_enabled;
}

void set_enabled(bool value) {
  std::lock_guard<std::mutex> lock(g_enabled_mutex);
  g_enabled = value;
  g_enabled_initialized = true;
}

bool supports(const std::string& covType) {
  CovKind kind;
  return covKindFromString(covType, &kind);
}

// Solves every column of B "in lockstep": one rmul_batched matvec + two
// batched_dot reductions per CG iteration cover ALL columns at once,
// instead of the ncols-separate-CG-loops the single-column version used to
// run (see git history) -- each of which paid its own matvec-launch +
// several-small-cuBLAS-round-trips overhead. Profiling found exactly that
// overhead, not FLOPs, dominating at this project's n (see
// CudaLinearAlgebraKernel.cu's rmul_batched_kernel comment), so cutting the
// number of launches/host-syncs by ~ncols is the actual point here, not
// reducing total compute (each column always needed its own full O(n^2)
// matvec per iteration; that's unchanged).
//
// A column that converges (or breaks down) before others gets FROZEN
// rather than dropped: its alpha/beta are zeroed from then on, so x/r/p
// stop changing for that column while the rest keep iterating. There is no
// cheap way to shrink an in-flight CUDA launch's column count mid-loop, so
// this trades a little wasted compute on early-converged columns (bounded
// by however many iterations the slowest column still needs) for keeping
// everything in one batched launch per step -- the right trade given
// launch/sync overhead, not FLOPs, was the measured bottleneck.
arma::mat conjugateGradient(const arma::mat& Xt, const arma::vec& theta, const std::string& covType,
                            const arma::mat& B, arma::uword max_iter, double tol) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraCuda::conjugateGradient: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);

  // Upload X and theta once -- reused for every CG iteration, never
  // re-transferred mid-solve.
  double *d_Xt, *d_theta;
  LK_CUDA_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  LK_CUDA_CHECK(cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX,
                          cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));

  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  LK_CUDA_CHECK(cudaMalloc(&d_b, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_x, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_r, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_p, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_Ap, mat_bytes));

  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(ncols);
  double *d_scratch, *d_alpha, *d_beta;
  LK_CUDA_CHECK(cudaMalloc(&d_scratch, col_bytes));  // reused for pAp then for r.r each iteration
  LK_CUDA_CHECK(cudaMalloc(&d_alpha, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_beta, col_bytes));

  // n and ncols don't change across a CG solve, so the matvec's own scratch
  // requirement (see lk_cuda_rmul_batched_scratch_elems) is fixed too --
  // allocate it once here rather than inside the per-iteration matvec call.
  const int rmul_scratch_elems = lk_cuda_rmul_batched_scratch_elems(n, ncols);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_CUDA_CHECK(cudaMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));

  LK_CUDA_CHECK(cudaMemcpy(d_b, B.memptr(), mat_bytes, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemset(d_x, 0, mat_bytes));
  LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b - A*0
  LK_CUDA_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z = r (no precond)

  std::vector<double> bnorm(ncols), rz_old(ncols), host_scratch(ncols), host_alpha(ncols), host_beta(ncols);
  std::vector<bool> active(ncols);
  for (arma::uword c = 0; c < B.n_cols; ++c)
    bnorm[c] = arma::norm(B.col(c));
  for (int c = 0; c < ncols; ++c) {
    active[c] = bnorm[c] != 0.0;  // x=0 already solves A*x=0 for a zero column
    rz_old[c] = bnorm[c] * bnorm[c];  // r=b initially, so r.r = |b|^2
  }

  constexpr arma::uword restart_every = 50;
  bool any_active = std::any_of(active.begin(), active.end(), [](bool a) { return a; });

  for (arma::uword it = 0; any_active && it < max_iter; ++it) {
    lk_cuda_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_p, ncols, d_Ap, d_rmul_scratch);
    LK_CUDA_CHECK(cudaGetLastError());
    lk_cuda_batched_dot_launch(d_p, d_Ap, n, ncols, d_scratch);  // pAp
    LK_CUDA_CHECK(cudaGetLastError());
    LK_CUDA_CHECK(cudaMemcpy(host_scratch.data(), d_scratch, col_bytes, cudaMemcpyDeviceToHost));

    for (int c = 0; c < ncols; ++c) {
      if (!active[c] || host_scratch[c] <= 0.0) {
        if (active[c])
          active[c] = false;  // breakdown guard, same as the CPU path
        host_alpha[c] = 0.0;
        continue;
      }
      host_alpha[c] = rz_old[c] / host_scratch[c];
    }
    LK_CUDA_CHECK(cudaMemcpy(d_alpha, host_alpha.data(), col_bytes, cudaMemcpyHostToDevice));
    lk_cuda_batched_axpy_launch(d_alpha, d_p, d_x, n, ncols);  // x += alpha*p
    LK_CUDA_CHECK(cudaGetLastError());

    if ((it + 1) % restart_every == 0) {
      // Full restart: recompute the exact residual r = b - A*x from
      // scratch for every still-active column, same rationale as
      // LinearAlgebra::conjugateGradient's restart_every (corrects
      // round-off drift in the recursively updated residual).
      lk_cuda_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_x, ncols, d_Ap,
                                  d_rmul_scratch);  // Ap = A*x
      LK_CUDA_CHECK(cudaGetLastError());
      LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b
      for (int c = 0; c < ncols; ++c)
        host_alpha[c] = -1.0;  // reuse d_alpha as an all-(-1) column vector
      LK_CUDA_CHECK(cudaMemcpy(d_alpha, host_alpha.data(), col_bytes, cudaMemcpyHostToDevice));
      lk_cuda_batched_axpy_launch(d_alpha, d_Ap, d_r, n, ncols);  // r += -1*(A*x) = b - A*x
      LK_CUDA_CHECK(cudaGetLastError());

      lk_cuda_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r == rnorm^2
      LK_CUDA_CHECK(cudaGetLastError());
      LK_CUDA_CHECK(cudaMemcpy(host_scratch.data(), d_scratch, col_bytes, cudaMemcpyDeviceToHost));
      for (int c = 0; c < ncols; ++c) {
        if (!active[c])
          continue;
        rz_old[c] = host_scratch[c];
        if (std::sqrt(host_scratch[c]) / bnorm[c] < tol)
          active[c] = false;
      }
      LK_CUDA_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // restart: p = z = r
      any_active = std::any_of(active.begin(), active.end(), [](bool a) { return a; });
      continue;
    }

    for (int c = 0; c < ncols; ++c)
      host_alpha[c] = active[c] ? -host_alpha[c] : 0.0;
    LK_CUDA_CHECK(cudaMemcpy(d_alpha, host_alpha.data(), col_bytes, cudaMemcpyHostToDevice));
    lk_cuda_batched_axpy_launch(d_alpha, d_Ap, d_r, n, ncols);  // r -= alpha*Ap
    LK_CUDA_CHECK(cudaGetLastError());

    lk_cuda_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r == rz_new == rnorm^2
    LK_CUDA_CHECK(cudaGetLastError());
    LK_CUDA_CHECK(cudaMemcpy(host_scratch.data(), d_scratch, col_bytes, cudaMemcpyDeviceToHost));

    for (int c = 0; c < ncols; ++c) {
      if (!active[c]) {
        host_beta[c] = 0.0;
        continue;
      }
      if (std::sqrt(host_scratch[c]) / bnorm[c] < tol) {
        active[c] = false;
        host_beta[c] = 0.0;
        continue;
      }
      host_beta[c] = host_scratch[c] / rz_old[c];
      rz_old[c] = host_scratch[c];
    }
    LK_CUDA_CHECK(cudaMemcpy(d_beta, host_beta.data(), col_bytes, cudaMemcpyHostToDevice));
    lk_cuda_batched_update_p_launch(d_r, d_beta, d_p, n, ncols);  // p = r + beta*p
    LK_CUDA_CHECK(cudaGetLastError());

    any_active = std::any_of(active.begin(), active.end(), [](bool a) { return a; });
  }

  arma::mat X(n, ncols, arma::fill::none);
  LK_CUDA_CHECK(cudaMemcpy(X.memptr(), d_x, mat_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_Xt);
  cudaFree(d_theta);
  cudaFree(d_b);
  cudaFree(d_x);
  cudaFree(d_r);
  cudaFree(d_p);
  cudaFree(d_Ap);
  cudaFree(d_scratch);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  if (d_rmul_scratch)
    cudaFree(d_rmul_scratch);

  return X;
}

}  // namespace LinearAlgebraCuda

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE
