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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define LK_CUDA_CHECK(expr)                                                    \
  do {                                                                         \
    cudaError_t lk_cuda_status__ = (expr);                                     \
    if (lk_cuda_status__ != cudaSuccess) {                                     \
      std::ostringstream lk_cuda_oss__;                                        \
      lk_cuda_oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                    << cudaGetErrorString(lk_cuda_status__);                   \
      throw std::runtime_error(lk_cuda_oss__.str());                           \
    }                                                                          \
  } while (0)

#define LK_CUBLAS_CHECK(expr)                                                     \
  do {                                                                            \
    cublasStatus_t lk_cublas_status__ = (expr);                                   \
    if (lk_cublas_status__ != CUBLAS_STATUS_SUCCESS) {                            \
      std::ostringstream lk_cublas_oss__;                                         \
      lk_cublas_oss__ << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << static_cast<int>(lk_cublas_status__);                    \
      throw std::runtime_error(lk_cublas_oss__.str());                           \
    }                                                                            \
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

// Lazily-created, process-wide cuBLAS handle (bound to the default stream,
// like every other launch in this file) -- creation is a real cost (device
// context setup), so it's paid once, not once per conjugateGradient/
// rmulBatched/dRmulBatched call. Mirrors the existing smCount()-style
// lazy-static pattern in CudaLinearAlgebraKernel.cu; this project's CUDA
// path already assumes calls aren't issued concurrently from multiple host
// threads (no stream/queue separation anywhere else either), so no extra
// locking is added here.
cublasHandle_t cublasHandle() {
  static cublasHandle_t handle = [] {
    cublasHandle_t h;
    const cublasStatus_t st = cublasCreate(&h);
    if (st != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("LinearAlgebraCuda: cublasCreate failed");
    return h;
  }();
  return handle;
}

// Dense-R fast path: for a separable kernel (all four CUDA-supported
// covTypes are), materialize R (and, for dRmulBatched, the dimX dR/dtheta_k
// blocks) ONCE per host-side call -- which already batches every column of
// that call's right-hand side -- instead of recomputing every covariance
// entry's transcendentals on every CG iteration / Lanczos step
// (rmul_batched_kernel / drmul_batched_kernel do the latter). Subsequent
// matvecs become a single cublasDgemm per iteration: cheap FLOPs against an
// already-evaluated matrix, instead of re-evaluating exp()/log1p() for
// every (i,j) pair every single time. Gated by a device-memory budget
// (LK_ITERATIVE_CUDA_DENSE_MAX_MB, default 4096 MiB; 0 forces the
// matrix-free kernels above, e.g. for n past the budget). Independent of
// the CPU path's LK_ITERATIVE_DENSE_MAX_MB (host RAM and device VRAM are
// different budgets on the same machine).
bool denseFitsBudget(double need_mb) {
  std::size_t budget_mb = 4096;
  if (const char* e = std::getenv("LK_ITERATIVE_CUDA_DENSE_MAX_MB")) {
    try {
      budget_mb = static_cast<std::size_t>(std::stoull(e));
    } catch (...) { /* keep default */
    }
  }
  return budget_mb > 0 && need_mb <= static_cast<double>(budget_mb);
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
arma::mat conjugateGradient(const arma::mat& Xt,
                            const arma::vec& theta,
                            const std::string& covType,
                            const arma::mat& B,
                            arma::uword max_iter,
                            double tol,
                            const arma::mat& precU,
                            const arma::vec& precDinv,
                            const arma::mat& precMcholLower) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraCuda::conjugateGradient: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);
  const bool preconditioned = (precU.n_elem > 0);
  const int pk = preconditioned ? static_cast<int>(precU.n_cols) : 0;

  // Upload X and theta once -- reused for every CG iteration, never
  // re-transferred mid-solve.
  double *d_Xt, *d_theta;
  LK_CUDA_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  LK_CUDA_CHECK(
      cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));

  // Dense fast path: materialize R ONCE for this whole solve (every
  // iteration of THIS loop shares it), then every matvec is a cublasDgemm
  // instead of a fresh matrix-free covariance sweep. See denseFitsBudget's
  // comment. Named d_Rmat (not d_R) to avoid any confusion with the CG
  // residual buffer d_r declared just below.
  const double r_need_mb = static_cast<double>(n) * n * 8.0 / (1024.0 * 1024.0);
  const bool dense = denseFitsBudget(r_need_mb);
  double* d_Rmat = nullptr;
  if (dense) {
    LK_CUDA_CHECK(cudaMalloc(&d_Rmat, sizeof(double) * static_cast<std::size_t>(n) * n));
    lk_cuda_build_cov_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_Rmat, nullptr);
    LK_CUDA_CHECK(cudaGetLastError());
  }

  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  LK_CUDA_CHECK(cudaMalloc(&d_b, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_x, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_r, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_p, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_Ap, mat_bytes));

  // Per-column CG scalars, kept ON DEVICE so the loop never round-trips
  // pAp / r.r / alpha / beta through the host (that was ~4 blocking
  // cudaMemcpy per iteration -- the dominant cost of an ill-conditioned
  // solve that runs thousands of iterations). The host only pulls back a single
  // "any column still active?" int, every `sync_every` iterations.
  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(ncols);
  const std::size_t col_bytes_i = sizeof(int) * static_cast<std::size_t>(ncols);
  double *d_scratch, *d_scratch2, *d_alpha, *d_neg_alpha, *d_beta, *d_rz_old, *d_bnorm, *d_neg_ones;
  int *d_active, *d_flag;
  LK_CUDA_CHECK(cudaMalloc(&d_scratch, col_bytes));   // pAp, then r.r
  LK_CUDA_CHECK(cudaMalloc(&d_scratch2, col_bytes));  // preconditioned: r.z alongside r.r
  LK_CUDA_CHECK(cudaMalloc(&d_alpha, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_neg_alpha, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_beta, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_rz_old, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_bnorm, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_neg_ones, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_active, col_bytes_i));
  LK_CUDA_CHECK(cudaMalloc(&d_flag, sizeof(int)));

  // Preconditioner device state: U (n x k), Dinv (n), Mchol (k x k lower),
  // z = Pinv(r) (n x ncols) and its two scratch buffers. Uploaded once.
  double *d_precU = nullptr, *d_precDinv = nullptr, *d_precMchol = nullptr, *d_z = nullptr, *d_prec_nc = nullptr,
         *d_prec_kc = nullptr;
  if (preconditioned) {
    LK_CUDA_CHECK(cudaMalloc(&d_precU, sizeof(double) * static_cast<std::size_t>(n) * pk));
    LK_CUDA_CHECK(cudaMalloc(&d_precDinv, sizeof(double) * static_cast<std::size_t>(n)));
    LK_CUDA_CHECK(cudaMalloc(&d_precMchol, sizeof(double) * static_cast<std::size_t>(pk) * pk));
    LK_CUDA_CHECK(cudaMalloc(&d_z, mat_bytes));
    LK_CUDA_CHECK(cudaMalloc(&d_prec_nc, mat_bytes));
    LK_CUDA_CHECK(cudaMalloc(&d_prec_kc, sizeof(double) * static_cast<std::size_t>(pk) * ncols));
    LK_CUDA_CHECK(cudaMemcpy(d_precU, precU.memptr(), sizeof(double) * static_cast<std::size_t>(n) * pk,
                             cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(
        cudaMemcpy(d_precDinv, precDinv.memptr(), sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(cudaMemcpy(d_precMchol, precMcholLower.memptr(), sizeof(double) * static_cast<std::size_t>(pk) * pk,
                             cudaMemcpyHostToDevice));
  }
  auto precondApply = [&](const double* d_in, double* d_out) {
    lk_cuda_precond_apply_launch(d_precU, n, pk, d_precDinv, d_precMchol, d_in, ncols, d_out, d_prec_nc, d_prec_kc);
    LK_CUDA_CHECK(cudaGetLastError());
  };

  // n and ncols don't change across a CG solve, so the matrix-free matvec's
  // own scratch requirement (see lk_cuda_rmul_batched_scratch_elems) is
  // fixed too -- allocate it once here rather than inside the per-iteration
  // matvec call. Not needed at all on the dense path (no tiled reduction).
  const int rmul_scratch_elems = dense ? 0 : lk_cuda_rmul_batched_scratch_elems(n, ncols);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_CUDA_CHECK(cudaMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));

  // Ap = R * V, either a single cublasDgemm against the materialized d_Rmat
  // or the matrix-free rmul_batched_kernel -- the only difference the rest
  // of this CG loop needs to know about.
  const double gemm_one = 1.0, gemm_zero = 0.0;
  auto matvec = [&](const double* d_in, double* d_out) {
    if (dense) {
      LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols, n, &gemm_one, d_Rmat, n, d_in, n,
                                  &gemm_zero, d_out, n));
    } else {
      lk_cuda_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_in, ncols, d_out, d_rmul_scratch);
      LK_CUDA_CHECK(cudaGetLastError());
    }
  };

  LK_CUDA_CHECK(cudaMemcpy(d_b, B.memptr(), mat_bytes, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemset(d_x, 0, mat_bytes));
  LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b - A*0

  std::vector<double> bnorm(ncols), rz0(ncols), neg_ones(ncols, -1.0);
  std::vector<int> active_h(ncols);
  bool any_active_h = false;
  for (int c = 0; c < ncols; ++c) {
    bnorm[c] = arma::norm(B.col(c));
    active_h[c] = (bnorm[c] != 0.0) ? 1 : 0;  // x=0 already solves A*x=0 for a zero column
    rz0[c] = bnorm[c] * bnorm[c];             // r=b initially, so r.r = |b|^2 (unpreconditioned rz_old)
    any_active_h = any_active_h || (active_h[c] != 0);
  }
  LK_CUDA_CHECK(cudaMemcpy(d_bnorm, bnorm.data(), col_bytes, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_neg_ones, neg_ones.data(), col_bytes, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_active, active_h.data(), col_bytes_i, cudaMemcpyHostToDevice));

  if (preconditioned) {
    precondApply(d_r, d_z);                                              // z = Pinv(r)
    LK_CUDA_CHECK(cudaMemcpy(d_p, d_z, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z
    lk_cuda_batched_dot_launch(d_r, d_z, n, ncols, d_rz_old);           // rz_old = <r, z>
    LK_CUDA_CHECK(cudaGetLastError());
  } else {
    LK_CUDA_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z = r
    LK_CUDA_CHECK(cudaMemcpy(d_rz_old, rz0.data(), col_bytes, cudaMemcpyHostToDevice));
  }

  constexpr arma::uword restart_every = 50;  // exact-residual recompute, corrects round-off drift
  constexpr arma::uword sync_every = 10;     // host "any active?" poll cadence
  int host_flag = any_active_h ? 1 : 0;

  for (arma::uword it = 0; host_flag && it < max_iter; ++it) {
    matvec(d_p, d_Ap);  // Ap = R*p
    lk_cuda_batched_dot_launch(d_p, d_Ap, n, ncols, d_scratch);  // pAp
    LK_CUDA_CHECK(cudaGetLastError());
    lk_cuda_cg_alpha_launch(d_rz_old, d_scratch, ncols, d_active, d_alpha, d_neg_alpha);
    LK_CUDA_CHECK(cudaGetLastError());
    lk_cuda_batched_axpy_launch(d_alpha, d_p, d_x, n, ncols);  // x += alpha*p
    LK_CUDA_CHECK(cudaGetLastError());

    if ((it + 1) % restart_every == 0) {
      // Full restart: recompute r = b - A*x exactly for every still-active
      // column (same rationale as LinearAlgebra::conjugateGradient's).
      matvec(d_x, d_Ap);  // Ap = R*x
      LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b
      lk_cuda_batched_axpy_launch(d_neg_ones, d_Ap, d_r, n, ncols);             // r = b - A*x
      LK_CUDA_CHECK(cudaGetLastError());
      lk_cuda_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r (true residual)
      LK_CUDA_CHECK(cudaGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_cuda_batched_dot_launch(d_r, d_z, n, ncols, d_scratch2);  // r.z
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_cg_restart_precond_launch(d_scratch, d_scratch2, d_bnorm, tol, ncols, d_active, d_rz_old);
        LK_CUDA_CHECK(cudaGetLastError());
        LK_CUDA_CHECK(cudaMemcpy(d_p, d_z, mat_bytes, cudaMemcpyDeviceToDevice));  // restart: p = z
      } else {
        lk_cuda_cg_restart_launch(d_scratch, d_bnorm, tol, ncols, d_active, d_rz_old);
        LK_CUDA_CHECK(cudaGetLastError());
        LK_CUDA_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // restart: p = r
      }
    } else {
      lk_cuda_batched_axpy_launch(d_neg_alpha, d_Ap, d_r, n, ncols);  // r -= alpha*Ap
      LK_CUDA_CHECK(cudaGetLastError());
      lk_cuda_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r (true residual)
      LK_CUDA_CHECK(cudaGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_cuda_batched_dot_launch(d_r, d_z, n, ncols, d_scratch2);  // r.z == rz_new
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_cg_beta_precond_launch(d_scratch, d_scratch2, d_bnorm, tol, ncols, d_active, d_rz_old, d_beta);
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_batched_update_p_launch(d_z, d_beta, d_p, n, ncols);  // p = z + beta*p
        LK_CUDA_CHECK(cudaGetLastError());
      } else {
        lk_cuda_cg_beta_launch(d_scratch, d_bnorm, tol, ncols, d_active, d_rz_old, d_beta);
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_batched_update_p_launch(d_r, d_beta, d_p, n, ncols);  // p = r + beta*p
        LK_CUDA_CHECK(cudaGetLastError());
      }
    }

    if ((it + 1) % sync_every == 0 || (it + 1) % restart_every == 0) {
      LK_CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
      lk_cuda_cg_any_active_launch(d_active, ncols, d_flag);
      LK_CUDA_CHECK(cudaGetLastError());
      LK_CUDA_CHECK(cudaMemcpy(&host_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    }
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
  cudaFree(d_scratch2);
  cudaFree(d_alpha);
  cudaFree(d_neg_alpha);
  cudaFree(d_beta);
  cudaFree(d_rz_old);
  cudaFree(d_bnorm);
  cudaFree(d_neg_ones);
  cudaFree(d_active);
  cudaFree(d_flag);
  if (d_rmul_scratch)
    cudaFree(d_rmul_scratch);
  if (d_Rmat)
    cudaFree(d_Rmat);
  if (preconditioned) {
    cudaFree(d_precU);
    cudaFree(d_precDinv);
    cudaFree(d_precMchol);
    cudaFree(d_z);
    cudaFree(d_prec_nc);
    cudaFree(d_prec_kc);
  }

  return X;
}

// R(Xt,theta) * V in one batched device launch (see the header). Same
// upload-once / single-download structure as conjugateGradient, minus the
// CG loop: this is exactly one matvec.
arma::mat rmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraCuda::rmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraCuda::rmulBatched: V has " + std::to_string(V.n_rows) + " rows, expected "
                                + std::to_string(n));

  double *d_Xt, *d_theta, *d_V, *d_Av;
  LK_CUDA_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  LK_CUDA_CHECK(cudaMalloc(&d_V, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_Av, mat_bytes));

  LK_CUDA_CHECK(
      cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_V, V.memptr(), mat_bytes, cudaMemcpyHostToDevice));

  // Dense fast path: materialize R once for this single matvec call --
  // still a win over the matrix-free kernel whenever ncols > 1 (SLQ/
  // Hutchinson callers always pass ncols = nprobe), since it turns ncols
  // covariance sweeps' worth of transcendentals into one build + a cheap
  // GEMM. See denseFitsBudget's comment.
  const double r_need_mb = static_cast<double>(n) * n * 8.0 / (1024.0 * 1024.0);
  const bool dense = denseFitsBudget(r_need_mb);
  double* d_Rmat = nullptr;
  double* d_scratch = nullptr;
  if (dense) {
    LK_CUDA_CHECK(cudaMalloc(&d_Rmat, sizeof(double) * static_cast<std::size_t>(n) * n));
    lk_cuda_build_cov_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_Rmat, nullptr);
    LK_CUDA_CHECK(cudaGetLastError());
    const double one = 1.0, zero = 0.0;
    LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols, n, &one, d_Rmat, n, d_V, n, &zero,
                                d_Av, n));
  } else {
    const int scratch_elems = lk_cuda_rmul_batched_scratch_elems(n, ncols);
    if (scratch_elems > 0)
      LK_CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(double) * static_cast<std::size_t>(scratch_elems)));
    lk_cuda_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_V, ncols, d_Av, d_scratch);
    LK_CUDA_CHECK(cudaGetLastError());
  }
  LK_CUDA_CHECK(cudaDeviceSynchronize());

  arma::mat Av(n, ncols, arma::fill::none);
  LK_CUDA_CHECK(cudaMemcpy(Av.memptr(), d_Av, mat_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_Xt);
  cudaFree(d_theta);
  cudaFree(d_V);
  cudaFree(d_Av);
  if (d_scratch)
    cudaFree(d_scratch);
  if (d_Rmat)
    cudaFree(d_Rmat);
  return Av;
}

arma::mat dRmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraCuda::dRmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (dimX > kMaxDimX)
    throw std::invalid_argument("LinearAlgebraCuda::dRmulBatched: dimX " + std::to_string(dimX) + " > "
                                + std::to_string(kMaxDimX));
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraCuda::dRmulBatched: V has " + std::to_string(V.n_rows)
                                + " rows, expected " + std::to_string(n));

  double *d_Xt, *d_theta, *d_V, *d_Out;
  LK_CUDA_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_CUDA_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  LK_CUDA_CHECK(cudaMalloc(&d_V, sizeof(double) * static_cast<std::size_t>(n) * ncols));
  const std::size_t out_bytes = sizeof(double) * static_cast<std::size_t>(n) * dimX * ncols;
  LK_CUDA_CHECK(cudaMalloc(&d_Out, out_bytes));

  LK_CUDA_CHECK(
      cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(
      d_V, V.memptr(), sizeof(double) * static_cast<std::size_t>(n) * ncols, cudaMemcpyHostToDevice));

  // Dense fast path: materialize the dimX dR/dtheta_k blocks once, then one
  // cublasDgemm per dimension writes straight into its interleaved slot of
  // d_Out (column c's k-th block lives at d_Out[(c*dimX+k)*n .. +n), i.e.
  // consecutive c's for a FIXED k are dimX*n apart -- exactly a column-major
  // n x ncols matrix with leading dimension dimX*n starting at d_Out + k*n,
  // so no separate scatter/copy kernel is needed).
  const double dr_need_mb = static_cast<double>(n) * n * dimX * 8.0 / (1024.0 * 1024.0);
  const bool dense = denseFitsBudget(dr_need_mb);
  if (dense) {
    double* d_dR = nullptr;
    LK_CUDA_CHECK(cudaMalloc(&d_dR, sizeof(double) * static_cast<std::size_t>(n) * n * dimX));
    lk_cuda_build_cov_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), nullptr, d_dR);
    LK_CUDA_CHECK(cudaGetLastError());
    const double one = 1.0, zero = 0.0;
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    for (int k = 0; k < dimX; ++k) {
      LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols, n, &one, d_dR + n2 * k, n, d_V,
                                  n, &zero, d_Out + static_cast<std::size_t>(k) * n, dimX * n));
    }
    cudaFree(d_dR);
  } else {
    lk_cuda_drmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_V, ncols, d_Out);
    LK_CUDA_CHECK(cudaGetLastError());
  }
  LK_CUDA_CHECK(cudaDeviceSynchronize());

  arma::mat Out(static_cast<arma::uword>(n), static_cast<arma::uword>(dimX) * ncols, arma::fill::none);
  LK_CUDA_CHECK(cudaMemcpy(Out.memptr(), d_Out, out_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_Xt);
  cudaFree(d_theta);
  cudaFree(d_V);
  cudaFree(d_Out);
  return Out;
}

}  // namespace LinearAlgebraCuda

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE
