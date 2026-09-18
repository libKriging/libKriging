// UNVERIFIED SYCL / Intel oneAPI port of src/lib/cuda/SyclLinearAlgebra.cpp
// -- see src/lib/sycl/SyclLinearAlgebra.hpp for the full caveat. Compiled by
// the SAME compiler as the rest of libKriging (NOT the SYCL compiler): it
// touches arma::mat/arma::vec and only ever calls the plain-C lk_sycl_*
// surface in SyclLinearAlgebraKernel.hpp, which hides every SYCL header /
// USM / queue detail. Same ABI-safety rule and same CG algorithm as the
// CUDA backend -- the only differences below are the device memory calls
// (lk_sycl_* instead of cuda*) and the availability probe.
#include "SyclLinearAlgebra.hpp"

#ifdef LIBKRIGING_USE_SYCL_ITERATIVE

#include "SyclLinearAlgebraKernel.hpp"

#include "libKriging/LinearAlgebra.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

// The CUDA host code below is kept verbatim by aliasing its device-memory
// calls onto the lk_sycl_* surface. cuda* names here are local shims, not
// the real CUDA runtime.
enum { lk_h2d = 0, lk_d2h = 1, lk_d2d = 2 };
template <class T>
inline void lk_dev_malloc(T** p, unsigned long bytes) {
  *p = static_cast<T*>(lk_sycl_malloc(bytes));
}
#define LK_SYCL_CHECK(expr) (expr)
#define cudaMalloc(pp, bytes) lk_dev_malloc(pp, bytes)
#define cudaFree(p) lk_sycl_free(p)
#define cudaMemset(p, v, bytes) lk_sycl_memset(p, v, bytes)
#define cudaMemcpy(dst, src, bytes, kind) lk_sycl_memcpy(dst, src, bytes, kind)
#define cudaMemcpyHostToDevice lk_h2d
#define cudaMemcpyDeviceToHost lk_d2h
#define cudaMemcpyDeviceToDevice lk_d2d
#define cudaGetLastError() 0
#define cudaDeviceSynchronize() (lk_sycl_sync(), 0)

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

namespace LinearAlgebraSycl {

bool available() {
  static const bool cached = lk_sycl_available() != 0;
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
// cheap way to shrink an in-flight device launch's column count mid-loop, so
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
                            const arma::mat& precMcholLower,
                            arma::uword* n_unconverged_out,
                            const arma::mat* /*X0*/) {
  // X0 (warm start) not yet implemented on this backend -- see the header's
  // doc comment; always solves from x0=0 regardless of what's passed here.
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraSycl::conjugateGradient: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);
  const bool preconditioned = (precU.n_elem > 0);
  const int pk = preconditioned ? static_cast<int>(precU.n_cols) : 0;

  // Upload X and theta once -- reused for every CG iteration, never
  // re-transferred mid-solve.
  double *d_Xt, *d_theta;
  LK_SYCL_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_SYCL_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  LK_SYCL_CHECK(
      cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));

  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  LK_SYCL_CHECK(cudaMalloc(&d_b, mat_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_x, mat_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_r, mat_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_p, mat_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_Ap, mat_bytes));

  // Per-column CG scalars, kept ON DEVICE so the loop never round-trips
  // pAp / r.r / alpha / beta through the host (that was ~4 blocking
  // cudaMemcpy per iteration -- the dominant cost of an ill-conditioned
  // solve that runs thousands of iterations). The host only pulls back a single
  // "any column still active?" int, every `sync_every` iterations.
  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(ncols);
  const std::size_t col_bytes_i = sizeof(int) * static_cast<std::size_t>(ncols);
  double *d_scratch, *d_scratch2, *d_alpha, *d_neg_alpha, *d_beta, *d_rz_old, *d_bnorm, *d_neg_ones;
  int *d_active, *d_flag;
  LK_SYCL_CHECK(cudaMalloc(&d_scratch, col_bytes));   // pAp, then r.r
  LK_SYCL_CHECK(cudaMalloc(&d_scratch2, col_bytes));  // preconditioned: r.z alongside r.r
  LK_SYCL_CHECK(cudaMalloc(&d_alpha, col_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_neg_alpha, col_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_beta, col_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_rz_old, col_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_bnorm, col_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_neg_ones, col_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_active, col_bytes_i));
  LK_SYCL_CHECK(cudaMalloc(&d_flag, sizeof(int)));

  // Preconditioner device state: U (n x k), Dinv (n), Mchol (k x k lower),
  // z = Pinv(r) (n x ncols) and its two scratch buffers. Uploaded once.
  double *d_precU = nullptr, *d_precDinv = nullptr, *d_precMchol = nullptr, *d_z = nullptr, *d_prec_nc = nullptr,
         *d_prec_kc = nullptr;
  if (preconditioned) {
    LK_SYCL_CHECK(cudaMalloc(&d_precU, sizeof(double) * static_cast<std::size_t>(n) * pk));
    LK_SYCL_CHECK(cudaMalloc(&d_precDinv, sizeof(double) * static_cast<std::size_t>(n)));
    LK_SYCL_CHECK(cudaMalloc(&d_precMchol, sizeof(double) * static_cast<std::size_t>(pk) * pk));
    LK_SYCL_CHECK(cudaMalloc(&d_z, mat_bytes));
    LK_SYCL_CHECK(cudaMalloc(&d_prec_nc, mat_bytes));
    LK_SYCL_CHECK(cudaMalloc(&d_prec_kc, sizeof(double) * static_cast<std::size_t>(pk) * ncols));
    LK_SYCL_CHECK(cudaMemcpy(d_precU, precU.memptr(), sizeof(double) * static_cast<std::size_t>(n) * pk,
                             cudaMemcpyHostToDevice));
    LK_SYCL_CHECK(
        cudaMemcpy(d_precDinv, precDinv.memptr(), sizeof(double) * static_cast<std::size_t>(n), cudaMemcpyHostToDevice));
    LK_SYCL_CHECK(cudaMemcpy(d_precMchol, precMcholLower.memptr(), sizeof(double) * static_cast<std::size_t>(pk) * pk,
                             cudaMemcpyHostToDevice));
  }
  auto precondApply = [&](const double* d_in, double* d_out) {
    lk_sycl_precond_apply_launch(d_precU, n, pk, d_precDinv, d_precMchol, d_in, ncols, d_out, d_prec_nc, d_prec_kc);
    LK_SYCL_CHECK(cudaGetLastError());
  };

  // n and ncols don't change across a CG solve, so the matvec's own scratch
  // requirement (see lk_sycl_rmul_batched_scratch_elems) is fixed too --
  // allocate it once here rather than inside the per-iteration matvec call.
  const int rmul_scratch_elems = lk_sycl_rmul_batched_scratch_elems(n, ncols);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_SYCL_CHECK(cudaMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));

  LK_SYCL_CHECK(cudaMemcpy(d_b, B.memptr(), mat_bytes, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemset(d_x, 0, mat_bytes));
  LK_SYCL_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b - A*0

  std::vector<double> bnorm(ncols), rz0(ncols), neg_ones(ncols, -1.0);
  std::vector<int> active_h(ncols);
  bool any_active_h = false;
  for (int c = 0; c < ncols; ++c) {
    bnorm[c] = arma::norm(B.col(c));
    active_h[c] = (bnorm[c] != 0.0) ? 1 : 0;  // x=0 already solves A*x=0 for a zero column
    rz0[c] = bnorm[c] * bnorm[c];             // r=b initially, so r.r = |b|^2 (unpreconditioned rz_old)
    any_active_h = any_active_h || (active_h[c] != 0);
  }
  LK_SYCL_CHECK(cudaMemcpy(d_bnorm, bnorm.data(), col_bytes, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(d_neg_ones, neg_ones.data(), col_bytes, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(d_active, active_h.data(), col_bytes_i, cudaMemcpyHostToDevice));

  if (preconditioned) {
    precondApply(d_r, d_z);                                              // z = Pinv(r)
    LK_SYCL_CHECK(cudaMemcpy(d_p, d_z, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z
    lk_sycl_batched_dot_launch(d_r, d_z, n, ncols, d_rz_old);           // rz_old = <r, z>
    LK_SYCL_CHECK(cudaGetLastError());
  } else {
    LK_SYCL_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z = r
    LK_SYCL_CHECK(cudaMemcpy(d_rz_old, rz0.data(), col_bytes, cudaMemcpyHostToDevice));
  }

  constexpr arma::uword restart_every = 50;  // exact-residual recompute, corrects round-off drift
  constexpr arma::uword sync_every = 10;     // host "any active?" poll cadence
  int host_flag = any_active_h ? 1 : 0;

  for (arma::uword it = 0; host_flag && it < max_iter; ++it) {
    lk_sycl_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_p, ncols, d_Ap, d_rmul_scratch);
    LK_SYCL_CHECK(cudaGetLastError());
    lk_sycl_batched_dot_launch(d_p, d_Ap, n, ncols, d_scratch);  // pAp
    LK_SYCL_CHECK(cudaGetLastError());
    lk_sycl_cg_alpha_launch(d_rz_old, d_scratch, ncols, d_active, d_alpha, d_neg_alpha);
    LK_SYCL_CHECK(cudaGetLastError());
    lk_sycl_batched_axpy_launch(d_alpha, d_p, d_x, n, ncols);  // x += alpha*p
    LK_SYCL_CHECK(cudaGetLastError());

    if ((it + 1) % restart_every == 0) {
      // Full restart: recompute r = b - A*x exactly for every still-active
      // column (same rationale as LinearAlgebra::conjugateGradient's).
      lk_sycl_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_x, ncols, d_Ap,
                                  d_rmul_scratch);  // Ap = A*x
      LK_SYCL_CHECK(cudaGetLastError());
      LK_SYCL_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b
      lk_sycl_batched_axpy_launch(d_neg_ones, d_Ap, d_r, n, ncols);             // r = b - A*x
      LK_SYCL_CHECK(cudaGetLastError());
      lk_sycl_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r (true residual)
      LK_SYCL_CHECK(cudaGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_sycl_batched_dot_launch(d_r, d_z, n, ncols, d_scratch2);  // r.z
        LK_SYCL_CHECK(cudaGetLastError());
        lk_sycl_cg_restart_precond_launch(d_scratch, d_scratch2, d_bnorm, tol, ncols, d_active, d_rz_old);
        LK_SYCL_CHECK(cudaGetLastError());
        LK_SYCL_CHECK(cudaMemcpy(d_p, d_z, mat_bytes, cudaMemcpyDeviceToDevice));  // restart: p = z
      } else {
        lk_sycl_cg_restart_launch(d_scratch, d_bnorm, tol, ncols, d_active, d_rz_old);
        LK_SYCL_CHECK(cudaGetLastError());
        LK_SYCL_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // restart: p = r
      }
    } else {
      lk_sycl_batched_axpy_launch(d_neg_alpha, d_Ap, d_r, n, ncols);  // r -= alpha*Ap
      LK_SYCL_CHECK(cudaGetLastError());
      lk_sycl_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r (true residual)
      LK_SYCL_CHECK(cudaGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_sycl_batched_dot_launch(d_r, d_z, n, ncols, d_scratch2);  // r.z == rz_new
        LK_SYCL_CHECK(cudaGetLastError());
        lk_sycl_cg_beta_precond_launch(d_scratch, d_scratch2, d_bnorm, tol, ncols, d_active, d_rz_old, d_beta);
        LK_SYCL_CHECK(cudaGetLastError());
        lk_sycl_batched_update_p_launch(d_z, d_beta, d_p, n, ncols);  // p = z + beta*p
        LK_SYCL_CHECK(cudaGetLastError());
      } else {
        lk_sycl_cg_beta_launch(d_scratch, d_bnorm, tol, ncols, d_active, d_rz_old, d_beta);
        LK_SYCL_CHECK(cudaGetLastError());
        lk_sycl_batched_update_p_launch(d_r, d_beta, d_p, n, ncols);  // p = r + beta*p
        LK_SYCL_CHECK(cudaGetLastError());
      }
    }

    if ((it + 1) % sync_every == 0 || (it + 1) % restart_every == 0) {
      LK_SYCL_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
      lk_sycl_cg_any_active_launch(d_active, ncols, d_flag);
      LK_SYCL_CHECK(cudaGetLastError());
      LK_SYCL_CHECK(cudaMemcpy(&host_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
    }
  }

  arma::mat X(n, ncols, arma::fill::none);
  LK_SYCL_CHECK(cudaMemcpy(X.memptr(), d_x, mat_bytes, cudaMemcpyDeviceToHost));

  // See CudaLinearAlgebra.cpp's matching readback: d_active still holds,
  // per column, whether the loop hit max_iter before that column reached
  // tol (a converged/deactivated column was already zeroed).
  std::vector<int> active_final(static_cast<std::size_t>(ncols));
  LK_SYCL_CHECK(cudaMemcpy(active_final.data(), d_active, col_bytes_i, cudaMemcpyDeviceToHost));
  arma::uword n_unconverged = 0;
  for (int c = 0; c < ncols; ++c)
    if (active_final[static_cast<std::size_t>(c)])
      ++n_unconverged;
  if (n_unconverged_out != nullptr)
    *n_unconverged_out = n_unconverged;
  LinearAlgebra::cgNonConvergenceWarning(n_unconverged, static_cast<arma::uword>(ncols),
                                        static_cast<arma::uword>(n), max_iter);

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
    throw std::invalid_argument("LinearAlgebraSycl::rmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraSycl::rmulBatched: V has " + std::to_string(V.n_rows) + " rows, expected "
                                + std::to_string(n));

  double *d_Xt, *d_theta, *d_V, *d_Av;
  LK_SYCL_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_SYCL_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  LK_SYCL_CHECK(cudaMalloc(&d_V, mat_bytes));
  LK_SYCL_CHECK(cudaMalloc(&d_Av, mat_bytes));

  const int scratch_elems = lk_sycl_rmul_batched_scratch_elems(n, ncols);
  double* d_scratch = nullptr;
  if (scratch_elems > 0)
    LK_SYCL_CHECK(cudaMalloc(&d_scratch, sizeof(double) * static_cast<std::size_t>(scratch_elems)));

  LK_SYCL_CHECK(
      cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(d_V, V.memptr(), mat_bytes, cudaMemcpyHostToDevice));

  lk_sycl_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_V, ncols, d_Av, d_scratch);
  LK_SYCL_CHECK(cudaGetLastError());
  LK_SYCL_CHECK(cudaDeviceSynchronize());

  arma::mat Av(n, ncols, arma::fill::none);
  LK_SYCL_CHECK(cudaMemcpy(Av.memptr(), d_Av, mat_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_Xt);
  cudaFree(d_theta);
  cudaFree(d_V);
  cudaFree(d_Av);
  if (d_scratch)
    cudaFree(d_scratch);
  return Av;
}

arma::mat dRmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraSycl::dRmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (dimX > kMaxDimX)
    throw std::invalid_argument("LinearAlgebraSycl::dRmulBatched: dimX " + std::to_string(dimX) + " > "
                                + std::to_string(kMaxDimX));
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraSycl::dRmulBatched: V has " + std::to_string(V.n_rows)
                                + " rows, expected " + std::to_string(n));

  double *d_Xt, *d_theta, *d_V, *d_Out;
  LK_SYCL_CHECK(cudaMalloc(&d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_SYCL_CHECK(cudaMalloc(&d_theta, sizeof(double) * dimX));
  LK_SYCL_CHECK(cudaMalloc(&d_V, sizeof(double) * static_cast<std::size_t>(n) * ncols));
  const std::size_t out_bytes = sizeof(double) * static_cast<std::size_t>(n) * dimX * ncols;
  LK_SYCL_CHECK(cudaMalloc(&d_Out, out_bytes));

  LK_SYCL_CHECK(
      cudaMemcpy(d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));
  LK_SYCL_CHECK(cudaMemcpy(
      d_V, V.memptr(), sizeof(double) * static_cast<std::size_t>(n) * ncols, cudaMemcpyHostToDevice));

  lk_sycl_drmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_V, ncols, d_Out);
  LK_SYCL_CHECK(cudaGetLastError());
  LK_SYCL_CHECK(cudaDeviceSynchronize());

  arma::mat Out(static_cast<arma::uword>(n), static_cast<arma::uword>(dimX) * ncols, arma::fill::none);
  LK_SYCL_CHECK(cudaMemcpy(Out.memptr(), d_Out, out_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_Xt);
  cudaFree(d_theta);
  cudaFree(d_V);
  cudaFree(d_Out);
  return Out;
}

}  // namespace LinearAlgebraSycl

#endif  // LIBKRIGING_USE_SYCL_ITERATIVE
