// HIP/ROCm port of src/lib/cuda/CudaLinearAlgebra.cpp -- see
// src/lib/hip/HipLinearAlgebra.hpp for the full caveat (verification status,
// build recipe, and the tiled-matvec bug found and fixed 2026-09-13).
// Mechanical cuda*->hip* / .cuh->.hpp translation of the CUDA backend;
// identical algorithm, batching and preconditioner math.
// Host-compiler-compiled (NOT hipcc) on purpose: this file touches
// arma::mat/arma::vec (via .memptr()/.n_elem/etc), and must be compiled by
// the SAME compiler as the rest of libKriging to guarantee it agrees with
// Kriging.cpp/KrigingImpl.cpp on those types' memory layout -- see the
// comment in HipLinearAlgebraKernel.hpp for why crossing that boundary
// with hipcc silently breaks (arma::vec::memptr() came back null on the
// hipcc side despite n_elem reading correctly). The only hipcc-compiled code
// this file talks to is the plain-C, pointer-only lk_hip_*_launch family.
#include "HipLinearAlgebra.hpp"

#ifdef LIBKRIGING_USE_HIP_ITERATIVE

#include "HipLinearAlgebraKernel.hpp"

#include "libKriging/LinearAlgebra.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

#define LK_HIP_CHECK(expr)                                                    \
  do {                                                                         \
    hipError_t lk_hip_status__ = (expr);                                     \
    if (lk_hip_status__ != hipSuccess) {                                     \
      std::ostringstream lk_hip_oss__;                                        \
      lk_hip_oss__ << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                    << hipGetErrorString(lk_hip_status__);                   \
      throw std::runtime_error(lk_hip_oss__.str());                           \
    }                                                                          \
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

// Dense-R fast path: for a separable kernel (all four HIP-supported
// covTypes are), materialize R (and, for dRmulBatched, the dimX dR/dtheta_k
// blocks) ONCE per host-side call -- which already batches every column of
// that call's right-hand side, or once per theta via DenseCovCache below --
// instead of recomputing every covariance entry's transcendentals on every
// CG iteration / Lanczos step (rmul_batched_kernel / drmul_batched_kernel
// do the latter). Subsequent matvecs become a single lk_hip_dense_matvec_
// launch call per iteration: cheap FMA against an already-evaluated matrix
// (no BLAS -- see the file header), instead of re-evaluating exp()/log1p()
// for every (i,j) pair every single time.
// Gated by a device-memory budget (LK_ITERATIVE_HIP_DENSE_MAX_MB, default
// 4096 MiB; 0 forces the matrix-free kernels above, e.g. for n past the
// budget). Independent of the CPU path's LK_ITERATIVE_DENSE_MAX_MB (host
// RAM and device VRAM are different budgets on the same machine) and of
// CUDA's own LK_ITERATIVE_CUDA_DENSE_MAX_MB. Mirrors
// CudaLinearAlgebra.cpp's denseFitsBudget() exactly.
bool denseFitsBudget(double need_mb) {
  std::size_t budget_mb = 4096;
  if (const char* e = std::getenv("LK_ITERATIVE_HIP_DENSE_MAX_MB")) {
    try {
      budget_mb = static_cast<std::size_t>(std::stoull(e));
    } catch (...) { /* keep default */
    }
  }
  return budget_mb > 0 && need_mb <= static_cast<double>(budget_mb);
}

// Device-resident cache of the materialized dense R for the LAST
// (Xt, theta, covType) seen. Mirrors CudaLinearAlgebra.cpp's DenseCovCache
// exactly EXCEPT it has no fp32 mixed-precision copy (that CUDA-only
// optimization -- plan item #8 -- is out of scope here; RDNA2 consumer
// parts like this project's reference RX 6600 don't have the matrix-core
// throughput advantage that makes it pay off).
//
// Every entry point below used to be stateless: conjugateGradient,
// rmulBatched, dRmulBatched and stochasticLogDetBatched each
// hipMalloc'd d_Xt/d_theta(/d_Rmat), rebuilt R from scratch, then hipFree'd
// everything before returning. That is fine for a single solve, but it is
// NOT what the iterative objective does: one _logLikelihoodIterative
// evaluation runs three Krylov passes at a FIXED theta (CG on [F|y], the
// SLQ Lanczos recurrence, then CG on the Hutchinson probes) -- the SLQ
// recurrence alone calls its matvec once per Lanczos step (40 steps by
// default). Caching on (n, dimX, kind, Xt, theta), compared by VALUE (not
// pointer -- callers pass temporaries and Armadillo reuses freed memory, so
// a pointer/shape comparison would alias two different designs onto the
// same cached R), collapses all of that to one build per theta.
struct DenseCovCache {
  int n = 0;
  int dimX = 0;
  int kind = -1;
  std::vector<double> xt;
  std::vector<double> theta;
  double* d_Xt = nullptr;
  double* d_theta = nullptr;
  double* d_R = nullptr;
};

DenseCovCache& covCache() {
  static DenseCovCache cache;
  return cache;
}

bool covCacheMatches(const DenseCovCache& c, const arma::mat& Xt, const arma::vec& theta, CovKind kind) {
  if (c.d_Xt == nullptr || c.n != static_cast<int>(Xt.n_cols) || c.dimX != static_cast<int>(Xt.n_rows)
      || c.kind != static_cast<int>(kind))
    return false;
  if (c.xt.size() != Xt.n_elem || c.theta.size() != theta.n_elem)
    return false;
  return std::equal(c.xt.begin(), c.xt.end(), Xt.memptr()) && std::equal(c.theta.begin(), c.theta.end(), theta.memptr());
}

// Binds the cache to (Xt, theta, kind), uploading Xt/theta if the key
// changed, and returns it. d_R is left null on a key change; covCacheR()
// below is what actually materializes it (callers that only need the
// matrix-free path must not pay for a build they won't use).
DenseCovCache& covCacheBind(const arma::mat& Xt, const arma::vec& theta, CovKind kind) {
  DenseCovCache& c = covCache();
  if (covCacheMatches(c, Xt, theta, kind))
    return c;

  if (c.d_Xt)
    hipFree(c.d_Xt);
  if (c.d_theta)
    hipFree(c.d_theta);
  if (c.d_R)
    hipFree(c.d_R);
  c.d_Xt = c.d_theta = c.d_R = nullptr;

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  LK_HIP_CHECK(hipMalloc(&c.d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_HIP_CHECK(hipMalloc(&c.d_theta, sizeof(double) * dimX));
  LK_HIP_CHECK(
      hipMemcpy(c.d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemcpy(c.d_theta, theta.memptr(), sizeof(double) * dimX, hipMemcpyHostToDevice));

  c.n = n;
  c.dimX = dimX;
  c.kind = static_cast<int>(kind);
  c.xt.assign(Xt.memptr(), Xt.memptr() + Xt.n_elem);
  c.theta.assign(theta.memptr(), theta.memptr() + theta.n_elem);
  return c;
}

// Cached dense R for the bound key, built on first use. Returns nullptr
// when the dense path is over budget, so callers keep their matrix-free
// fallback.
double* covCacheR(DenseCovCache& c) {
  if (c.d_R)
    return c.d_R;
  const double need_mb = static_cast<double>(c.n) * c.n * 8.0 / (1024.0 * 1024.0);
  if (!denseFitsBudget(need_mb))
    return nullptr;
  LK_HIP_CHECK(hipMalloc(&c.d_R, sizeof(double) * static_cast<std::size_t>(c.n) * c.n));
  lk_hip_build_cov_launch(c.d_Xt, c.n, c.dimX, c.d_theta, c.kind, c.d_R, nullptr);
  LK_HIP_CHECK(hipGetLastError());
  return c.d_R;
}

}  // namespace

namespace LinearAlgebraHip {

bool available() {
  static const bool cached = [] {
    int count = 0;
    hipError_t status = hipGetDeviceCount(&count);
    return status == hipSuccess && count > 0;
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
// HipLinearAlgebraKernel.hip.cpp's rmul_batched_kernel comment), so cutting the
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
                            const arma::vec& tol,
                            const arma::mat& precU,
                            const arma::vec& precDinv,
                            const arma::mat& precMcholLower,
                            arma::uword* n_unconverged_out,
                            const arma::mat* /*X0*/) {
  // X0 (warm start) not yet implemented on this backend -- see the header's
  // doc comment; always solves from x0=0 regardless of what's passed here.
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraHip::conjugateGradient: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);
  const bool preconditioned = (precU.n_elem > 0);
  const int pk = preconditioned ? static_cast<int>(precU.n_cols) : 0;
  if (static_cast<int>(tol.n_elem) != ncols)
    throw std::invalid_argument("LinearAlgebraHip::conjugateGradient: tol has " + std::to_string(tol.n_elem)
                                + " entries, expected " + std::to_string(ncols) + " (one per column of B)");

  // X, theta and R come from the process-wide dense cache (see
  // DenseCovCache): a whole objective evaluation runs at a fixed theta, so
  // the first pass pays the upload + build and the SLQ/probe passes that
  // follow reuse them. Nothing here is freed on the way out -- the cache
  // owns d_Xt/d_theta/d_Rmat and drops them when theta next moves.
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  double* d_Xt = cov.d_Xt;
  double* d_theta = cov.d_theta;
  double* d_Rmat = covCacheR(cov);
  const bool dense = (d_Rmat != nullptr);

  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  LK_HIP_CHECK(hipMalloc(&d_b, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_x, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_r, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_p, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_Ap, mat_bytes));

  // Per-column CG scalars, kept ON DEVICE so the loop never round-trips
  // pAp / r.r / alpha / beta through the host (that was ~4 blocking
  // hipMemcpy per iteration -- the dominant cost of an ill-conditioned
  // solve that runs thousands of iterations). The host only pulls back a single
  // "any column still active?" int, every `sync_every` iterations.
  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(ncols);
  const std::size_t col_bytes_i = sizeof(int) * static_cast<std::size_t>(ncols);
  double *d_scratch, *d_scratch2, *d_alpha, *d_neg_alpha, *d_beta, *d_rz_old, *d_bnorm, *d_neg_ones, *d_tol;
  int *d_active, *d_flag;
  LK_HIP_CHECK(hipMalloc(&d_scratch, col_bytes));   // pAp, then r.r
  LK_HIP_CHECK(hipMalloc(&d_scratch2, col_bytes));  // preconditioned: r.z alongside r.r
  LK_HIP_CHECK(hipMalloc(&d_alpha, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_neg_alpha, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_beta, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_rz_old, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_bnorm, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_neg_ones, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_tol, col_bytes));
  LK_HIP_CHECK(hipMemcpy(d_tol, tol.memptr(), col_bytes, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMalloc(&d_active, col_bytes_i));
  LK_HIP_CHECK(hipMalloc(&d_flag, sizeof(int)));

  // Preconditioner device state: U (n x k), Dinv (n), Mchol (k x k lower),
  // z = Pinv(r) (n x ncols) and its two scratch buffers. Uploaded once.
  double *d_precU = nullptr, *d_precDinv = nullptr, *d_precMchol = nullptr, *d_z = nullptr, *d_prec_nc = nullptr,
         *d_prec_kc = nullptr;
  if (preconditioned) {
    LK_HIP_CHECK(hipMalloc(&d_precU, sizeof(double) * static_cast<std::size_t>(n) * pk));
    LK_HIP_CHECK(hipMalloc(&d_precDinv, sizeof(double) * static_cast<std::size_t>(n)));
    LK_HIP_CHECK(hipMalloc(&d_precMchol, sizeof(double) * static_cast<std::size_t>(pk) * pk));
    LK_HIP_CHECK(hipMalloc(&d_z, mat_bytes));
    LK_HIP_CHECK(hipMalloc(&d_prec_nc, mat_bytes));
    LK_HIP_CHECK(hipMalloc(&d_prec_kc, sizeof(double) * static_cast<std::size_t>(pk) * ncols));
    LK_HIP_CHECK(hipMemcpy(d_precU, precU.memptr(), sizeof(double) * static_cast<std::size_t>(n) * pk,
                             hipMemcpyHostToDevice));
    LK_HIP_CHECK(
        hipMemcpy(d_precDinv, precDinv.memptr(), sizeof(double) * static_cast<std::size_t>(n), hipMemcpyHostToDevice));
    LK_HIP_CHECK(hipMemcpy(d_precMchol, precMcholLower.memptr(), sizeof(double) * static_cast<std::size_t>(pk) * pk,
                             hipMemcpyHostToDevice));
  }
  auto precondApply = [&](const double* d_in, double* d_out) {
    lk_hip_precond_apply_launch(d_precU, n, pk, d_precDinv, d_precMchol, d_in, ncols, d_out, d_prec_nc, d_prec_kc);
    LK_HIP_CHECK(hipGetLastError());
  };

  // n and ncols don't change across a CG solve, so the matrix-free matvec's
  // own scratch requirement (see lk_hip_rmul_batched_scratch_elems) is
  // fixed too -- allocate it once here rather than inside the per-iteration
  // matvec call. Not needed at all on the dense path (no tiled reduction).
  const int rmul_scratch_elems = dense ? 0 : lk_hip_rmul_batched_scratch_elems(n, ncols);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_HIP_CHECK(hipMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));

  // Ap = R * V, either lk_hip_dense_matvec_launch against the materialized
  // d_Rmat or the matrix-free rmul_batched_kernel -- the only difference
  // the rest of this CG loop needs to know about.
  auto matvec = [&](const double* d_in, double* d_out) {
    if (dense) {
      lk_hip_dense_matvec_launch(d_Rmat, n, d_in, ncols, d_out, n);
      LK_HIP_CHECK(hipGetLastError());
    } else {
      lk_hip_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_in, ncols, d_out, d_rmul_scratch);
      LK_HIP_CHECK(hipGetLastError());
    }
  };

  LK_HIP_CHECK(hipMemcpy(d_b, B.memptr(), mat_bytes, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemset(d_x, 0, mat_bytes));
  LK_HIP_CHECK(hipMemcpy(d_r, d_b, mat_bytes, hipMemcpyDeviceToDevice));  // r = b - A*0

  std::vector<double> bnorm(ncols), rz0(ncols), neg_ones(ncols, -1.0);
  std::vector<int> active_h(ncols);
  bool any_active_h = false;
  for (int c = 0; c < ncols; ++c) {
    bnorm[c] = arma::norm(B.col(c));
    active_h[c] = (bnorm[c] != 0.0) ? 1 : 0;  // x=0 already solves A*x=0 for a zero column
    rz0[c] = bnorm[c] * bnorm[c];             // r=b initially, so r.r = |b|^2 (unpreconditioned rz_old)
    any_active_h = any_active_h || (active_h[c] != 0);
  }
  LK_HIP_CHECK(hipMemcpy(d_bnorm, bnorm.data(), col_bytes, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemcpy(d_neg_ones, neg_ones.data(), col_bytes, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemcpy(d_active, active_h.data(), col_bytes_i, hipMemcpyHostToDevice));

  if (preconditioned) {
    precondApply(d_r, d_z);                                              // z = Pinv(r)
    LK_HIP_CHECK(hipMemcpy(d_p, d_z, mat_bytes, hipMemcpyDeviceToDevice));  // p = z
    lk_hip_batched_dot_launch(d_r, d_z, n, ncols, d_rz_old);           // rz_old = <r, z>
    LK_HIP_CHECK(hipGetLastError());
  } else {
    LK_HIP_CHECK(hipMemcpy(d_p, d_r, mat_bytes, hipMemcpyDeviceToDevice));  // p = z = r
    LK_HIP_CHECK(hipMemcpy(d_rz_old, rz0.data(), col_bytes, hipMemcpyHostToDevice));
  }

  constexpr arma::uword restart_every = 50;  // exact-residual recompute, corrects round-off drift
  constexpr arma::uword sync_every = 10;     // host "any active?" poll cadence
  int host_flag = any_active_h ? 1 : 0;

  for (arma::uword it = 0; host_flag && it < max_iter; ++it) {
    matvec(d_p, d_Ap);
    lk_hip_batched_dot_launch(d_p, d_Ap, n, ncols, d_scratch);  // pAp
    LK_HIP_CHECK(hipGetLastError());
    lk_hip_cg_alpha_launch(d_rz_old, d_scratch, ncols, d_active, d_alpha, d_neg_alpha);
    LK_HIP_CHECK(hipGetLastError());
    lk_hip_batched_axpy_launch(d_alpha, d_p, d_x, n, ncols);  // x += alpha*p
    LK_HIP_CHECK(hipGetLastError());

    if ((it + 1) % restart_every == 0) {
      // Full restart: recompute r = b - A*x exactly for every still-active
      // column (same rationale as LinearAlgebra::conjugateGradient's).
      matvec(d_x, d_Ap);  // Ap = A*x
      LK_HIP_CHECK(hipMemcpy(d_r, d_b, mat_bytes, hipMemcpyDeviceToDevice));  // r = b
      lk_hip_batched_axpy_launch(d_neg_ones, d_Ap, d_r, n, ncols);             // r = b - A*x
      LK_HIP_CHECK(hipGetLastError());
      lk_hip_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r (true residual)
      LK_HIP_CHECK(hipGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_hip_batched_dot_launch(d_r, d_z, n, ncols, d_scratch2);  // r.z
        LK_HIP_CHECK(hipGetLastError());
        lk_hip_cg_restart_precond_launch(d_scratch, d_scratch2, d_bnorm, d_tol, ncols, d_active, d_rz_old);
        LK_HIP_CHECK(hipGetLastError());
        LK_HIP_CHECK(hipMemcpy(d_p, d_z, mat_bytes, hipMemcpyDeviceToDevice));  // restart: p = z
      } else {
        lk_hip_cg_restart_launch(d_scratch, d_bnorm, d_tol, ncols, d_active, d_rz_old);
        LK_HIP_CHECK(hipGetLastError());
        LK_HIP_CHECK(hipMemcpy(d_p, d_r, mat_bytes, hipMemcpyDeviceToDevice));  // restart: p = r
      }
    } else {
      lk_hip_batched_axpy_launch(d_neg_alpha, d_Ap, d_r, n, ncols);  // r -= alpha*Ap
      LK_HIP_CHECK(hipGetLastError());
      lk_hip_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);  // r.r (true residual)
      LK_HIP_CHECK(hipGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_hip_batched_dot_launch(d_r, d_z, n, ncols, d_scratch2);  // r.z == rz_new
        LK_HIP_CHECK(hipGetLastError());
        lk_hip_cg_beta_precond_launch(d_scratch, d_scratch2, d_bnorm, d_tol, ncols, d_active, d_rz_old, d_beta);
        LK_HIP_CHECK(hipGetLastError());
        lk_hip_batched_update_p_launch(d_z, d_beta, d_p, n, ncols);  // p = z + beta*p
        LK_HIP_CHECK(hipGetLastError());
      } else {
        lk_hip_cg_beta_launch(d_scratch, d_bnorm, d_tol, ncols, d_active, d_rz_old, d_beta);
        LK_HIP_CHECK(hipGetLastError());
        lk_hip_batched_update_p_launch(d_r, d_beta, d_p, n, ncols);  // p = r + beta*p
        LK_HIP_CHECK(hipGetLastError());
      }
    }

    if ((it + 1) % sync_every == 0 || (it + 1) % restart_every == 0) {
      LK_HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));
      lk_hip_cg_any_active_launch(d_active, ncols, d_flag);
      LK_HIP_CHECK(hipGetLastError());
      LK_HIP_CHECK(hipMemcpy(&host_flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));
    }
  }

  arma::mat X(n, ncols, arma::fill::none);
  LK_HIP_CHECK(hipMemcpy(X.memptr(), d_x, mat_bytes, hipMemcpyDeviceToHost));

  // See CudaLinearAlgebra.cpp's matching readback: d_active still holds,
  // per column, whether the loop hit max_iter before that column reached
  // tol (a converged/deactivated column was already zeroed).
  std::vector<int> active_final(static_cast<std::size_t>(ncols));
  LK_HIP_CHECK(hipMemcpy(active_final.data(), d_active, col_bytes_i, hipMemcpyDeviceToHost));
  arma::uword n_unconverged = 0;
  for (int c = 0; c < ncols; ++c)
    if (active_final[static_cast<std::size_t>(c)])
      ++n_unconverged;
  if (n_unconverged_out != nullptr)
    *n_unconverged_out = n_unconverged;
  LinearAlgebra::cgNonConvergenceWarning(n_unconverged, static_cast<arma::uword>(ncols),
                                        static_cast<arma::uword>(n), max_iter);

  hipFree(d_b);
  hipFree(d_x);
  hipFree(d_r);
  hipFree(d_p);
  hipFree(d_Ap);
  hipFree(d_scratch);
  hipFree(d_scratch2);
  hipFree(d_alpha);
  hipFree(d_neg_alpha);
  hipFree(d_beta);
  hipFree(d_rz_old);
  hipFree(d_bnorm);
  hipFree(d_neg_ones);
  hipFree(d_tol);
  hipFree(d_active);
  hipFree(d_flag);
  if (d_rmul_scratch)
    hipFree(d_rmul_scratch);
  if (preconditioned) {
    hipFree(d_precU);
    hipFree(d_precDinv);
    hipFree(d_precMchol);
    hipFree(d_z);
    hipFree(d_prec_nc);
    hipFree(d_prec_kc);
  }

  return X;
}

// Scalar-tol convenience overload for the common case (every column shares
// one tolerance) -- just broadcasts into the per-column vector above. X0
// not honored here either (same caveat as above) -- threaded through only
// so the call compiles against the vector-tol overload's signature.
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
                            const arma::mat* X0) {
  return conjugateGradient(Xt, theta, covType, B, max_iter, arma::vec(B.n_cols, arma::fill::value(tol)), precU,
                           precDinv, precMcholLower, n_unconverged_out, X0);
}

// R(Xt,theta) * V in one batched device launch (see the header). Same
// upload-once / single-download structure as conjugateGradient, minus the
// CG loop: this is exactly one matvec.
arma::mat rmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraHip::rmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraHip::rmulBatched: V has " + std::to_string(V.n_rows) + " rows, expected "
                                + std::to_string(n));

  // X, theta and R come from the dense cache: the SLQ recurrence calls this
  // once per Lanczos step at a FIXED theta, so R is built on the first step
  // and reused by every later one (it used to be rebuilt, and a fresh
  // buffer re-allocated, on all 40 of them).
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  double* d_Rmat = covCacheR(cov);
  const bool dense = (d_Rmat != nullptr);

  double *d_V, *d_Av;
  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  LK_HIP_CHECK(hipMalloc(&d_V, mat_bytes));
  LK_HIP_CHECK(hipMalloc(&d_Av, mat_bytes));
  LK_HIP_CHECK(hipMemcpy(d_V, V.memptr(), mat_bytes, hipMemcpyHostToDevice));

  double* d_scratch = nullptr;
  if (dense) {
    lk_hip_dense_matvec_launch(d_Rmat, n, d_V, ncols, d_Av, n);
    LK_HIP_CHECK(hipGetLastError());
  } else {
    const int scratch_elems = lk_hip_rmul_batched_scratch_elems(n, ncols);
    if (scratch_elems > 0)
      LK_HIP_CHECK(hipMalloc(&d_scratch, sizeof(double) * static_cast<std::size_t>(scratch_elems)));
    lk_hip_rmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), d_V, ncols, d_Av, d_scratch);
    LK_HIP_CHECK(hipGetLastError());
  }
  LK_HIP_CHECK(hipDeviceSynchronize());

  arma::mat Av(n, ncols, arma::fill::none);
  LK_HIP_CHECK(hipMemcpy(Av.memptr(), d_Av, mat_bytes, hipMemcpyDeviceToHost));

  hipFree(d_V);
  hipFree(d_Av);
  if (d_scratch)
    hipFree(d_scratch);
  return Av;
}

arma::mat dRmulBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType, const arma::mat& V) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraHip::dRmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (dimX > kMaxDimX)
    throw std::invalid_argument("LinearAlgebraHip::dRmulBatched: dimX " + std::to_string(dimX) + " > "
                                + std::to_string(kMaxDimX));
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraHip::dRmulBatched: V has " + std::to_string(V.n_rows)
                                + " rows, expected " + std::to_string(n));

  // Xt/theta are shared with the dense cache (the gradient pass runs at the
  // same theta as the CG/SLQ passes that precede it, so they are already
  // uploaded). The dR blocks themselves are NOT cached -- see DenseCovCache
  // -- they are used exactly once per gradient evaluation, so there is
  // nothing to amortize, and holding them would multiply the steady-state
  // device footprint by dimX.
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);

  double *d_V, *d_Out;
  LK_HIP_CHECK(hipMalloc(&d_V, sizeof(double) * static_cast<std::size_t>(n) * ncols));
  const std::size_t out_bytes = sizeof(double) * static_cast<std::size_t>(n) * dimX * ncols;
  LK_HIP_CHECK(hipMalloc(&d_Out, out_bytes));
  LK_HIP_CHECK(hipMemcpy(
      d_V, V.memptr(), sizeof(double) * static_cast<std::size_t>(n) * ncols, hipMemcpyHostToDevice));

  // Dense fast path: materialize the dimX dR/dtheta_k blocks once, then one
  // lk_hip_dense_matvec_launch per dimension writes straight into its
  // interleaved slot of d_Out (column c's k-th block lives at
  // d_Out[(c*dimX+k)*n .. +n), i.e. consecutive c's for a FIXED k are
  // dimX*n apart -- exactly a column-major n x ncols matrix with leading
  // dimension dimX*n starting at d_Out + k*n, so no separate scatter/copy
  // kernel is needed -- see lk_hip_dense_matvec_launch's `ldc` parameter).
  const double dr_need_mb = static_cast<double>(n) * n * dimX * 8.0 / (1024.0 * 1024.0);
  const bool dense = denseFitsBudget(dr_need_mb);
  if (dense) {
    double* d_dR = nullptr;
    LK_HIP_CHECK(hipMalloc(&d_dR, sizeof(double) * static_cast<std::size_t>(n) * n * dimX));
    lk_hip_build_cov_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), nullptr, d_dR);
    LK_HIP_CHECK(hipGetLastError());
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    for (int k = 0; k < dimX; ++k) {
      lk_hip_dense_matvec_launch(d_dR + n2 * k, n, d_V, ncols, d_Out + static_cast<std::size_t>(k) * n, dimX * n);
      LK_HIP_CHECK(hipGetLastError());
    }
    hipFree(d_dR);
  } else {
    lk_hip_drmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), d_V, ncols, d_Out);
    LK_HIP_CHECK(hipGetLastError());
  }
  LK_HIP_CHECK(hipDeviceSynchronize());

  arma::mat Out(static_cast<arma::uword>(n), static_cast<arma::uword>(dimX) * ncols, arma::fill::none);
  LK_HIP_CHECK(hipMemcpy(Out.memptr(), d_Out, out_bytes, hipMemcpyDeviceToHost));

  hipFree(d_V);
  hipFree(d_Out);
  return Out;
}

// See the .hpp doc comment for the algorithm/scoping summary -- a
// mechanical port of LinearAlgebraCuda::stochasticLogDetBatched, now
// sharing the same DenseCovCache/lk_hip_dense_matvec_launch dense fast
// path as conjugateGradient/rmulBatched above (a single dense matvec per
// Lanczos step against the once-per-theta materialized R, instead of
// recomputing every covariance entry's transcendentals on all
// `lanczos_steps` steps), with the reorthogonalization's two
// cublasDgemmStridedBatched-equivalent calls replaced by
// lk_hip_lanczos_reorth_dot/sub_launch (hand-written kernels -- this
// backend has no BLAS dependency anywhere, see the file header).
// `d_V` holds every probe's ENTIRE Krylov
// history for the whole call, step-major: step j's n x npr block lives at
// d_V + j*npr*n -- same layout CUDA's version uses, and for the same
// reason (step j's block is already exactly what the matvec/reorth kernels
// want as input, no gather pass needed).
double stochasticLogDetBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType,
                               arma::uword lanczos_steps_in, const arma::mat& probes) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraHip::stochasticLogDetBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int npr = static_cast<int>(probes.n_cols);
  const int ls = static_cast<int>(std::min<arma::uword>(lanczos_steps_in, Xt.n_cols));
  if (npr == 0 || ls == 0)
    return 0.0;

  // znorm==0 probes never iterate (matches the CPU version) -- Rademacher
  // probes never hit this in practice, but a caller could pass anything.
  std::vector<double> znorm(static_cast<std::size_t>(npr));
  std::vector<int> active_h(static_cast<std::size_t>(npr));
  arma::mat V0(static_cast<arma::uword>(n), static_cast<arma::uword>(npr), arma::fill::zeros);
  for (int p = 0; p < npr; ++p) {
    znorm[static_cast<std::size_t>(p)] = arma::norm(probes.col(static_cast<arma::uword>(p)));
    if (znorm[static_cast<std::size_t>(p)] != 0.0) {
      V0.col(static_cast<arma::uword>(p)) = probes.col(static_cast<arma::uword>(p)) / znorm[static_cast<std::size_t>(p)];
      active_h[static_cast<std::size_t>(p)] = 1;
    } else {
      active_h[static_cast<std::size_t>(p)] = 0;
    }
  }

  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  double* d_Rmat = covCacheR(cov);
  const bool dense = (d_Rmat != nullptr);
  const int rmul_scratch_elems = dense ? 0 : lk_hip_rmul_batched_scratch_elems(n, npr);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_HIP_CHECK(hipMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));
  auto matvec = [&](const double* d_in, double* d_out) {
    if (dense) {
      lk_hip_dense_matvec_launch(d_Rmat, n, d_in, npr, d_out, n);
      LK_HIP_CHECK(hipGetLastError());
    } else {
      lk_hip_rmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), d_in, npr, d_out,
                                 d_rmul_scratch);
      LK_HIP_CHECK(hipGetLastError());
    }
  };

  const std::size_t step_bytes = sizeof(double) * static_cast<std::size_t>(n) * npr;
  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(npr);
  const std::size_t col_bytes_i = sizeof(int) * static_cast<std::size_t>(npr);
  double *d_V, *d_W, *d_dot, *d_dot2, *d_alpha_step, *d_neg_alpha, *d_beta_step, *d_inv_bj, *d_neg_beta_prev, *d_t;
  double *d_alpha_all, *d_beta_all;
  int *d_active, *d_m_eff;
  LK_HIP_CHECK(hipMalloc(&d_V, step_bytes * static_cast<std::size_t>(ls)));
  LK_HIP_CHECK(hipMemset(d_V, 0, step_bytes * static_cast<std::size_t>(ls)));  // see .hpp: zero = "not iterating"
  LK_HIP_CHECK(hipMalloc(&d_W, step_bytes));
  LK_HIP_CHECK(hipMalloc(&d_dot, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_dot2, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_alpha_step, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_neg_alpha, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_beta_step, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_inv_bj, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_neg_beta_prev, col_bytes));
  LK_HIP_CHECK(hipMalloc(&d_t, sizeof(double) * static_cast<std::size_t>(ls) * npr));
  LK_HIP_CHECK(hipMalloc(&d_active, col_bytes_i));
  LK_HIP_CHECK(hipMalloc(&d_m_eff, col_bytes_i));
  LK_HIP_CHECK(hipMalloc(&d_alpha_all, sizeof(double) * static_cast<std::size_t>(ls) * npr));
  LK_HIP_CHECK(hipMalloc(&d_beta_all, sizeof(double) * static_cast<std::size_t>(ls) * npr));

  LK_HIP_CHECK(hipMemcpy(d_V, V0.memptr(), step_bytes, hipMemcpyHostToDevice));
  LK_HIP_CHECK(hipMemcpy(d_active, active_h.data(), col_bytes_i, hipMemcpyHostToDevice));
  std::vector<int> m_eff_init(static_cast<std::size_t>(npr), ls);
  LK_HIP_CHECK(hipMemcpy(d_m_eff, m_eff_init.data(), col_bytes_i, hipMemcpyHostToDevice));

  for (int j = 0; j < ls; ++j) {
    double* Vj = d_V + static_cast<std::size_t>(j) * npr * n;
    matvec(Vj, d_W);
    if (j > 0) {
      double* Vprev = d_V + static_cast<std::size_t>(j - 1) * npr * n;
      lk_hip_batched_axpy_launch(d_neg_beta_prev, Vprev, d_W, n, npr);  // w -= beta_prev * v_prev
      LK_HIP_CHECK(hipGetLastError());
    }
    lk_hip_batched_dot_launch(d_W, Vj, n, npr, d_dot);  // alpha_j = <w, Vj>
    LK_HIP_CHECK(hipGetLastError());
    lk_hip_lanczos_alpha_launch(d_dot, npr, d_active, d_alpha_step, d_neg_alpha);
    LK_HIP_CHECK(hipGetLastError());
    LK_HIP_CHECK(hipMemcpy(d_alpha_all + static_cast<std::size_t>(j) * npr, d_alpha_step, col_bytes,
                             hipMemcpyDeviceToDevice));
    lk_hip_batched_axpy_launch(d_neg_alpha, Vj, d_W, n, npr);  // w -= alpha_j * Vj
    LK_HIP_CHECK(hipGetLastError());

    // Full reorthogonalization against V[:, 0..j] for every probe at once,
    // via the two hand-written kernels this backend uses in place of
    // CUDA's cublasDgemmStridedBatched pair (see the .hpp doc comments).
    const int m = j + 1;
    lk_hip_lanczos_reorth_dot_launch(d_V, d_W, n, npr, ls, m, d_t);
    LK_HIP_CHECK(hipGetLastError());
    lk_hip_lanczos_reorth_sub_launch(d_V, d_t, n, npr, ls, m, d_W);
    LK_HIP_CHECK(hipGetLastError());

    lk_hip_batched_dot_launch(d_W, d_W, n, npr, d_dot2);  // bj^2
    LK_HIP_CHECK(hipGetLastError());
    const int is_last = (j + 1 == ls) ? 1 : 0;
    lk_hip_lanczos_beta_launch(d_dot2, npr, j, is_last, d_active, d_m_eff, d_beta_step, d_inv_bj, d_neg_beta_prev);
    LK_HIP_CHECK(hipGetLastError());
    LK_HIP_CHECK(hipMemcpy(d_beta_all + static_cast<std::size_t>(j) * npr, d_beta_step, col_bytes,
                             hipMemcpyDeviceToDevice));

    if (j + 1 < ls) {
      double* Vnext = d_V + static_cast<std::size_t>(j + 1) * npr * n;
      lk_hip_batched_axpy_launch(d_inv_bj, d_W, Vnext, n, npr);  // Vnext (== 0) += inv_bj * w
      LK_HIP_CHECK(hipGetLastError());
    }
  }

  std::vector<double> alpha_h(static_cast<std::size_t>(ls) * npr), beta_h(static_cast<std::size_t>(ls) * npr);
  LK_HIP_CHECK(hipMemcpy(alpha_h.data(), d_alpha_all, sizeof(double) * alpha_h.size(), hipMemcpyDeviceToHost));
  LK_HIP_CHECK(hipMemcpy(beta_h.data(), d_beta_all, sizeof(double) * beta_h.size(), hipMemcpyDeviceToHost));
  std::vector<int> m_eff_h(static_cast<std::size_t>(npr));
  LK_HIP_CHECK(hipMemcpy(m_eff_h.data(), d_m_eff, col_bytes_i, hipMemcpyDeviceToHost));

  hipFree(d_V);
  hipFree(d_W);
  hipFree(d_dot);
  hipFree(d_dot2);
  hipFree(d_alpha_step);
  hipFree(d_neg_alpha);
  hipFree(d_beta_step);
  hipFree(d_inv_bj);
  hipFree(d_neg_beta_prev);
  hipFree(d_t);
  hipFree(d_active);
  hipFree(d_m_eff);
  hipFree(d_alpha_all);
  hipFree(d_beta_all);
  if (d_rmul_scratch)
    hipFree(d_rmul_scratch);

  // Same tail as LinearAlgebra::stochasticLogDetBatched: per-probe
  // tridiagonal eigendecomposition (O(nprobe*lanczos_steps^2), negligible
  // next to the matvecs above -- not worth a device port).
  double total = 0.0;
  for (int p = 0; p < npr; ++p) {
    if (znorm[static_cast<std::size_t>(p)] == 0.0)
      continue;
    const int me = m_eff_h[static_cast<std::size_t>(p)];
    arma::mat T(static_cast<arma::uword>(me), static_cast<arma::uword>(me), arma::fill::zeros);
    for (int jj = 0; jj < me; ++jj)
      T(static_cast<arma::uword>(jj), static_cast<arma::uword>(jj)) = alpha_h[static_cast<std::size_t>(jj) * npr + p];
    for (int jj = 0; jj + 1 < me; ++jj) {
      const double b = beta_h[static_cast<std::size_t>(jj) * npr + p];
      T(static_cast<arma::uword>(jj), static_cast<arma::uword>(jj + 1)) = b;
      T(static_cast<arma::uword>(jj + 1), static_cast<arma::uword>(jj)) = b;
    }
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, T);
    double quad = 0.0;
    for (int jj = 0; jj < me; ++jj) {
      const double lambda = std::max(eigval(static_cast<arma::uword>(jj)), LinearAlgebra::num_nugget);
      quad += eigvec(0, static_cast<arma::uword>(jj)) * eigvec(0, static_cast<arma::uword>(jj)) * std::log(lambda);
    }
    total += quad;
  }
  return (static_cast<double>(n) / static_cast<double>(npr)) * total;
}

}  // namespace LinearAlgebraHip

#endif  // LIBKRIGING_USE_HIP_ITERATIVE
