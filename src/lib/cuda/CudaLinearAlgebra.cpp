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

#include "libKriging/LinearAlgebra.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// Plan item #8: mixed-precision matvec, opt-in while it's being validated
// (default off -- every existing test/bench keeps running full fp64 unless
// this is set). Only takes effect on the dense path: the matrix-free
// kernels' cost is transcendental evaluation (exp/log1p per pair), not
// GEMM FLOPs, so TF32 tensor cores -- which only accelerate matrix
// multiply -- don't help there. See CHANGELOG.md / todo_reach_gpytorch.md
// for the measurement behind restricting the fp64 correction step to the
// GPU (never the CPU, even on a throttled card) and for why this is scoped
// to conjugateGradient's own matvec rather than the SLQ Lanczos recursion
// too (not yet extended there).
bool mixedPrecisionEnabled() {
  return std::getenv("LK_ITERATIVE_CUDA_MIXED_PRECISION") != nullptr;
}

// Device-resident cache of the materialized dense R for the LAST
// (Xt, theta, covType) seen.
//
// Every entry point below used to be stateless: conjugateGradient,
// rmulBatched and dRmulBatched each cudaMalloc'd d_Xt/d_theta/d_Rmat,
// rebuilt R from scratch, then cudaFree'd everything before returning.
// That is fine for a single solve, but it is NOT what the iterative
// objective does: one _logLikelihoodIterative evaluation runs three Krylov
// passes at a FIXED theta (CG on [F|y], the SLQ Lanczos recurrence, then CG
// on the Hutchinson probes), and the SLQ recurrence calls rmulBatched once
// per Lanczos step -- 40 steps in the benchmark. So a constant n x n matrix
// was being re-allocated (128 MB at n = 4000) and re-evaluated (16 M fp64
// exp/log1p) 40+ times per objective evaluation, for ~0.02 ms of actually
// useful GEMM each time.
//
// Caching on (n, dimX, kind, Xt, theta) collapses all of that to one build
// per theta. The key is compared by VALUE, not by pointer: callers pass
// temporaries and Armadillo reuses freed memory, so a pointer/shape
// comparison would alias two different designs onto the same cached R.
// Comparing contents is O(n*dimX + dimX) -- 16 k doubles at n = 4000, d = 4
// -- i.e. nothing next to the O(n^2) build it guards.
//
// Only R is kept resident. The dRmulBatched blocks (dimX * n^2, 512 MB at
// n = 4000, d = 4) are deliberately NOT cached: they are used exactly once
// per gradient evaluation, so there is nothing to amortize, and holding
// them would quintuple the steady-state device footprint.
struct DenseCovCache {
  int n = 0;
  int dimX = 0;
  int kind = -1;
  std::vector<double> xt;
  std::vector<double> theta;
  double* d_Xt = nullptr;
  double* d_theta = nullptr;
  double* d_R = nullptr;
  // Lazily-built fp32 copy of d_R (plan item #8: mixed-precision matvec),
  // built once per theta by covCacheRf32() below, on top of the ALREADY
  // materialized double d_R -- a cast, not a second build, so it never pays
  // for its own covariance evaluation.
  float* d_R_f32 = nullptr;
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
    cudaFree(c.d_Xt);
  if (c.d_theta)
    cudaFree(c.d_theta);
  if (c.d_R)
    cudaFree(c.d_R);
  if (c.d_R_f32)
    cudaFree(c.d_R_f32);
  c.d_Xt = c.d_theta = c.d_R = nullptr;
  c.d_R_f32 = nullptr;

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  LK_CUDA_CHECK(cudaMalloc(&c.d_Xt, sizeof(double) * static_cast<std::size_t>(n) * dimX));
  LK_CUDA_CHECK(cudaMalloc(&c.d_theta, sizeof(double) * dimX));
  LK_CUDA_CHECK(
      cudaMemcpy(c.d_Xt, Xt.memptr(), sizeof(double) * static_cast<std::size_t>(n) * dimX, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(c.d_theta, theta.memptr(), sizeof(double) * dimX, cudaMemcpyHostToDevice));

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
  LK_CUDA_CHECK(cudaMalloc(&c.d_R, sizeof(double) * static_cast<std::size_t>(c.n) * c.n));
  lk_cuda_build_cov_launch(c.d_Xt, c.n, c.dimX, c.d_theta, c.kind, c.d_R, nullptr);
  LK_CUDA_CHECK(cudaGetLastError());
  return c.d_R;
}

// Lazily-built fp32 copy of the (already dense-materialized) R, for the
// mixed-precision matvec (plan item #8). Requires covCacheR() to already
// have built the double R -- callers only reach for this on the dense
// path, same gating as the double cache. A cast, not a rebuild: costs one
// O(n^2) elementwise kernel, not a second covariance evaluation.
float* covCacheRf32(DenseCovCache& c) {
  if (c.d_R_f32)
    return c.d_R_f32;
  const std::size_t count = static_cast<std::size_t>(c.n) * c.n;
  LK_CUDA_CHECK(cudaMalloc(&c.d_R_f32, sizeof(float) * count));
  lk_cuda_cast_d2f_launch(c.d_R, static_cast<long long>(count), c.d_R_f32);
  LK_CUDA_CHECK(cudaGetLastError());
  return c.d_R_f32;
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
// Per-column tol: lets a batched solve fuse right-hand-side groups that
// need different tolerances into ONE Krylov pass (see the .cuh doc
// comment) -- each column still freezes independently at its own tol[c],
// same lockstep-batching contract as every other per-column quantity here.
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
                            const arma::mat* X0) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraCuda::conjugateGradient: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(B.n_cols);
  const bool preconditioned = (precU.n_elem > 0);
  const int pk = preconditioned ? static_cast<int>(precU.n_cols) : 0;
  if (static_cast<int>(tol.n_elem) != ncols)
    throw std::invalid_argument("LinearAlgebraCuda::conjugateGradient: tol has " + std::to_string(tol.n_elem)
                                + " entries, expected " + std::to_string(ncols) + " (one per column of B)");
  if (X0 != nullptr && (static_cast<int>(X0->n_rows) != n || static_cast<int>(X0->n_cols) != ncols))
    throw std::invalid_argument("LinearAlgebraCuda::conjugateGradient: X0 is " + std::to_string(X0->n_rows) + "x"
                                + std::to_string(X0->n_cols) + ", expected " + std::to_string(n) + "x"
                                + std::to_string(ncols) + " (same shape as B)");

  // X, theta and R now come from the process-wide dense cache (see
  // DenseCovCache): a whole objective evaluation runs at a fixed theta, so
  // the first pass pays the upload + build and the SLQ/probe passes that
  // follow reuse them. Nothing here is freed on the way out -- the cache
  // owns d_Xt/d_theta/d_Rmat and drops them when theta next moves.
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  double* d_Xt = cov.d_Xt;
  double* d_theta = cov.d_theta;
  double* d_Rmat = covCacheR(cov);
  const bool dense = (d_Rmat != nullptr);
  const bool mixed_precision = dense && mixedPrecisionEnabled();
  float* d_Rmat_f32 = mixed_precision ? covCacheRf32(cov) : nullptr;

  // Current working-set width: starts at ncols, shrinks as columns
  // converge and get compacted out (see compactConverged below). Declared
  // this early because matvec/precondApply (defined further down) capture
  // it by reference and must see it already in scope.
  int ncols_cur = ncols;
  auto matBytesCur = [&] {
    return sizeof(double) * static_cast<std::size_t>(n) * static_cast<std::size_t>(ncols_cur);
  };

  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  double *d_b, *d_x, *d_r, *d_p, *d_Ap;
  LK_CUDA_CHECK(cudaMalloc(&d_b, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_x, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_r, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_p, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_Ap, mat_bytes));

  // fp32 sandwich buffers for the mixed-precision matvec (plan item #8):
  // cast the current search direction down once, run the TF32/fp32 GEMM,
  // cast the result back up. Sized at the ORIGINAL ncols (like every other
  // buffer here) and used at their leading ncols_cur portion after a
  // compaction, same convention as d_p/d_Ap.
  float *d_p_f32 = nullptr, *d_Ap_f32 = nullptr;
  if (mixed_precision) {
    LK_CUDA_CHECK(cudaMalloc(&d_p_f32, sizeof(float) * static_cast<std::size_t>(n) * ncols));
    LK_CUDA_CHECK(cudaMalloc(&d_Ap_f32, sizeof(float) * static_cast<std::size_t>(n) * ncols));
  }

  // Per-column CG scalars, kept ON DEVICE so the loop never round-trips
  // pAp / r.r / alpha / beta through the host (that was ~4 blocking
  // cudaMemcpy per iteration -- the dominant cost of an ill-conditioned
  // solve that runs thousands of iterations). The host only pulls back a single
  // "any column still active?" int, every `sync_every` iterations.
  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(ncols);
  const std::size_t col_bytes_i = sizeof(int) * static_cast<std::size_t>(ncols);
  double *d_scratch, *d_scratch2, *d_alpha, *d_neg_alpha, *d_beta, *d_rz_old, *d_bnorm, *d_neg_ones, *d_tol;
  int *d_active, *d_flag;
  LK_CUDA_CHECK(cudaMalloc(&d_scratch, col_bytes));   // pAp, then r.r
  LK_CUDA_CHECK(cudaMalloc(&d_scratch2, col_bytes));  // preconditioned: r.z alongside r.r
  LK_CUDA_CHECK(cudaMalloc(&d_alpha, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_neg_alpha, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_beta, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_rz_old, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_bnorm, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_neg_ones, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_tol, col_bytes));
  LK_CUDA_CHECK(cudaMemcpy(d_tol, tol.memptr(), col_bytes, cudaMemcpyHostToDevice));
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
  // z = P^-1 r with P = D + U U^T, applied by the Woodbury identity:
  //   z = Dinv .* (r - U (L L^T)^-1 U^T (Dinv .* r))
  // Everything that is not elementwise goes through cuBLAS: two DGEMMs and
  // one DTRSM pair on the k x ncols block. The previous hand-written
  // kernels made this the dominant cost of a preconditioned solve -- the
  // k x k triangular solve in particular ran one thread per right-hand-side
  // column with an O(k^2) serial substitution inside it, i.e. `ncols`
  // threads total (300 on a 132-SM H100) and O(k^2) serialized FLOPs each,
  // so `LLIterative(30,128,40)` cost 13x `LLIterative(30,0,40)` for an
  // identical log-likelihood value. DTRSM solves exactly this shape with
  // the whole device.
  const double prec_one = 1.0, prec_zero = 0.0;
  auto precondApply = [&](const double* d_in, double* d_out) {
    lk_cuda_scale_rows_launch(d_precDinv, d_in, n, ncols_cur, d_prec_nc);  // d_prec_nc = Dinv .* r
    LK_CUDA_CHECK(cudaGetLastError());
    // t = U^T (Dinv .* r)   (k x ncols_cur)
    LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, pk, ncols_cur, n, &prec_one, d_precU, n,
                                d_prec_nc, n, &prec_zero, d_prec_kc, pk));
    // s = (L L^T)^-1 t, in place: L y = t then L^T s = y
    LK_CUBLAS_CHECK(cublasDtrsm(cublasHandle(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT, pk, ncols_cur, &prec_one, d_precMchol, pk, d_prec_kc, pk));
    LK_CUBLAS_CHECK(cublasDtrsm(cublasHandle(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, pk, ncols_cur, &prec_one, d_precMchol, pk, d_prec_kc, pk));
    // d_prec_nc is free again (its Dinv .* r content was consumed by the
    // first DGEMM): reuse it for U s (n x ncols_cur) instead of a 7th buffer.
    LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols_cur, pk, &prec_one, d_precU, n,
                                d_prec_kc, pk, &prec_zero, d_prec_nc, n));
    lk_cuda_precond_finish_launch(d_precDinv, d_in, d_prec_nc, n, ncols_cur, d_out);
    LK_CUDA_CHECK(cudaGetLastError());
  };

  // n and ncols don't change across a CG solve, so the matrix-free matvec's
  // own scratch requirement (see lk_cuda_rmul_batched_scratch_elems) is
  // fixed too -- allocate it once here rather than inside the per-iteration
  // matvec call. Not needed at all on the dense path (no tiled reduction).
  // Mutable: column compaction (below) shrinks ncols_cur mid-solve, and
  // chooseJBlocks can pick a DIFFERENT tile count for a smaller column
  // count, so this gets recomputed and reallocated at every compaction
  // rather than assumed to still fit the allocation sized for the original
  // ncols.
  int rmul_scratch_elems = dense ? 0 : lk_cuda_rmul_batched_scratch_elems(n, ncols);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_CUDA_CHECK(cudaMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));

  // Ap = R * V, either a single cublasDgemm against the materialized d_Rmat
  // or the matrix-free rmul_batched_kernel -- the only difference the rest
  // of this CG loop needs to know about. Reads ncols_cur (not ncols): after
  // a compaction this is the current, possibly-shrunk, working-set width.
  //
  // `exact` (plan item #8): when mixed_precision is on, most calls run the
  // TF32/fp32 GEMM instead (d_in cast down, GEMM'd against d_Rmat_f32, cast
  // back up) -- cheap but with accumulating fp32-level error. The caller
  // passes exact=true ONLY at the periodic restart's residual recompute,
  // which stays full fp64 end to end: that's what corrects away the fp32
  // path's error, the same iterative-refinement argument that already
  // justifies the restart's round-off correction on the plain fp64 path.
  // Matrix-free calls always ignore `exact` (no fp32 kernel written for
  // that path yet -- its cost is transcendental evaluation, not GEMM
  // FLOPs, so TF32 tensor cores don't help it the same way).
  const double gemm_one = 1.0, gemm_zero = 0.0;
  const float gemm_one_f = 1.0f, gemm_zero_f = 0.0f;
  auto matvec = [&](const double* d_in, double* d_out, bool exact) {
    if (dense) {
      if (mixed_precision && !exact) {
        const long long count = static_cast<long long>(n) * ncols_cur;
        lk_cuda_cast_d2f_launch(d_in, count, d_p_f32);
        LK_CUDA_CHECK(cudaGetLastError());
        LK_CUBLAS_CHECK(cublasGemmEx(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols_cur, n, &gemm_one_f,
                                     d_Rmat_f32, CUDA_R_32F, n, d_p_f32, CUDA_R_32F, n, &gemm_zero_f, d_Ap_f32,
                                     CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        lk_cuda_cast_f2d_launch(d_Ap_f32, count, d_out);
        LK_CUDA_CHECK(cudaGetLastError());
      } else {
        LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols_cur, n, &gemm_one, d_Rmat, n,
                                    d_in, n, &gemm_zero, d_out, n));
      }
    } else {
      lk_cuda_rmul_batched_launch(d_Xt, n, dimX, d_theta, static_cast<int>(kind), d_in, ncols_cur, d_out,
                                  d_rmul_scratch);
      LK_CUDA_CHECK(cudaGetLastError());
    }
  };

  // neg_ones is a constant (doesn't depend on B/X0): upload it before the
  // X0 branch below, which needs it for the "r = b - A*x0" axpy.
  std::vector<double> neg_ones(ncols, -1.0);
  LK_CUDA_CHECK(cudaMemcpy(d_neg_ones, neg_ones.data(), col_bytes, cudaMemcpyHostToDevice));

  LK_CUDA_CHECK(cudaMemcpy(d_b, B.memptr(), mat_bytes, cudaMemcpyHostToDevice));
  if (X0 != nullptr) {
    LK_CUDA_CHECK(cudaMemcpy(d_x, X0->memptr(), mat_bytes, cudaMemcpyHostToDevice));
    matvec(d_x, d_Ap, /*exact=*/true);                                    // Ap = A*x0, full fp64
    LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b
    lk_cuda_batched_axpy_launch(d_neg_ones, d_Ap, d_r, n, ncols);         // r -= Ap  =>  r = b - A*x0
    LK_CUDA_CHECK(cudaGetLastError());
  } else {
    LK_CUDA_CHECK(cudaMemset(d_x, 0, mat_bytes));
    LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, mat_bytes, cudaMemcpyDeviceToDevice));  // r = b - A*0
  }

  std::vector<double> bnorm(ncols);
  for (int c = 0; c < ncols; ++c)
    bnorm[c] = arma::norm(B.col(c));
  LK_CUDA_CHECK(cudaMemcpy(d_bnorm, bnorm.data(), col_bytes, cudaMemcpyHostToDevice));

  // rz_old is now always computed ON DEVICE from the actual initial residual
  // (r=b when X0 is null, so <r,r> == |b|^2 exactly -- this reproduces the
  // old host-computed bnorm^2 shortcut bit-for-bit in that case, just via
  // one extra one-time kernel launch instead of a host loop; with X0 given,
  // r != b in general, so the shortcut would have been wrong there).
  if (preconditioned) {
    precondApply(d_r, d_z);                                              // z = Pinv(r)
    LK_CUDA_CHECK(cudaMemcpy(d_p, d_z, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z
    lk_cuda_batched_dot_launch(d_r, d_z, n, ncols, d_rz_old);           // rz_old = <r, z>
    LK_CUDA_CHECK(cudaGetLastError());
  } else {
    LK_CUDA_CHECK(cudaMemcpy(d_p, d_r, mat_bytes, cudaMemcpyDeviceToDevice));  // p = z = r
    lk_cuda_batched_dot_launch(d_r, d_r, n, ncols, d_rz_old);           // rz_old = <r, r>
    LK_CUDA_CHECK(cudaGetLastError());
  }

  // active_h: a column is started inactive when b=0 (x=0 trivially solves
  // it, same as before X0 existed) OR when X0 already meets tol for it (a
  // good warm start needing zero further iterations) -- the latter can only
  // happen when X0 != nullptr. Needs the true residual norm r.r, which is
  // exactly d_rz_old in the unpreconditioned case above but is r.z (not
  // r.r) once preconditioned, so fetch it separately there.
  std::vector<double> resid_sq_h(ncols);
  if (preconditioned) {
    lk_cuda_batched_dot_launch(d_r, d_r, n, ncols, d_scratch);
    LK_CUDA_CHECK(cudaGetLastError());
    LK_CUDA_CHECK(cudaMemcpy(resid_sq_h.data(), d_scratch, col_bytes, cudaMemcpyDeviceToHost));
  } else {
    LK_CUDA_CHECK(cudaMemcpy(resid_sq_h.data(), d_rz_old, col_bytes, cudaMemcpyDeviceToHost));
  }
  std::vector<int> active_h(ncols);
  bool any_active_h = false;
  for (int c = 0; c < ncols; ++c) {
    const bool zero_b = (bnorm[c] == 0.0);
    const bool already_at_tol
        = (X0 != nullptr) && !zero_b && (std::sqrt(std::max(resid_sq_h[c], 0.0)) < tol(c) * bnorm[c]);
    active_h[c] = (!zero_b && !already_at_tol) ? 1 : 0;
    any_active_h = any_active_h || (active_h[c] != 0);
  }
  LK_CUDA_CHECK(cudaMemcpy(d_active, active_h.data(), col_bytes_i, cudaMemcpyHostToDevice));

  constexpr arma::uword restart_every = 50;  // exact-residual recompute, corrects round-off drift
  constexpr arma::uword sync_every = 10;     // host "any active?" poll cadence
  int host_flag = any_active_h ? 1 : 0;

  // Every column of a batched CG solve is an independent solve, coupled
  // only through the shared matvec -- and that matvec is itself
  // columnwise-independent, (A*P)[:,c] = A*P[:,c]. So dropping an already
  // converged/frozen column out of the working set changes nothing about
  // the remaining columns' math, only how many columns the next
  // matvec/dot/axpy touch. col_map[k] is the ORIGINAL column index of
  // working-set slot k; X_out accumulates each column's final value at its
  // original position as it leaves the working set (by compaction or, for
  // whatever remains, at loop exit).
  std::vector<int> col_map(static_cast<std::size_t>(ncols));
  for (int c = 0; c < ncols; ++c)
    col_map[static_cast<std::size_t>(c)] = c;
  arma::mat X_out(n, ncols, arma::fill::zeros);

  // Piggybacks on the restart's existing host sync -- not a new
  // synchronization point. Gathers every persistent per-column and
  // per-(row,column) CG state buffer (x, r, p, z, rz_old, bnorm, tol) into
  // the front of their allocations, keeping only the still-active columns,
  // and writes the just-deactivated ones' frozen x into X_out. A host
  // round trip of the working n x ncols_cur matrices, but only when a
  // restart's exact-residual recompute reveals the active set actually
  // shrank -- i.e. at most once per restart_every=50 iterations.
  auto compactConverged = [&]() {
    std::vector<int> active_h_cur(static_cast<std::size_t>(ncols_cur));
    LK_CUDA_CHECK(cudaMemcpy(active_h_cur.data(), d_active, sizeof(int) * static_cast<std::size_t>(ncols_cur),
                             cudaMemcpyDeviceToHost));
    int n_still_active = 0;
    for (int c = 0; c < ncols_cur; ++c)
      n_still_active += active_h_cur[static_cast<std::size_t>(c)];
    if (n_still_active == 0 || n_still_active == ncols_cur)
      return;  // nothing to compact: either the loop exits next, or nothing dropped

    const std::size_t cur_bytes = matBytesCur();
    // d_b (the right-hand side) must be compacted too: the restart branch
    // recomputes r = b - A*x by copying straight from d_b's leading
    // ncols_cur columns, so once the working set is reordered, b needs the
    // SAME reorder or a later restart reads the wrong right-hand side for
    // a kept column.
    arma::mat Xh(n, ncols_cur, arma::fill::none), Rh(n, ncols_cur, arma::fill::none),
        Ph(n, ncols_cur, arma::fill::none), Bh(n, ncols_cur, arma::fill::none);
    LK_CUDA_CHECK(cudaMemcpy(Xh.memptr(), d_x, cur_bytes, cudaMemcpyDeviceToHost));
    LK_CUDA_CHECK(cudaMemcpy(Rh.memptr(), d_r, cur_bytes, cudaMemcpyDeviceToHost));
    LK_CUDA_CHECK(cudaMemcpy(Ph.memptr(), d_p, cur_bytes, cudaMemcpyDeviceToHost));
    LK_CUDA_CHECK(cudaMemcpy(Bh.memptr(), d_b, cur_bytes, cudaMemcpyDeviceToHost));
    arma::mat Zh;
    if (preconditioned) {
      Zh.set_size(n, ncols_cur);
      LK_CUDA_CHECK(cudaMemcpy(Zh.memptr(), d_z, cur_bytes, cudaMemcpyDeviceToHost));
    }
    const std::size_t cur_col_bytes = sizeof(double) * static_cast<std::size_t>(ncols_cur);
    std::vector<double> rz_old_h(static_cast<std::size_t>(ncols_cur)), bnorm_h(static_cast<std::size_t>(ncols_cur)),
        tol_h(static_cast<std::size_t>(ncols_cur));
    LK_CUDA_CHECK(cudaMemcpy(rz_old_h.data(), d_rz_old, cur_col_bytes, cudaMemcpyDeviceToHost));
    LK_CUDA_CHECK(cudaMemcpy(bnorm_h.data(), d_bnorm, cur_col_bytes, cudaMemcpyDeviceToHost));
    LK_CUDA_CHECK(cudaMemcpy(tol_h.data(), d_tol, cur_col_bytes, cudaMemcpyDeviceToHost));

    std::vector<int> keep;
    keep.reserve(static_cast<std::size_t>(n_still_active));
    std::vector<int> new_col_map(static_cast<std::size_t>(n_still_active));
    for (int c = 0; c < ncols_cur; ++c) {
      if (active_h_cur[static_cast<std::size_t>(c)]) {
        new_col_map[keep.size()] = col_map[static_cast<std::size_t>(c)];
        keep.push_back(c);
      } else {
        X_out.col(static_cast<arma::uword>(col_map[static_cast<std::size_t>(c)])) = Xh.col(c);
      }
    }
    const arma::uvec keep_u = arma::conv_to<arma::uvec>::from(keep);
    const arma::mat Xh2 = Xh.cols(keep_u);
    const arma::mat Rh2 = Rh.cols(keep_u);
    const arma::mat Ph2 = Ph.cols(keep_u);
    const arma::mat Bh2 = Bh.cols(keep_u);
    std::vector<double> rz_old2(static_cast<std::size_t>(n_still_active)),
        bnorm2(static_cast<std::size_t>(n_still_active)), tol2(static_cast<std::size_t>(n_still_active));
    for (std::size_t k = 0; k < keep.size(); ++k) {
      rz_old2[k] = rz_old_h[static_cast<std::size_t>(keep[k])];
      bnorm2[k] = bnorm_h[static_cast<std::size_t>(keep[k])];
      tol2[k] = tol_h[static_cast<std::size_t>(keep[k])];
    }

    ncols_cur = n_still_active;
    const std::size_t new_bytes = matBytesCur();
    LK_CUDA_CHECK(cudaMemcpy(d_x, Xh2.memptr(), new_bytes, cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(cudaMemcpy(d_r, Rh2.memptr(), new_bytes, cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(cudaMemcpy(d_p, Ph2.memptr(), new_bytes, cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(cudaMemcpy(d_b, Bh2.memptr(), new_bytes, cudaMemcpyHostToDevice));
    if (preconditioned) {
      const arma::mat Zh2 = Zh.cols(keep_u);
      LK_CUDA_CHECK(cudaMemcpy(d_z, Zh2.memptr(), new_bytes, cudaMemcpyHostToDevice));
    }
    const std::size_t new_col_bytes = sizeof(double) * static_cast<std::size_t>(ncols_cur);
    LK_CUDA_CHECK(cudaMemcpy(d_rz_old, rz_old2.data(), new_col_bytes, cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(cudaMemcpy(d_bnorm, bnorm2.data(), new_col_bytes, cudaMemcpyHostToDevice));
    LK_CUDA_CHECK(cudaMemcpy(d_tol, tol2.data(), new_col_bytes, cudaMemcpyHostToDevice));
    const std::vector<int> ones(static_cast<std::size_t>(ncols_cur), 1);  // every kept column is, by construction, active
    LK_CUDA_CHECK(cudaMemcpy(d_active, ones.data(), sizeof(int) * static_cast<std::size_t>(ncols_cur),
                             cudaMemcpyHostToDevice));
    col_map = std::move(new_col_map);

    // The matrix-free matvec's tiling scratch requirement depends on
    // ncols_cur (chooseJBlocks may pick MORE tiles for a smaller column
    // count to keep the GPU busy) -- reallocate for the new width rather
    // than assume it still fits inside the allocation sized for the
    // original ncols.
    if (!dense) {
      const int new_scratch_elems = lk_cuda_rmul_batched_scratch_elems(n, ncols_cur);
      if (new_scratch_elems != rmul_scratch_elems) {
        if (d_rmul_scratch)
          cudaFree(d_rmul_scratch);
        d_rmul_scratch = nullptr;
        rmul_scratch_elems = new_scratch_elems;
        if (rmul_scratch_elems > 0)
          LK_CUDA_CHECK(cudaMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));
      }
    }
  };

  arma::uword iters_done = 0;
  for (arma::uword it = 0; host_flag && it < max_iter; ++it) {
    iters_done = it + 1;
    matvec(d_p, d_Ap, /*exact=*/false);  // Ap = R*p -- the cheap fp32/TF32 path when mixed_precision is on
    lk_cuda_batched_dot_launch(d_p, d_Ap, n, ncols_cur, d_scratch);  // pAp
    LK_CUDA_CHECK(cudaGetLastError());
    lk_cuda_cg_alpha_launch(d_rz_old, d_scratch, ncols_cur, d_active, d_alpha, d_neg_alpha);
    LK_CUDA_CHECK(cudaGetLastError());
    lk_cuda_batched_axpy_launch(d_alpha, d_p, d_x, n, ncols_cur);  // x += alpha*p
    LK_CUDA_CHECK(cudaGetLastError());

    if ((it + 1) % restart_every == 0) {
      // Periodic TRUE-residual replacement (r = b - A*x): corrects the round-off
      // drift of the recursive residual but, unlike the former full restart
      // (p = r or z), KEEPS the search direction via the usual beta update:
      // resetting p destroyed CG conjugacy (see LinearAlgebra::conjugateGradient).
      // Full restart: recompute r = b - A*x exactly for every still-active
      // column (same rationale as LinearAlgebra::conjugateGradient's) --
      // always full fp64, this is the correction step mixed_precision relies on.
      matvec(d_x, d_Ap, /*exact=*/true);  // Ap = R*x
      LK_CUDA_CHECK(cudaMemcpy(d_r, d_b, matBytesCur(), cudaMemcpyDeviceToDevice));  // r = b
      lk_cuda_batched_axpy_launch(d_neg_ones, d_Ap, d_r, n, ncols_cur);             // r = b - A*x
      LK_CUDA_CHECK(cudaGetLastError());
      lk_cuda_batched_dot_launch(d_r, d_r, n, ncols_cur, d_scratch);  // r.r (true residual)
      LK_CUDA_CHECK(cudaGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_cuda_batched_dot_launch(d_r, d_z, n, ncols_cur, d_scratch2);  // r.z
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_cg_beta_precond_launch(d_scratch, d_scratch2, d_bnorm, d_tol, ncols_cur, d_active, d_rz_old, d_beta);
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_batched_update_p_launch(d_z, d_beta, d_p, n, ncols_cur);  // p = z + beta*p
      } else {
        lk_cuda_cg_beta_launch(d_scratch, d_bnorm, d_tol, ncols_cur, d_active, d_rz_old, d_beta);
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_batched_update_p_launch(d_r, d_beta, d_p, n, ncols_cur);  // p = r + beta*p
      }
    } else {
      lk_cuda_batched_axpy_launch(d_neg_alpha, d_Ap, d_r, n, ncols_cur);  // r -= alpha*Ap
      LK_CUDA_CHECK(cudaGetLastError());
      lk_cuda_batched_dot_launch(d_r, d_r, n, ncols_cur, d_scratch);  // r.r (true residual)
      LK_CUDA_CHECK(cudaGetLastError());
      if (preconditioned) {
        precondApply(d_r, d_z);
        lk_cuda_batched_dot_launch(d_r, d_z, n, ncols_cur, d_scratch2);  // r.z == rz_new
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_cg_beta_precond_launch(d_scratch, d_scratch2, d_bnorm, d_tol, ncols_cur, d_active, d_rz_old, d_beta);
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_batched_update_p_launch(d_z, d_beta, d_p, n, ncols_cur);  // p = z + beta*p
        LK_CUDA_CHECK(cudaGetLastError());
      } else {
        lk_cuda_cg_beta_launch(d_scratch, d_bnorm, d_tol, ncols_cur, d_active, d_rz_old, d_beta);
        LK_CUDA_CHECK(cudaGetLastError());
        lk_cuda_batched_update_p_launch(d_r, d_beta, d_p, n, ncols_cur);  // p = r + beta*p
        LK_CUDA_CHECK(cudaGetLastError());
      }
    }

    if ((it + 1) % sync_every == 0 || (it + 1) % restart_every == 0) {
      // A COUNT, not just a 0/1 flag: the same single-int round trip this
      // loop already pays for every sync_every/restart_every iterations
      // also tells us whether the active set actually shrank, so
      // compactConverged's (comparatively expensive) full gather/scatter
      // only runs when it has something to do -- calling it unconditionally
      // every restart, even when nothing had converged since the last one,
      // measurably slowed down solves that never need to compact at all
      // (e.g. the dense path, where every column tends to converge in
      // lockstep) with a host round trip that bought nothing.
      LK_CUDA_CHECK(cudaMemset(d_flag, 0, sizeof(int)));
      lk_cuda_cg_active_count_launch(d_active, ncols_cur, d_flag);
      LK_CUDA_CHECK(cudaGetLastError());
      int active_count = 0;
      LK_CUDA_CHECK(cudaMemcpy(&active_count, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
      host_flag = active_count > 0 ? 1 : 0;
      if (active_count > 0 && active_count < ncols_cur)
        compactConverged();
    }
  }

  // Opt-in diagnostic: the single most useful number when an iterative
  // objective is slow is "did CG converge, or did it run to max_iter?".
  // Reaching max_iter means every later quantity (Hutchinson trace, mean,
  // variance) is built on an unconverged solve.
  if (std::getenv("LK_ITERATIVE_VERBOSE") != nullptr) {
    // tol is now per-column (mBCG fusion can mix e.g. cg_tol and
    // probes_cg_tol in one call) -- print the range rather than a single
    // value; min==max for every non-fused call site.
    const double tol_min = tol.min();
    const double tol_max = tol.max();
    char tol_str[64];
    if (tol_min == tol_max)
      std::snprintf(tol_str, sizeof(tol_str), "%g", tol_min);
    else
      std::snprintf(tol_str, sizeof(tol_str), "%g..%g", tol_min, tol_max);
    std::fprintf(stderr,
                 "[lk-cuda] CG n=%d ncols=%d tol=%s precond_rank=%d iters=%llu/%llu compacted_to=%d%s%s\n", n, ncols,
                 tol_str, pk, static_cast<unsigned long long>(iters_done), static_cast<unsigned long long>(max_iter),
                 ncols_cur, iters_done >= max_iter ? "  <-- NOT CONVERGED" : "", mixed_precision ? " mixed_fp32" : "");
  }

  // Whatever remains in the (possibly compacted-down) working set is each
  // column's final value, whether it converged on the last iteration or
  // hit max_iter still active -- scatter it into X_out at col_map[c], same
  // as every earlier compaction did for the columns that dropped out
  // before it.
  if (ncols_cur > 0) {
    arma::mat Xh_final(n, ncols_cur, arma::fill::none);
    LK_CUDA_CHECK(cudaMemcpy(Xh_final.memptr(), d_x, matBytesCur(), cudaMemcpyDeviceToHost));
    for (int c = 0; c < ncols_cur; ++c)
      X_out.col(static_cast<arma::uword>(col_map[static_cast<std::size_t>(c)])) = Xh_final.col(c);
  }

  // d_active still holds, per (working-set) column, whether the loop above
  // exited via max_iter with that column not yet at tol -- a column
  // dropped by compactConverged was, by construction, already converged,
  // so it can never contribute to n_unconverged; reading the final,
  // possibly-shrunk d_active gives the same count as reading the original
  // ncols-wide one would. Same convergence contract/reporting as
  // LinearAlgebra::conjugateGradientBatched.
  std::vector<int> active_final(static_cast<std::size_t>(ncols_cur));
  if (ncols_cur > 0)
    LK_CUDA_CHECK(cudaMemcpy(active_final.data(), d_active, sizeof(int) * static_cast<std::size_t>(ncols_cur),
                             cudaMemcpyDeviceToHost));
  arma::uword n_unconverged = 0;
  for (int c = 0; c < ncols_cur; ++c)
    if (active_final[static_cast<std::size_t>(c)])
      ++n_unconverged;
  if (n_unconverged_out != nullptr)
    *n_unconverged_out = n_unconverged;
  LinearAlgebra::cgNonConvergenceWarning(n_unconverged, static_cast<arma::uword>(ncols),
                                        static_cast<arma::uword>(n), max_iter);

  // d_Xt / d_theta / d_Rmat / d_Rmat_f32 are owned by the dense cache, not by this call.
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
  cudaFree(d_tol);
  cudaFree(d_active);
  cudaFree(d_flag);
  if (d_rmul_scratch)
    cudaFree(d_rmul_scratch);
  if (d_p_f32) {
    cudaFree(d_p_f32);
    cudaFree(d_Ap_f32);
  }
  if (preconditioned) {
    cudaFree(d_precU);
    cudaFree(d_precDinv);
    cudaFree(d_precMchol);
    cudaFree(d_z);
    cudaFree(d_prec_nc);
    cudaFree(d_prec_kc);
  }

  return X_out;
}

// Scalar-tol convenience overload for the common case (every column shares
// one tolerance) -- just broadcasts into the per-column vector above.
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
    throw std::invalid_argument("LinearAlgebraCuda::rmulBatched: unsupported covType '" + covType + "'");

  const int n = static_cast<int>(Xt.n_cols);
  const int dimX = static_cast<int>(Xt.n_rows);
  const int ncols = static_cast<int>(V.n_cols);
  if (static_cast<int>(V.n_rows) != n)
    throw std::invalid_argument("LinearAlgebraCuda::rmulBatched: V has " + std::to_string(V.n_rows) + " rows, expected "
                                + std::to_string(n));

  // X, theta and R come from the dense cache: the SLQ recurrence calls this
  // once per Lanczos step at a FIXED theta, so R is built on the first step
  // and reused by every later one (it used to be rebuilt, and a 128 MB
  // buffer re-allocated, on all 40 of them).
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);
  double* d_Rmat = covCacheR(cov);
  const bool dense = (d_Rmat != nullptr);

  double *d_V, *d_Av;
  const std::size_t mat_bytes = sizeof(double) * static_cast<std::size_t>(n) * ncols;
  LK_CUDA_CHECK(cudaMalloc(&d_V, mat_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_Av, mat_bytes));
  LK_CUDA_CHECK(cudaMemcpy(d_V, V.memptr(), mat_bytes, cudaMemcpyHostToDevice));

  double* d_scratch = nullptr;
  if (dense) {
    const double one = 1.0, zero = 0.0;
    LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols, n, &one, d_Rmat, n, d_V, n, &zero,
                                d_Av, n));
  } else {
    const int scratch_elems = lk_cuda_rmul_batched_scratch_elems(n, ncols);
    if (scratch_elems > 0)
      LK_CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(double) * static_cast<std::size_t>(scratch_elems)));
    lk_cuda_rmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), d_V, ncols, d_Av, d_scratch);
    LK_CUDA_CHECK(cudaGetLastError());
  }
  LK_CUDA_CHECK(cudaDeviceSynchronize());

  arma::mat Av(n, ncols, arma::fill::none);
  LK_CUDA_CHECK(cudaMemcpy(Av.memptr(), d_Av, mat_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_V);
  cudaFree(d_Av);
  if (d_scratch)
    cudaFree(d_scratch);
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

  // Xt/theta are shared with the dense cache (the gradient pass runs at the
  // same theta as the CG/SLQ passes that precede it, so they are already
  // uploaded). The dR blocks themselves are NOT cached -- see DenseCovCache.
  DenseCovCache& cov = covCacheBind(Xt, theta, kind);

  double *d_V, *d_Out;
  LK_CUDA_CHECK(cudaMalloc(&d_V, sizeof(double) * static_cast<std::size_t>(n) * ncols));
  const std::size_t out_bytes = sizeof(double) * static_cast<std::size_t>(n) * dimX * ncols;
  LK_CUDA_CHECK(cudaMalloc(&d_Out, out_bytes));
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
    lk_cuda_build_cov_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), nullptr, d_dR);
    LK_CUDA_CHECK(cudaGetLastError());
    const double one = 1.0, zero = 0.0;
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    for (int k = 0; k < dimX; ++k) {
      LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, ncols, n, &one, d_dR + n2 * k, n, d_V,
                                  n, &zero, d_Out + static_cast<std::size_t>(k) * n, dimX * n));
    }
    cudaFree(d_dR);
  } else {
    lk_cuda_drmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), d_V, ncols, d_Out);
    LK_CUDA_CHECK(cudaGetLastError());
  }
  LK_CUDA_CHECK(cudaDeviceSynchronize());

  arma::mat Out(static_cast<arma::uword>(n), static_cast<arma::uword>(dimX) * ncols, arma::fill::none);
  LK_CUDA_CHECK(cudaMemcpy(Out.memptr(), d_Out, out_bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_V);
  cudaFree(d_Out);
  return Out;
}

// See the .cuh doc comment for the algorithm/scoping summary. `d_V` holds
// every probe's ENTIRE Krylov history for the whole call, step-major: step
// j's n x nprobe block lives at d_V + j*nprobe*n. Two properties fall out
// of that layout that the reorthogonalization/matvec below both lean on:
//   - step j's block is already a contiguous n x nprobe matrix (lda=n),
//     exactly what the matvec kernel/cuBLAS Dgemm want as input -- no
//     gather pass needed to go from "history" to "this step's vectors".
//   - probe p's history across steps 0..j is a STANDARD column-major n x
//     (j+1) matrix with leading dimension nprobe*n (columns land nprobe*n
//     elements apart, each column itself contiguous) -- exactly what
//     cublasDgemmStridedBatched wants for a per-probe batch item, base
//     pointer d_V+p*n, batch stride n. No transposition/copy needed there
//     either; both matvec and reorthogonalization read straight out of one
//     buffer, in the layout each of them wants natively.
double stochasticLogDetBatched(const arma::mat& Xt, const arma::vec& theta, const std::string& covType,
                               arma::uword lanczos_steps_in, const arma::mat& probes) {
  CovKind kind;
  if (!covKindFromString(covType, &kind))
    throw std::invalid_argument("LinearAlgebraCuda::stochasticLogDetBatched: unsupported covType '" + covType + "'");

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
  const int rmul_scratch_elems = dense ? 0 : lk_cuda_rmul_batched_scratch_elems(n, npr);
  double* d_rmul_scratch = nullptr;
  if (rmul_scratch_elems > 0)
    LK_CUDA_CHECK(cudaMalloc(&d_rmul_scratch, sizeof(double) * static_cast<std::size_t>(rmul_scratch_elems)));
  const double gemm_one = 1.0, gemm_zero = 0.0;
  auto matvec = [&](const double* d_in, double* d_out) {
    if (dense) {
      LK_CUBLAS_CHECK(cublasDgemm(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, npr, n, &gemm_one, d_Rmat, n, d_in, n,
                                  &gemm_zero, d_out, n));
    } else {
      lk_cuda_rmul_batched_launch(cov.d_Xt, n, dimX, cov.d_theta, static_cast<int>(kind), d_in, npr, d_out,
                                  d_rmul_scratch);
      LK_CUDA_CHECK(cudaGetLastError());
    }
  };

  const std::size_t step_bytes = sizeof(double) * static_cast<std::size_t>(n) * npr;
  const std::size_t col_bytes = sizeof(double) * static_cast<std::size_t>(npr);
  const std::size_t col_bytes_i = sizeof(int) * static_cast<std::size_t>(npr);
  double *d_V, *d_W, *d_dot, *d_dot2, *d_alpha_step, *d_neg_alpha, *d_beta_step, *d_inv_bj, *d_neg_beta_prev, *d_t;
  double *d_alpha_all, *d_beta_all;
  int *d_active, *d_m_eff;
  LK_CUDA_CHECK(cudaMalloc(&d_V, step_bytes * static_cast<std::size_t>(ls)));
  LK_CUDA_CHECK(cudaMemset(d_V, 0, step_bytes * static_cast<std::size_t>(ls)));  // see .cuh: zero = "not iterating"
  LK_CUDA_CHECK(cudaMalloc(&d_W, step_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_dot, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_dot2, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_alpha_step, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_neg_alpha, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_beta_step, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_inv_bj, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_neg_beta_prev, col_bytes));
  LK_CUDA_CHECK(cudaMalloc(&d_t, sizeof(double) * static_cast<std::size_t>(ls) * npr));
  LK_CUDA_CHECK(cudaMalloc(&d_active, col_bytes_i));
  LK_CUDA_CHECK(cudaMalloc(&d_m_eff, col_bytes_i));
  LK_CUDA_CHECK(cudaMalloc(&d_alpha_all, sizeof(double) * static_cast<std::size_t>(ls) * npr));
  LK_CUDA_CHECK(cudaMalloc(&d_beta_all, sizeof(double) * static_cast<std::size_t>(ls) * npr));

  LK_CUDA_CHECK(cudaMemcpy(d_V, V0.memptr(), step_bytes, cudaMemcpyHostToDevice));
  LK_CUDA_CHECK(cudaMemcpy(d_active, active_h.data(), col_bytes_i, cudaMemcpyHostToDevice));
  std::vector<int> m_eff_init(static_cast<std::size_t>(npr), ls);
  LK_CUDA_CHECK(cudaMemcpy(d_m_eff, m_eff_init.data(), col_bytes_i, cudaMemcpyHostToDevice));

  const double gemm_neg_one = -1.0;
  const long long stride_probe = n;              // consecutive probes' base pointers in d_V / d_W, in elements
  const long long stride_t = ls;                 // consecutive probes' slots in d_t
  const int lda_hist = npr * n;                  // leading dim of a probe's n x (<=ls) history view into d_V

  for (int j = 0; j < ls; ++j) {
    double* Vj = d_V + static_cast<std::size_t>(j) * npr * n;
    matvec(Vj, d_W);
    if (j > 0) {
      double* Vprev = d_V + static_cast<std::size_t>(j - 1) * npr * n;
      lk_cuda_batched_axpy_launch(d_neg_beta_prev, Vprev, d_W, n, npr);  // w -= beta_prev * v_prev
      LK_CUDA_CHECK(cudaGetLastError());
    }
    lk_cuda_batched_dot_launch(d_W, Vj, n, npr, d_dot);  // alpha_j = <w, Vj>
    LK_CUDA_CHECK(cudaGetLastError());
    lk_cuda_lanczos_alpha_launch(d_dot, npr, d_active, d_alpha_step, d_neg_alpha);
    LK_CUDA_CHECK(cudaGetLastError());
    LK_CUDA_CHECK(cudaMemcpy(d_alpha_all + static_cast<std::size_t>(j) * npr, d_alpha_step, col_bytes,
                             cudaMemcpyDeviceToDevice));
    lk_cuda_batched_axpy_launch(d_neg_alpha, Vj, d_W, n, npr);  // w -= alpha_j * Vj
    LK_CUDA_CHECK(cudaGetLastError());

    // Full reorthogonalization against V[:, 0..j] for every probe at once:
    // w_p -= V_hist_p (V_hist_p^T w_p), as two cublasDgemmStridedBatched
    // calls (batchCount = npr) rather than nprobe*(j+1) separate dot/axpy
    // pairs -- see the function doc comment for why V_hist_p is already in
    // exactly the shape/stride cuBLAS wants without a gather pass.
    const int m = j + 1;
    LK_CUBLAS_CHECK(cublasDgemmStridedBatched(cublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, m, 1, n, &gemm_one, d_V,
                                              lda_hist, stride_probe, d_W, n, stride_probe, &gemm_zero, d_t, ls,
                                              stride_t, npr));
    LK_CUBLAS_CHECK(cublasDgemmStridedBatched(cublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, 1, m, &gemm_neg_one, d_V,
                                              lda_hist, stride_probe, d_t, ls, stride_t, &gemm_one, d_W, n,
                                              stride_probe, npr));

    lk_cuda_batched_dot_launch(d_W, d_W, n, npr, d_dot2);  // bj^2
    LK_CUDA_CHECK(cudaGetLastError());
    const int is_last = (j + 1 == ls) ? 1 : 0;
    lk_cuda_lanczos_beta_launch(d_dot2, npr, j, is_last, d_active, d_m_eff, d_beta_step, d_inv_bj, d_neg_beta_prev);
    LK_CUDA_CHECK(cudaGetLastError());
    LK_CUDA_CHECK(cudaMemcpy(d_beta_all + static_cast<std::size_t>(j) * npr, d_beta_step, col_bytes,
                             cudaMemcpyDeviceToDevice));

    if (j + 1 < ls) {
      double* Vnext = d_V + static_cast<std::size_t>(j + 1) * npr * n;
      lk_cuda_batched_axpy_launch(d_inv_bj, d_W, Vnext, n, npr);  // Vnext (== 0) += inv_bj * w
      LK_CUDA_CHECK(cudaGetLastError());
    }
  }

  std::vector<double> alpha_h(static_cast<std::size_t>(ls) * npr), beta_h(static_cast<std::size_t>(ls) * npr);
  LK_CUDA_CHECK(cudaMemcpy(alpha_h.data(), d_alpha_all, sizeof(double) * alpha_h.size(), cudaMemcpyDeviceToHost));
  LK_CUDA_CHECK(cudaMemcpy(beta_h.data(), d_beta_all, sizeof(double) * beta_h.size(), cudaMemcpyDeviceToHost));
  std::vector<int> m_eff_h(static_cast<std::size_t>(npr));
  LK_CUDA_CHECK(cudaMemcpy(m_eff_h.data(), d_m_eff, col_bytes_i, cudaMemcpyDeviceToHost));

  cudaFree(d_V);
  cudaFree(d_W);
  cudaFree(d_dot);
  cudaFree(d_dot2);
  cudaFree(d_alpha_step);
  cudaFree(d_neg_alpha);
  cudaFree(d_beta_step);
  cudaFree(d_inv_bj);
  cudaFree(d_neg_beta_prev);
  cudaFree(d_t);
  cudaFree(d_active);
  cudaFree(d_m_eff);
  cudaFree(d_alpha_all);
  cudaFree(d_beta_all);
  if (d_rmul_scratch)
    cudaFree(d_rmul_scratch);

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

}  // namespace LinearAlgebraCuda

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE
