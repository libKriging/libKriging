#include "CudaLinearAlgebraKernel.cuh"

#ifdef LIBKRIGING_USE_CUDA_ITERATIVE

#include <cuda_runtime.h>

#include <cstddef>

namespace {

enum class CovKind : int { Gauss = 0, Exp = 1, Matern32 = 2, Matern52 = 3 };

__device__ __forceinline__ double lk_cov_pair(CovKind kind,
                                              const double* __restrict__ Xi,
                                              const double* __restrict__ Xj,
                                              const double* __restrict__ theta,
                                              int dimX) {
  double c, sum = 0.0, sum_sq = 0.0;
  switch (kind) {
    case CovKind::Gauss:
      for (int k = 0; k < dimX; ++k) {
        double val = (Xi[k] - Xj[k]) / theta[k];
        sum_sq += val * val;
      }
      c = exp(-0.5 * sum_sq);
      break;
    case CovKind::Exp:
      for (int k = 0; k < dimX; ++k)
        sum += fabs((Xi[k] - Xj[k]) / theta[k]);
      c = exp(-sum);
      break;
    case CovKind::Matern32:
      for (int k = 0; k < dimX; ++k) {
        double d = 1.7320508075688772 * fabs((Xi[k] - Xj[k]) / theta[k]);
        sum += d - log1p(d);
      }
      c = exp(-sum);
      break;
    case CovKind::Matern52:
    default:
      for (int k = 0; k < dimX; ++k) {
        double d = 2.23606797749979 * fabs((Xi[k] - Xj[k]) / theta[k]);
        sum += d - log1p(d + (d * d) / 3.0);
      }
      c = exp(-sum);
      break;
  }
  return c;
}

// Largest ARD input dimension the dR/dtheta kernel keeps a per-thread stack
// accumulator for. Far beyond any realistic GP (checked host-side in
// LinearAlgebraCuda::dRmulBatched, which falls back to the CPU path above
// this).
#define LK_CUDA_MAX_DIMX 32

// out[k] = d(ln cov)/d(theta_k) for the pair (Xi, Xj), matching
// Covariance::DlnCovDtheta_{gauss,exp,matern32,matern52} in
// src/lib/Covariance.cpp EXACTLY (same closed forms). cov()*out[k] is then
// d(cov)/d(theta_k), the quantity Kriging::_logLikelihoodIterative's CPU
// dRmul_all accumulates.
__device__ __forceinline__ void lk_dlncov_pair(CovKind kind,
                                               const double* __restrict__ Xi,
                                               const double* __restrict__ Xj,
                                               const double* __restrict__ theta,
                                               int dimX,
                                               double* __restrict__ out) {
  switch (kind) {
    case CovKind::Gauss:
      for (int k = 0; k < dimX; ++k) {
        double dx = Xi[k] - Xj[k];
        out[k] = (dx * dx) / (theta[k] * theta[k] * theta[k]);
      }
      break;
    case CovKind::Exp:
      for (int k = 0; k < dimX; ++k)
        out[k] = fabs(Xi[k] - Xj[k]) / (theta[k] * theta[k]);
      break;
    case CovKind::Matern32:
      for (int k = 0; k < dimX; ++k) {
        double d = 1.7320508075688772 * fabs((Xi[k] - Xj[k]) / theta[k]);
        out[k] = (d * d) / (1.0 + d) / theta[k];
      }
      break;
    case CovKind::Matern52:
    default:
      for (int k = 0; k < dimX; ++k) {
        double d = 2.23606797749979 * fabs((Xi[k] - Xj[k]) / theta[k]);
        double a = 1.0 + d;
        double b = (d * d) / 3.0;
        out[k] = (a * b) / (a + b) / theta[k];
      }
      break;
  }
}

// One thread per (row i, column c): Out[i, k + c*dimX] = sum_{j != i}
// d(R_ij)/d(theta_k) * V[j, c], for every k in [0, dimX). R is never
// materialized (matrix-free, matching the CPU dRmul_all). Out is
// n x (dimX*ncols) column-major -- column c's dimX-wide block of
// d(R)/d(theta) . V[:,c] lives at Out + c*dimX*n. Diagonal j==i contributes
// nothing (d(1)/d(theta) = 0), same as the CPU loop's j = i+1 start.
__global__ void drmul_batched_kernel(const double* __restrict__ Xt,
                                     int n,
                                     int dimX,
                                     const double* __restrict__ theta,
                                     CovKind kind,
                                     const double* __restrict__ V,
                                     int ncols,
                                     double* __restrict__ Out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n || c >= ncols)
    return;

  double acc[LK_CUDA_MAX_DIMX];
  for (int k = 0; k < dimX; ++k)
    acc[k] = 0.0;
  double dln[LK_CUDA_MAX_DIMX];

  const double* Vc = V + static_cast<std::size_t>(c) * n;
  const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
  for (int j = 0; j < n; ++j) {
    if (j == i)
      continue;
    const double* Xj = Xt + static_cast<std::size_t>(j) * dimX;
    const double cij = lk_cov_pair(kind, Xi, Xj, theta, dimX);
    lk_dlncov_pair(kind, Xi, Xj, theta, dimX, dln);
    const double vj = Vc[j];
    for (int k = 0; k < dimX; ++k)
      acc[k] += cij * dln[k] * vj;
  }
  double* Oc = Out + static_cast<std::size_t>(c) * dimX * n;
  for (int k = 0; k < dimX; ++k)
    Oc[static_cast<std::size_t>(k) * n + i] = acc[k];
}

// One thread per (row i, column c): R is never materialized (matching the
// CPU Rmul's O(n) memory invariant) -- each thread recomputes R(i,j) on the
// fly while walking j = 0..n-1 and accumulates R(i,:) . P[:,c] into
// Ap[i,c]. Xt is (dimX x n) column-major; P/Ap are (n x ncols) column-major
// (column c at P + c*n), matching arma::mat::memptr() layout.
//
// grid.y = ncols batches every right-hand-side column into ONE launch
// instead of the caller looping ncols times: total covariance evaluations
// are unchanged (each column's matvec always needed its own full n^2 pass;
// R was never cached across columns even before batching, by the same
// matrix-free design), but this cuts the number of kernel launches and
// host<->device round trips in CudaLinearAlgebra.cpp's CG loop by a factor
// of ncols, AND multiplies the block count by ncols -- which also directly
// addresses the occupancy problem profiling found (see the single-column
// history of this file / bench/bench-iterative-cuda.cpp's n-sweep): at
// small n, grid.y>1 keeps far more SMs busy than grid=(ceil(n/128),1) ever
// could on its own.
__global__ void rmul_batched_kernel(const double* __restrict__ Xt,
                                    int n,
                                    int dimX,
                                    const double* __restrict__ theta,
                                    CovKind kind,
                                    const double* __restrict__ P,
                                    int ncols,
                                    double* __restrict__ Ap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n || c >= ncols)
    return;

  const double* Pc = P + static_cast<std::size_t>(c) * n;
  double acc = Pc[i];  // diag = 1
  const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
  for (int j = 0; j < n; ++j) {
    if (j == i)
      continue;
    const double* Xj = Xt + static_cast<std::size_t>(j) * dimX;
    acc += lk_cov_pair(kind, Xi, Xj, theta, dimX) * Pc[j];
  }
  Ap[static_cast<std::size_t>(c) * n + i] = acc;
}

// Same computation as rmul_batched_kernel, but each block only covers a
// [j_start, j_end) slice of the reduction dimension (blockIdx.z selects the
// slice), writing its partial sum to a DISTINCT slot of d_partial (shape
// n x ncols x j_blocks, slice jb at d_partial + jb*n*ncols) rather than
// accumulating into a shared Ap[i,c] cell. Needed because
// rmul_batched_kernel's grid is only (ceil(n/128), ncols): for a small
// ncols (e.g. the 2-3-column [F|y] solve), that leaves most SMs idle
// regardless of n -- profiling with `nsys stats --report gpukernsum`
// confirmed this (rmul_batched_kernel at ~12.4ms/launch, ~2700x the
// batched_dot/axpy kernels in the same CG loop, for that same small-ncols
// call). Splitting j across blockIdx.z multiplies the block count
// independently of both n and ncols, so even a 2-column call at moderate n
// fills the GPU.
//
// Deliberately NOT using atomicAdd to combine the j-slices directly into
// Ap: that was tried first and reverted -- atomicAdd's accumulation order
// depends on which thread's write lands first, which is not guaranteed
// reproducible run-to-run, and this project already fixed the exact same
// class of nondeterminism once for the CPU path's row-parallel matvec (see
// git history: "Fix nondeterminism in LLIterative's row-parallel matvec").
// Two tests ("... predict routes to predictIterative", "LLIterative honors
// optim=none identically to optim=BFGS") call this matvec twice with
// identical inputs and require bit-close identical outputs; atomics broke
// that. Writing to distinct slots and reducing them in a fixed jb-order
// afterward (sum_partials_kernel) keeps the summation order deterministic
// regardless of thread scheduling, at the cost of one extra kernel pass
// and O(n*ncols*j_blocks) scratch memory.
__global__ void rmul_batched_tiled_kernel(const double* __restrict__ Xt,
                                          int n,
                                          int dimX,
                                          const double* __restrict__ theta,
                                          CovKind kind,
                                          const double* __restrict__ P,
                                          int ncols,
                                          double* __restrict__ d_partial,
                                          int j_tile) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  const int jb = blockIdx.z;
  if (i >= n || c >= ncols)
    return;
  const int j_start = jb * j_tile;

  const double* Pc = P + static_cast<std::size_t>(c) * n;
  const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
  double partial = (jb == 0) ? Pc[i] : 0.0;  // diag = 1, added exactly once
  // j_tile = ceil(n / j_blocks) can overshoot n (e.g. n=35, j_blocks=8 ->
  // j_tile=5, last slice's j_start=35 == n): that slice covers no columns
  // but must still WRITE its d_partial slot (as 0, or the lone diag term for
  // jb==0) -- sum_partials_kernel unconditionally sums all j_blocks slots,
  // and cuda/hipMalloc doesn't zero-init, so skipping the write here left
  // sum_partials_kernel adding garbage device memory into the matvec result
  // (confirmed on the HIP port at n=35/ncols=2 -- the LLIterative(60) CG
  // solve for [F|y] -- which produced a negative SSE; see git history).
  if (j_start < n) {
    const int j_end = min(j_start + j_tile, n);
    for (int j = j_start; j < j_end; ++j) {
      if (j == i)
        continue;
      const double* Xj = Xt + static_cast<std::size_t>(j) * dimX;
      partial += lk_cov_pair(kind, Xi, Xj, theta, dimX) * Pc[j];
    }
  }
  d_partial[(static_cast<std::size_t>(jb) * ncols + c) * n + i] = partial;
}

// One thread per (row i, column c): sums the j_blocks partial sums for
// that cell in fixed jb = 0, 1, ..., j_blocks-1 order (NOT whatever order
// threads happen to finish rmul_batched_tiled_kernel in), so the result is
// reproducible across repeated calls with identical inputs -- see the
// determinism note on rmul_batched_tiled_kernel above.
__global__ void sum_partials_kernel(const double* __restrict__ d_partial,
                                    int n,
                                    int ncols,
                                    int j_blocks,
                                    double* __restrict__ Ap) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n || c >= ncols)
    return;
  double acc = 0.0;
  for (int jb = 0; jb < j_blocks; ++jb)
    acc += d_partial[(static_cast<std::size_t>(jb) * ncols + c) * n + i];
  Ap[static_cast<std::size_t>(c) * n + i] = acc;
}

// One block per column: a standard shared-memory tree reduction of
// sum_i A[i,c]*B[i,c], with each thread first striding over n/blockDim.x
// elements before the intra-block reduction.
__global__ void batched_dot_kernel(const double* __restrict__ A,
                                   const double* __restrict__ B,
                                   int n,
                                   double* __restrict__ out) {
  extern __shared__ double sdata[];
  const int c = blockIdx.x;
  const double* Ac = A + static_cast<std::size_t>(c) * n;
  const double* Bc = B + static_cast<std::size_t>(c) * n;

  double sum = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    sum += Ac[i] * Bc[i];
  sdata[threadIdx.x] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[c] = sdata[0];
}

// Y[:,c] += alpha[c] * X[:,c], one thread per (i, c).
__global__ void batched_axpy_kernel(const double* __restrict__ alpha,
                                    const double* __restrict__ X,
                                    double* __restrict__ Y,
                                    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n)
    return;
  const std::size_t idx = static_cast<std::size_t>(c) * n + i;
  Y[idx] += alpha[c] * X[idx];
}

// P[:,c] = R[:,c] + beta[c] * P[:,c], one thread per (i, c) -- fused
// scal+axpy for CG's search-direction update (avoids a second pass over P).
__global__ void batched_update_p_kernel(const double* __restrict__ R,
                                        const double* __restrict__ beta,
                                        double* __restrict__ P,
                                        int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n)
    return;
  const std::size_t idx = static_cast<std::size_t>(c) * n + i;
  P[idx] = R[idx] + beta[c] * P[idx];
}

// Cached once per process (cudaDeviceGetAttribute is a host-side round trip
// to the driver, not worth repeating on every matvec call in a CG loop that
// may issue thousands of them per fit).
int smCount() {
  static const int count = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
      return 1;
    int n = 0;
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, device) != cudaSuccess || n <= 0)
      return 1;
    return n;
  }();
  return count;
}

// How many j-slices (blockIdx.z) to split the reduction into, given the
// (ceil(n/128), ncols) block count rmul_batched_kernel would use on its
// own. Only tiles when that base grid would leave the GPU under-filled;
// otherwise returns 1 (row_blocks*ncols alone already gives every SM
// enough work, and the tiled kernel's extra reduction pass + scratch
// memory would be pure overhead). Capped at 8: each additional j-slice
// adds a full n*ncols-sized scratch write/read pass (sum_partials_kernel),
// and empirically (see tests/KrigingIterativeTest.cpp's Nystrom-precond
// comparison) a higher cap bought more parallelism than it was worth once
// this project's own tests started noticing the added floating-point
// reassociation from more slices.
int chooseJBlocks(int n, int row_blocks, int ncols) {
  const int target_blocks = 4 * smCount();
  const int base_blocks = row_blocks * ncols;
  if (base_blocks >= target_blocks)
    return 1;
  int j_blocks = (target_blocks + base_blocks - 1) / base_blocks;
  if (j_blocks > 8)
    j_blocks = 8;
  if (j_blocks > n)
    j_blocks = n;
  return j_blocks < 1 ? 1 : j_blocks;
}

int computeJBlocks(int n, int ncols) {
  const dim3 block(128, 1, 1);
  const int row_blocks = (n + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
  return chooseJBlocks(n, row_blocks, ncols);
}

}  // namespace

// Required length (in doubles, not bytes) of the scratch buffer
// lk_cuda_rmul_batched_launch needs for a given (n, ncols) -- 0 if no
// tiling will happen (small-ncols/large-n calls where the untiled kernel
// already fills the GPU). n and ncols are fixed for an entire CG solve, so
// the caller (CudaLinearAlgebra.cpp) calls this ONCE per conjugateGradient
// invocation and allocates the scratch buffer once, rather than
// malloc/free-ing it on every one of a CG loop's potentially thousands of
// matvec calls.
extern "C" int lk_cuda_rmul_batched_scratch_elems(int n, int ncols) {
  const int j_blocks = computeJBlocks(n, ncols);
  return j_blocks <= 1 ? 0 : j_blocks * n * ncols;
}

// d_scratch must be non-null and sized (in doubles) at least
// lk_cuda_rmul_batched_scratch_elems(n, ncols) whenever that is > 0;
// ignored (may be null) otherwise.
extern "C" void lk_cuda_rmul_batched_launch(const double* d_Xt,
                                            int n,
                                            int dimX,
                                            const double* d_theta,
                                            int covKind,
                                            const double* d_P,
                                            int ncols,
                                            double* d_Ap,
                                            double* d_scratch) {
  const dim3 block(128, 1, 1);
  const int row_blocks = (n + static_cast<int>(block.x) - 1) / static_cast<int>(block.x);
  const int j_blocks = chooseJBlocks(n, row_blocks, ncols);

  if (j_blocks == 1) {
    const dim3 grid(row_blocks, static_cast<unsigned int>(ncols), 1);
    rmul_batched_kernel<<<grid, block>>>(d_Xt, n, dimX, d_theta, static_cast<CovKind>(covKind), d_P, ncols, d_Ap);
    return;
  }

  const int j_tile = (n + j_blocks - 1) / j_blocks;
  const dim3 tiled_grid(row_blocks, static_cast<unsigned int>(ncols), static_cast<unsigned int>(j_blocks));
  rmul_batched_tiled_kernel<<<tiled_grid, block>>>(
      d_Xt, n, dimX, d_theta, static_cast<CovKind>(covKind), d_P, ncols, d_scratch, j_tile);

  const dim3 reduce_grid(row_blocks, static_cast<unsigned int>(ncols), 1);
  sum_partials_kernel<<<reduce_grid, block>>>(d_scratch, n, ncols, j_blocks, d_Ap);
}

extern "C" void lk_cuda_batched_dot_launch(const double* d_A, const double* d_B, int n, int ncols, double* d_out) {
  const int block = 256;
  batched_dot_kernel<<<ncols, block, block * sizeof(double)>>>(d_A, d_B, n, d_out);
}

extern "C" void lk_cuda_batched_axpy_launch(const double* d_alpha, const double* d_X, double* d_Y, int n, int ncols) {
  const dim3 block(128, 1, 1);
  const dim3 grid((n + block.x - 1) / block.x, static_cast<unsigned int>(ncols), 1);
  batched_axpy_kernel<<<grid, block>>>(d_alpha, d_X, d_Y, n);
}

extern "C" void lk_cuda_batched_update_p_launch(const double* d_R,
                                                const double* d_beta,
                                                double* d_P,
                                                int n,
                                                int ncols) {
  const dim3 block(128, 1, 1);
  const dim3 grid((n + block.x - 1) / block.x, static_cast<unsigned int>(ncols), 1);
  batched_update_p_kernel<<<grid, block>>>(d_R, d_beta, d_P, n);
}

// --- CG per-iteration scalar updates, kept ON DEVICE ------------------------
// The CG loop's alpha/beta/convergence arithmetic is one value per column
// (ncols small). Doing it on the host meant a blocking cudaMemcpy of the
// ncols-long pAp / r.r vectors D2H and the alpha/beta vectors H2D on EVERY
// iteration -- ~4 device synchronizations per step. These kernels do that
// arithmetic in place on device buffers so the host only needs a single
// int ("any column still active?") copied back every few iterations. Each
// mirrors LinearAlgebra::conjugateGradient's / the host loop's logic
// exactly, including the pAp<=0 breakdown guard and the relative-residual
// convergence test.

__global__ void cg_alpha_kernel(const double* __restrict__ rz_old,
                                const double* __restrict__ pAp,
                                int ncols,
                                int* __restrict__ active,
                                double* __restrict__ alpha,
                                double* __restrict__ neg_alpha) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols)
    return;
  if (!active[c]) {
    alpha[c] = 0.0;
    neg_alpha[c] = 0.0;
    return;
  }
  const double q = pAp[c];
  if (!(q > 0.0)) {  // breakdown guard: freeze this column
    active[c] = 0;
    alpha[c] = 0.0;
    neg_alpha[c] = 0.0;
    return;
  }
  const double a = rz_old[c] / q;
  alpha[c] = a;
  neg_alpha[c] = -a;
}

__global__ void cg_beta_kernel(const double* __restrict__ rr_new,
                               const double* __restrict__ bnorm,
                               double tol,
                               int ncols,
                               int* __restrict__ active,
                               double* __restrict__ rz_old,
                               double* __restrict__ beta) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols)
    return;
  if (!active[c]) {
    beta[c] = 0.0;
    return;
  }
  const double rn = rr_new[c];
  if (sqrt(rn) / bnorm[c] < tol) {  // converged: freeze (rz_old left as-is, matches host 'continue')
    active[c] = 0;
    beta[c] = 0.0;
    return;
  }
  beta[c] = rn / rz_old[c];
  rz_old[c] = rn;
}

__global__ void cg_restart_kernel(const double* __restrict__ rr,
                                  const double* __restrict__ bnorm,
                                  double tol,
                                  int ncols,
                                  int* __restrict__ active,
                                  double* __restrict__ rz_old) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols)
    return;
  if (!active[c])
    return;
  rz_old[c] = rr[c];
  if (sqrt(rr[c]) / bnorm[c] < tol)
    active[c] = 0;
}

__global__ void cg_any_active_kernel(const int* __restrict__ active, int ncols, int* __restrict__ flag) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols)
    return;
  if (active[c])
    atomicOr(flag, 1);
}

extern "C" void lk_cuda_cg_alpha_launch(const double* d_rz_old,
                                        const double* d_pAp,
                                        int ncols,
                                        int* d_active,
                                        double* d_alpha,
                                        double* d_neg_alpha) {
  const int block = 128;
  cg_alpha_kernel<<<(ncols + block - 1) / block, block>>>(d_rz_old, d_pAp, ncols, d_active, d_alpha, d_neg_alpha);
}

extern "C" void lk_cuda_cg_beta_launch(const double* d_rr_new,
                                       const double* d_bnorm,
                                       double tol,
                                       int ncols,
                                       int* d_active,
                                       double* d_rz_old,
                                       double* d_beta) {
  const int block = 128;
  cg_beta_kernel<<<(ncols + block - 1) / block, block>>>(d_rr_new, d_bnorm, tol, ncols, d_active, d_rz_old, d_beta);
}

extern "C" void lk_cuda_cg_restart_launch(const double* d_rr,
                                          const double* d_bnorm,
                                          double tol,
                                          int ncols,
                                          int* d_active,
                                          double* d_rz_old) {
  const int block = 128;
  cg_restart_kernel<<<(ncols + block - 1) / block, block>>>(d_rr, d_bnorm, tol, ncols, d_active, d_rz_old);
}

extern "C" void lk_cuda_cg_any_active_launch(const int* d_active, int ncols, int* d_flag) {
  const int block = 128;
  cg_any_active_kernel<<<(ncols + block - 1) / block, block>>>(d_active, ncols, d_flag);
}

// --- Dense fast-path build: materialize R (+ optionally dR/dtheta) --------
// One thread per (i,j): both R[i,j] and R[j,i] are computed independently
// (no symmetry exploited, unlike the CPU build_separable_cov's
// thread-count-limited half-loop) -- with a GPU's thread count this is
// cheap, and it avoids any write-write hazard entirely. i==j needs no
// special case: Xi==Xj drives every kernel's u=|dx|/theta to 0, and
// lk_cov_pair/lk_dlncov_pair already give R=1 / dlncov=0 there on their
// own (same functions the matrix-free kernels above use per-pair, just
// paid ONCE here instead of once per CG iteration / Lanczos step).
__global__ void build_cov_kernel(const double* __restrict__ Xt,
                                 int n,
                                 int dimX,
                                 const double* __restrict__ theta,
                                 CovKind kind,
                                 double* __restrict__ R,
                                 double* __restrict__ dR) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n)
    return;

  const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
  const double* Xj = Xt + static_cast<std::size_t>(j) * dimX;
  const double c = lk_cov_pair(kind, Xi, Xj, theta, dimX);
  const std::size_t idx = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * n;
  if (R != nullptr)
    R[idx] = c;
  if (dR != nullptr) {
    double dln[LK_CUDA_MAX_DIMX];
    lk_dlncov_pair(kind, Xi, Xj, theta, dimX, dln);
    const std::size_t n2 = static_cast<std::size_t>(n) * n;
    for (int k = 0; k < dimX; ++k)
      dR[static_cast<std::size_t>(k) * n2 + idx] = c * dln[k];
  }
}

// --- Nystrom/Woodbury preconditioner apply, ON DEVICE --------------------
// z = Dinv .* (r - U * (M^-1 (U^T (Dinv .* r)))), M = Mchol Mchol^T, k x k.
// Matches LinearAlgebra::WoodburyFactorization::solve exactly (same
// factors, passed in from the host). Lets LLIterative(m,precond_rank) /
// predictIterative(use_nystrom_precond=True) run their CG on the GPU
// instead of falling back to the CPU path.

// DinvR[i,c] = Dinv[i] * R[i,c]  (n x ncols)
__global__ void scale_rows_kernel(const double* __restrict__ Dinv, const double* __restrict__ R, int n, double* __restrict__ Out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n)
    return;
  const std::size_t idx = static_cast<std::size_t>(c) * n + i;
  Out[idx] = Dinv[i] * R[idx];
}

// z[i,c] = Dinv[i] * (r[i,c] - Us[i,c])  (all n x ncols)
__global__ void precond_finish_kernel(const double* __restrict__ Dinv, const double* __restrict__ r, const double* __restrict__ Us, int n, double* __restrict__ z) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n)
    return;
  const std::size_t idx = static_cast<std::size_t>(c) * n + i;
  z[idx] = Dinv[i] * (r[idx] - Us[idx]);
}

// beta[c] = active ? rz_new[c]/rz_old[c] : 0 ; rz_old[c] = rz_new[c] ;
// convergence tested on the TRUE residual norm rr[c] (not rz). Matches the
// preconditioned branch of LinearAlgebra::conjugateGradient.
__global__ void cg_beta_precond_kernel(const double* __restrict__ rr, const double* __restrict__ rz_new, const double* __restrict__ bnorm, double tol, int ncols, int* __restrict__ active, double* __restrict__ rz_old, double* __restrict__ beta) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols)
    return;
  if (!active[c]) {
    beta[c] = 0.0;
    return;
  }
  if (sqrt(rr[c]) / bnorm[c] < tol) {
    active[c] = 0;
    beta[c] = 0.0;
    return;
  }
  beta[c] = rz_new[c] / rz_old[c];
  rz_old[c] = rz_new[c];
}

extern "C" void lk_cuda_scale_rows_launch(const double* d_Dinv, const double* d_R, int n, int ncols, double* d_Out) {
  const dim3 blk(128, 1, 1);
  const dim3 grid_n((n + blk.x - 1) / blk.x, static_cast<unsigned int>(ncols), 1);
  scale_rows_kernel<<<grid_n, blk>>>(d_Dinv, d_R, n, d_Out);
}

extern "C" void lk_cuda_precond_finish_launch(const double* d_Dinv, const double* d_r, const double* d_Us, int n,
                                              int ncols, double* d_z) {
  const dim3 blk(128, 1, 1);
  const dim3 grid_n((n + blk.x - 1) / blk.x, static_cast<unsigned int>(ncols), 1);
  precond_finish_kernel<<<grid_n, blk>>>(d_Dinv, d_r, d_Us, n, d_z);
}

extern "C" void lk_cuda_cg_beta_precond_launch(const double* d_rr, const double* d_rz_new, const double* d_bnorm,
                                               double tol, int ncols, int* d_active, double* d_rz_old, double* d_beta) {
  const int block = 128;
  cg_beta_precond_kernel<<<(ncols + block - 1) / block, block>>>(d_rr, d_rz_new, d_bnorm, tol, ncols, d_active, d_rz_old,
                                                                d_beta);
}

// Restart-iteration variant for preconditioned CG: rz_old <- rz[c] (the
// preconditioned inner product), converge on the true residual rr[c].
__global__ void cg_restart_precond_kernel(const double* __restrict__ rr, const double* __restrict__ rz, const double* __restrict__ bnorm, double tol, int ncols, int* __restrict__ active, double* __restrict__ rz_old) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncols || !active[c])
    return;
  rz_old[c] = rz[c];
  if (sqrt(rr[c]) / bnorm[c] < tol)
    active[c] = 0;
}

extern "C" void lk_cuda_cg_restart_precond_launch(const double* d_rr, const double* d_rz, const double* d_bnorm,
                                                  double tol, int ncols, int* d_active, double* d_rz_old) {
  const int block = 128;
  cg_restart_precond_kernel<<<(ncols + block - 1) / block, block>>>(d_rr, d_rz, d_bnorm, tol, ncols, d_active, d_rz_old);
}

// dimX must be <= LK_CUDA_MAX_DIMX (guaranteed by the host caller). d_Out
// must be sized n * dimX * ncols doubles.
extern "C" void lk_cuda_drmul_batched_launch(const double* d_Xt,
                                             int n,
                                             int dimX,
                                             const double* d_theta,
                                             int covKind,
                                             const double* d_V,
                                             int ncols,
                                             double* d_Out) {
  const dim3 block(128, 1, 1);
  const dim3 grid((n + block.x - 1) / block.x, static_cast<unsigned int>(ncols), 1);
  drmul_batched_kernel<<<grid, block>>>(d_Xt, n, dimX, d_theta, static_cast<CovKind>(covKind), d_V, ncols, d_Out);
}

// d_dR (when non-null) must be sized n*n*dimX doubles and dimX <= 32 (same
// bound as lk_cuda_drmul_batched_launch) -- checked by the host caller.
extern "C" void lk_cuda_build_cov_launch(const double* d_Xt,
                                         int n,
                                         int dimX,
                                         const double* d_theta,
                                         int covKind,
                                         double* d_R,
                                         double* d_dR) {
  const dim3 block(16, 16, 1);
  const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
  build_cov_kernel<<<grid, block>>>(d_Xt, n, dimX, d_theta, static_cast<CovKind>(covKind), d_R, d_dR);
}

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE
