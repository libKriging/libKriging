#include "CudaLinearAlgebraKernel.cuh"

#ifdef LIBKRIGING_USE_CUDA_ITERATIVE

#include <cuda_runtime.h>

#include <cstddef>

namespace {

enum class CovKind : int { Gauss = 0, Exp = 1, Matern32 = 2, Matern52 = 3 };

__device__ __forceinline__ double lk_cov_pair(CovKind kind, const double* __restrict__ Xi,
                                              const double* __restrict__ Xj, const double* __restrict__ theta,
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
__global__ void rmul_batched_kernel(const double* __restrict__ Xt, int n, int dimX,
                                    const double* __restrict__ theta, CovKind kind, const double* __restrict__ P,
                                    int ncols, double* __restrict__ Ap) {
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
__global__ void rmul_batched_tiled_kernel(const double* __restrict__ Xt, int n, int dimX,
                                          const double* __restrict__ theta, CovKind kind,
                                          const double* __restrict__ P, int ncols, double* __restrict__ d_partial,
                                          int j_tile) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  const int jb = blockIdx.z;
  const int j_start = jb * j_tile;
  if (i >= n || c >= ncols || j_start >= n)
    return;
  const int j_end = min(j_start + j_tile, n);

  const double* Pc = P + static_cast<std::size_t>(c) * n;
  const double* Xi = Xt + static_cast<std::size_t>(i) * dimX;
  double partial = (jb == 0) ? Pc[i] : 0.0;  // diag = 1, added exactly once
  for (int j = j_start; j < j_end; ++j) {
    if (j == i)
      continue;
    const double* Xj = Xt + static_cast<std::size_t>(j) * dimX;
    partial += lk_cov_pair(kind, Xi, Xj, theta, dimX) * Pc[j];
  }
  d_partial[(static_cast<std::size_t>(jb) * ncols + c) * n + i] = partial;
}

// One thread per (row i, column c): sums the j_blocks partial sums for
// that cell in fixed jb = 0, 1, ..., j_blocks-1 order (NOT whatever order
// threads happen to finish rmul_batched_tiled_kernel in), so the result is
// reproducible across repeated calls with identical inputs -- see the
// determinism note on rmul_batched_tiled_kernel above.
__global__ void sum_partials_kernel(const double* __restrict__ d_partial, int n, int ncols, int j_blocks,
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
__global__ void batched_dot_kernel(const double* __restrict__ A, const double* __restrict__ B, int n,
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
__global__ void batched_axpy_kernel(const double* __restrict__ alpha, const double* __restrict__ X,
                                    double* __restrict__ Y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y;
  if (i >= n)
    return;
  const std::size_t idx = static_cast<std::size_t>(c) * n + i;
  Y[idx] += alpha[c] * X[idx];
}

// P[:,c] = R[:,c] + beta[c] * P[:,c], one thread per (i, c) -- fused
// scal+axpy for CG's search-direction update (avoids a second pass over P).
__global__ void batched_update_p_kernel(const double* __restrict__ R, const double* __restrict__ beta,
                                        double* __restrict__ P, int n) {
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
extern "C" void lk_cuda_rmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                            const double* d_P, int ncols, double* d_Ap, double* d_scratch) {
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
  rmul_batched_tiled_kernel<<<tiled_grid, block>>>(d_Xt, n, dimX, d_theta, static_cast<CovKind>(covKind), d_P, ncols,
                                                   d_scratch, j_tile);

  const dim3 reduce_grid(row_blocks, static_cast<unsigned int>(ncols), 1);
  sum_partials_kernel<<<reduce_grid, block>>>(d_scratch, n, ncols, j_blocks, d_Ap);
}

extern "C" void lk_cuda_batched_dot_launch(const double* d_A, const double* d_B, int n, int ncols, double* d_out) {
  const int block = 256;
  batched_dot_kernel<<<ncols, block, block * sizeof(double)>>>(d_A, d_B, n, d_out);
}

extern "C" void lk_cuda_batched_axpy_launch(const double* d_alpha, const double* d_X, double* d_Y, int n,
                                            int ncols) {
  const dim3 block(128, 1, 1);
  const dim3 grid((n + block.x - 1) / block.x, static_cast<unsigned int>(ncols), 1);
  batched_axpy_kernel<<<grid, block>>>(d_alpha, d_X, d_Y, n);
}

extern "C" void lk_cuda_batched_update_p_launch(const double* d_R, const double* d_beta, double* d_P, int n,
                                                int ncols) {
  const dim3 block(128, 1, 1);
  const dim3 grid((n + block.x - 1) / block.x, static_cast<unsigned int>(ncols), 1);
  batched_update_p_kernel<<<grid, block>>>(d_R, d_beta, d_P, n);
}

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE
