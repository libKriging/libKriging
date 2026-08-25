#ifndef LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRAKERNEL_CUH
#define LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRAKERNEL_CUH

#ifdef LIBKRIGING_USE_CUDA_ITERATIVE

// Plain-C, pointer-only surface between the nvcc-compiled kernels
// (CudaLinearAlgebraKernel.cu) and the host-compiler-compiled orchestration
// code (CudaLinearAlgebra.cpp, built by the SAME compiler as the rest of
// libKriging). No Armadillo (or other non-POD C++) type may cross this
// boundary: nvcc and the project's host compiler can silently disagree on
// class layout for a nontrivial type like arma::Col/Mat (different feature-
// detection macros picked up while compiling Armadillo's headers), which
// showed up in practice as arma::vec::memptr() returning a null pointer on
// the nvcc side while n_elem still read correctly -- a genuine ABI mismatch,
// not a logic bug. Raw pointers/ints/doubles have no such ambiguity.
//
// All of these are BATCHED across a matrix's ncols columns in one launch,
// rather than looping one column at a time on the host: conjugateGradient's
// CG loop (CudaLinearAlgebra.cpp) solves every column of a right-hand-side
// matrix B "in lockstep" (columns that converge early are frozen via a
// zeroed alpha/beta rather than dropped, since CUDA has no cheap per-column
// early-exit mid-launch), replacing what used to be ncols independent CG
// loops -- each launching its own matvec + several cuBLAS calls, so ncols
// separate rounds of kernel-launch and host-sync overhead -- with ONE
// matvec + a couple of reductions per iteration, regardless of ncols.
// Profiling (see CudaLinearAlgebraKernel.cu's rmul_batched_kernel comment)
// found kernel-launch/host-sync overhead, not FLOPs, as the dominant cost
// at the n this project targets, which is exactly what batching amortizes.
extern "C" {

// Length (in doubles) of the scratch buffer lk_cuda_rmul_batched_launch
// needs for a given (n, ncols); 0 if no scratch is needed. n and ncols are
// fixed for a whole CG solve, so call this ONCE per conjugateGradient
// invocation and allocate the scratch buffer once (see
// CudaLinearAlgebra.cpp), not on every matvec call inside the CG loop.
int lk_cuda_rmul_batched_scratch_elems(int n, int ncols);

// covKind: 0=gauss, 1=exp, 2=matern3_2, 3=matern5_2 (LinearAlgebraCuda's
// CovKind enum in CudaLinearAlgebra.cpp, passed through as a plain int).
// Xt is (dimX x n), column-major (element (k,j) at Xt[k + j*dimX]) -- same
// layout as arma::mat::memptr(), so the host side can upload it verbatim.
// P/Ap are (n x ncols), column-major (column c at P + c*n) -- same layout
// as an arma::mat's memptr(). Ap[:,c] = R(Xt,theta)*P[:,c] for every column
// c in one launch; R is never materialized (O(n) device memory, matching
// the CPU Rmul's invariant). d_scratch must be sized (in doubles) at least
// lk_cuda_rmul_batched_scratch_elems(n, ncols) whenever that's > 0 (may be
// null otherwise). All pointers are device pointers. Launches
// asynchronously; the caller is responsible for checking
// cudaGetLastError()/cudaDeviceSynchronize() afterward.
void lk_cuda_rmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                 const double* d_P, int ncols, double* d_Ap, double* d_scratch);

// out[c] = sum_i A[i,c]*B[i,c] for every column c (A, B are n x ncols,
// column-major); pass the same pointer for A and B to get a per-column
// squared-norm. out is a device pointer of length ncols.
void lk_cuda_batched_dot_launch(const double* d_A, const double* d_B, int n, int ncols, double* d_out);

// Y[:,c] += alpha[c] * X[:,c] for every column c. alpha is a device pointer
// of length ncols; X, Y are n x ncols, column-major.
void lk_cuda_batched_axpy_launch(const double* d_alpha, const double* d_X, double* d_Y, int n, int ncols);

// P[:,c] = R[:,c] + beta[c] * P[:,c] for every column c (fused scal+axpy
// for CG's search-direction update). beta is a device pointer of length
// ncols; R, P are n x ncols, column-major.
void lk_cuda_batched_update_p_launch(const double* d_R, const double* d_beta, double* d_P, int n, int ncols);

}  // extern "C"

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRAKERNEL_CUH
