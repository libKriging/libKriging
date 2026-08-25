#ifndef LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRAKERNEL_HPP
#define LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRAKERNEL_HPP

#ifdef LIBKRIGING_USE_HIP_ITERATIVE

// UNVERIFIED port of src/lib/cuda/CudaLinearAlgebraKernel.cuh -- see
// HipLinearAlgebra.hpp for why. Plain-C, pointer-only surface between the
// hipcc-compiled kernels (HipLinearAlgebraKernel.hip.cpp) and the
// host-compiler-compiled orchestration code (HipLinearAlgebra.cpp, built
// by the SAME compiler as the rest of libKriging). No Armadillo (or other
// non-POD C++) type may cross this boundary: the HIP compiler and the
// project's host compiler can silently disagree on class layout for a
// nontrivial type like arma::Col/Mat (this bit the CUDA backend in
// practice -- arma::vec::memptr() came back null on the nvcc side despite
// n_elem reading correctly -- a genuine ABI mismatch, not a logic bug, and
// nothing about HIP's toolchain makes that risk go away). Raw
// pointers/ints/doubles have no such ambiguity.
//
// All of these are BATCHED across a matrix's ncols columns in one launch,
// rather than looping one column at a time on the host -- see
// CudaLinearAlgebraKernel.cuh's file comment for the full rationale (the
// HIP and CUDA kernels are otherwise line-for-line the same algorithm).
extern "C" {

// Length (in doubles) of the scratch buffer lk_hip_rmul_batched_launch
// needs for a given (n, ncols); 0 if no scratch is needed. n and ncols are
// fixed for a whole CG solve, so call this ONCE per conjugateGradient
// invocation and allocate the scratch buffer once (see
// HipLinearAlgebra.cpp), not on every matvec call inside the CG loop.
int lk_hip_rmul_batched_scratch_elems(int n, int ncols);

// covKind: 0=gauss, 1=exp, 2=matern3_2, 3=matern5_2 (LinearAlgebraHip's
// CovKind enum in HipLinearAlgebra.cpp, passed through as a plain int).
// Xt is (dimX x n), column-major (element (k,j) at Xt[k + j*dimX]) -- same
// layout as arma::mat::memptr(), so the host side can upload it verbatim.
// P/Ap are (n x ncols), column-major (column c at P + c*n) -- same layout
// as an arma::mat's memptr(). Ap[:,c] = R(Xt,theta)*P[:,c] for every column
// c in one launch; R is never materialized (O(n) device memory, matching
// the CPU Rmul's invariant). d_scratch must be sized (in doubles) at least
// lk_hip_rmul_batched_scratch_elems(n, ncols) whenever that's > 0 (may be
// null otherwise). All pointers are device pointers. Launches
// asynchronously; the caller is responsible for checking
// hipGetLastError()/hipDeviceSynchronize() afterward.
void lk_hip_rmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                const double* d_P, int ncols, double* d_Ap, double* d_scratch);

// out[c] = sum_i A[i,c]*B[i,c] for every column c (A, B are n x ncols,
// column-major); pass the same pointer for A and B to get a per-column
// squared-norm. out is a device pointer of length ncols.
void lk_hip_batched_dot_launch(const double* d_A, const double* d_B, int n, int ncols, double* d_out);

// Y[:,c] += alpha[c] * X[:,c] for every column c. alpha is a device pointer
// of length ncols; X, Y are n x ncols, column-major.
void lk_hip_batched_axpy_launch(const double* d_alpha, const double* d_X, double* d_Y, int n, int ncols);

// P[:,c] = R[:,c] + beta[c] * P[:,c] for every column c (fused scal+axpy
// for CG's search-direction update). beta is a device pointer of length
// ncols; R, P are n x ncols, column-major.
void lk_hip_batched_update_p_launch(const double* d_R, const double* d_beta, double* d_P, int n, int ncols);

}  // extern "C"

#endif  // LIBKRIGING_USE_HIP_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRAKERNEL_HPP
