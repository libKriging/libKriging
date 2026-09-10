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

// --- CG per-iteration scalar updates, done ON DEVICE ----------------------
// All operate on ncols-length device vectors so the CG loop never has to
// round-trip pAp / r.r / alpha / beta through the host. See
// CudaLinearAlgebraKernel.cu for the exact per-column semantics (they
// mirror the host loop's, breakdown guard and relative-residual
// convergence test included). d_active is int (0/1) per column.

// alpha[c] = active[c] && pAp[c] > 0 ? rz_old[c]/pAp[c] : 0 ; neg_alpha[c] = -alpha[c].
// Marks active[c]=0 on the pAp[c] <= 0 breakdown.
void lk_cuda_cg_alpha_launch(const double* d_rz_old, const double* d_pAp, int ncols, int* d_active, double* d_alpha,
                             double* d_neg_alpha);

// If active[c]: converged when sqrt(rr_new[c])/bnorm[c] < tol (then active[c]=0, beta[c]=0,
// rz_old unchanged); else beta[c] = rr_new[c]/rz_old[c] and rz_old[c] = rr_new[c].
void lk_cuda_cg_beta_launch(const double* d_rr_new, const double* d_bnorm, double tol, int ncols, int* d_active,
                            double* d_rz_old, double* d_beta);

// Restart-iteration variant: if active[c], rz_old[c] = rr[c] and active[c]=0 when converged.
void lk_cuda_cg_restart_launch(const double* d_rr, const double* d_bnorm, double tol, int ncols, int* d_active,
                               double* d_rz_old);

// d_flag (single int) must be zeroed by the caller first; set to 1 if any active[c] != 0.
void lk_cuda_cg_any_active_launch(const int* d_active, int ncols, int* d_flag);

// Batched d(R)/d(theta) . V : d_Out[i, k + c*dimX] = sum_{j!=i}
// d(R_ij)/d(theta_k) * V[j,c], for k in [0,dimX), one launch over all
// (i, c). d_Out is n x (dimX*ncols) column-major (see the .cu). dimX must
// be <= 32. Matches Kriging::_logLikelihoodIterative's CPU dRmul_all.
void lk_cuda_drmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                  const double* d_V, int ncols, double* d_Out);

// Nystrom/Woodbury preconditioner apply: d_z = Dinv .* (r - U M^-1 U^T (Dinv .* r)),
// M = d_Mchol d_Mchol^T (k x k lower). d_U is n x k col-major, d_Dinv is n,
// d_r/d_z are n x ncols. d_scratch_nc must be >= n*ncols doubles,
// d_scratch_kc >= k*ncols. Matches LinearAlgebra::WoodburyFactorization::solve.
void lk_cuda_precond_apply_launch(const double* d_U, int n, int k, const double* d_Dinv, const double* d_Mchol,
                                  const double* d_r, int ncols, double* d_z, double* d_scratch_nc, double* d_scratch_kc);

// Preconditioned-CG beta update: beta[c] = rz_new/rz_old, rz_old <- rz_new,
// convergence tested on the TRUE residual norm rr[c]/bnorm[c].
void lk_cuda_cg_beta_precond_launch(const double* d_rr, const double* d_rz_new, const double* d_bnorm, double tol,
                                    int ncols, int* d_active, double* d_rz_old, double* d_beta);

// Restart variant for preconditioned CG: rz_old <- rz[c], converge on rr[c].
void lk_cuda_cg_restart_precond_launch(const double* d_rr, const double* d_rz, const double* d_bnorm, double tol,
                                       int ncols, int* d_active, double* d_rz_old);

}  // extern "C"

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRAKERNEL_CUH
