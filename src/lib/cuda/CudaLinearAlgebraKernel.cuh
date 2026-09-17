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

// d_tol is a PER-COLUMN device pointer (length ncols) -- lets a batched
// solve fuse right-hand sides that need different tolerances (e.g.
// Kriging.cpp's mBCG fusion of [F|y|probes]: F/y want the tighter cg_tol,
// probes want the looser probes_cg_tol) into ONE Krylov pass instead of a
// separate call per tolerance group, each column still freezing at its own
// tol independently (same lockstep-batching contract as everywhere else in
// this file). If active[c]: converged when sqrt(rr_new[c])/bnorm[c] <
// tol[c] (then active[c]=0, beta[c]=0, rz_old unchanged); else beta[c] =
// rr_new[c]/rz_old[c] and rz_old[c] = rr_new[c].
void lk_cuda_cg_beta_launch(const double* d_rr_new, const double* d_bnorm, const double* d_tol, int ncols,
                            int* d_active, double* d_rz_old, double* d_beta);

// Restart-iteration variant: if active[c], rz_old[c] = rr[c] and active[c]=0
// when converged (sqrt(rr[c])/bnorm[c] < tol[c], tol per-column as above).
void lk_cuda_cg_restart_launch(const double* d_rr, const double* d_bnorm, const double* d_tol, int ncols,
                               int* d_active, double* d_rz_old);

// d_flag (single int) must be zeroed by the caller first; set to 1 if any active[c] != 0.
void lk_cuda_cg_any_active_launch(const int* d_active, int ncols, int* d_flag);

// Batched d(R)/d(theta) . V : d_Out[i, k + c*dimX] = sum_{j!=i}
// d(R_ij)/d(theta_k) * V[j,c], for k in [0,dimX), one launch over all
// (i, c). d_Out is n x (dimX*ncols) column-major (see the .cu). dimX must
// be <= 32. Matches Kriging::_logLikelihoodIterative's CPU dRmul_all.
void lk_cuda_drmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                  const double* d_V, int ncols, double* d_Out);

// Dense fast-path build: fills d_R (n x n, column-major, element (i,j) at
// i + j*n; pass nullptr to skip -- dRmulBatched only needs d_dR) with
// R(Xt,theta)[i,j], for a SEPARABLE kernel -- gauss / exp / matern3_2 /
// matern5_2, the same four covKind is a code for everywhere else in this
// file. One thread per (i,j) pair with i <= j (the upper triangle
// including the diagonal): R is symmetric, so the launch computes each
// pair's (transcendental-heavy: exp/log1p) covariance value ONCE and
// writes it to both R[i,j] and R[j,i], instead of every thread in a full
// n x n grid evaluating its own cell independently and paying that cost
// twice per pair for no reason (lower-triangle threads simply return).
// The diagonal needs no special case beyond that: Xi==Xj makes every
// kernel's u=|dx|/theta term 0, and R(0)=1 falls out of lk_cov_pair on its
// own; writing "both" R[i,i] slots is one redundant but harmless store
// (same address, one owning thread, not a race). When d_dR is non-null,
// ALSO fills the dimX contiguous n x n blocks (both triangle slots, same
// reasoning) d_dR[k*n*n + i + j*n] = R[i,j] * d(ln Cov)/d(theta_k)[i,j]
// (same nonnegative log-derivative lk_dlncov_pair computes elsewhere in this
// file) -- requires dimX <= 32 (the caller must check, same bound
// lk_cuda_drmul_batched_launch already enforces). Lets
// CudaLinearAlgebra.cpp replace the matrix-free rmul_batched_kernel /
// drmul_batched_kernel calls with ONE build (paid once per host-side call,
// which already batches many columns) + a cublasDgemm per matvec, instead
// of recomputing every covariance entry's transcendentals on every
// iteration.
void lk_cuda_build_cov_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind, double* d_R,
                              double* d_dR);

// Nystrom/Woodbury preconditioner apply, elementwise halves only: the two
// GEMMs (U^T Z, U S) and the k x k triangular solve in between are issued
// from CudaLinearAlgebra.cpp as cublasDgemm/cublasDtrsm -- a hand-written
// kernel for those was one thread per right-hand-side column doing an O(k^2)
// strictly serial substitution (300 threads busy on a 132-SM device), which
// made the preconditioner an order of magnitude more expensive than the CG
// iterations it was meant to save.
//
// d_Out[i,c] = d_Dinv[i] * d_R[i,c]  (n x ncols)
void lk_cuda_scale_rows_launch(const double* d_Dinv, const double* d_R, int n, int ncols, double* d_Out);

// d_z[i,c] = d_Dinv[i] * (d_r[i,c] - d_Us[i,c])  (all n x ncols)
void lk_cuda_precond_finish_launch(const double* d_Dinv, const double* d_r, const double* d_Us, int n, int ncols,
                                   double* d_z);

// Preconditioned-CG beta update: beta[c] = rz_new/rz_old, rz_old <- rz_new,
// convergence tested on the TRUE residual norm rr[c]/bnorm[c] < d_tol[c]
// (per-column, see lk_cuda_cg_beta_launch's comment for why).
void lk_cuda_cg_beta_precond_launch(const double* d_rr, const double* d_rz_new, const double* d_bnorm,
                                    const double* d_tol, int ncols, int* d_active, double* d_rz_old, double* d_beta);

// Restart variant for preconditioned CG: rz_old <- rz[c], converge on rr[c]
// (per-column d_tol as above).
void lk_cuda_cg_restart_precond_launch(const double* d_rr, const double* d_rz, const double* d_bnorm,
                                       const double* d_tol, int ncols, int* d_active, double* d_rz_old);

// --- Device-resident batched Lanczos (Stochastic Lanczos Quadrature) ------
// Lets LinearAlgebraCuda::stochasticLogDetBatched keep every probe's Krylov
// vectors resident on the GPU across the whole recurrence (see
// CudaLinearAlgebra.cpp) instead of round-tripping the matvec result through
// the host every step to do reorthogonalization/bookkeeping there (the old
// scheme: LinearAlgebra::stochasticLogDetBatched called with rmulBatched as
// AmulBatched, each call its own upload/download).

// alpha[c] = active[c] ? dot[c] : 0 ; neg_alpha[c] = -alpha[c].
void lk_cuda_lanczos_alpha_launch(const double* d_dot, int ncols, const int* d_active, double* d_alpha,
                                  double* d_neg_alpha);

// Per-probe end-of-step bookkeeping, mirroring
// LinearAlgebra::stochasticLogDetBatched's host loop body exactly:
//   bj = sqrt(dot2[c])
//   if active[c]:
//     if bj < 1e-12: active[c] = 0 ; m_eff[c] = step_idx + 1  (invariant subspace, frozen from here on)
//     beta_out[c]      = (is_last_step || bj < 1e-12) ? 0 : bj        -- this step's T off-diagonal
//     inv_bj_out[c]    = (is_last_step || bj < 1e-12) ? 0 : 1/bj      -- normalizes w into V[:, step_idx+1]
//     neg_beta_prev_out[c] = (is_last_step || bj < 1e-12) ? 0 : -bj   -- feeds next step's w -= beta_prev*v_prev
//   else: beta_out[c] = inv_bj_out[c] = neg_beta_prev_out[c] = 0 (no-ops: that probe's V columns
//     stay exactly zero from m_eff[c] on, since the caller zero-initializes the whole V buffer
//     once and every later write to it is gated by inv_bj_out being 0 here).
// active/m_eff are both read-modify-write (int, length ncols); m_eff must be
// pre-filled with lanczos_steps (the "never went inactive" default) by the caller.
void lk_cuda_lanczos_beta_launch(const double* d_dot2, int ncols, int step_idx, int is_last_step, int* d_active,
                                 int* d_m_eff, double* d_beta_out, double* d_inv_bj_out, double* d_neg_beta_prev_out);

}  // extern "C"

#endif  // LIBKRIGING_USE_CUDA_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_CUDA_CUDALINEARALGEBRAKERNEL_CUH
