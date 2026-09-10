#ifndef LIBKRIGING_SRC_LIB_SYCL_SYCLLINEARALGEBRAKERNEL_HPP
#define LIBKRIGING_SRC_LIB_SYCL_SYCLLINEARALGEBRAKERNEL_HPP

// UNVERIFIED SYCL / Intel oneAPI port of src/lib/cuda/CudaLinearAlgebraKernel.cuh
// -- see src/lib/sycl/SyclLinearAlgebra.hpp for the full caveat. Structural
// translation of the CUDA kernels (the CUDA/HIP backends were build-and-run
// verified; this one has no oneAPI toolchain or Intel GPU to compile or run
// against). SYCL's programming model differs (single-source, USM, an
// explicit in-order queue), so unlike the HIP port this is a rewrite, not a
// rename -- the algorithm, batching and preconditioner math are the same.
//
// This is still the plain-C, pointer-only surface: SyclLinearAlgebra.cpp
// (host orchestration, armadillo types) is compiled by the SAME compiler as
// the rest of libKriging, and only ever calls these lk_sycl_* functions --
// the SYCL headers stay confined to SyclLinearAlgebraKernel.cpp, compiled
// by the SYCL compiler. Same ABI-safety rule as the CUDA backend.
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE

extern "C" {

// --- device memory / sync surface (so SyclLinearAlgebra.cpp needs no SYCL
// header). All allocations/copies use one process-wide in-order GPU queue;
// the memcpy `kind` (0 = host->device, 1 = device->host, 2 = device->device)
// is informational -- every call waits for completion, matching the
// strategic sync points of the CUDA host code. lk_sycl_available() is 1 iff
// a SYCL GPU queue could be constructed at first use.
int lk_sycl_available(void);
void* lk_sycl_malloc(unsigned long bytes);
void lk_sycl_free(void* p);
void lk_sycl_memcpy(void* dst, const void* src, unsigned long bytes, int kind);
void lk_sycl_memset(void* p, int value, unsigned long bytes);
void lk_sycl_sync(void);

// --- the batched kernels (same contract and column-major layout as the
// CUDA lk_cuda_* family; see CudaLinearAlgebraKernel.cuh). d_scratch is
// unused (this port does not tile the reduction) and lk_sycl_..._scratch_elems
// always returns 0.
int lk_sycl_rmul_batched_scratch_elems(int n, int ncols);
void lk_sycl_rmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                 const double* d_P, int ncols, double* d_Ap, double* d_scratch);
void lk_sycl_batched_dot_launch(const double* d_A, const double* d_B, int n, int ncols, double* d_out);
void lk_sycl_batched_axpy_launch(const double* d_alpha, const double* d_X, double* d_Y, int n, int ncols);
void lk_sycl_batched_update_p_launch(const double* d_R, const double* d_beta, double* d_P, int n, int ncols);

void lk_sycl_cg_alpha_launch(const double* d_rz_old, const double* d_pAp, int ncols, int* d_active, double* d_alpha,
                             double* d_neg_alpha);
void lk_sycl_cg_beta_launch(const double* d_rr_new, const double* d_bnorm, double tol, int ncols, int* d_active,
                            double* d_rz_old, double* d_beta);
void lk_sycl_cg_restart_launch(const double* d_rr, const double* d_bnorm, double tol, int ncols, int* d_active,
                               double* d_rz_old);
void lk_sycl_cg_any_active_launch(const int* d_active, int ncols, int* d_flag);

void lk_sycl_drmul_batched_launch(const double* d_Xt, int n, int dimX, const double* d_theta, int covKind,
                                  const double* d_V, int ncols, double* d_Out);

void lk_sycl_precond_apply_launch(const double* d_U, int n, int k, const double* d_Dinv, const double* d_Mchol,
                                  const double* d_r, int ncols, double* d_z, double* d_scratch_nc, double* d_scratch_kc);
void lk_sycl_cg_beta_precond_launch(const double* d_rr, const double* d_rz_new, const double* d_bnorm, double tol,
                                    int ncols, int* d_active, double* d_rz_old, double* d_beta);
void lk_sycl_cg_restart_precond_launch(const double* d_rr, const double* d_rz, const double* d_bnorm, double tol,
                                       int ncols, int* d_active, double* d_rz_old);

}  // extern "C"

#endif  // LIBKRIGING_USE_SYCL_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_SYCL_SYCLLINEARALGEBRAKERNEL_HPP
