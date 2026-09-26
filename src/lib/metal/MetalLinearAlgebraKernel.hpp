#ifndef LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRAKERNEL_HPP
#define LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRAKERNEL_HPP

// UNVERIFIED Apple-Metal port of src/lib/cuda/CudaLinearAlgebraKernel.cuh --
// see src/lib/metal/MetalLinearAlgebra.hpp for the full caveat (never
// compiled or run; FLOAT32-only because MSL has no double on Apple GPUs).
//
// Plain-C, pointer-only surface: MetalLinearAlgebra.cpp (host orchestration,
// armadillo) is compiled by the normal compiler and only calls these
// lk_metal_* functions; all Metal / metal-cpp symbols stay in
// MetalLinearAlgebraKernel.cpp. Device "pointers" here are MTL::Buffer*
// handles cast to void* -- opaque to the host file, never dereferenced.
//
// Buffers hold FLOAT32 (or int32). The host file must convert: the
// lk_metal_upload_/download_ helpers do the double<->float narrowing so the
// truncation is explicit and localised, not a silent raw memcpy.
#ifdef LIBKRIGING_USE_METAL_ITERATIVE

extern "C" {

int lk_metal_available(void);
void* lk_metal_malloc(unsigned long bytes);          // bytes already float-sized by the caller
void lk_metal_free(void* dbuf);
void lk_metal_upload_f64_as_f32(void* dbuf, const double* host, unsigned long count);
void lk_metal_download_f32_as_f64(double* host, const void* dbuf, unsigned long count);
void lk_metal_upload_i32(void* dbuf, const int* host, unsigned long count);
void lk_metal_download_i32(int* host, const void* dbuf, unsigned long count);
void lk_metal_memset_dev(void* dbuf, int value, unsigned long bytes);
void lk_metal_copy_dev(void* dst, const void* src, unsigned long bytes);  // device->device, same dtype

// Batched kernels: same contract / column-major layout as lk_cuda_* (see
// CudaLinearAlgebraKernel.cuh). No tiling -> _scratch_elems returns 0 and
// the scratch arg is unused. Each launch ENCODES onto a shared pending
// command buffer rather than committing immediately -- see
// MetalLinearAlgebraKernel.cpp's Ctx comment; only lk_metal_upload_*/
// lk_metal_download_* actually commit+wait (flush).
int lk_metal_rmul_batched_scratch_elems(int n, int ncols);
// p_byte_offset: byte offset of d_P's ncols columns into whatever buffer
// d_P actually points at (0 for a standalone n x ncols buffer; a Lanczos
// step's block offset into the device-resident history buffer for
// stochasticLogDetBatched -- see MetalLinearAlgebra.cpp).
void lk_metal_rmul_batched_launch(const void* d_Xt, int n, int dimX, const void* d_theta, int covKind, const void* d_P,
                                  int ncols, void* d_Ap, void* d_scratch, long p_byte_offset = 0);
// a_byte_offset/b_byte_offset: same sub-buffer-offset use as
// lk_metal_rmul_batched_launch's p_byte_offset, for d_A/d_B respectively.
void lk_metal_batched_dot_launch(const void* d_A, const void* d_B, int n, int ncols, void* d_out,
                                 long a_byte_offset = 0, long b_byte_offset = 0);
// x_byte_offset/y_byte_offset: same sub-buffer-offset use as
// lk_metal_rmul_batched_launch's p_byte_offset, for d_X/d_Y respectively.
void lk_metal_batched_axpy_launch(const void* d_alpha, const void* d_X, void* d_Y, int n, int ncols,
                                  long x_byte_offset = 0, long y_byte_offset = 0);
void lk_metal_batched_update_p_launch(const void* d_R, const void* d_beta, void* d_P, int n, int ncols);
void lk_metal_cg_alpha_launch(const void* d_rz_old, const void* d_pAp, int ncols, void* d_active, void* d_alpha,
                              void* d_neg_alpha);
// d_tol: PER-COLUMN tolerance buffer (length ncols), not a scalar -- see
// lk_metal_kernels.metal's CgP/cg_beta doc comment.
void lk_metal_cg_beta_launch(const void* d_rr_new, const void* d_bnorm, const void* d_tol, int ncols, void* d_active,
                             void* d_rz_old, void* d_beta);
void lk_metal_cg_restart_launch(const void* d_rr, const void* d_bnorm, const void* d_tol, int ncols, void* d_active,
                                void* d_rz_old);
void lk_metal_cg_any_active_launch(const void* d_active, int ncols, void* d_flag);
void lk_metal_drmul_batched_launch(const void* d_Xt, int n, int dimX, const void* d_theta, int covKind, const void* d_V,
                                   int ncols, void* d_Out);
void lk_metal_precond_apply_launch(const void* d_U, int n, int k, const void* d_Dinv, const void* d_Mchol,
                                   const void* d_r, int ncols, void* d_z, void* d_scratch_nc, void* d_scratch_kc);
void lk_metal_cg_beta_precond_launch(const void* d_rr, const void* d_rz_new, const void* d_bnorm, const void* d_tol,
                                     int ncols, void* d_active, void* d_rz_old, void* d_beta);
void lk_metal_cg_restart_precond_launch(const void* d_rr, const void* d_rz, const void* d_bnorm, const void* d_tol,
                                        int ncols, void* d_active, void* d_rz_old);

// Device-resident batched Lanczos (see MetalLinearAlgebra.hpp's
// stochasticLogDetBatched doc comment) -- mirrors lk_hip_lanczos_*.
void lk_metal_lanczos_alpha_launch(const void* d_dot, int ncols, int step_idx, const void* d_active,
                                   void* d_alpha_all, void* d_neg_alpha);
void lk_metal_lanczos_beta_launch(const void* d_dot2, int ncols, int step_idx, int is_last_step, void* d_active,
                                  void* d_m_eff, void* d_beta_out, void* d_inv_bj_out, void* d_neg_beta_prev_out);
void lk_metal_lanczos_reorth_dot_launch(const void* d_V, const void* d_W, int n, int npr, int ls, int m, void* d_t);
void lk_metal_lanczos_reorth_sub_launch(const void* d_V, const void* d_t, int n, int npr, int ls, int m, void* d_W);

// Dense-R fast path (see MetalLinearAlgebra.cpp's DenseCovCache): builds
// R once (build_cov), then dense_matvec is a cheap FMA sweep against it
// instead of rmul_batched's per-pair covariance re-evaluation.
void lk_metal_build_cov_launch(const void* d_Xt, int n, int dimX, const void* d_theta, int covKind, void* d_R);
// v_byte_offset: same sub-buffer-offset use as lk_metal_rmul_batched_launch's
// p_byte_offset, for d_V.
void lk_metal_dense_matvec_launch(const void* d_R, int n, const void* d_V, int ncols, void* d_Out, int ldc,
                                  long v_byte_offset = 0);

}  // extern "C"

#endif  // LIBKRIGING_USE_METAL_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRAKERNEL_HPP
