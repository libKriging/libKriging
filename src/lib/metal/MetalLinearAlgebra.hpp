#ifndef LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRA_HPP

// UNVERIFIED Apple-Metal port of src/lib/cuda/CudaLinearAlgebra.cuh -- see
// src/lib/hip/HipLinearAlgebra.hpp for the "never compiled or run" caveat
// (no macOS / Metal toolchain here), which applies here with one EXTRA,
// load-bearing caveat:
//
//   Metal Shading Language has NO double on Apple-Silicon GPUs (M1/M2/M3
//   and later are float32 / float16 only). So this backend runs the entire
//   iterative path -- the matrix-free R*v matvec, the CG recurrence, the
//   SLQ Lanczos, the Woodbury apply -- in FLOAT32. That is a genuine
//   numerical difference from the double CUDA/HIP/SYCL backends, not a
//   mechanical detail: CG on an ill-conditioned interpolation kernel loses
//   digits fast in single precision, so predictIterative's `tol` floors
//   itself at ~1e-4 here (see MetalLinearAlgebra.cpp) and the SLQ
//   log-determinant is correspondingly noisier. Treat this as a
//   proof-of-portability skeleton that needs a real design review (mixed
//   precision? Kahan sums? refuse ill-conditioned theta?) before use, not
//   as a drop-in equal of the other backends.
//
// Only compiled/declared with -DENABLE_METAL_ITERATIVE=ON (macOS + Metal +
// metal-cpp headers via -DMETAL_CPP_DIR). LIBKRIGING_USE_METAL_ITERATIVE is
// only ever defined by CMake.
#ifdef LIBKRIGING_USE_METAL_ITERATIVE

#include "libKriging/utils/lk_armadillo.hpp"

#include <string>

#include "libKriging/libKriging_exports.h"

// Same public API as LinearAlgebraCuda / LinearAlgebraHip / LinearAlgebraSycl
// (double at the boundary; the float32 truncation happens inside). See the
// CUDA header for the per-function contract.
namespace LinearAlgebraMetal {

LIBKRIGING_EXPORT bool available();
LIBKRIGING_EXPORT bool enabled();
LIBKRIGING_EXPORT void set_enabled(bool value);
LIBKRIGING_EXPORT bool supports(const std::string& covType);

// n_unconverged_out, when non-null, receives how many of B's columns were
// still active when the loop hit max_iter -- see
// CudaLinearAlgebra.cuh's matching parameter.
//
// X0, when non-null, warm-starts the solve from that initial guess instead
// of x0=0 -- ported from CudaLinearAlgebra.cpp's warm-start preamble
// (upload x0, r = b - A*x0, rz_old computed on-device from the real
// residual, columns already within tol start inactive).
//
// tol is PER-COLUMN (length B.n_cols): each column freezes independently
// once it reaches ITS OWN tol[c], letting a caller fuse right-hand-side
// groups that need different tolerances (e.g. Kriging.cpp's mBCG fusion of
// [F|y|probes] -- F/y want the tighter cg_tol, probes want the looser
// probes_cg_tol) into ONE Krylov pass instead of a separate call per group
// -- matches LinearAlgebraCuda::conjugateGradient's per-column tol vector.
LIBKRIGING_EXPORT arma::mat conjugateGradient(const arma::mat& Xt,
                                              const arma::vec& theta,
                                              const std::string& covType,
                                              const arma::mat& B,
                                              arma::uword max_iter,
                                              const arma::vec& tol,
                                              const arma::mat& precU = arma::mat(),
                                              const arma::vec& precDinv = arma::vec(),
                                              const arma::mat& precMcholLower = arma::mat(),
                                              arma::uword* n_unconverged_out = nullptr,
                                              const arma::mat* X0 = nullptr);

// Scalar-tol convenience overload for the common case (every column shares
// one tolerance) -- broadcasts into the per-column vector above.
LIBKRIGING_EXPORT arma::mat conjugateGradient(const arma::mat& Xt,
                                              const arma::vec& theta,
                                              const std::string& covType,
                                              const arma::mat& B,
                                              arma::uword max_iter,
                                              double tol = 1e-8,
                                              const arma::mat& precU = arma::mat(),
                                              const arma::vec& precDinv = arma::vec(),
                                              const arma::mat& precMcholLower = arma::mat(),
                                              arma::uword* n_unconverged_out = nullptr,
                                              const arma::mat* X0 = nullptr);

LIBKRIGING_EXPORT arma::mat rmulBatched(const arma::mat& Xt,
                                        const arma::vec& theta,
                                        const std::string& covType,
                                        const arma::mat& V);

LIBKRIGING_EXPORT arma::mat dRmulBatched(const arma::mat& Xt,
                                         const arma::vec& theta,
                                         const std::string& covType,
                                         const arma::mat& V);

constexpr int kMaxDimX = 32;

// Stochastic Lanczos Quadrature log-determinant estimate for R(Xt,theta),
// device-resident: unlike calling LinearAlgebra::stochasticLogDetBatched
// with rmulBatched as its AmulBatched callback (every Lanczos step then
// uploads that step's probe vectors, computes R*V, and downloads the
// result so the host can do the next step's reorthogonalization/dot
// products in Armadillo before uploading again), this keeps EVERY probe's
// entire Krylov history resident on the device for the whole recurrence --
// the matvec, the dot products, the full reorthogonalization and the
// per-step bookkeeping all run without leaving the GPU (batched onto the
// same pending-command-buffer machinery as conjugateGradient -- see
// MetalLinearAlgebraKernel.cpp). Only nprobe*lanczos_steps scalars
// (alpha/beta) come back to the host, once, after the last step, for the
// final small tridiagonal eigendecompositions (still done on host via
// arma::eig_sym -- O(nprobe*lanczos_steps^2), not worth porting). Same
// estimator, same fixed-seed probes, same full-reorthogonalization
// numerics as the CPU/ping-pong version -- performance-only, not an
// algorithm change. Ported from LinearAlgebraHip::stochasticLogDetBatched
// (itself a mechanical port of the CUDA version, with the
// cublasDgemmStridedBatched reorthogonalization pair replaced by two
// hand-written kernels since neither HIP nor Metal carries a BLAS
// dependency here) -- see lk_metal_kernels.metal's lanczos_reorth_dot/sub.
//
// UNPRECONDITIONED ONLY: does not accept a Woodbury-whitened operator (same
// scope limitation as the CUDA/HIP versions) -- Kriging.cpp falls back to
// the CPU-orchestrated LinearAlgebra::stochasticLogDetBatched (with this
// namespace's rmulBatched as AmulBatched, whitened by
// WoodburyFactorization::whitenL/whitenLt on the host in between) when a
// Nystrom preconditioner is active.
LIBKRIGING_EXPORT double stochasticLogDetBatched(const arma::mat& Xt,
                                                 const arma::vec& theta,
                                                 const std::string& covType,
                                                 arma::uword lanczos_steps,
                                                 const arma::mat& probes);

}  // namespace LinearAlgebraMetal

#endif  // LIBKRIGING_USE_METAL_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_METAL_METALLINEARALGEBRA_HPP
