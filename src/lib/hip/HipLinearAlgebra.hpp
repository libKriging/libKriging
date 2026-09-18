// HIP/ROCm port of src/lib/cuda/CudaLinearAlgebra.cuh -- mechanical
// cuda*->hip* / .cuh->.hpp translation of the CUDA backend; identical
// algorithm, batching and preconditioner math.
//
// Verified 2026-09-13 on real hardware: AMD Radeon RX 6600 (gfx1032),
// Windows, ROCm/HIP SDK 6.2.4 (gfx1032 is not on AMD's officially supported
// GPU list for any HIP SDK release, but the 6.2.4 runtime detects and runs
// on it fine -- newer ROCm 7.x reportedly does not). Built with CMake's
// Ninja generator + plain (non -cl) Clang for C/CXX/HIP -- this machine had
// no VS/MSBuild HIP toolset integration, and CMake 3.31's HIP-language ABI
// detection has a real bug when the HIP compiler is clang-cl (no
// CMAKE_HIP_SIMULATE_VERSION branch in Windows-MSVC.cmake). All of
// LinearAlgebraTest, KrigingIterativeTest and KrigingPredictIterativeTest
// pass. This run also found and fixed a real bug (now fixed here and in the
// identical CUDA original, CudaLinearAlgebraKernel.cu): rmul_batched_tiled_kernel
// could leave a trailing d_partial slice unwritten when j_tile*j_blocks
// overshot n, and sum_partials_kernel summed that uninitialized device
// memory into the matvec result -- surfaced as a negative SSE (hence a NaN
// log-likelihood) in the LLIterative gradient/SLQ path at n=35, ncols=2 on
// this GPU's 14 CUs. Not yet exercised: SYCL/Metal remain UNVERIFIED, and
// this was only tried on gfx1032 with one small-n test suite -- other GPUs/
// problem sizes may still expose new issues.
#ifndef LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRA_HPP

// Only compiled/declared when the project is configured with
// -DENABLE_CUDA_ITERATIVE=ON (see root CMakeLists.txt). Kept behind this
// macro so a default (non-CUDA) build never sees CUDA types/symbols, and
// LIBKRIGING_USE_HIP_ITERATIVE is only ever defined by CMake, never
// auto-detected at configure time.
#ifdef LIBKRIGING_USE_HIP_ITERATIVE

#include "libKriging/utils/lk_armadillo.hpp"

#include <string>

#include "libKriging/libKriging_exports.h"

// GPU counterpart of LinearAlgebra::conjugateGradient, scoped to the
// matrix-free R*v matvec used by LLIterative (Kriging::_logLikelihoodIterative)
// and predictIterative (KrigingImpl::predictIterative_impl). R is never
// materialized on the GPU either -- see CudaLinearAlgebra.cu's rmul_kernel --
// matching the CPU path's O(n) memory invariant. Covers the SLQ
// log-determinant matvec, the dR/dtheta trace matvec, and the optional
// Nystrom/Woodbury CG preconditioner; callers fall back to
// LinearAlgebra::conjugateGradient only for an unsupported covType.
namespace LinearAlgebraHip {

// True iff a CUDA device was found at runtime (lazy-initialized, cached).
// Independent of the compile-time flag: a build compiled with CUDA support
// can still run on a machine with no GPU, in which case this is false and
// callers must fall back to the CPU path.
LIBKRIGING_EXPORT bool available();

// Runtime on/off switch, defaulting to available(). Exists so the same
// binary can compare CPU vs GPU (e.g. in a benchmark) without recompiling.
LIBKRIGING_EXPORT bool enabled();
LIBKRIGING_EXPORT void set_enabled(bool value);

// Covariance kernels with a device-side implementation (see
// CudaLinearAlgebra.cu): "gauss", "exp", "matern3_2", "matern5_2".
LIBKRIGING_EXPORT bool supports(const std::string& covType);

// Matrix-free CG solve of R(Xt,theta)*Y = B (R = correlation matrix implied
// by Xt (d x n, one point per column), theta and covType). Solves every
// column of B in lockstep -- one batched matvec + a couple of batched
// reductions per iteration cover all columns at once (see
// HipLinearAlgebraKernel.hip.cpp), rather than looping one column at a time --
// with the whole CG loop running on the GPU (single upload of Xt/theta/B,
// single download of the result, no host<->device round trip per
// iteration beyond the small per-column scalars CG's convergence check
// needs). A column that converges before others is frozen (its
// contribution to further updates zeroed) rather than dropped, since a
// batched launch can't cheaply shrink its own column count mid-loop. Same
// convergence contract as LinearAlgebra::conjugateGradient (relative
// residual < tol or max_iter iterations, periodic exact-residual restart).
// When precU is non-empty, runs PRECONDITIONED CG with the Nystrom/Woodbury
// preconditioner z = precDinv .* (r - precU M^-1 precU^T (precDinv .* r)),
// M = precMcholLower precMcholLower^T -- pass the factors straight from a
// LinearAlgebra::WoodburyFactorization (U(), Dinv(), McholLower()) so the
// GPU apply is bit-for-bit the same preconditioner as the CPU path.
// n_unconverged_out, when non-null, receives how many of B's columns were
// still active when the loop hit max_iter -- see
// CudaLinearAlgebra.cuh's matching parameter.
//
// X0 exists for signature parity with LinearAlgebraCuda::conjugateGradient
// (so the shared LK_ITER_GPU_BIND macro in Kriging.cpp compiles against
// whichever GPU backend is enabled) but is NOT YET honored here -- accepted
// and ignored, same effective behavior as x0=0 always. No HIP hardware was
// available to validate a real port of CudaLinearAlgebra.cpp's warm-start
// preamble (upload X0, matvec once for the exact initial residual, skip
// columns already under tol) when this parameter was added; that CUDA
// implementation is the reference to port.
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

// Batched matrix-free matvec: returns R(Xt,theta) * V (V is n x ncols,
// column-major; result n x ncols) in ONE device launch covering every
// column, R never materialized -- the same lk_hip_rmul_batched kernel the
// CG loop uses, exposed for callers that need R*V outside a CG solve
// (Stochastic Lanczos Quadrature log-determinant, Hutchinson trace). Xt is
// uploaded once per call; the O(n*ncols) result is the only download.
// Falls back to the caller's CPU path when covType is unsupported (same
// contract as conjugateGradient) -- callers must check supports() first.
LIBKRIGING_EXPORT arma::mat rmulBatched(const arma::mat& Xt,
                                        const arma::vec& theta,
                                        const std::string& covType,
                                        const arma::mat& V);

// Batched d(R)/d(theta) . V for the Hutchinson gradient trace: returns an
// n x (dimX * V.n_cols) matrix whose column-c block [c*dimX, (c+1)*dimX)
// is d(R)/d(theta) . V[:,c] (n x dimX), matching
// Kriging::_logLikelihoodIterative's CPU dRmul_all. Requires dimX <= 32 and
// a device-supported covType (callers check supports() / dimX first).
LIBKRIGING_EXPORT arma::mat dRmulBatched(const arma::mat& Xt,
                                         const arma::vec& theta,
                                         const std::string& covType,
                                         const arma::mat& V);

// Largest ARD dimension dRmulBatched supports (mirrors LK_HIP_MAX_DIMX in
// the kernel TU).
constexpr int kMaxDimX = 32;

}  // namespace LinearAlgebraHip

#endif  // LIBKRIGING_USE_HIP_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRA_HPP
