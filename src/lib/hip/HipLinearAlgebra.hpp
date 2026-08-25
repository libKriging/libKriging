#ifndef LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRA_HPP
#define LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRA_HPP

// UNVERIFIED: written as a careful mechanical port of the CUDA backend
// (src/lib/cuda/CudaLinearAlgebra.cuh) with no way to compile or run HIP
// code in the environment this was written in (no ROCm toolchain, no AMD
// GPU). Every file in the CUDA backend was build-and-execute verified, and
// that process caught two real bugs (a determinism regression from
// atomicAdd, a wrong Woodbury-solve formula) that code review alone did
// not -- this file has had no equivalent verification. Treat it as a
// starting point that needs a real ROCm build + AMD GPU run before trusting
// it, not as validated as the CUDA counterpart.
//
// Only compiled/declared when the project is configured with
// -DENABLE_HIP_ITERATIVE=ON (see root CMakeLists.txt). Kept behind this
// macro so a default (non-HIP) build never sees HIP types/symbols, and
// LIBKRIGING_USE_HIP_ITERATIVE is only ever defined by CMake, never
// auto-detected at configure time.
#ifdef LIBKRIGING_USE_HIP_ITERATIVE

#include "libKriging/utils/lk_armadillo.hpp"

#include <string>

#include "libKriging/libKriging_exports.h"

// AMD/ROCm counterpart of LinearAlgebraCuda (src/lib/cuda/CudaLinearAlgebra.cuh),
// scoped to the same matrix-free R*v matvec used by LLIterative
// (Kriging::_logLikelihoodIterative) and predictIterative
// (KrigingImpl::predictIterative_impl). R is never materialized on the GPU
// either -- see HipLinearAlgebraKernel.hip.cpp's rmul_batched_kernel --
// matching the CPU path's O(n) memory invariant. Only handles the
// no-preconditioner case; callers fall back to LinearAlgebra::conjugateGradient
// otherwise (unsupported covType, or a Nystrom preconditioner requested).
//
// Solves every column of B "in lockstep": one batched matvec + a couple of
// batched reductions per CG iteration cover all columns at once, rather
// than looping one column at a time. A column that converges before others
// is frozen (its contribution to further updates zeroed) rather than
// dropped, since a batched launch can't cheaply shrink its own column
// count mid-loop. Same convergence contract as
// LinearAlgebra::conjugateGradient (relative residual < tol or max_iter
// iterations, periodic exact-residual restart).
namespace LinearAlgebraHip {

// True iff a ROCm/HIP device was found at runtime (lazy-initialized,
// cached). Independent of the compile-time flag: a build compiled with HIP
// support can still run on a machine with no AMD GPU, in which case this is
// false and callers must fall back to the CPU path.
LIBKRIGING_EXPORT bool available();

// Runtime on/off switch, defaulting to available(). Exists so the same
// binary can compare CPU vs GPU (e.g. in a benchmark) without recompiling.
LIBKRIGING_EXPORT bool enabled();
LIBKRIGING_EXPORT void set_enabled(bool value);

// Covariance kernels with a device-side implementation (see
// HipLinearAlgebraKernel.hip.cpp): "gauss", "exp", "matern3_2", "matern5_2".
LIBKRIGING_EXPORT bool supports(const std::string& covType);

// Matrix-free CG solve of R(Xt,theta)*Y = B (R = correlation matrix implied
// by Xt (d x n, one point per column), theta and covType), the whole CG
// loop running on the GPU (single upload of Xt/theta/B, single download of
// the result, no host<->device round trip per iteration beyond the small
// per-column scalars CG's convergence check needs).
LIBKRIGING_EXPORT arma::mat conjugateGradient(const arma::mat& Xt,
                                              const arma::vec& theta,
                                              const std::string& covType,
                                              const arma::mat& B,
                                              arma::uword max_iter,
                                              double tol = 1e-8);

}  // namespace LinearAlgebraHip

#endif  // LIBKRIGING_USE_HIP_ITERATIVE

#endif  // LIBKRIGING_SRC_LIB_HIP_HIPLINEARALGEBRA_HPP
