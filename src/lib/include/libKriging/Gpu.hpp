#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_GPU_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_GPU_HPP

#include <string>

#include "libKriging/libKriging_exports.h"

// GPU acceleration of the iterative method (method "iterative(...)": CG
// solves, SLQ log-determinant, Hutchinson gradient of fit / predict / update).
// Always compiled: in a build without a GPU backend (the default, CPU-only
// packages) the queries report "" / "none" / false and the settings have no
// effect, so every binding exposes the same calls whatever the build.
//
// Backends, in the dispatch order: CUDA > HIP > SYCL > Metal. A backend is
// available when it is compiled in and finds a usable device at run time
// (CUDA: and a loadable cuBLAS).
//
// Process-wide, configuration-before-use as Parallel's settings
// (Parallel.hpp): set them before the calls, not while calls run in other
// threads. Each setting reads its default from an environment variable.
class Gpu {
 public:
  // --- read only
  /// Comma-separated GPU backends compiled into this build ("cuda", "hip",
  /// "sycl", "metal"), "" when none.
  LIBKRIGING_EXPORT static std::string compiled_backends();
  /// True when a compiled-in backend found a usable device.
  LIBKRIGING_EXPORT static bool available();
  /// Backend the iterative method dispatches to ("cuda", "hip", "sycl",
  /// "metal"), "none" for the CPU path.
  LIBKRIGING_EXPORT static std::string active_backend();

  // --- settings
  /// Use the GPU when available (default true, false when LK_ITERATIVE_GPU is
  /// 0 / off / false / no). Enabling without a usable device keeps the CPU
  /// path: a call never fails because of a missing GPU.
  LIBKRIGING_EXPORT static void set_enabled(bool enabled);
  LIBKRIGING_EXPORT static bool get_enabled();
  /// Device-memory budget, in MiB, of the dense correlation matrix kept on the
  /// device (one matrix product per CG iteration instead of a covariance
  /// evaluation per pair); 0 forces the matrix-free kernels. Default 4096, or
  /// LK_GPU_DENSE_MAX_MB.
  LIBKRIGING_EXPORT static void set_dense_max_mb(int mb);
  LIBKRIGING_EXPORT static int get_dense_max_mb();
  /// Mixed precision (TF32 / fp32 products, fp64 correction) of the dense CG
  /// products, CUDA only. Default false, or LK_GPU_MIXED_PRECISION set.
  LIBKRIGING_EXPORT static void set_mixed_precision(bool on);
  LIBKRIGING_EXPORT static bool get_mixed_precision();
  /// Diagnostic of each GPU CG call (iterations used, columns not converged)
  /// on stderr. Default false, or LK_GPU_VERBOSE set.
  LIBKRIGING_EXPORT static void set_verbose(bool on);
  LIBKRIGING_EXPORT static bool get_verbose();
};

namespace libKriging {
namespace gpu {
// Initial on / off state of a backend whose device availability is
// `device_available`: Gpu::get_enabled() && device_available. Used by the
// backends (src/lib/cuda, hip, sycl, metal) on their first call.
LIBKRIGING_EXPORT bool default_enabled(bool device_available);
}  // namespace gpu
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_GPU_HPP
