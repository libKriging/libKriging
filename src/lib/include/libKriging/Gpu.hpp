#ifndef LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_GPU_HPP
#define LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_GPU_HPP

// Backend-agnostic runtime switch for the GPU-accelerated iterative path
// (LLIterative fit solves and predictIterative). Always compiled -- in a
// build without any GPU backend every query simply reports "none"/false and
// set_enabled() is a no-op -- so every binding can expose the same three
// calls regardless of how libKriging was configured.
//
// Backends are tried in a fixed priority order, CUDA > HIP > SYCL > Metal
// (the same order Kriging.cpp/KrigingImpl.cpp dispatch in). A backend is
// "available" when it was compiled in AND finds a usable device (plus, for
// CUDA, a loadable cuBLAS) at runtime; it starts "enabled" iff available,
// unless the LK_ITERATIVE_GPU environment variable says otherwise.

#include <string>

#include "libKriging/libKriging_exports.h"

namespace libKriging {
namespace gpu {

// Comma-separated list of the GPU backends compiled into this build
// ("cuda", "hip", "sycl", "metal"), or "" when none.
LIBKRIGING_EXPORT std::string compiled_backends();

// True iff at least one compiled-in backend found a usable device.
LIBKRIGING_EXPORT bool available();

// Name of the backend the iterative path currently dispatches to ("cuda",
// "hip", "sycl", "metal"), or "none" (CPU path).
LIBKRIGING_EXPORT std::string active_backend();

// True iff active_backend() != "none".
LIBKRIGING_EXPORT bool enabled();

// Turn every compiled-in backend on or off at runtime. Turning on a backend
// whose device is not available is ignored (it stays off), so
// set_enabled(true) never makes a call fail on a machine without a GPU.
LIBKRIGING_EXPORT void set_enabled(bool value);

// Initial on/off state for a backend whose device availability is
// `device_available`: false when LK_ITERATIVE_GPU is set to 0/off/false/no
// (case-insensitive), device_available otherwise. Used by each backend's
// own enabled() on first call; exposed for those backends, not for users.
LIBKRIGING_EXPORT bool default_enabled(bool device_available);

}  // namespace gpu
}  // namespace libKriging

#endif  // LIBKRIGING_SRC_LIB_INCLUDE_LIBKRIGING_GPU_HPP
