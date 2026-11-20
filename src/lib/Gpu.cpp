#include "libKriging/Gpu.hpp"

#include "cuda/CudaLinearAlgebra.cuh"    // no-op unless built with -DENABLE_CUDA_ITERATIVE=ON
#include "hip/HipLinearAlgebra.hpp"      // no-op unless built with -DENABLE_HIP_ITERATIVE=ON
#include "metal/MetalLinearAlgebra.hpp"  // no-op unless built with -DENABLE_METAL_ITERATIVE=ON
#include "sycl/SyclLinearAlgebra.hpp"    // no-op unless built with -DENABLE_SYCL_ITERATIVE=ON

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {
// Settings: default from the environment, read at load. Relaxed atomics:
// configuration-before-use, as Parallel's settings.
bool env_flag(const char* name) {
  return std::getenv(name) != nullptr;
}

bool env_enabled() {
  const char* e = std::getenv("LK_ITERATIVE_GPU");
  if (e == nullptr)
    return true;
  std::string v(e);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "off" || v == "false" || v == "no");
}

int env_int(const char* name, int dflt) {
  const char* e = std::getenv(name);
  if (e == nullptr)
    return dflt;
  try {
    const int v = std::stoi(e);
    return v >= 0 ? v : dflt;
  } catch (const std::exception&) {
    return dflt;
  }
}

std::atomic<bool> g_enabled{env_enabled()};
std::atomic<int> g_dense_max_mb{env_int("LK_GPU_DENSE_MAX_MB", 4096)};
std::atomic<bool> g_mixed_precision{env_flag("LK_GPU_MIXED_PRECISION")};
std::atomic<bool> g_verbose{env_flag("LK_GPU_VERBOSE")};
}  // namespace

LIBKRIGING_EXPORT std::string Gpu::compiled_backends() {
  std::string out;
  auto add = [&out](const char* name) {
    if (!out.empty())
      out += ",";
    out += name;
  };
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE
  add("cuda");
#endif
#ifdef LIBKRIGING_USE_HIP_ITERATIVE
  add("hip");
#endif
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE
  add("sycl");
#endif
#ifdef LIBKRIGING_USE_METAL_ITERATIVE
  add("metal");
#endif
  (void)add;
  return out;
}

LIBKRIGING_EXPORT bool Gpu::available() {
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE
  if (LinearAlgebraCuda::available())
    return true;
#endif
#ifdef LIBKRIGING_USE_HIP_ITERATIVE
  if (LinearAlgebraHip::available())
    return true;
#endif
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE
  if (LinearAlgebraSycl::available())
    return true;
#endif
#ifdef LIBKRIGING_USE_METAL_ITERATIVE
  if (LinearAlgebraMetal::available())
    return true;
#endif
  return false;
}

LIBKRIGING_EXPORT std::string Gpu::active_backend() {
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE
  if (LinearAlgebraCuda::enabled())
    return "cuda";
#endif
#ifdef LIBKRIGING_USE_HIP_ITERATIVE
  if (LinearAlgebraHip::enabled())
    return "hip";
#endif
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE
  if (LinearAlgebraSycl::enabled())
    return "sycl";
#endif
#ifdef LIBKRIGING_USE_METAL_ITERATIVE
  if (LinearAlgebraMetal::enabled())
    return "metal";
#endif
  return "none";
}

LIBKRIGING_EXPORT void Gpu::set_enabled(bool enabled) {
  g_enabled.store(enabled, std::memory_order_relaxed);
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE
  LinearAlgebraCuda::set_enabled(enabled && LinearAlgebraCuda::available());
#endif
#ifdef LIBKRIGING_USE_HIP_ITERATIVE
  LinearAlgebraHip::set_enabled(enabled && LinearAlgebraHip::available());
#endif
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE
  LinearAlgebraSycl::set_enabled(enabled && LinearAlgebraSycl::available());
#endif
#ifdef LIBKRIGING_USE_METAL_ITERATIVE
  LinearAlgebraMetal::set_enabled(enabled && LinearAlgebraMetal::available());
#endif
}

LIBKRIGING_EXPORT bool Gpu::get_enabled() {
  return g_enabled.load(std::memory_order_relaxed);
}

LIBKRIGING_EXPORT void Gpu::set_dense_max_mb(int mb) {
  if (mb < 0)
    throw std::invalid_argument("Gpu::set_dense_max_mb: expected mb >= 0");
  g_dense_max_mb.store(mb, std::memory_order_relaxed);
}

LIBKRIGING_EXPORT int Gpu::get_dense_max_mb() {
  return g_dense_max_mb.load(std::memory_order_relaxed);
}

LIBKRIGING_EXPORT void Gpu::set_mixed_precision(bool on) {
  g_mixed_precision.store(on, std::memory_order_relaxed);
}

LIBKRIGING_EXPORT bool Gpu::get_mixed_precision() {
  return g_mixed_precision.load(std::memory_order_relaxed);
}

LIBKRIGING_EXPORT void Gpu::set_verbose(bool on) {
  g_verbose.store(on, std::memory_order_relaxed);
}

LIBKRIGING_EXPORT bool Gpu::get_verbose() {
  return g_verbose.load(std::memory_order_relaxed);
}

namespace libKriging {
namespace gpu {
LIBKRIGING_EXPORT bool default_enabled(bool device_available) {
  return device_available && Gpu::get_enabled();
}
}  // namespace gpu
}  // namespace libKriging
