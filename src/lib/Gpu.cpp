#include "libKriging/Gpu.hpp"

#include "cuda/CudaLinearAlgebra.cuh"    // no-op unless built with -DENABLE_CUDA_ITERATIVE=ON/AUTO
#include "hip/HipLinearAlgebra.hpp"      // no-op unless built with -DENABLE_HIP_ITERATIVE=ON
#include "metal/MetalLinearAlgebra.hpp"  // no-op unless built with -DENABLE_METAL_ITERATIVE=ON
#include "sycl/SyclLinearAlgebra.hpp"    // no-op unless built with -DENABLE_SYCL_ITERATIVE=ON

#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace libKriging {
namespace gpu {

std::string compiled_backends() {
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

bool available() {
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

std::string active_backend() {
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

bool enabled() {
  return active_backend() != "none";
}

void set_enabled(bool value) {
#ifdef LIBKRIGING_USE_CUDA_ITERATIVE
  LinearAlgebraCuda::set_enabled(value && LinearAlgebraCuda::available());
#endif
#ifdef LIBKRIGING_USE_HIP_ITERATIVE
  LinearAlgebraHip::set_enabled(value && LinearAlgebraHip::available());
#endif
#ifdef LIBKRIGING_USE_SYCL_ITERATIVE
  LinearAlgebraSycl::set_enabled(value && LinearAlgebraSycl::available());
#endif
#ifdef LIBKRIGING_USE_METAL_ITERATIVE
  LinearAlgebraMetal::set_enabled(value && LinearAlgebraMetal::available());
#endif
  (void)value;
}

bool default_enabled(bool device_available) {
  if (!device_available)
    return false;
  const char* e = std::getenv("LK_ITERATIVE_GPU");
  if (e == nullptr)
    return true;
  std::string v(e);
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return !(v == "0" || v == "off" || v == "false" || v == "no");
}

}  // namespace gpu
}  // namespace libKriging
