#include "Gpu_binding.hpp"

#include "libKriging/Gpu.hpp"

#include "tools/MxMapper.hpp"

namespace GpuBinding {

void compiled_backends(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, libKriging::gpu::compiled_backends(), "compiled_backends");
}

void available(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, libKriging::gpu::available(), "available");
}

void backend(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, libKriging::gpu::active_backend(), "backend");
}

void enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, libKriging::gpu::enabled(), "enabled");
}

void set_enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  libKriging::gpu::set_enabled(input.get<bool>(0, "enabled"));
}

}  // namespace GpuBinding
