#ifndef LIBKRIGING_BINDINGS_OCTAVE_GPU_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_GPU_BINDING_HPP

#include <mex.h>

namespace GpuBinding {
void compiled_backends(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void available(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void backend(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_enabled(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace GpuBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_GPU_BINDING_HPP
