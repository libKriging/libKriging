#ifndef LIBKRIGING_BINDINGS_OCTAVE_PARAMS_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_PARAMS_BINDING_HPP

#include <mex.h>

namespace ParamsBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void display(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace ParamsBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_PARAMS_BINDING_HPP
