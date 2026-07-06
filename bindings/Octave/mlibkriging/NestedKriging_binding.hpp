#ifndef LIBKRIGING_BINDINGS_OCTAVE_NESTEDKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_NESTEDKRIGING_BINDING_HPP

#include <mex.h>

namespace NestedKrigingBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void aggregation(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void nb_groups(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void beta0(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace NestedKrigingBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_NESTEDKRIGING_BINDING_HPP
