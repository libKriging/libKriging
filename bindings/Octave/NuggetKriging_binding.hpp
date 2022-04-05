#ifndef LIBKRIGING_BINDINGS_OCTAVE_NUGGETKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_NUGGETKRIGING_BINDING_HPP

#include <mex.h>

namespace NuggetKrigingBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logMargPost(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace NuggetKrigingBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_NUGGETKRIGING_BINDING_HPP
