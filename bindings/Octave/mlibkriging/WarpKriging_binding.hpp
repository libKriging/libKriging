#ifndef LIBKRIGING_BINDINGS_OCTAVE_WARPKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_WARPKRIGING_BINDING_HPP

#include <mex.h>

namespace WarpKrigingBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void copy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void is_fitted(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void feature_dim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void warping(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace WarpKrigingBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_WARPKRIGING_BINDING_HPP
