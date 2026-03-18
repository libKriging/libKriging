#ifndef LIBKRIGING_BINDINGS_OCTAVE_LINEARREGRESSION_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_LINEARREGRESSION_BINDING_HPP

#include <mex.h>

namespace LinearRegressionBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace LinearRegressionBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_LINEARREGRESSION_BINDING_HPP
