#ifndef LIBKRIGING_BINDINGS_OCTAVE_LINEARREGRESSION_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_LINEARREGRESSION_BINDING_HPP

#include <mex.h>

namespace LinearRegressionBinding {
void build(int nlhs, void** plhs, int nrhs, const void** prhs);
void destroy(int nlhs, void** plhs, int nrhs, const void** prhs);
void fit(int nlhs, void** plhs, int nrhs, const void** prhs);
void predict(int nlhs, void** plhs, int nrhs, const void** prhs);
}  // namespace LinearRegressionBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_LINEARREGRESSION_BINDING_HPP
