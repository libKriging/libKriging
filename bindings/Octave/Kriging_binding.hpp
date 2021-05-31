#ifndef LIBKRIGING_BINDINGS_OCTAVE_KRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_KRIGING_BINDING_HPP

#include <mex.h>

namespace KrigingBinding {
void build(int nlhs, void** plhs, int nrhs, const void** prhs);
void destroy(int nlhs, void** plhs, int nrhs, const void** prhs);
void fit(int nlhs, void** plhs, int nrhs, const void** prhs);
void predict(int nlhs, void** plhs, int nrhs, const void** prhs);
void simulate(int nlhs, void** plhs, int nrhs, const void** prhs);
void update(int nlhs, void** plhs, int nrhs, const void** prhs);
void describeModel(int nlhs, void** plhs, int nrhs, const void** prhs);
void leaveOneOut(int nlhs, void** plhs, int nrhs, const void** prhs);
void logLikelihood(int nlhs, void** plhs, int nrhs, const void** prhs);
void logMargPost(int nlhs, void** plhs, int nrhs, const void** prhs);
}  // namespace KrigingBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_KRIGING_BINDING_HPP
