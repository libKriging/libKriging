#ifndef LIBKRIGING_BINDINGS_OCTAVE_MULTIOUTPUTKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_MULTIOUTPUTKRIGING_BINDING_HPP

#include <mex.h>

namespace MultiOutputKrigingBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void build_empty(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void save(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void load(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void output_theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void set_output_coordinates(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void update_simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void leaveOneOutMat(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void leaveOneOut(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void leaveOneOutFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predictCovFactors(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void component(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void output_model(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void nb_outputs(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void Y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void output_coordinates(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void regmodel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void normalize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void optim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void objective(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void centerY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void scaleY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void beta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void output_cov(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void nb_components(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void pca_basis(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void pca_explained(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void pca_residual(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace MultiOutputKrigingBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_MULTIOUTPUTKRIGING_BINDING_HPP
