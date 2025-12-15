#ifndef LIBKRIGING_BINDINGS_OCTAVE_NUGGETKRIGING_BINDING_HPP
#define LIBKRIGING_BINDINGS_OCTAVE_NUGGETKRIGING_BINDING_HPP

#include <mex.h>

namespace NuggetKrigingBinding {
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void copy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void update_simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void save(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void load(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logMargPostFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void logMargPost(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void covMat(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void model(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void optim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void objective(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void centerX(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void scaleX(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void centerY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void scaleY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void normalize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void regmodel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void F(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void T(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void M(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void z(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void beta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void is_beta_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void is_theta_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void is_sigma2_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void nugget(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void is_nugget_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
}  // namespace NuggetKrigingBinding

#endif  // LIBKRIGING_BINDINGS_OCTAVE_NUGGETKRIGING_BINDING_HPP
