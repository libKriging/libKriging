#include "Kriging_binding.hpp"
#include "LinearRegression_binding.hpp"
#include "NuggetKriging_binding.hpp"
#include "mex.h"  // cf https://fr.mathworks.com/help/
#include "tools/MxException.hpp"
#include "tools/MxMapper.hpp"
#include "tools/string_hash.hpp"

void help_page() {
  mexPrintf(R"(
mLibKriging help page
  the Octave/Matlab interface for libKriging

  see https://github.com/libKriging/libKriging
  
)");
}

/*
 clear all
 y = randn (20,1);
 X = randn (20,1);
 X2 = randn (20,1);
 a=LinearRegression();
 a.fit(y,X);
 [y2,stderr] = a.predict(X2);

 load "y.mat"
 load "X.mat"
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) try {
  const char* nm = mexFunctionName();
#ifdef MEX_DEBUG
  mexPrintf("  You called MEX function: %s\n", nm);
  mexPrintf("  with %d inputs and %d outputs\n", nrhs, nlhs);
  // mexCallMATLAB(0, NULL, 1, (mxArray**)&prhs[0], "disp");
#endif
  if (std::strcmp(nm, "mLibKriging") != 0) {
    throw MxException(LOCATION(), "mLibKriging:badMexFunction", "you call mLibKriging with an illegal name");
  }

  if (nrhs < 1)
    return help_page();

  MxMapper input{"Input", std::min(1, nrhs), const_cast<mxArray**>(prhs)};
  std::string command = input.get<0, std::string>("command");

#ifdef MEX_DEBUG
  mexPrintf("  with command: %s\n", command.c_str());
#endif

  switch (fnv_hash(command)) {
    case "help"_hash:
      return help_page();
    case "LinearRegression::new"_hash:
      return LinearRegressionBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::delete"_hash:
      return LinearRegressionBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::fit"_hash:
      return LinearRegressionBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::predict"_hash:
      return LinearRegressionBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);

    case "Kriging::new"_hash:
      return KrigingBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::delete"_hash:
      return KrigingBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::fit"_hash:
      return KrigingBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::predict"_hash:
      return KrigingBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::simulate"_hash:
      return KrigingBinding::simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::update"_hash:
      return KrigingBinding::update(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::summary"_hash:
      return KrigingBinding::summary(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::leaveOneOut"_hash:
      return KrigingBinding::leaveOneOut(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::logLikelihood"_hash:
      return KrigingBinding::logLikelihood(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::logMargPost"_hash:
      return KrigingBinding::logMargPost(nlhs, plhs, nrhs - 1, prhs + 1);

    case "Kriging::kernel"_hash:
      return KrigingBinding::kernel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::optim"_hash:
      return KrigingBinding::optim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::objective"_hash:
      return KrigingBinding::objective(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::X"_hash:
      return KrigingBinding::X(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::centerX"_hash:
      return KrigingBinding::centerX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::scaleX"_hash:
      return KrigingBinding::scaleX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::y"_hash:
      return KrigingBinding::y(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::centerY"_hash:
      return KrigingBinding::centerY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::scaleY"_hash:
      return KrigingBinding::scaleY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::regmodel"_hash:
      return KrigingBinding::regmodel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::F"_hash:
      return KrigingBinding::F(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::T"_hash:
      return KrigingBinding::T(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::M"_hash:
      return KrigingBinding::M(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::z"_hash:
      return KrigingBinding::z(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::beta"_hash:
      return KrigingBinding::beta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_beta_estim"_hash:
      return KrigingBinding::is_beta_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::theta"_hash:
      return KrigingBinding::theta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_theta_estim"_hash:
      return KrigingBinding::is_theta_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::sigma2"_hash:
      return KrigingBinding::sigma2(nlhs, plhs, nrhs - 1, prhs + 1);
    case "Kriging::is_sigma2_estim "_hash:
      return KrigingBinding::is_sigma2_estim(nlhs, plhs, nrhs - 1, prhs + 1);

    case "NuggetKriging::new"_hash:
      return NuggetKrigingBinding::build(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::delete"_hash:
      return NuggetKrigingBinding::destroy(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::fit"_hash:
      return NuggetKrigingBinding::fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::predict"_hash:
      return NuggetKrigingBinding::predict(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::simulate"_hash:
      return NuggetKrigingBinding::simulate(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::update"_hash:
      return NuggetKrigingBinding::update(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::summary"_hash:
      return NuggetKrigingBinding::summary(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::logLikelihood"_hash:
      return NuggetKrigingBinding::logLikelihood(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::logMargPost"_hash:
      return NuggetKrigingBinding::logMargPost(nlhs, plhs, nrhs - 1, prhs + 1);

    case "NuggetKriging::kernel"_hash:
      return NuggetKrigingBinding::kernel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::optim"_hash:
      return NuggetKrigingBinding::optim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::objective"_hash:
      return NuggetKrigingBinding::objective(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::X"_hash:
      return NuggetKrigingBinding::X(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::centerX"_hash:
      return NuggetKrigingBinding::centerX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::scaleX"_hash:
      return NuggetKrigingBinding::scaleX(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::y"_hash:
      return NuggetKrigingBinding::y(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::centerY"_hash:
      return NuggetKrigingBinding::centerY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::scaleY"_hash:
      return NuggetKrigingBinding::scaleY(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::regmodel"_hash:
      return NuggetKrigingBinding::regmodel(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::F"_hash:
      return NuggetKrigingBinding::F(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::T"_hash:
      return NuggetKrigingBinding::T(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::M"_hash:
      return NuggetKrigingBinding::M(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::z"_hash:
      return NuggetKrigingBinding::z(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::beta"_hash:
      return NuggetKrigingBinding::beta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::is_beta_estim"_hash:
      return NuggetKrigingBinding::is_beta_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::theta"_hash:
      return NuggetKrigingBinding::theta(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::is_theta_estim"_hash:
      return NuggetKrigingBinding::is_theta_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::sigma2"_hash:
      return NuggetKrigingBinding::sigma2(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::is_sigma2_estim "_hash:
      return NuggetKrigingBinding::is_sigma2_estim(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::nugget"_hash:
      return NuggetKrigingBinding::nugget(nlhs, plhs, nrhs - 1, prhs + 1);
    case "NuggetKriging::is_nugget_estim "_hash:
      return NuggetKrigingBinding::is_nugget_estim(nlhs, plhs, nrhs - 1, prhs + 1);

    default:
      throw MxException(LOCATION(), "mLibKriging:noRoute", "No route to such command [", command, "]");
  }
} catch (MxException& e) {
  mexErrMsgIdAndTxt(e.id, e.msg.c_str());
} catch (std::exception& e) {
  mexErrMsgIdAndTxt("mLibKriging:kernelException", e.what());
} catch (...) {  // catch everything even end-of-scope event
  mexErrMsgIdAndTxt("mLibKriging:exception", "unexpected exception");
}
