#include "LinearRegression_binding.hpp"
#include "mex.h"  // cf https://fr.mathworks.com/help/
#include "mlibKriging_exports.h"
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
MLIBKRIGING_EXPORT
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

    default:
      throw MxException(LOCATION(), "mLibKriging:noRoute", "No route to such command [", command, "]");
  }
} catch (MxException& e) {
  mexErrMsgIdAndTxt(e.id, e.msg.c_str());
} catch (...) {  // catch everything even end-of-scope event
  mexErrMsgIdAndTxt("mLibKriging:exception", "unexcepted exception");
}
