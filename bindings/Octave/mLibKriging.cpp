#include <armadillo>
#include <cassert>
#include <map>
#include <memory>
#include <tuple>

#include "mex.h"  // cf https://fr.mathworks.com/help/
#include "tools/string_hash.hpp"
#include "toys/func1.hpp"
#include "toys/func_new.hpp"

void help_page() {
  mexPrintf(R"(
mLibKriging help page
  the Octave/Matlab interface for libKriging

  see https://github.com/MASCOTNUM/libKriging
  
)");
}

/*
 clear all
 y = randn (20,1);
 X = randn (20,1);
 X2 = randn (20,1);
 a=LinearRegression();
 a.fit(y,X);
 [y2,stderr] = a.predic(X2);

 load "y.mat"
 load "X.mat"
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) try {
  const char* nm = mexFunctionName();
#ifdef MEX_DEBUG
  mexPrintf("You called MEX function: %s\n", nm);
  mexPrintf("have %d inputs and %d outputs\n", nrhs, nlhs);
#endif
  if (std::strcmp(nm, "mLibKriging") != 0) {
    mexErrMsgTxt("you call mLibKriging with an illegal name");
  }

  if (nrhs < 1)
    return help_page();

  if (!mxIsChar(prhs[0]) || mxGetNumberOfDimensions(prhs[0]) != 2 || mxGetM(prhs[0]) != 1 || mxGetM(prhs[0]) != 1) {
    mexErrMsgTxt("arg1 should be a command name (scalar string)");
  }

  char command[256];
  if (mxGetString(prhs[0], command, 256) != 0) {
    mexErrMsgTxt("cannot decode command name");
  }

  switch (fnv_hash(command)) {
    case "help"_hash:
      return help_page();
    case "LinearRegression::new"_hash:
      return func_new(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::delete"_hash:
      return func_delete(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::fit"_hash:
      return func_fit(nlhs, plhs, nrhs - 1, prhs + 1);
    case "LinearRegression::predict"_hash:
      return func_predict(nlhs, plhs, nrhs - 1, prhs + 1);

    case "func1"_hash:
      return func1(nlhs, plhs, nrhs - 1, prhs + 1);
    default:
      mexErrMsgIdAndTxt("mLibKriging:NoRoute", "No route to such command [%s]", command);
  }
} catch (...) { // catch everything even end-of-scope event
  mexErrMsgIdAndTxt("mLibKriging:exception", "unexcepted exception");
}
