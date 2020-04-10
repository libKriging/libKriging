#include "mex.h"
/*
  clear mypow2
  b = randn (4,1) + 1i * randn (4,1);
  all (b.^2 == mypow2 (b))
 * [a,b]=func(a,a)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  mwSize n;
  mwIndex i;
  double *vri, *vro;

  if (nrhs != 1 || !mxIsDouble(prhs[0]))
    mexErrMsgTxt("ARG1 must be a double matrix");

  const char* nm = mexFunctionName();
  mexPrintf("You called function: %s\n", nm);
  mexPrintf("have %d inputs and %d outputs\n", nrhs, nlhs);

  const size_t nrow = mxGetM(prhs[0]);
  const size_t ncol = mxGetN(prhs[0]);
  const size_t ndim = mxGetNumberOfDimensions(prhs[0]);
  mexPrintf("Matrix have %d dimensions: %d x %d\n", ndim, nrow, ncol);

  n = mxGetNumberOfElements(prhs[0]);
  mxComplexity flag = (mxIsComplex(prhs[0])) ? mxCOMPLEX : mxREAL;

  plhs[0]
      = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]), mxGetDimensions(prhs[0]), mxGetClassID(prhs[0]), flag);
  vri = mxGetPr(prhs[0]);
  vro = mxGetPr(plhs[0]);

  if (mxIsComplex(prhs[0])) {
    double *vii, *vio;
    vii = mxGetPi(prhs[0]);
    vio = mxGetPi(plhs[0]);

    for (i = 0; i < n; i++) {
      vro[i] = vri[i] * vri[i] - vii[i] * vii[i];
      vio[i] = 2 * vri[i] * vii[i];
    }
  } else {
    for (i = 0; i < n; i++)
      vro[i] = vri[i] * vri[i];
  }

  //  for (int i = 0; i < n; ++i)
  //    vro[i] = i;
}