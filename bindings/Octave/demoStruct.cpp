#include "mex.h"

/*
 field1 = 'f1';  value1 = zeros(1,10);
 field2 = 'f2';  value2 = {'a', 'b'};
 field3 = 'f3';  value3 = {pi, pi.^2};
 field4 = 'f4';  value4 = {'fourth'};
 s = struct(field1,value1,field2,value2,field3,value3,field4,value4)
*/
void
mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
  int i;
  mwIndex j;
  mxArray *v;
  const char *keys[] = { "this", "that" };

  if (nrhs != 1 || ! mxIsStruct (prhs[0]))
    mexErrMsgTxt ("ARG1 must be a struct");

  for (i = 0; i < mxGetNumberOfFields (prhs[0]); i++)
    for (j = 0; j < mxGetNumberOfElements (prhs[0]); j++)
    {
      mexPrintf ("field %s(%d) = ", mxGetFieldNameByNumber (prhs[0], i), j);
      v = mxGetFieldByNumber (prhs[0], j, i);
      mexCallMATLAB (0, NULL, 1, &v, "disp");
    }

  v = mxCreateStructMatrix (2, 2, 2, keys);

  mxSetFieldByNumber (v, 0, 0, mxCreateString ("this1"));
  mxSetFieldByNumber (v, 0, 1, mxCreateString ("that1"));
  mxSetFieldByNumber (v, 1, 0, mxCreateString ("this2"));
  mxSetFieldByNumber (v, 1, 1, mxCreateString ("that2"));
  mxSetFieldByNumber (v, 2, 0, mxCreateString ("this3"));
  mxSetFieldByNumber (v, 2, 1, mxCreateString ("that3"));
  mxSetFieldByNumber (v, 3, 0, mxCreateString ("this4"));
  mxSetFieldByNumber (v, 3, 1, mxCreateString ("that4"));

  if (nlhs)
    plhs[0] = v;
}