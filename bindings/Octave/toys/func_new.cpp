//
// Created by Pascal Havé on 08/04/2020.
//

#include "func_new.hpp"

#include <cstring>
#include <iostream>

#include "../tools/ObjectCollector.hpp"

void func_new(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {  // NOLINT (do not declare C-style arrays)
  if (nrhs != 0 || nlhs != 1)
    mexErrMsgTxt("No argument expected; one object reference as output");

  if (false) {  // as Cell

  } else {                                       // as Struct
    const char* fieldnames[2] = {"Id", "Type"};  // This will hold field names.
    plhs[0] = mxCreateStructMatrix(1, 1, 2, fieldnames);

    mxArray* id = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    mxArray* type = mxCreateString("libKriging Data Id");
    uint64_t * id_ptr = reinterpret_cast<uint64_t *>(mxGetData(id));
    *id_ptr = 42;

    mxSetFieldByNumber(plhs[0], 0, 0, id);
    mxSetFieldByNumber(plhs[0], 0, 1, type);

    // mexCallMATLAB(0, NULL, 1, &plhs[0], "disp");
  }
}

void func_delete(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 1 || nlhs != 0)
    mexErrMsgTxt("ref expected and no output");

  const char* fieldnames[2] = {"Id", "Type"};  // This will hold field names.

  if (nrhs != 1 || !mxIsStruct(prhs[0]) || mxGetNumberOfFields(prhs[0]) != 2)
    mexErrMsgTxt("ARG1 must be a 2-field struct");

  for (int i = 0; i < sizeof(fieldnames) / sizeof(fieldnames[0]); ++i) {
    if (mxGetNumberOfElements(prhs[0]) != 1)
      mexErrMsgTxt("Corrupted ref");
    const char* rhs_fieldname = mxGetFieldNameByNumber(prhs[0], i);
    if (std::strcmp(rhs_fieldname, fieldnames[i]) != 0)
      mexErrMsgTxt("Corrupted ref");
  }

  mxArray * mxtype =  mxGetFieldByNumber(prhs[0], 0, 1);
  char type[256];
  mxGetString(mxtype, type, 256);
  if (std::strcmp(type,"libKriging Data Id") !=0)
    mexErrMsgTxt("Corrupted ref");
  
  mxArray * mxid = mxGetFieldByNumber(prhs[0], 0, 0);
  const uint64_t id = *reinterpret_cast<uint64_t*>(mxGetData(mxid));
  ObjectCollector::unregisterObject(id);
}
