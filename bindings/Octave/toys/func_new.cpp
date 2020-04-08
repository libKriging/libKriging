//
// Created by Pascal Havé on 08/04/2020.
//

#include "func_new.hpp"

#include <armadillo>
#include <cstring>
#include <iostream>

#include "../tools/ObjectCollector.hpp"
#include "libKriging/LinearRegression.hpp"
#include "relative_error.hpp"

void func_new(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {  // NOLINT (do not declare C-style arrays)
  if (nrhs != 1 || nlhs != 0)
    mexErrMsgTxt("No argument expected; one object reference as output");

  mxArray* objectRef = mxGetProperty(prhs[0], 0, "ref");
  if (objectRef == nullptr)
    mexErrMsgIdAndTxt("mLibKriging:badObject", "object does not contain 'ref' property");

  if (!mxIsEmpty(objectRef)) {
    mexErrMsgIdAndTxt("mLibKriging:alreadyBuilt", "object already contain a non empty 'ref' property");
  }

  auto ref = ObjectCollector::registerObject(new LinearRegression{});
  mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *((uint64_t*)mxGetData(out)) = ref;

  mxSetProperty((mxArray*)prhs[0], 0, "ref", out);

  // mexCallMATLAB(0, NULL, 1, &plhs[0], "disp");
}

void func_delete(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  if (nrhs != 1 || nlhs != 0)
    mexErrMsgTxt("ref expected and no output");

  mxArray* objectRef = mxGetProperty(prhs[0], 0, "ref");
  if (objectRef == nullptr)
    mexErrMsgIdAndTxt("mLibKriging:badObject", "object does not contain 'ref' property");

  if (mxIsEmpty(objectRef)) {
    mexErrMsgIdAndTxt("mLibKriging:alreadyBuilt", "object already contain an empty 'ref' property");
  }

  auto ref = *((uint64_t*)mxGetData(objectRef));
  std::cout << "ref=" << ref << std::endl;
  ObjectCollector::unregisterObject(ref);
  mxArray* out = mxCreateNumericMatrix(0, 0, mxUINT64_CLASS, mxREAL);
  mxSetProperty((mxArray*)prhs[0], 0, "ref", out);
}

void func_fit(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {  // NOLINT (do not declare C-style arrays)
  if (nrhs != 3 || nlhs != 0) {
    mexErrMsgTxt("ARGS must be an initialized obj, a vector y and a matrix X of double values");
  }

  mxArray* objectRef = mxGetProperty(prhs[0], 0, "ref");
  if (objectRef == nullptr)
    mexErrMsgIdAndTxt("mLibKriging:badObject", "object does not contain 'ref' property");

  if (mxIsEmpty(objectRef)) {
    mexErrMsgIdAndTxt("mLibKriging:alreadyBuilt", "object already contain an empty 'ref' property");
  }

  auto ref = *((uint64_t*)mxGetData(objectRef));
  auto* lin_reg = ObjectCollector::getObject<LinearRegression>(ref);

  if (!mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2])) {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    mexErrMsgTxt("ARGS must be a vector y and a matrix X of double values");
  }

  const size_t nyrow = mxGetM(prhs[1]);
  const size_t nycol = mxGetN(prhs[1]);
  const size_t nydim = mxGetNumberOfDimensions(prhs[1]);
  const size_t nXrow = mxGetM(prhs[2]);
  const size_t nXcol = mxGetN(prhs[2]);
  const size_t nXdim = mxGetNumberOfDimensions(prhs[2]);

  mexPrintf("Arg 1 have %d dimensions: %d x %d\n", nydim, nyrow, nycol);
  mexPrintf("Arg 2 have %d dimensions: %d x %d\n", nXdim, nXrow, nXcol);

  if (nycol != 1 || nydim > 2) {
    mexErrMsgTxt("ARG 1 must be a column vector y");
  }

  if (nXdim > 2) {
    mexErrMsgTxt("ARG 2 must be a matrix with only 2 dimensions");
  }

  if (nyrow != nXrow) {
    mexErrMsgTxt("ARG vector y and matrix X don't have same number of rows");
  }

  // Matlab matrices use column-major order like armadillo
  double* py = mxGetPr(prhs[1]);
  double* pX = mxGetPr(prhs[2]);
  const arma::vec y(py, nyrow, false, true);
  const arma::mat X(pX, nXrow, nXcol, false, true);

  lin_reg->fit(y, X);
}

void func_predict(int nlhs,
                  mxArray* plhs[],
                  int nrhs,
                  const mxArray* prhs[]) {  // NOLINT (do not declare C-style arrays)
  if (nrhs != 2) {
    mexErrMsgTxt("ARGS must be an initialized obj, a matrix X");
  }

  if (nlhs > 2) {
    mexErrMsgTxt("Too many lhs requested; only [y_pred, residual] is available");
  }

  mxArray* objectRef = mxGetProperty(prhs[0], 0, "ref");
  if (objectRef == nullptr)
    mexErrMsgIdAndTxt("mLibKriging:badObject", "object does not contain 'ref' property");

  if (mxIsEmpty(objectRef)) {
    mexErrMsgIdAndTxt("mLibKriging:alreadyBuilt", "object already contain an empty 'ref' property");
  }

  auto ref = *((uint64_t*)mxGetData(objectRef));
  auto* lin_reg = ObjectCollector::getObject<LinearRegression>(ref);

  if (!mxIsDouble(prhs[1])) {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    mexErrMsgTxt("ARGS must be a matrix X of double values");
  }

  const size_t nXrow = mxGetM(prhs[1]);
  const size_t nXcol = mxGetN(prhs[1]);
  const size_t nXdim = mxGetNumberOfDimensions(prhs[1]);

  mexPrintf("Arg 1 have %d dimensions: %d x %d\n", nXdim, nXrow, nXcol);

  if (nXdim > 2) {
    mexErrMsgTxt("ARG 2 must be a matrix with only 2 dimensions");
  }

  //  if (nyrow != nXrow) {
  //    mexErrMsgTxt("ARG vector y and matrix X don't have same number of rows");
  //  }

  // Matlab matrices use column-major order like armadillo
  double* pX = mxGetPr(prhs[1]);
  const arma::mat X(pX, nXrow, nXcol, false, true);

  auto [y_pred, stderr_v] = lin_reg->predict(X);

  //  auto err = relative_error(y, y_pred);
  //
  //  std::cout << "Error = " << err << '\n';
  //
  //  for (unsigned long i = 0; i < nyrow * nycol; ++i)
  //    py[i] = i;

  if (nlhs > 0) {
    plhs[0] = mxCreateNumericMatrix(y_pred.n_rows, y_pred.n_cols, mxDOUBLE_CLASS, mxREAL);

    if (false && y_pred.mem_state == 0 && y_pred.n_elem > arma::arma_config::mat_prealloc) {
      // FIXME hard trick; use internal implementation of arma::~Mat
      arma::access::rw(y_pred.mem_state) = 2;
      mxSetPr(plhs[0], y_pred.memptr());
    } else {
      std::memcpy(mxGetPr(plhs[0]), y_pred.memptr(), sizeof(double) * y_pred.n_rows * y_pred.n_cols);
    }
  }

  if (nlhs > 1) {
    plhs[1] = mxCreateNumericMatrix(stderr_v.n_rows, stderr_v.n_cols, mxDOUBLE_CLASS, mxREAL);

    if (false && stderr_v.mem_state == 0 && stderr_v.n_elem > arma::arma_config::mat_prealloc) {
      // FIXME hard trick; use internal implementation of arma::~Mat
      arma::access::rw(stderr_v.mem_state) = 2;
      mxSetPr(plhs[1], stderr_v.memptr());
    } else {
      std::memcpy(mxGetPr(plhs[1]), stderr_v.memptr(), sizeof(double) * stderr_v.n_rows * stderr_v.n_cols);
    }
  }
}
