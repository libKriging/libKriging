//
// Created by Pascal Havé on 08/04/2020.
//

#include "func1.hpp"

#include <armadillo>
#include <cstring>

#include "../tools/ObjectCollector.hpp"
#include "libKriging/LinearRegression.hpp"
#include "relative_error.hpp"

void func1(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {  // NOLINT (do not declare C-style arrays)
  if (nrhs != 2 || !mxIsDouble(prhs[0])
      || !mxIsDouble(prhs[1])) {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    mexErrMsgTxt("ARGS must be a vector y and a matrix X of double values");
  }

  if (nlhs > 2) {
    mexErrMsgTxt("Too many lhs requested; only [y_pred, residual] is available");
  }

  const size_t nyrow = mxGetM(prhs[0]);
  const size_t nycol = mxGetN(prhs[0]);
  const size_t nydim = mxGetNumberOfDimensions(prhs[0]);
  const size_t nXrow = mxGetM(prhs[1]);
  const size_t nXcol = mxGetN(prhs[1]);
  const size_t nXdim = mxGetNumberOfDimensions(prhs[1]);

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
  double* py = mxGetPr(prhs[0]);
  double* pX = mxGetPr(prhs[1]);
  const arma::vec y(py, nyrow, false, true);
  const arma::mat X(pX, nXrow, nXcol, false, true);

  auto ref = ObjectCollector::registerObject(new LinearRegression{});
  auto lin_reg = ObjectCollector::getObject<LinearRegression>(ref);

  lin_reg->fit(y, X);

  std::cout << ">> " << X.at(0, 0) << "\n";

  auto [y_pred, stderr_v] = lin_reg->predict(X);

  auto err = relative_error(y, y_pred);

  std::cout << "Error = " << err << '\n';

  for (unsigned long i = 0; i < nyrow * nycol; ++i)
    py[i] = i;

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
