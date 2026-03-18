#include "LinearRegression_binding.hpp"

#include "libKriging/LinearRegression.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

namespace LinearRegressionBinding {

void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{0}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  output.set(0, buildObject<LinearRegression>(), "new object reference");
}

void destroy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  destroyObject(input.get<uint64_t>(0, "object reference"));
  output.set(0, EmptyObject{}, "deleted object reference");
}

void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* lin_reg = input.getObjectFromRef<LinearRegression>(0, "LinearRegression reference");
  lin_reg->fit(input.get<arma::vec>(1, "vector"), input.get<arma::mat>(2, "matrix"));
}

void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* lin_reg = input.getObjectFromRef<LinearRegression>(0, "LinearRegression reference");
  auto [y_pred, stderr_v] = lin_reg->predict(input.get<arma::mat>(1, "matrix"));
  output.set(0, y_pred, "predicted response");
  output.setOptional(1, stderr_v, "prediction error");
}

}  // namespace LinearRegressionBinding