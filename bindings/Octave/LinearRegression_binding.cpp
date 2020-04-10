#include "LinearRegression_binding.hpp"

#include "libKriging/LinearRegression.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

namespace LinearRegressionBinding {

void build(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  buildObject<LinearRegression>(input.get<0, mxArray*>("object reference"));
}

void destroy(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  destroyObject(input.get<0, mxArray*>("object reference"));
}

void fit(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* lin_reg = input.getObject<0, LinearRegression>("LinearRegression reference");
  lin_reg->fit(input.get<1, arma::vec>("vector"), input.get<2, arma::mat>("matrix"));
}

void predict(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* lin_reg = input.getObject<0, LinearRegression>("LinearRegression reference");
  auto [y_pred, stderr_v] = lin_reg->predict(input.get<1, arma::mat>("matrix"));
  output.set<0>(y_pred, "predicted response");
  output.setOptional<1>(stderr_v, "prediction error");
}

}  // namespace LinearRegressionBinding