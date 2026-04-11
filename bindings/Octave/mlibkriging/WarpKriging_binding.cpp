#include "WarpKriging_binding.hpp"

#include "libKriging/WarpKriging.hpp"

#include "common_binding.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

using libKriging::WarpKriging;

// Convert a MATLAB cell array of strings to std::vector<std::string>
static std::vector<std::string> cellToStringVec(const mxArray* cell, const char* param_name) {
  if (!mxIsCell(cell)) {
    throw MxException(LOCATION(), "mLibKriging:badType", param_name, " must be a cell array of strings");
  }
  const mwSize n = mxGetNumberOfElements(cell);
  std::vector<std::string> result(n);
  for (mwSize i = 0; i < n; ++i) {
    mxArray* elem = mxGetCell(cell, i);
    if (elem == nullptr) {
      throw MxException(LOCATION(), "mLibKriging:badType", param_name, " contains an empty cell element");
    }
    char* str = mxArrayToString(elem);
    if (str == nullptr) {
      throw MxException(LOCATION(), "mLibKriging:badType", param_name, " cell element is not a string");
    }
    result[i] = std::string(str);
    mxFree(str);
  }
  return result;
}

// Convert std::vector<std::string> to a MATLAB cell array
static mxArray* stringVecToCell(const std::vector<std::string>& vec) {
  mxArray* cell = mxCreateCellMatrix(1, vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    mxSetCell(cell, i, mxCreateString(vec[i].c_str()));
  }
  return cell;
}

namespace WarpKrigingBinding {

void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  // args: y, X, warping_cell, kernel, [regmodel], [normalize], [optim], [objective]
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto y_vec = input.get<arma::vec>(0, "y vector");
  auto X_mat = input.get<arma::mat>(1, "X matrix");
  auto warping = cellToStringVec(prhs[2], "warping");
  auto kernel = input.getOptional<std::string>(3, "kernel").value_or("gauss");
  auto regmodel = input.getOptional<std::string>(4, "regression model").value_or("constant");
  auto normalize = input.getOptional<bool>(5, "normalize").value_or(false);
  auto optim = input.getOptional<std::string>(6, "optim").value_or("BFGS+Adam");
  auto objective = input.getOptional<std::string>(7, "objective").value_or("LL");
  auto wk = buildObject<WarpKriging>(y_vec, X_mat, warping, kernel, regmodel, normalize, optim, objective);
  output.set(0, wk, "new object reference");
}

void copy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  // Re-construct since WarpKriging is not copyable (contains unique_ptr)
  auto wk_copy = buildObject<WarpKriging>(wk->warping_strings(), wk->kernel());
  if (wk->is_fitted()) {
    auto* wk_ptr = reinterpret_cast<WarpKriging*>(wk_copy);
    wk_ptr->fit(wk->y(), wk->X());
  }
  output.set(0, wk_copy, "copied object reference");
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
                 RequiresArg::Range{3, 7}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto regmodel = input.getOptional<std::string>(3, "regression model").value_or("constant");
  auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
  auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS+Adam");
  auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  wk->fit(
      input.get<arma::vec>(1, "y vector"), input.get<arma::mat>(2, "X matrix"), regmodel, normalize, optim, objective);
}

void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 5}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");

  const bool withStd = flag_output_compliance(input, 2, "with standard deviation", output, 1);
  const bool withCov = flag_output_compliance(input, 3, "with covariance", output, 2);
  const bool withDeriv = flag_output_compliance(input, 4, "with derivatives", output, 3);

  auto [y_pred, stdev_pred, cov_pred, mean_deriv, stdev_deriv]
      = wk->predict(input.get<arma::mat>(1, "X_n matrix"), withStd, withCov, withDeriv);
  output.set(0, y_pred, "predicted y");
  if (withStd)
    output.set(1, stdev_pred, "predicted stdev");
  if (withCov)
    output.set(2, cov_pred, "predicted cov");
  if (withDeriv) {
    output.set(3, mean_deriv, "predicted mean derivative");
    output.set(4, stdev_deriv, "predicted stdev derivative");
  }
}

void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  auto nsim = input.get<int32_t>(1, "nsim");
  auto seed = input.get<int32_t>(2, "seed");
  auto result = wk->simulate(nsim, static_cast<uint64_t>(seed), input.get<arma::mat>(3, "X_n matrix"));
  output.set(0, result, "simulated values");
}

void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  wk->update(input.get<arma::vec>(1, "y vector"), input.get<arma::mat>(2, "X matrix"));
}

void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{0, 1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  auto desc = wk->summary();
  if (output.count() == 0) {
    mexPrintf("%s\n", desc.c_str());
  } else {
    output.set(0, desc, "summary string");
  }
}

void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 3}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");

  const bool withGrad = flag_output_compliance(input, 2, "with gradient", output, 1);
  const bool withHess = flag_output_compliance(input, 3, "with hessian", output, 2);

  auto [ll, grad, hess] = wk->logLikelihoodFun(input.get<arma::vec>(1, "theta vector"), withGrad, withHess);
  output.set(0, ll, "log-likelihood value");
  if (withGrad)
    output.set(1, grad, "log-likelihood gradient");
  if (withHess)
    output.set(2, hess, "log-likelihood hessian");
}

void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->logLikelihood(), "log-likelihood value");
}

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->kernel(), "kernel name");
}

void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->X(), "X matrix");
}

void y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->y(), "y vector");
}

void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->theta(), "theta vector");
}

void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->sigma2(), "sigma2 value");
}

void is_fitted(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, wk->is_fitted(), "is_fitted flag");
}

void feature_dim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  output.set(0, static_cast<double>(wk->feature_dim()), "feature dimension");
}

void warping(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  // Output: we set directly since MxMapper doesn't handle cell arrays
  if (nlhs < 1) {
    throw MxException(LOCATION(), "mLibKriging:badOutput", "warping requires at least one output");
  }
  auto* wk = input.getObjectFromRef<WarpKriging>(0, "WarpKriging reference");
  plhs[0] = stringVecToCell(wk->warping_strings());
}

}  // namespace WarpKrigingBinding
