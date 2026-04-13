#include "MLPKriging_binding.hpp"

#include "libKriging/MLPKriging.hpp"

#include <map>
#include <string>
#include "common_binding.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

using libKriging::MLPKriging;

// Convert a MATLAB struct to std::map<std::string, std::string>
static std::map<std::string, std::string> structToStringMap(const mxArray* s) {
  std::map<std::string, std::string> result;
  if (s == nullptr || mxIsEmpty(s))
    return result;
  if (!mxIsStruct(s)) {
    throw MxException(LOCATION(), "mLibKriging:badType", "parameters must be a struct");
  }
  int nfields = mxGetNumberOfFields(s);
  for (int i = 0; i < nfields; ++i) {
    const char* fname = mxGetFieldNameByNumber(s, i);
    mxArray* val = mxGetFieldByNumber(s, 0, i);
    if (val == nullptr)
      continue;
    char* str = mxArrayToString(val);
    if (str == nullptr) {
      if (mxIsNumeric(val) && mxGetNumberOfElements(val) == 1) {
        double dval = mxGetScalar(val);
        result[fname] = std::to_string(dval);
      } else {
        throw MxException(LOCATION(), "mLibKriging:badType", "parameter value for '", fname, "' must be a string or scalar");
      }
    } else {
      result[fname] = std::string(str);
      mxFree(str);
    }
  }
  return result;
}

// Convert an arma::vec of doubles (as given by the user) to std::vector<arma::uword>
static std::vector<arma::uword> toUwordVec(const arma::vec& v) {
  std::vector<arma::uword> out;
  out.reserve(v.n_elem);
  for (arma::uword i = 0; i < v.n_elem; ++i) {
    out.push_back(static_cast<arma::uword>(v(i)));
  }
  return out;
}

static mxArray* uwordVecToMat(const std::vector<arma::uword>& v) {
  mxArray* a = mxCreateDoubleMatrix(1, v.size(), mxREAL);
  double* p = mxGetPr(a);
  for (size_t i = 0; i < v.size(); ++i)
    p[i] = static_cast<double>(v[i]);
  return a;
}

namespace MLPKrigingBinding {

void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  // args: y, X, hidden_dims, d_out, activation, kernel, [regmodel], [normalize], [optim], [objective], [parameters_struct]
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 11}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto y_vec = input.get<arma::vec>(0, "y vector");
  auto X_mat = input.get<arma::mat>(1, "X matrix");
  auto hidden_dims = toUwordVec(input.get<arma::vec>(2, "hidden_dims vector"));
  auto d_out = static_cast<arma::uword>(input.getOptional<double>(3, "d_out").value_or(2.0));
  auto activation = input.getOptional<std::string>(4, "activation").value_or("selu");
  auto kernel = input.getOptional<std::string>(5, "kernel").value_or("gauss");
  auto regmodel = input.getOptional<std::string>(6, "regression model").value_or("constant");
  auto normalize = input.getOptional<bool>(7, "normalize").value_or(false);
  auto optim = input.getOptional<std::string>(8, "optim").value_or("BFGS+Adam");
  auto objective = input.getOptional<std::string>(9, "objective").value_or("LL");
  std::map<std::string, std::string> params;
  if (nrhs > 10)
    params = structToStringMap(prhs[10]);
  auto mk = buildObject<MLPKriging>(
      y_vec, X_mat, hidden_dims, d_out, activation, kernel, regmodel, normalize, optim, objective, params);
  output.set(0, mk, "new object reference");
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
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto regmodel = input.getOptional<std::string>(3, "regression model").value_or("constant");
  auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
  auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS+Adam");
  auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  std::map<std::string, std::string> params;
  if (nrhs > 7)
    params = structToStringMap(prhs[7]);
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  mk->fit(input.get<arma::vec>(1, "y vector"), input.get<arma::mat>(2, "X matrix"), regmodel, normalize, optim,
          objective, params);
}

void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 5}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");

  const bool withStd = flag_output_compliance(input, 2, "with standard deviation", output, 1);
  const bool withCov = flag_output_compliance(input, 3, "with covariance", output, 2);
  const bool withDeriv = flag_output_compliance(input, 4, "with derivatives", output, 3);

  auto [y_pred, stdev_pred, cov_pred, mean_deriv, stdev_deriv]
      = mk->predict(input.get<arma::mat>(1, "X_n matrix"), withStd, withCov, withDeriv);
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
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  auto nsim = input.get<int32_t>(1, "nsim");
  auto seed = input.get<int32_t>(2, "seed");
  auto result = mk->simulate(nsim, static_cast<uint64_t>(seed), input.get<arma::mat>(3, "X_n matrix"));
  output.set(0, result, "simulated values");
}

void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  mk->update(input.get<arma::vec>(1, "y vector"), input.get<arma::mat>(2, "X matrix"));
}

void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{0, 1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  auto desc = mk->summary();
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
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");

  const bool withGrad = flag_output_compliance(input, 2, "with gradient", output, 1);
  const bool withHess = flag_output_compliance(input, 3, "with hessian", output, 2);

  auto [ll, grad, hess] = mk->logLikelihoodFun(input.get<arma::vec>(1, "theta vector"), withGrad, withHess);
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
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->logLikelihood(), "log-likelihood value");
}

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->kernel(), "kernel name");
}

void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->X(), "X matrix");
}

void y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->y(), "y vector");
}

void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->theta(), "theta vector");
}

void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->sigma2(), "sigma2 value");
}

void is_fitted(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->is_fitted(), "is_fitted flag");
}

void feature_dim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, static_cast<double>(mk->feature_dim()), "feature dimension");
}

void hidden_dims(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  if (nlhs < 1) {
    throw MxException(LOCATION(), "mLibKriging:badOutput", "hidden_dims requires at least one output");
  }
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  plhs[0] = uwordVecToMat(mk->hidden_dims());
}

void activation(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mk = input.getObjectFromRef<MLPKriging>(0, "MLPKriging reference");
  output.set(0, mk->activation(), "activation name");
}

}  // namespace MLPKrigingBinding
