#include "Kriging_binding.hpp"

#include "libKriging/Kriging.hpp"
#include "libKriging/Trend.hpp"

#include "Params.hpp"
#include "common_binding.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

static Kriging::Parameters makeParameters(std::optional<Params*> dict) {
  if (dict) {
    const Params& params = *dict.value();
    return Kriging::Parameters{params.get<double>("sigma2"),
                               params.get<bool>("is_sigma2_estim").value_or(true),
                               params.get<arma::mat>("theta"),
                               params.get<bool>("is_theta_estim").value_or(true),
                               params.get<arma::mat>("beta"),  // should be converted as arma::colvec by execution
                               params.get<bool>("is_beta_estim").value_or(true),
                               params.get<double>("nugget"),
                               params.get<bool>("is_nugget_estim").value_or(true)};
  } else {
    return Kriging::Parameters{};
  }
}

static Kriging::NoiseModel parseNoiseModel(const std::string& s) {
  if (s == "none" || s.empty())
    return Kriging::NoiseModel::None;
  if (s == "nugget")
    return Kriging::NoiseModel::Nugget;
  if (s == "heterogeneous")
    return Kriging::NoiseModel::Heterogeneous;
  throw std::runtime_error("Unknown noise_model: '" + s + "'. Expected 'none', 'nugget', or 'heterogeneous'.");
}

static std::string noiseModelToString(Kriging::NoiseModel nm) {
  switch (nm) {
    case Kriging::NoiseModel::None:
      return "none";
    case Kriging::NoiseModel::Nugget:
      return "nugget";
    case Kriging::NoiseModel::Heterogeneous:
      return "heterogeneous";
  }
  return "none";
}

namespace KrigingBinding {

void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 10}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(3, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(7, "parameters"));
  const auto noise_model_str = input.getOptional<std::string>(8, "noise_model").value_or("none");
  const auto noise_model = parseNoiseModel(noise_model_str);
  const auto noise = input.getOptional<arma::vec>(9, "noise");

  auto y = input.get<arma::vec>(0, "vector");
  auto X = input.get<arma::mat>(1, "matrix");
  auto kernel = input.get<std::string>(2, "kernel");

  Kriging km_obj(kernel, noise_model);
  if (noise_model == Kriging::NoiseModel::Heterogeneous && noise) {
    km_obj.fit(y, noise.value(), X, regmodel, normalize, optim, objective, parameters);
  } else {
    km_obj.fit(y, X, regmodel, normalize, optim, objective, parameters);
  }
  auto km = buildObject<Kriging>(std::move(km_obj));
  output.set(0, km, "new object reference");
}

void copy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  auto km_copy = buildObject<Kriging>(*km, ExplicitCopySpecifier{});
  output.set(0, km_copy, "copied object reference");
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
                 RequiresArg::Range{3, 9}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  if (km->noise_model() == Kriging::NoiseModel::Heterogeneous) {
    // Heterogeneous: fit(ref, y, noise, X, [regmodel], [normalize], [optim], [objective], [parameters])
    const auto regmodel
        = Trend::fromString(input.getOptional<std::string>(4, "regression model").value_or("constant"));
    const auto normalize = input.getOptional<bool>(5, "normalize").value_or(false);
    const auto optim = input.getOptional<std::string>(6, "optim").value_or("BFGS");
    const auto objective = input.getOptional<std::string>(7, "objective").value_or("LL");
    const auto parameters = makeParameters(input.getOptionalObject<Params>(8, "parameters"));
    km->fit(input.get<arma::vec>(1, "vector"),
            input.get<arma::vec>(2, "noise"),
            input.get<arma::mat>(3, "matrix"),
            regmodel,
            normalize,
            optim,
            objective,
            parameters);
  } else {
    // None/Nugget: fit(ref, y, X, [regmodel], [normalize], [optim], [objective], [parameters])
    const auto regmodel
        = Trend::fromString(input.getOptional<std::string>(3, "regression model").value_or("constant"));
    const auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
    const auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS");
    const auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
    const auto parameters = makeParameters(input.getOptionalObject<Params>(7, "parameters"));
    km->fit(input.get<arma::vec>(1, "vector"),
            input.get<arma::mat>(2, "matrix"),
            regmodel,
            normalize,
            optim,
            objective,
            parameters);
  }
}

void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 5}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const bool return_stdev = flag_output_compliance(input, 2, "return_stdev", output, 1);
  const bool return_cov = flag_output_compliance(input, 3, "return_cov", output, 2);
  const bool return_deriv = flag_output_compliance(input, 4, "return_deriv", output, 3);
  auto [y_pred, stderr_v, cov_m, mean_deriv_m, stderr_deriv_m]
      = km->predict(input.get<arma::mat>(1, "matrix"), return_stdev, return_cov, return_deriv);
  output.set(0, y_pred, "predicted response");
  output.setOptional(1, stderr_v, "stderr vector");
  output.setOptional(2, cov_m, "cov matrix");
  output.setOptional(3, mean_deriv_m, "predicted mean deriv matrix");
  output.setOptional(4, stderr_deriv_m, "predicted stdev deriv matrix");
}

void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{5, 6}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const auto nsim = input.get<int>(1, "nsim");
  const auto seed = input.get<int>(2, "seed");
  const auto X_n = input.get<arma::mat>(3, "X_n");
  if (km->noise_model() == Kriging::NoiseModel::None) {
    // simulate(nsim, seed, X_n, will_update)
    auto result = km->simulate(nsim, seed, X_n, input.get<bool>(4, "will_update"));
    output.set(0, result, "simulated response");
  } else if (km->noise_model() == Kriging::NoiseModel::Nugget) {
    // simulate(nsim, seed, X_n, with_nugget, will_update)
    auto result
        = km->simulate(nsim, seed, X_n, input.get<bool>(4, "with_nugget"), input.get<bool>(5, "will_update"));
    output.set(0, result, "simulated response");
  } else {
    // Heterogeneous: simulate(nsim, seed, X_n, with_noise, will_update)
    auto result
        = km->simulate(nsim, seed, X_n, input.get<arma::vec>(4, "with_noise"), input.get<bool>(5, "will_update"));
    output.set(0, result, "simulated response");
  }
}

void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{4, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  if (km->noise_model() == Kriging::NoiseModel::Heterogeneous) {
    // update(ref, y_u, noise_u, X_u, refit)
    km->update(input.get<arma::vec>(1, "y_u"),
               input.get<arma::vec>(2, "noise_u"),
               input.get<arma::mat>(3, "X_u"),
               input.get<bool>(4, "refit"));
  } else {
    // update(ref, y_u, X_u, refit)
    km->update(input.get<arma::vec>(1, "y_u"), input.get<arma::mat>(2, "X_u"), input.get<bool>(3, "refit"));
  }
}

void update_simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  if (km->noise_model() == Kriging::NoiseModel::Heterogeneous) {
    // update_simulate(ref, y_u, noise_u, X_u)
    auto result = km->update_simulate(
        input.get<arma::vec>(1, "y_u"), input.get<arma::vec>(2, "noise_u"), input.get<arma::mat>(3, "X_u"));
    output.set(0, result, "updated simulated values");
  } else {
    // update_simulate(ref, y_u, X_u)
    auto result = km->update_simulate(input.get<arma::vec>(1, "y_u"), input.get<arma::mat>(2, "X_u"));
    output.set(0, result, "updated simulated values");
  }
}

void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->summary(), "Model description");
}

void save(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const auto filename = input.get<std::string>(1, "filename");
  km->save(filename);
}

void load(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto filename = input.get<std::string>(0, "filename");
  auto km = buildObject<Kriging>(Kriging::load(filename));
  output.set(0, km, "new object reference");
}

void leaveOneOutFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const bool return_grad = flag_output_compliance(input, 2, "return_grad", output, 1);
  auto [loo, loograd] = km->leaveOneOutFun(input.get<arma::vec>(1, "theta"), return_grad, false);
  output.set(0, loo, "loo");                  // FIXME better name
  output.setOptional(1, loograd, "loograd");  // FIXME better name
}

void leaveOneOutVec(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{2}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  auto [yhat_mean, yhat_sd] = km->leaveOneOutVec(input.get<arma::vec>(1, "theta"));
  output.set(0, yhat_mean, "mean");  // FIXME better name
  output.set(1, yhat_sd, "stdev");   // FIXME better name
}

void leaveOneOut(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->leaveOneOut(), "Model leaveOneOut");
}

void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const bool return_grad = flag_output_compliance(input, 2, "return_grad", output, 1);
  auto [ll, llgrad] = km->logLikelihoodFun(input.get<arma::vec>(1, "theta"), return_grad, false);
  output.set(0, ll, "ll");                  // FIXME better name
  output.setOptional(1, llgrad, "llgrad");  // FIXME better name
}

void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->logLikelihood(), "Model logLikelihood");
}

void logMargPostFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const bool return_grad = flag_output_compliance(input, 2, "return_grad", output, 1);
  auto [lmp, lmpgrad] = km->logMargPostFun(input.get<arma::vec>(1, "theta"), return_grad, false);
  output.set(0, lmp, "lmp");                  // FIXME better name
  output.setOptional(1, lmpgrad, "lmpgrad");  // FIXME better name
}

void logMargPost(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->logMargPost(), "Model logMargPost");
}

void covMat(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->covMat(input.get<arma::mat>(1, "X1"), input.get<arma::mat>(2, "X2")), "Covariance matrix");
}

void model(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");

  const char* fieldnames[] = {"kernel",          "optim",     "objective",       "noise_model",
                               "theta",           "is_theta_estim",
                               "sigma2",          "is_sigma2_estim",
                               "nugget",          "is_nugget_estim",
                               "X",               "centerX",   "scaleX",
                               "y",               "centerY",   "scaleY",
                               "noise",           "normalize", "regmodel",
                               "beta",            "is_beta_estim",
                               "F",               "T",         "M",
                               "z"};
  mxArray* model_struct = mxCreateStructMatrix(1, 1, 25, fieldnames);

  // Helper lambda to create mxArray* from arma types using setter
  auto createMxArray = [](const auto& value) -> mxArray* {
    mxArray* result = nullptr;
    setter(value, result);
    return result;
  };

  mxSetField(model_struct, 0, "kernel", mxCreateString(km->kernel().c_str()));
  mxSetField(model_struct, 0, "optim", mxCreateString(km->optim().c_str()));
  mxSetField(model_struct, 0, "objective", mxCreateString(km->objective().c_str()));
  mxSetField(model_struct, 0, "noise_model", mxCreateString(noiseModelToString(km->noise_model()).c_str()));
  mxSetField(model_struct, 0, "theta", createMxArray(km->theta()));
  mxSetField(model_struct, 0, "is_theta_estim", mxCreateLogicalScalar(km->is_theta_estim()));
  mxSetField(model_struct, 0, "sigma2", mxCreateDoubleScalar(km->sigma2()));
  mxSetField(model_struct, 0, "is_sigma2_estim", mxCreateLogicalScalar(km->is_sigma2_estim()));
  mxSetField(model_struct, 0, "nugget", mxCreateDoubleScalar(km->nugget()));
  mxSetField(model_struct, 0, "is_nugget_estim", mxCreateLogicalScalar(km->is_nugget_estim()));
  mxSetField(model_struct, 0, "X", createMxArray(km->X()));
  mxSetField(model_struct, 0, "centerX", createMxArray(km->centerX()));
  mxSetField(model_struct, 0, "scaleX", createMxArray(km->scaleX()));
  mxSetField(model_struct, 0, "y", createMxArray(km->y()));
  mxSetField(model_struct, 0, "centerY", mxCreateDoubleScalar(km->centerY()));
  mxSetField(model_struct, 0, "scaleY", mxCreateDoubleScalar(km->scaleY()));
  mxSetField(model_struct, 0, "noise", createMxArray(km->noise()));
  mxSetField(model_struct, 0, "normalize", mxCreateLogicalScalar(km->normalize()));
  mxSetField(model_struct, 0, "regmodel", mxCreateString(Trend::toString(km->regmodel()).c_str()));
  mxSetField(model_struct, 0, "beta", createMxArray(km->beta()));
  mxSetField(model_struct, 0, "is_beta_estim", mxCreateLogicalScalar(km->is_beta_estim()));
  mxSetField(model_struct, 0, "F", createMxArray(km->F()));
  mxSetField(model_struct, 0, "T", createMxArray(km->T()));
  mxSetField(model_struct, 0, "M", createMxArray(km->M()));
  mxSetField(model_struct, 0, "z", createMxArray(km->z()));

  output.set(0, model_struct, "model");
}

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->kernel(), "kernel");
}

void optim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->optim(), "optim");
}

void objective(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->objective(), "objective");
}

void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->X(), "X");
}

void centerX(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->centerX(), "centerX");
}

void scaleX(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->scaleX(), "scaleX");
}

void y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->y(), "y");
}

void centerY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->centerY(), "centerY");
}

void scaleY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->scaleY(), "scaleY");
}

void normalize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->normalize(), "normalize");
}

void regmodel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, Trend::toString(km->regmodel()), "regmodel");
}

void F(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->F(), "F");
}

void T(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->T(), "T");
}

void M(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->M(), "M");
}

void z(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->z(), "z");
}

void beta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->beta(), "beta");
}

void is_beta_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->is_beta_estim(), "is_beta_estim");
}

void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->theta(), "theta");
}

void is_theta_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->is_theta_estim(), "is_theta_estim");
}

void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->sigma2(), "sigma2");
}

void is_sigma2_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->is_sigma2_estim(), "is_sigma2_estim ");
}

void noise_model(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, noiseModelToString(km->noise_model()), "noise_model");
}

void nugget(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->nugget(), "nugget");
}

void is_nugget_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->is_nugget_estim(), "is_nugget_estim");
}

void noise(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  output.set(0, km->noise(), "noise");
}

}  // namespace KrigingBinding
