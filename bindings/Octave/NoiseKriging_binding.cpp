#include "NoiseKriging_binding.hpp"

#include "libKriging/NoiseKriging.hpp"
#include "libKriging/Trend.hpp"

#include "Params.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

namespace NoiseKrigingBinding {

static NoiseKriging::Parameters makeParameters(std::optional<Params*> dict) {
  if (dict) {
    const Params& params = *dict.value();
    return NoiseKriging::Parameters{params.get<arma::mat>("sigma2"),  // should be converted as arma::vec by execution
                                    params.get<bool>("is_sigma2_estim").value_or(true),
                                    params.get<arma::mat>("theta"),
                                    params.get<bool>("is_theta_estim").value_or(true),
                                    params.get<arma::mat>("beta"),  // should be converted as arma::colvec by execution
                                    params.get<bool>("is_beta_estim").value_or(true)};
  } else {
    return NoiseKriging::Parameters{};
  }
}

void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{4, 9}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(4, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(5, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(6, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(7, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(8, "parameters"));
  auto km = buildObject<NoiseKriging>(input.get<arma::vec>(0, "vector"),
                                      input.get<arma::vec>(1, "vector"),
                                      input.get<arma::mat>(2, "matrix"),
                                      input.get<std::string>(3, "kernel"),
                                      regmodel,
                                      normalize,
                                      optim,
                                      objective,
                                      parameters);
  output.set(0, km, "new object reference");
}

void copy(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  auto km_copy = buildObject<NoiseKriging>(*km, ExplicitCopySpecifier{});
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
                 RequiresArg::Range{4, 9}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(4, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(5, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(6, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(7, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(8, "parameters"));
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  km->fit(input.get<arma::vec>(1, "vector"),
          input.get<arma::vec>(2, "vector"),
          input.get<arma::mat>(3, "matrix"),
          regmodel,
          normalize,
          optim,
          objective,
          parameters);
}

// Used to check if an input flag option implies the related output
bool flag_output_compliance(MxMapper& input, int I, const char* msg, const MxMapper& output, int output_position) {
  const auto flag = input.template getOptional<bool>(I, msg);
  if (flag) {
    const bool flag_value = flag.value();
    if (flag_value) {
      if (output.count() <= output_position) {
        throw MxException(LOCATION(),
                          "mLibKriging:inconsistentOutput",
                          MxMapper::parameterStr(I, msg),
                          " is set without related output");
      }
    }
    return flag_value;
  } else {
    return output.count() > output_position;
  }
}

void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 5}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
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
                 RequiresArg::Exactly{6}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  auto result = km->simulate(input.get<int>(1, "nsim"),
                             input.get<int>(2, "seed"),
                             input.get<arma::mat>(3, "X_n"),
                             input.get<arma::vec>(4, "with_noise"),
                             input.get<bool>(5, "will_update"));
  output.set(0, result, "simulated response");
}

void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  km->update(input.get<arma::vec>(1, "y_u"),
             input.get<arma::vec>(2, "noise_u"),
             input.get<arma::mat>(3, "X_u"),
             input.get<bool>(4, "refit"));
}

void update_simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  km->update_simulate(
      input.get<arma::vec>(1, "y_u"), input.get<arma::vec>(2, "noise_u"), input.get<arma::mat>(3, "X_u"));
}

void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->summary(), "Model description");
}

void save(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "Kriging reference");
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
  auto km = buildObject<NoiseKriging>(NoiseKriging::load(filename));
  output.set(0, km, "new object reference");
}

void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  const bool return_grad = flag_output_compliance(input, 3, "return_grad", output, 1);
  auto [ll, llgrad] = km->logLikelihoodFun(input.get<arma::vec>(1, "theta_alpha"), return_grad, false);
  output.set(0, ll, "ll");                  // FIXME better name
  output.setOptional(1, llgrad, "llgrad");  // FIXME better name
}

void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->logLikelihood(), "Model logLikelihood");
}

void covMat(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->covMat(input.get<arma::mat>(1, "X1"), input.get<arma::mat>(2, "X2")), "Covariance matrix");
}

void model(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");

  const char* fieldnames[]
      = {"kernel",  "optim",    "objective",      "theta", "is_theta_estim", "sigma2", "is_sigma2_estim",
         "X",       "centerX",  "scaleX",         "y",     "centerY",        "scaleY", "noise",
         "normalize", "regmodel", "beta",           "is_beta_estim", "F",      "T",      "M",
         "z"};
  mxArray* model_struct = mxCreateStructMatrix(1, 1, 22, fieldnames);

  mxSetField(model_struct, 0, "kernel", mxCreateString(km->kernel().c_str()));
  mxSetField(model_struct, 0, "optim", mxCreateString(km->optim().c_str()));
  mxSetField(model_struct, 0, "objective", mxCreateString(km->objective().c_str()));

  mxArray* theta_val = nullptr;
  setter(km->theta(), theta_val);
  mxSetField(model_struct, 0, "theta", theta_val);

  mxSetField(model_struct, 0, "is_theta_estim", mxCreateLogicalScalar(km->is_theta_estim()));
  mxSetField(model_struct, 0, "sigma2", mxCreateDoubleScalar(km->sigma2()));
  mxSetField(model_struct, 0, "is_sigma2_estim", mxCreateLogicalScalar(km->is_sigma2_estim()));

  mxArray* X_val = nullptr;
  setter(km->X(), X_val);
  mxSetField(model_struct, 0, "X", X_val);

  mxArray* centerX_val = nullptr;
  setter(km->centerX(), centerX_val);
  mxSetField(model_struct, 0, "centerX", centerX_val);

  mxArray* scaleX_val = nullptr;
  setter(km->scaleX(), scaleX_val);
  mxSetField(model_struct, 0, "scaleX", scaleX_val);

  mxArray* y_val = nullptr;
  setter(km->y(), y_val);
  mxSetField(model_struct, 0, "y", y_val);

  mxSetField(model_struct, 0, "centerY", mxCreateDoubleScalar(km->centerY()));
  mxSetField(model_struct, 0, "scaleY", mxCreateDoubleScalar(km->scaleY()));

  mxArray* noise_val = nullptr;
  setter(km->noise(), noise_val);
  mxSetField(model_struct, 0, "noise", noise_val);

  mxSetField(model_struct, 0, "normalize", mxCreateLogicalScalar(km->normalize()));
  mxSetField(model_struct, 0, "regmodel", mxCreateString(Trend::toString(km->regmodel()).c_str()));

  mxArray* beta_val = nullptr;
  setter(km->beta(), beta_val);
  mxSetField(model_struct, 0, "beta", beta_val);

  mxSetField(model_struct, 0, "is_beta_estim", mxCreateLogicalScalar(km->is_beta_estim()));

  mxArray* F_val = nullptr;
  setter(km->F(), F_val);
  mxSetField(model_struct, 0, "F", F_val);

  mxArray* T_val = nullptr;
  setter(km->T(), T_val);
  mxSetField(model_struct, 0, "T", T_val);

  mxArray* M_val = nullptr;
  setter(km->M(), M_val);
  mxSetField(model_struct, 0, "M", M_val);

  mxArray* z_val = nullptr;
  setter(km->z(), z_val);
  mxSetField(model_struct, 0, "z", z_val);

  output.set(0, model_struct, "model");
}

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->kernel(), "kernel");
}

void optim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->optim(), "optim");
}

void objective(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->objective(), "objective");
}

void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->X(), "X");
}

void centerX(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->centerX(), "centerX");
}

void scaleX(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->scaleX(), "scaleX");
}

void y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->y(), "y");
}

void centerY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->centerY(), "centerY");
}

void scaleY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->scaleY(), "scaleY");
}

void normalize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->normalize(), "normalize");
}

void noise(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->noise(), "noise");
}

void regmodel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, Trend::toString(km->regmodel()), "regmodel");
}

void F(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->F(), "F");
}

void T(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->T(), "T");
}

void M(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->M(), "M");
}

void z(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->z(), "z");
}

void beta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->beta(), "beta");
}

void is_beta_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->is_beta_estim(), "is_beta_estim");
}

void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->theta(), "theta");
}

void is_theta_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->is_theta_estim(), "is_theta_estim");
}

void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->sigma2(), "sigma2");
}

void is_sigma2_estim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<NoiseKriging>(0, "NoiseKriging reference");
  output.set(0, km->is_sigma2_estim(), "is_sigma2_estim ");
}

}  // namespace NoiseKrigingBinding