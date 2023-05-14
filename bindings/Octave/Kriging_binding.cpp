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
                               params.get<bool>("is_beta_estim").value_or(true)};
  } else {
    return Kriging::Parameters{};
  }
}

namespace KrigingBinding {

void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(3, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(7, "parameters"));
  auto km = buildObject<Kriging>(input.get<arma::vec>(0, "vector"),
                                 input.get<arma::mat>(1, "matrix"),
                                 input.get<std::string>(2, "kernel"),
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
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(3, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(7, "parameters"));
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  km->fit(input.get<arma::vec>(1, "vector"),
          input.get<arma::mat>(2, "matrix"),
          regmodel,
          normalize,
          optim,
          objective,
          parameters);
}

void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 5}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const bool withStd = flag_output_compliance(input, 2, "withStd", output, 1);
  const bool withCov = flag_output_compliance(input, 3, "withCov", output, 2);
  const bool withDeriv = flag_output_compliance(input, 4, "withDeriv", output, 3);
  auto [y_pred, stderr_v, cov_m, mean_deriv_m, stderr_deriv_m]
      = km->predict(input.get<arma::mat>(1, "matrix"), withStd, withCov, withDeriv);
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
                 RequiresArg::Exactly{4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  auto result = km->simulate(input.get<int>(1, "nsim"), input.get<int>(2, "seed"), input.get<arma::mat>(3, "Xp"));
  output.set(0, result, "simulated response");
}

void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  km->update(input.get<arma::vec>(1, "new y"), input.get<arma::mat>(2, "new X"));
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
  const bool want_grad = flag_output_compliance(input, 2, "want_grad", output, 1);
  auto [loo, loograd] = km->leaveOneOutFun(input.get<arma::vec>(1, "theta"), want_grad);
  output.set(0, loo, "loo");                  // FIXME better name
  output.setOptional(1, loograd, "loograd");  // FIXME better name
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
                 RequiresArg::Range{2, 4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 3}};
  auto* km = input.getObjectFromRef<Kriging>(0, "Kriging reference");
  const bool want_grad = flag_output_compliance(input, 2, "want_grad", output, 1);
  const bool want_hess = flag_output_compliance(input, 3, "want_hess", output, 2);
  auto [ll, llgrad, llhess] = km->logLikelihoodFun(input.get<arma::vec>(1, "theta"), want_grad, want_hess);
  output.set(0, ll, "ll");                  // FIXME better name
  output.setOptional(1, llgrad, "llgrad");  // FIXME better name
  output.setOptional(2, llhess, "llhess");  // FIXME better name
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
  const bool want_grad = flag_output_compliance(input, 2, "want_grad", output, 1);
  auto [lmp, lmpgrad] = km->logMargPostFun(input.get<arma::vec>(1, "theta"), want_grad);
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

}  // namespace KrigingBinding
