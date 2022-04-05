#include "Kriging_binding.hpp"

#include "libKriging/NuggetKriging.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

namespace NuggetKrigingBinding {

// Old short constructor only
// void build(int nlhs, void** plhs, int nrhs, const void** prhs) {
//  MxMapper input{"Input",
//                 nrhs,
//                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
//                 RequiresArg::Exactly{1}};
//  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
//  output.set<0>(buildObject<NuggetKriging>(input.get<0, std::string>("kernel")), "new object reference");
//}

void build(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  const auto regmodel = NuggetKriging::RegressionModelUtils::fromString(
      input.getOptional<3, std::string>("regression model").value_or("constant"));
  const auto normalize = input.getOptional<4, bool>("normalize").value_or(false);
  const auto optim = input.getOptional<5, std::string>("optim").value_or("BFGS");
  const auto objective = input.getOptional<6, std::string>("objective").value_or("LL");
  const auto parameters
      = NuggetKriging::Parameters{};  // input.getOptional<7, std::string>("parameters").value_or(); //
  // FIXME Parameters not done
  auto km = buildObject<NuggetKriging>(input.get<0, arma::vec>("vector"),
                                       input.get<1, arma::mat>("matrix"),
                                       input.get<2, std::string>("kernel"),
                                       regmodel,
                                       normalize,
                                       optim,
                                       objective,
                                       parameters);
  output.set<0>(km, "new object reference");
}

void destroy(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  destroyObject(input.get<0, uint64_t>("object reference"));
  output.set<0>(EmptyObject{}, "deleted object reference");
}

void fit(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  const auto regmodel = NuggetKriging::RegressionModelUtils::fromString(
      input.getOptional<3, std::string>("regression model").value_or("constant"));
  const auto normalize = input.getOptional<4, bool>("normalize").value_or(false);
  const auto optim = input.getOptional<5, std::string>("optim").value_or("BFGS");
  const auto objective = input.getOptional<6, std::string>("objective").value_or("LL");
  const auto parameters
      = NuggetKriging::Parameters{};  // input.getOptional<7, std::string>("parameters").value_or(); //
                                      // FIXME Parameters not done
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  km->fit(input.get<1, arma::vec>("vector"),
          input.get<2, arma::mat>("matrix"),
          regmodel,
          normalize,
          optim,
          objective,
          parameters);
}

// Used to check if an input flag option implies the related output
template <int I>
bool flag_output_compliance(MxMapper& input, const char* msg, const MxMapper& output, int output_position) {
  const auto flag = input.template getOptional<I, bool>(msg);
  if (flag) {
    const bool flag_value = flag.value();
    if (flag_value) {
      if (output.count() <= output_position) {
        throw MxException(LOCATION(),
                          "mLibKriging:inconsistentOutput",
                          MxMapper::parameterStr<I>(msg),
                          " is set without related output");
      }
    }
    return flag_value;
  } else {
    return output.count() > output_position;
  }
}

void predict(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 3}};
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  const bool withStd = flag_output_compliance<2>(input, "withStd", output, 1);
  const bool withCov = flag_output_compliance<3>(input, "withCov", output, 2);
  auto [y_pred, stderr_v, cov_m] = km->predict(input.get<1, arma::mat>("matrix"), withStd, withCov);
  output.set<0>(y_pred, "predicted response");
  output.setOptional<1>(stderr_v, "stderr vector");
  output.setOptional<2>(cov_m, "cov matrix");
}

void simulate(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  auto result = km->simulate(input.get<1, int>("nsim"), input.get<2, int>("seed"), input.get<3, arma::mat>("Xp"));
  output.set<0>(result, "simulated response");
}

void update(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  km->update(input.get<1, arma::vec>("new y"), input.get<2, arma::mat>("new X"), input.get<3, bool>("normalize"));
}

void summary(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  output.set<0>(km->summary(), "Model description");
}

void logLikelihood(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  const bool want_grad = flag_output_compliance<2>(input, "want_grad", output, 1);
  auto [ll, llgrad] = km->logLikelihoodEval(input.get<1, arma::vec>("theta"), want_grad);
  output.set<0>(ll, "ll");                  // FIXME better name
  output.setOptional<1>(llgrad, "llgrad");  // FIXME better name
}

void logMargPost(int nlhs, void** plhs, int nrhs, const void** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* km = input.getObject<0, NuggetKriging>("NuggetKriging reference");
  const bool want_grad = flag_output_compliance<2>(input, "want_grad", output, 1);
  auto [lmp, lmpgrad] = km->logMargPostEval(input.get<1, arma::vec>("theta"), want_grad);
  output.set<0>(lmp, "lmp");                  // FIXME better name
  output.setOptional<1>(lmpgrad, "lmpgrad");  // FIXME better name
}

}  // namespace NuggetKrigingBinding