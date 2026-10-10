#include "MultiOutputKriging_binding.hpp"

#include "libKriging/MultiOutputKriging.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"

#include "Params.hpp"
#include "common_binding.hpp"
#include "tools/MxMapper.hpp"
#include "tools/ObjectAccessor.hpp"

// matrix entry, a scalar being taken as 1 x 1 (one-dimensional theta)
static std::optional<arma::mat> get_mat(const Params& params, const std::string& key) {
  try {
    return params.get<arma::mat>(key);
  } catch (const MxException&) {
    return arma::mat(1, 1, arma::fill::value(params.get<double>(key).value()));
  }
}

// theta (one row per starting point), is_theta_estim and output_theta ("separable(<kernel>)")
static MultiOutputKriging::Parameters makeParameters(std::optional<Params*> dict) {
  MultiOutputKriging::Parameters p;
  if (dict) {
    const Params& params = *dict.value();
    p.theta = get_mat(params, "theta");
    p.is_theta_estim = params.get<bool>("is_theta_estim").value_or(true);
    p.output_theta = get_mat(params, "output_theta");
  }
  return p;
}

// 1-argument accessor: MultiOutputKriging::<name>(ref) -> value
template <typename F>
static void accessor(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs, const char* what, F&& f) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  output.set(0, f(*mo), what);
}

namespace MultiOutputKrigingBinding {

// MultiOutputKriging::new(Y, X, kernel, [output_model], [regmodel], [normalize], [optim], [objective],
//                         [parameters], [output_coordinates])
void build(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 10}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto Y = input.get<arma::mat>(0, "Y matrix");
  auto X = input.get<arma::mat>(1, "X matrix");
  auto kernel = input.get<std::string>(2, "kernel");
  const auto output_model = input.getOptional<std::string>(3, "output model").value_or("pca");
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(4, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(5, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(6, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(7, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(8, "parameters"));
  const auto t = input.getOptional<arma::mat>(9, "output coordinates");

  MultiOutputKriging mo_obj(kernel, output_model);
  if (t && t->n_elem > 0)
    mo_obj.set_output_coordinates(*t);
  mo_obj.fit(Y, X, regmodel, normalize, optim, objective, parameters);
  auto mo = buildObject<MultiOutputKriging>(std::move(mo_obj));
  output.set(0, mo, "new object reference");
}

// MultiOutputKriging::new_empty(kernel, [output_model])
void build_empty(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{1, 2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto kernel = input.get<std::string>(0, "kernel");
  const auto output_model = input.getOptional<std::string>(1, "output model").value_or("pca");
  auto mo = buildObject<MultiOutputKriging>(kernel, output_model);
  output.set(0, mo, "new object reference");
}

// MultiOutputKriging::save(ref, filename)
void save(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  mo->save(input.get<std::string>(1, "filename"));
}

// MultiOutputKriging::load(filename) -> new object reference
void load(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto mo = buildObject<MultiOutputKriging>(MultiOutputKriging::load(input.get<std::string>(0, "filename")));
  output.set(0, mo, "new object reference");
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

// MultiOutputKriging::fit(ref, Y, X, [regmodel], [normalize], [optim], [objective], [parameters])
void fit(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 8}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const auto regmodel = Trend::fromString(input.getOptional<std::string>(3, "regression model").value_or("constant"));
  const auto normalize = input.getOptional<bool>(4, "normalize").value_or(false);
  const auto optim = input.getOptional<std::string>(5, "optim").value_or("BFGS");
  const auto objective = input.getOptional<std::string>(6, "objective").value_or("LL");
  const auto parameters = makeParameters(input.getOptionalObject<Params>(7, "parameters"));
  mo->fit(input.get<arma::mat>(1, "Y matrix"),
          input.get<arma::mat>(2, "X matrix"),
          regmodel,
          normalize,
          optim,
          objective,
          parameters);
}

void set_output_coordinates(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  mo->set_output_coordinates(input.get<arma::mat>(1, "output coordinates"));
}

// [mean, stdev, cov, mean_deriv] = MultiOutputKriging::predict(ref, X, [return_stdev], [return_cov], [return_deriv])
void predict(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 4}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const bool return_stdev = flag_output_compliance(input, 2, "return_stdev", output, 1);
  const bool return_cov = flag_output_compliance(input, 3, "return_cov", output, 2);
  const bool return_deriv = flag_output_compliance(input, 4, "return_deriv", output, 3);
  auto [mean, stdev, cov, deriv]
      = mo->predict(input.get<arma::mat>(1, "X matrix"), return_stdev, return_cov, return_deriv);
  output.set(0, mean, "predicted mean (m x q)");
  output.setOptional(1, stdev, "predicted stdev (m x q)");
  output.setOptional(2, cov, "predicted covariance (mq x mq)");
  output.setOptional(3, deriv, "mean derivative (m x d x q)");
}

// sims = MultiOutputKriging::simulate(ref, nsim, seed, X, [will_update]) -> m x q x nsim
void simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{4, 5}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const auto nsim = input.get<int32_t>(1, "nsim");
  const auto seed = input.get<int32_t>(2, "seed");
  const bool will_update = input.getOptional<bool>(4, "will_update").value_or(false);
  output.set(0, mo->simulate(nsim, seed, input.get<arma::mat>(3, "X matrix"), will_update), "simulations");
}

void update_simulate(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  output.set(0,
             mo->update_simulate(input.get<arma::mat>(1, "Y_u matrix"), input.get<arma::mat>(2, "X_u matrix")),
             "updated simulations");
}

void update(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{3, 4}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{0}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const bool refit = input.getOptional<bool>(3, "refit").value_or(true);
  mo->update(input.get<arma::mat>(1, "Y_u matrix"), input.get<arma::mat>(2, "X_u matrix"), refit);
}

void leaveOneOutMat(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{1}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  auto [mean, stdev] = mo->leaveOneOutMat();
  output.set(0, mean, "LOO mean (n x q)");
  output.setOptional(1, stdev, "LOO stdev (n x q)");
}

void leaveOneOut(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "LOO error", [](MultiOutputKriging& mo) { return mo.leaveOneOut(); });
}

void logLikelihood(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "log-likelihood", [](MultiOutputKriging& mo) { return mo.logLikelihood(); });
}

// [ll, grad] = MultiOutputKriging::logLikelihoodFun(ref, theta, [return_grad])
void logLikelihoodFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const bool return_grad = flag_output_compliance(input, 2, "return_grad", output, 1);
  auto [ll, grad] = mo->logLikelihoodFun(input.get<arma::vec>(1, "theta"), return_grad);
  output.set(0, ll, "log-likelihood");
  output.setOptional(1, grad, "log-likelihood gradient");
}

// [loo, grad] = MultiOutputKriging::leaveOneOutFun(ref, theta, [return_grad])
void leaveOneOutFun(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Range{2, 3}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Range{1, 2}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const bool return_grad = flag_output_compliance(input, 2, "return_grad", output, 1);
  auto [loo, grad] = mo->leaveOneOutFun(input.get<arma::vec>(1, "theta"), return_grad);
  output.set(0, loo, "LOO error");
  output.setOptional(1, grad, "LOO gradient");
}

// [Cx, Sigma] = MultiOutputKriging::predictCovFactors(ref, X)
void predictCovFactors(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{2}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  auto [Cx, S] = mo->predictCovFactors(input.get<arma::mat>(1, "X matrix"));
  output.set(0, Cx, "correlation factor (m x m)");
  output.set(1, S, "output covariance factor (q x q)");
}

// MultiOutputKriging::component(ref, k) -> reference to a copy of latent Kriging k (1-based)
void component(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  MxMapper input{"Input",
                 nrhs,
                 const_cast<mxArray**>(prhs),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
                 RequiresArg::Exactly{2}};
  MxMapper output{"Output", nlhs, plhs, RequiresArg::Exactly{1}};
  auto* mo = input.getObjectFromRef<MultiOutputKriging>(0, "MultiOutputKriging reference");
  const auto k = static_cast<arma::sword>(input.get<double>(1, "component index"));
  if (k < 1)
    throw MxException(LOCATION(), "mLibKriging:badArg", "component index is 1-based");
  auto km = buildObject<Kriging>(mo->component(static_cast<arma::uword>(k - 1)), ExplicitCopySpecifier{});
  output.set(0, km, "Kriging reference");
}

void summary(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "summary", [](MultiOutputKriging& mo) { return mo.summary(); });
}

void kernel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "kernel", [](MultiOutputKriging& mo) { return mo.kernel(); });
}

void output_model(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "output model", [](MultiOutputKriging& mo) { return mo.output_model_string(); });
}

void nb_outputs(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "nb_outputs", [](MultiOutputKriging& mo) {
    return static_cast<double>(mo.nb_outputs());
  });
}

void X(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "X", [](MultiOutputKriging& mo) { return mo.X(); });
}

void Y(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "Y", [](MultiOutputKriging& mo) { return mo.Y(); });
}

void output_coordinates(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(
      nlhs, plhs, nrhs, prhs, "output coordinates", [](MultiOutputKriging& mo) { return mo.output_coordinates(); });
}

void regmodel(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "regmodel", [](MultiOutputKriging& mo) { return Trend::toString(mo.regmodel()); });
}

void normalize(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "normalize", [](MultiOutputKriging& mo) { return mo.normalize(); });
}

void optim(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "optim", [](MultiOutputKriging& mo) { return mo.optim(); });
}

void objective(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "objective", [](MultiOutputKriging& mo) { return mo.objective(); });
}

void centerY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "centerY", [](MultiOutputKriging& mo) { return mo.centerY(); });
}

void scaleY(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "scaleY", [](MultiOutputKriging& mo) { return mo.scaleY(); });
}

void theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "theta", [](MultiOutputKriging& mo) { return mo.theta(); });
}

void output_theta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "output_theta", [](MultiOutputKriging& mo) { return mo.output_theta(); });
}

void sigma2(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "sigma2", [](MultiOutputKriging& mo) { return mo.sigma2(); });
}

void beta(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "beta", [](MultiOutputKriging& mo) { return mo.beta(); });
}

void output_cov(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "output covariance", [](MultiOutputKriging& mo) { return mo.output_cov(); });
}

void nb_components(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "nb_components", [](MultiOutputKriging& mo) {
    return static_cast<double>(mo.nb_components());
  });
}

void pca_basis(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "pca_basis", [](MultiOutputKriging& mo) { return mo.pca_basis(); });
}

void pca_explained(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "pca_explained", [](MultiOutputKriging& mo) { return mo.pca_explained(); });
}

void pca_residual(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  accessor(nlhs, plhs, nrhs, prhs, "pca_residual", [](MultiOutputKriging& mo) { return mo.pca_residual(); });
}

}  // namespace MultiOutputKrigingBinding
