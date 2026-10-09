// clang-format off
// Must before any Rcpp code
#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/MultiOutputKriging.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/ExplicitCopySpecifier.hpp"

#include "retrofit_utils.hpp"

// MultiOutputKriging::Parameters from an R named list: theta (one row per
// starting point), is_theta_estim and output_theta ("separable(<kernel>)")
static MultiOutputKriging::Parameters mo_parameters_from_list(Rcpp::Nullable<Rcpp::List> parameters) {
  MultiOutputKriging::Parameters out;
  if (parameters.isNotNull()) {
    Rcpp::List params(parameters);
    if (params.size() > 0) {
      Rcpp::CharacterVector names = params.names();
      for (R_xlen_t i = 0; i < names.size(); ++i) {
        const std::string key = Rcpp::as<std::string>(names[i]);
        if (key != "theta" && key != "is_theta_estim" && key != "output_theta")
          Rcpp::stop(
              "MultiOutputKriging: unsupported parameter '%s' (only 'theta', 'is_theta_estim' and 'output_theta')",
              key.c_str());
      }
    }
    if (params.containsElementNamed("theta"))
      out.theta = Rcpp::as<arma::mat>(params["theta"]);
    if (params.containsElementNamed("is_theta_estim"))
      out.is_theta_estim = Rcpp::as<bool>(params["is_theta_estim"]);
    if (params.containsElementNamed("output_theta"))
      out.output_theta = Rcpp::as<arma::mat>(params["output_theta"]);
  }
  return out;
}

static Rcpp::XPtr<MultiOutputKriging> mo_ptr(Rcpp::List k) {
  if (!k.inherits("MultiOutputKriging"))
    Rcpp::stop("Input must be a MultiOutputKriging object");
  SEXP impl = k.attr("object");
  return Rcpp::XPtr<MultiOutputKriging>(impl);
}

static Rcpp::List mo_wrap(MultiOutputKriging* mo) {
  Rcpp::XPtr<MultiOutputKriging> impl_ptr(mo);
  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "MultiOutputKriging";
  return obj;
}

// arma::cube -> R array with the same dimensions
static Rcpp::NumericVector cube_to_array(const arma::cube& c) {
  Rcpp::NumericVector out(c.begin(), c.end());
  out.attr("dim") = Rcpp::IntegerVector::create(c.n_rows, c.n_cols, c.n_slices);
  return out;
}

// [[Rcpp::export]]
Rcpp::List new_MultiOutputKriging(std::string kernel, std::string output_model = "pca") {
  return mo_wrap(new MultiOutputKriging(kernel, output_model));
}

// [[Rcpp::export]]
Rcpp::List new_MultiOutputKrigingFit(arma::mat Y,
                                     arma::mat X,
                                     std::string kernel,
                                     std::string output_model = "pca",
                                     std::string regmodel = "constant",
                                     bool normalize = false,
                                     std::string optim = "BFGS",
                                     std::string objective = "LL",
                                     Rcpp::Nullable<Rcpp::List> parameters = R_NilValue,
                                     Rcpp::Nullable<Rcpp::NumericMatrix> output_coordinates = R_NilValue) {
  auto* mo = new MultiOutputKriging(kernel, output_model);
  try {
    if (output_coordinates.isNotNull())
      mo->set_output_coordinates(Rcpp::as<arma::mat>(output_coordinates));
    mo->fit(Y, X, Trend::fromString(regmodel), normalize, optim, objective, mo_parameters_from_list(parameters));
  } catch (...) {
    delete mo;
    throw;
  }
  return mo_wrap(mo);
}

// [[Rcpp::export]]
void multioutputkriging_fit(Rcpp::List k,
                            arma::mat Y,
                            arma::mat X,
                            std::string regmodel = "constant",
                            bool normalize = false,
                            std::string optim = "BFGS",
                            std::string objective = "LL",
                            Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  mo_ptr(k)->fit(Y, X, Trend::fromString(regmodel), normalize, optim, objective, mo_parameters_from_list(parameters));
}

// [[Rcpp::export]]
void multioutputkriging_set_output_coordinates(Rcpp::List k, arma::mat t) {
  mo_ptr(k)->set_output_coordinates(t);
}

// [[Rcpp::export]]
Rcpp::List multioutputkriging_predict(Rcpp::List k,
                                      arma::mat X_n,
                                      bool return_stdev = true,
                                      bool return_cov = false,
                                      bool return_deriv = false) {
  auto [mean, stdev, cov, deriv] = mo_ptr(k)->predict(X_n, return_stdev, return_cov, return_deriv);
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("mean") = mean);
  if (return_stdev)
    out.push_back(stdev, "stdev");
  if (return_cov)
    out.push_back(cov, "cov");
  if (return_deriv)
    out.push_back(cube_to_array(deriv), "mean_deriv");
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector multioutputkriging_simulate(Rcpp::List k,
                                                int nsim,
                                                int seed,
                                                arma::mat X_n,
                                                bool will_update = false) {
  return cube_to_array(mo_ptr(k)->simulate(nsim, seed, X_n, will_update));
}

// [[Rcpp::export]]
Rcpp::NumericVector multioutputkriging_update_simulate(Rcpp::List k, arma::mat Y_u, arma::mat X_u) {
  return cube_to_array(mo_ptr(k)->update_simulate(Y_u, X_u));
}

// [[Rcpp::export]]
void multioutputkriging_update(Rcpp::List k, arma::mat Y_u, arma::mat X_u, bool refit = true) {
  mo_ptr(k)->update(Y_u, X_u, refit);
}

// [[Rcpp::export]]
Rcpp::List multioutputkriging_leaveOneOutMat(Rcpp::List k) {
  auto [mean, stdev] = mo_ptr(k)->leaveOneOutMat();
  return Rcpp::List::create(Rcpp::Named("mean") = mean, Rcpp::Named("stdev") = stdev);
}

// [[Rcpp::export]]
double multioutputkriging_leaveOneOut(Rcpp::List k) {
  return mo_ptr(k)->leaveOneOut();
}

// [[Rcpp::export]]
double multioutputkriging_logLikelihood(Rcpp::List k) {
  return mo_ptr(k)->logLikelihood();
}

// [[Rcpp::export]]
Rcpp::List multioutputkriging_logLikelihoodFun(Rcpp::List k, arma::vec theta, bool return_grad = false) {
  auto [ll, grad] = mo_ptr(k)->logLikelihoodFun(theta, return_grad);
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("logLikelihood") = ll);
  if (return_grad)
    out.push_back(grad, "logLikelihoodGrad");
  return out;
}

// [[Rcpp::export]]
Rcpp::List multioutputkriging_leaveOneOutFun(Rcpp::List k, arma::vec theta, bool return_grad = false) {
  auto [loo, grad] = mo_ptr(k)->leaveOneOutFun(theta, return_grad);
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("leaveOneOut") = loo);
  if (return_grad)
    out.push_back(grad, "leaveOneOutGrad");
  return out;
}

// [[Rcpp::export]]
Rcpp::List multioutputkriging_predictCovFactors(Rcpp::List k, arma::mat X_n) {
  auto [Cx, S] = mo_ptr(k)->predictCovFactors(X_n);
  return Rcpp::List::create(Rcpp::Named("Cx") = Cx, Rcpp::Named("Sigma") = S);
}

// [[Rcpp::export]]
std::string multioutputkriging_summary(Rcpp::List k) {
  return mo_ptr(k)->summary();
}

// [[Rcpp::export]]
std::string multioutputkriging_kernel(Rcpp::List k) {
  return mo_ptr(k)->kernel();
}

// [[Rcpp::export]]
std::string multioutputkriging_output_model(Rcpp::List k) {
  return mo_ptr(k)->output_model_string();
}

// [[Rcpp::export]]
unsigned long multioutputkriging_nb_outputs(Rcpp::List k) {
  return mo_ptr(k)->nb_outputs();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_X(Rcpp::List k) {
  return mo_ptr(k)->X();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_Y(Rcpp::List k) {
  return mo_ptr(k)->Y();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_output_coordinates(Rcpp::List k) {
  return mo_ptr(k)->output_coordinates();
}

// [[Rcpp::export]]
std::string multioutputkriging_regmodel(Rcpp::List k) {
  return Trend::toString(mo_ptr(k)->regmodel());
}

// [[Rcpp::export]]
bool multioutputkriging_normalize(Rcpp::List k) {
  return mo_ptr(k)->normalize();
}

// [[Rcpp::export]]
std::string multioutputkriging_optim(Rcpp::List k) {
  return mo_ptr(k)->optim();
}

// [[Rcpp::export]]
std::string multioutputkriging_objective(Rcpp::List k) {
  return mo_ptr(k)->objective();
}

// [[Rcpp::export]]
arma::rowvec multioutputkriging_centerY(Rcpp::List k) {
  return mo_ptr(k)->centerY();
}

// [[Rcpp::export]]
arma::rowvec multioutputkriging_scaleY(Rcpp::List k) {
  return mo_ptr(k)->scaleY();
}

// [[Rcpp::export]]
arma::vec multioutputkriging_theta(Rcpp::List k) {
  return mo_ptr(k)->theta();
}

// [[Rcpp::export]]
arma::vec multioutputkriging_sigma2(Rcpp::List k) {
  return mo_ptr(k)->sigma2();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_beta(Rcpp::List k) {
  return mo_ptr(k)->beta();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_output_cov(Rcpp::List k) {
  return mo_ptr(k)->output_cov();
}

// [[Rcpp::export]]
arma::vec multioutputkriging_output_theta(Rcpp::List k) {
  return mo_ptr(k)->output_theta();
}

// [[Rcpp::export]]
void multioutputkriging_save(Rcpp::List k, std::string filename) {
  mo_ptr(k)->save(filename);
}

// [[Rcpp::export]]
Rcpp::List multioutputkriging_load(std::string filename) {
  return mo_wrap(new MultiOutputKriging(MultiOutputKriging::load(filename)));
}

// [[Rcpp::export]]
unsigned long multioutputkriging_nb_components(Rcpp::List k) {
  return mo_ptr(k)->nb_components();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_pca_basis(Rcpp::List k) {
  return mo_ptr(k)->pca_basis();
}

// [[Rcpp::export]]
arma::vec multioutputkriging_pca_explained(Rcpp::List k) {
  return mo_ptr(k)->pca_explained();
}

// [[Rcpp::export]]
arma::mat multioutputkriging_pca_residual(Rcpp::List k) {
  return mo_ptr(k)->pca_residual();
}

// Copy of latent Kriging number `index` (1-based), as a "Kriging" object
// [[Rcpp::export]]
Rcpp::List multioutputkriging_component(Rcpp::List k, unsigned long index) {
  auto mo = mo_ptr(k);
  if (index < 1)
    Rcpp::stop("MultiOutputKriging: component index is 1-based");
  Rcpp::XPtr<Kriging> impl_copy(new Kriging(mo->component(index - 1), ExplicitCopySpecifier{}));
  Rcpp::List obj;
  obj.attr("object") = impl_copy;
  obj.attr("class") = "Kriging";
  return obj;
}
