// clang-format off
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/WarpKriging.hpp"

using namespace libKriging;

// Store C++ object via Rcpp::XPtr
typedef Rcpp::XPtr<WarpKriging> WarpKrigingPtr;

// ---------------------------------------------------------------------------
//  Constructor:  warping is a character vector of spec strings
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP warpKriging_new(const arma::vec& y,
                     const arma::mat& X,
                     Rcpp::CharacterVector warping,
                     std::string kernel,
                     std::string regmodel = "constant",
                     bool normalize = false,
                     std::string optim = "BFGS+Adam",
                     std::string objective = "LL",
                     Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  // Convert R CharacterVector → std::vector<std::string>
  std::vector<std::string> warp_strs;
  for (int i = 0; i < warping.size(); ++i)
    warp_strs.push_back(Rcpp::as<std::string>(warping[i]));

  // Parse optional parameters
  std::map<std::string, std::string> params;
  if (parameters.isNotNull()) {
    Rcpp::List plist(parameters);
    Rcpp::CharacterVector names = plist.names();
    for (int i = 0; i < plist.size(); ++i)
      params[Rcpp::as<std::string>(names[i])] = Rcpp::as<std::string>(plist[i]);
  }

  WarpKriging* model = new WarpKriging(y, X, warp_strs, kernel, regmodel, normalize, optim, objective, params);

  return WarpKrigingPtr(model, true);
}

// ---------------------------------------------------------------------------
//  fit
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
void warpKriging_fit(SEXP model_ptr,
                     const arma::vec& y,
                     const arma::mat& X,
                     std::string regmodel = "constant",
                     bool normalize = false,
                     std::string optim = "BFGS+Adam",
                     std::string objective = "LL",
                     Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  WarpKrigingPtr model(model_ptr);

  std::map<std::string, std::string> params;
  if (parameters.isNotNull()) {
    Rcpp::List plist(parameters);
    Rcpp::CharacterVector names = plist.names();
    for (int i = 0; i < plist.size(); ++i)
      params[Rcpp::as<std::string>(names[i])] = Rcpp::as<std::string>(plist[i]);
  }

  model->fit(y, X, regmodel, normalize, optim, objective, params);
}

// ---------------------------------------------------------------------------
//  predict
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List warpKriging_predict(SEXP model_ptr,
                               const arma::mat& x_new,
                               bool withStd = true,
                               bool withCov = false,
                               bool withDeriv = false) {
  WarpKrigingPtr model(model_ptr);
  auto [mean, stdev, cov, mean_deriv, stdev_deriv] = model->predict(x_new, withStd, withCov, withDeriv);

  Rcpp::List result;
  result["mean"] = mean;
  if (withStd)
    result["stdev"] = stdev;
  if (withCov)
    result["cov"] = cov;
  if (withDeriv) {
    result["mean_deriv"] = mean_deriv;
    result["stdev_deriv"] = stdev_deriv;
  }
  return result;
}

// ---------------------------------------------------------------------------
//  simulate
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
arma::mat warpKriging_simulate(SEXP model_ptr, int nsim, int seed, const arma::mat& x_new, bool will_update = false) {
  WarpKrigingPtr model(model_ptr);
  return model->simulate(nsim, static_cast<uint64_t>(seed), x_new, will_update);
}

// [[Rcpp::export]]
arma::mat warpKriging_update_simulate(SEXP model_ptr, const arma::vec& y_u, const arma::mat& X_u) {
  WarpKrigingPtr model(model_ptr);
  return model->update_simulate(y_u, X_u);
}

// ---------------------------------------------------------------------------
//  update
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
void warpKriging_update(SEXP model_ptr, const arma::vec& y_new, const arma::mat& X_new, bool refit = true) {
  WarpKrigingPtr model(model_ptr);
  model->update(y_new, X_new, refit);
}

// ---------------------------------------------------------------------------
//  logLikelihood
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double warpKriging_logLikelihood(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->logLikelihood();
}

// [[Rcpp::export]]
Rcpp::List warpKriging_logLikelihoodFun(SEXP model_ptr,
                                        const arma::vec& theta_gp,
                                        bool withGrad = true,
                                        bool withHess = false) {
  WarpKrigingPtr model(model_ptr);
  auto [ll, grad, hess] = model->logLikelihoodFun(theta_gp, withGrad, withHess);

  Rcpp::List result;
  result["logLikelihood"] = ll;
  if (withGrad)
    result["gradient"] = grad;
  if (withHess)
    result["hessian"] = hess;
  return result;
}

// ---------------------------------------------------------------------------
//  Accessors
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
std::string warpKriging_summary(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->summary();
}

// [[Rcpp::export]]
arma::vec warpKriging_theta(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->theta();
}

// [[Rcpp::export]]
double warpKriging_sigma2(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->sigma2();
}

// [[Rcpp::export]]
std::string warpKriging_kernel(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->kernel();
}

// [[Rcpp::export]]
int warpKriging_featureDim(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return static_cast<int>(model->feature_dim());
}

// [[Rcpp::export]]
Rcpp::CharacterVector warpKriging_warping(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  auto ws = model->warping_strings();
  Rcpp::CharacterVector out(ws.size());
  for (size_t i = 0; i < ws.size(); ++i)
    out[i] = ws[i];
  return out;
}

// [[Rcpp::export]]
bool warpKriging_isFitted(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->is_fitted();
}

// [[Rcpp::export]]
arma::mat warpKriging_X(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->X();
}

// [[Rcpp::export]]
arma::vec warpKriging_y(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  return model->y();
}

// [[Rcpp::export]]
SEXP warpKriging_copy(SEXP model_ptr) {
  WarpKrigingPtr model(model_ptr);
  auto clone = new WarpKriging(model->warping_strings(), model->kernel());
  if (model->is_fitted()) {
    clone->fit(model->y(), model->X());
  }
  Rcpp::XPtr<WarpKriging> clone_ptr(clone);
  return clone_ptr;
}

// [[Rcpp::export]]
void warpKriging_save(SEXP model_ptr, std::string filename) {
  WarpKrigingPtr model(model_ptr);
  model->save(filename);
}
