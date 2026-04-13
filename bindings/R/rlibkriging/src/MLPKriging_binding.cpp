// clang-format off
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/MLPKriging.hpp"

using namespace libKriging;

typedef Rcpp::XPtr<MLPKriging> MLPKrigingPtr;

// ---------------------------------------------------------------------------
//  Constructor
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
SEXP mlpKriging_new(const arma::vec& y,
                    const arma::mat& X,
                    Rcpp::IntegerVector hidden_dims,
                    int d_out = 2,
                    std::string activation = "selu",
                    std::string kernel = "gauss",
                    std::string regmodel = "constant",
                    bool normalize = false,
                    std::string optim = "BFGS+Adam",
                    std::string objective = "LL",
                    Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  std::vector<arma::uword> hd;
  for (int i = 0; i < hidden_dims.size(); ++i)
    hd.push_back(static_cast<arma::uword>(hidden_dims[i]));

  std::map<std::string, std::string> params;
  if (parameters.isNotNull()) {
    Rcpp::List plist(parameters);
    Rcpp::CharacterVector names = plist.names();
    for (int i = 0; i < plist.size(); ++i)
      params[Rcpp::as<std::string>(names[i])] = Rcpp::as<std::string>(plist[i]);
  }

  MLPKriging* model = new MLPKriging(
      y, X, hd, static_cast<arma::uword>(d_out), activation, kernel, regmodel, normalize, optim, objective, params);

  return MLPKrigingPtr(model, true);
}

// ---------------------------------------------------------------------------
//  fit
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
void mlpKriging_fit(SEXP model_ptr,
                    const arma::vec& y,
                    const arma::mat& X,
                    std::string regmodel = "constant",
                    bool normalize = false,
                    std::string optim = "BFGS+Adam",
                    std::string objective = "LL",
                    Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  MLPKrigingPtr model(model_ptr);

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
Rcpp::List mlpKriging_predict(SEXP model_ptr,
                              const arma::mat& x_new,
                              bool withStd = true,
                              bool withCov = false,
                              bool withDeriv = false) {
  MLPKrigingPtr model(model_ptr);
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
arma::mat mlpKriging_simulate(SEXP model_ptr, int nsim, int seed, const arma::mat& x_new) {
  MLPKrigingPtr model(model_ptr);
  return model->simulate(nsim, static_cast<uint64_t>(seed), x_new);
}

// ---------------------------------------------------------------------------
//  update
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
void mlpKriging_update(SEXP model_ptr, const arma::vec& y_new, const arma::mat& X_new) {
  MLPKrigingPtr model(model_ptr);
  model->update(y_new, X_new);
}

// ---------------------------------------------------------------------------
//  logLikelihood
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double mlpKriging_logLikelihood(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->logLikelihood();
}

// [[Rcpp::export]]
Rcpp::List mlpKriging_logLikelihoodFun(SEXP model_ptr,
                                       const arma::vec& theta_gp,
                                       bool withGrad = true,
                                       bool withHess = false) {
  MLPKrigingPtr model(model_ptr);
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
std::string mlpKriging_summary(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->summary();
}

// [[Rcpp::export]]
arma::vec mlpKriging_theta(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->theta();
}

// [[Rcpp::export]]
double mlpKriging_sigma2(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->sigma2();
}

// [[Rcpp::export]]
std::string mlpKriging_kernel(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->kernel();
}

// [[Rcpp::export]]
int mlpKriging_featureDim(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return static_cast<int>(model->feature_dim());
}

// [[Rcpp::export]]
Rcpp::IntegerVector mlpKriging_hiddenDims(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  const auto& hd = model->hidden_dims();
  Rcpp::IntegerVector out(hd.size());
  for (size_t i = 0; i < hd.size(); ++i)
    out[i] = static_cast<int>(hd[i]);
  return out;
}

// [[Rcpp::export]]
std::string mlpKriging_activation(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->activation();
}

// [[Rcpp::export]]
bool mlpKriging_isFitted(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->is_fitted();
}

// [[Rcpp::export]]
arma::mat mlpKriging_X(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->X();
}

// [[Rcpp::export]]
arma::vec mlpKriging_y(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  return model->y();
}

// [[Rcpp::export]]
SEXP mlpKriging_copy(SEXP model_ptr) {
  MLPKrigingPtr model(model_ptr);
  auto* clone = new MLPKriging(model->hidden_dims(), model->d_out(), model->activation(), model->kernel());
  if (model->is_fitted()) {
    clone->fit(model->y(), model->X());
  }
  Rcpp::XPtr<MLPKriging> clone_ptr(clone);
  return clone_ptr;
}

// [[Rcpp::export]]
void mlpKriging_save(SEXP model_ptr, std::string filename) {
  MLPKrigingPtr model(model_ptr);
  model->save(filename);
}
