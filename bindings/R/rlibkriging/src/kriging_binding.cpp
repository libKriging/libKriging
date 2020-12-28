// clang-format off
// Must before any other include
#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Kriging.hpp"

// [[Rcpp::export]]
Rcpp::List kriging(arma::vec y,
                   arma::mat X,
                   std::string kernel,
                   std::string regmodel = "constant",
                   bool normalize = false,
                   std::string optim = "BFGS",
                   std::string objective = "LL",
                   Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  Kriging* ok = new Kriging(kernel);

  Rcpp::List _parameters;
  if (parameters.isNotNull()) {
    Rcpp::List params(parameters);
    _parameters = Rcpp::List::create();
    if (params.containsElementNamed("sigma2")) {
      _parameters.push_back(params["sigma2"], "sigma2");
      _parameters.push_back(true, "has_sigma2");
    } else {
      _parameters.push_back(-1, "sigma2");
      _parameters.push_back(false, "has_sigma2");
    }
    if (params.containsElementNamed("theta")) {
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(params["theta"]), "theta");
      _parameters.push_back(true, "has_theta");
    } else {
      _parameters.push_back(Rcpp::NumericMatrix(0), "theta");
      _parameters.push_back(false, "has_theta");
    }
  } else {
    _parameters = Rcpp::List::create(Rcpp::Named("sigma2") = -1,
                                     Rcpp::Named("has_sigma2") = false,
                                     Rcpp::Named("theta") = Rcpp::NumericMatrix(0),
                                     Rcpp::Named("has_theta") = false);
  }

  ok->fit(std::move(y),
          std::move(X),
          Kriging::RegressionModelUtils::fromString(regmodel),
          normalize,
          optim,
          objective,
          Kriging::Parameters{
              _parameters["sigma2"], _parameters["has_sigma2"], _parameters["theta"], _parameters["has_theta"]});

  Rcpp::XPtr<Kriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "Kriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List kriging_model(Rcpp::List ordinaryKriging) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return Rcpp::List::create(Rcpp::Named("theta") = impl_ptr->theta(),
                            Rcpp::Named("sigma2") = impl_ptr->sigma2(),
                            Rcpp::Named("X") = impl_ptr->X(),
                            Rcpp::Named("centerX") = impl_ptr->centerX(),
                            Rcpp::Named("scaleX") = impl_ptr->scaleX(),
                            Rcpp::Named("y") = impl_ptr->y(),
                            Rcpp::Named("centerY") = impl_ptr->centerY(),
                            Rcpp::Named("scaleY") = impl_ptr->scaleY(),
                            Rcpp::Named("regmodel") = Kriging::RegressionModelUtils::toString(impl_ptr->regmodel()),
                            Rcpp::Named("F") = impl_ptr->F(),
                            Rcpp::Named("T") = impl_ptr->T(),
                            Rcpp::Named("M") = impl_ptr->M(),
                            Rcpp::Named("z") = impl_ptr->z(),
                            Rcpp::Named("beta") = impl_ptr->beta());
}

// [[Rcpp::export]]
double kriging_logLikelihood(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->logLikelihoodFun(theta);
}

// [[Rcpp::export]]
double kriging_leaveOneOut(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->leaveOneOutFun(theta);
}

// [[Rcpp::export]]
arma::vec kriging_leaveOneOutGrad(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->leaveOneOutGrad(theta);
}

// [[Rcpp::export]]
arma::vec kriging_logLikelihoodGrad(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->logLikelihoodGrad(theta);
}

// [[Rcpp::export]]
arma::mat kriging_logLikelihoodHess(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->logLikelihoodHess(theta);
}

// [[Rcpp::export]]
Rcpp::List kriging_predict(Rcpp::List ordinaryKriging, arma::mat X, bool stdev, bool cov) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  if (stdev & cov) {
    auto pred = impl_ptr->predict(X, true, true);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred),
                              Rcpp::Named("stdev") = std::get<1>(pred),
                              Rcpp::Named("cov") = std::get<2>(pred));
  } else if (stdev & !cov) {
    auto pred = impl_ptr->predict(X, true, false);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred), Rcpp::Named("stdev") = std::get<1>(pred));
  } else if (!stdev & cov) {
    auto pred = impl_ptr->predict(X, false, true);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred), Rcpp::Named("cov") = std::get<2>(pred));
  } else if (!stdev & !cov) {
    auto pred = impl_ptr->predict(X, false, false);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred));
  }

  // FIXME no default return
}

// [[Rcpp::export]]
arma::mat kriging_simulate(Rcpp::List ordinaryKriging, int nsim, arma::mat X) {
  if (!ordinaryKriging.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->simulate(nsim, X);
}
