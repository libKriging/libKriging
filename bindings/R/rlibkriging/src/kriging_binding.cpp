// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Kriging.hpp"

// [[Rcpp::export]]
Rcpp::List new_Kriging(arma::vec y,
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
    if (params.containsElementNamed("beta")) {
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(params["beta"]), "beta");
      _parameters.push_back(true, "has_beta");
    } else {
      _parameters.push_back(Rcpp::NumericVector(0), "beta");
      _parameters.push_back(false, "has_beta");
    }
  } else {
    _parameters = Rcpp::List::create(Rcpp::Named("sigma2") = -1,
                                     Rcpp::Named("has_sigma2") = false,
                                     Rcpp::Named("theta") = Rcpp::NumericMatrix(0),
                                     Rcpp::Named("has_theta") = false,
                                     Rcpp::Named("beta") = Rcpp::NumericVector(0),
                                     Rcpp::Named("has_beta") = false);
  }

  ok->fit(std::move(y),
          std::move(X),
          Kriging::RegressionModelUtils::fromString(regmodel),
          normalize,
          optim,
          objective,
          Kriging::Parameters{_parameters["sigma2"],
                              _parameters["has_sigma2"],
                              _parameters["theta"],
                              _parameters["has_theta"],
                              _parameters["beta"],
                              _parameters["has_beta"]});

  Rcpp::XPtr<Kriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "Kriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List kriging_model(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return Rcpp::List::create(Rcpp::Named("kernel") = impl_ptr->kernel(),
                            Rcpp::Named("optim") = impl_ptr->optim(),
                            Rcpp::Named("objective") = impl_ptr->objective(),
                            Rcpp::Named("theta") = impl_ptr->theta(),
                            Rcpp::Named("estim_theta") = impl_ptr->estim_theta(),
                            Rcpp::Named("sigma2") = impl_ptr->sigma2(),
                            Rcpp::Named("estim_sigma2") = impl_ptr->estim_sigma2(),
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
                            Rcpp::Named("beta") = impl_ptr->beta(),
                            Rcpp::Named("estim_beta") = impl_ptr->estim_beta());
}

// [[Rcpp::export]]
Rcpp::List kriging_predict(Rcpp::List k, arma::mat X, bool stdev = true, bool cov = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  auto pred = impl_ptr->predict(X, stdev, cov);
  if (stdev & cov) {
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred),
                              Rcpp::Named("stdev") = std::get<1>(pred),
                              Rcpp::Named("cov") = std::get<2>(pred));
  } else if (stdev & !cov) {
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred), Rcpp::Named("stdev") = std::get<1>(pred));
  } else if (!stdev & cov) {
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred), Rcpp::Named("cov") = std::get<2>(pred));
  } else if (!stdev & !cov) {
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred));
  }
}

// [[Rcpp::export]]
arma::mat kriging_simulate(Rcpp::List k, int nsim, int seed, arma::mat X) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->simulate(nsim, seed, X);
}

// [[Rcpp::export]]
void kriging_update(Rcpp::List k, arma::vec y, arma::mat X, bool normalize = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  impl_ptr->update(y, X, normalize);

  // Rcpp::List obj;
  // obj.attr("object") = impl_ptr;
  // obj.attr("class") = "Kriging";
  // return obj;
}

// [[Rcpp::export]]
Rcpp::List kriging_logLikelihood(Rcpp::List k, arma::vec theta, bool grad = false, bool hess = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<double, arma::vec, arma::mat> ll = impl_ptr->logLikelihoodEval(theta, grad, hess);
  if (hess) {
    return Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll),
                              Rcpp::Named("logLikelihoodGrad") = std::get<1>(ll),
                              Rcpp::Named("logLikelihoodHess") = std::get<2>(ll));
  } else if (grad & !hess) {
    return Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll),
                              Rcpp::Named("logLikelihoodGrad") = std::get<1>(ll));
  } else if (!grad & !hess) {
    return Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll));
  }
}

// [[Rcpp::export]]
Rcpp::List kriging_leaveOneOut(Rcpp::List k, arma::vec theta, bool grad = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<double, arma::vec> loo = impl_ptr->leaveOneOutEval(theta, grad);
  if (grad) {
    return Rcpp::List::create(Rcpp::Named("leaveOneOut") = std::get<0>(loo),
                              Rcpp::Named("leaveOneOutGrad") = std::get<1>(loo));
  } else {
    return Rcpp::List::create(Rcpp::Named("leaveOneOut") = std::get<0>(loo));
  }
}

// [[Rcpp::export]]
Rcpp::List kriging_logMargPost(Rcpp::List k, arma::vec theta, bool grad = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<double, arma::vec> lmp = impl_ptr->logMargPostEval(theta, grad);
  if (grad) {
    return Rcpp::List::create(Rcpp::Named("logMargPost") = std::get<0>(lmp),
                              Rcpp::Named("logMargPostGrad") = std::get<1>(lmp));
  } else {
    return Rcpp::List::create(Rcpp::Named("logMargPost") = std::get<0>(lmp));
  }
}
