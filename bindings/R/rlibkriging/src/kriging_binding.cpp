// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"

#include <optional>
#include "retrofit_utils.hpp"

// [[Rcpp::export]]
Rcpp::List new_Kriging(std::string kernel) {
  Kriging* ok = new Kriging(kernel);

  Rcpp::XPtr<Kriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "Kriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List new_KrigingFit(arma::vec y,
                          arma::mat X,
                          std::string kernel,
                          std::string regmodel = "constant",
                          bool normalize = false,
                          std::string optim = "BFGS",
                          std::string objective = "LL",
                          Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  Rcpp::List _parameters;
  if (parameters.isNotNull()) {
    Rcpp::List params(parameters);
    _parameters = Rcpp::List::create();
    if (params.containsElementNamed("sigma2")) {
      _parameters.push_back(params["sigma2"], "sigma2");
      _parameters.push_back(true, "has_sigma2");
      _parameters.push_back(
          !(params.containsElementNamed("is_sigma2_estim") && !params["is_sigma2_estim"]) && optim != "none",
          "is_sigma2_estim");
    } else {
      //_parameters.push_back(Rcpp::runif(1), "sigma2"); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(true, "has_sigma2");
      _parameters.push_back(-1, "sigma2");
      _parameters.push_back(false, "has_sigma2");
      _parameters.push_back(true, "is_sigma2_estim");
    }
    if (params.containsElementNamed("theta")) {
      Rcpp::NumericVector theta = Rcpp::as<Rcpp::NumericVector>(params["theta"]);
      if (!theta.hasAttribute("dim")) {
        theta.attr("dim") = Rcpp::Dimension(1, theta.length());
      }
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(theta), "theta");
      _parameters.push_back(true, "has_theta");
      _parameters.push_back(
          !(params.containsElementNamed("is_theta_estim") && !params["is_theta_estim"]) && optim != "none",
          "is_theta_estim");
    } else {
      // Rcpp::NumericVector r = Rcpp::runif(X.n_cols); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(Rcpp::NumericMatrix(1, X.n_cols,
      // r.begin()), "theta"); _parameters.push_back(true, "has_theta");
      _parameters.push_back(Rcpp::NumericVector(0), "theta");
      _parameters.push_back(false, "has_theta");
      _parameters.push_back(true, "is_theta_estim");
    }
    if (params.containsElementNamed("beta")) {
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(params["beta"]), "beta");
      _parameters.push_back(true, "has_beta");
      _parameters.push_back(
          !(params.containsElementNamed("is_beta_estim") && !params["is_beta_estim"]) && optim != "none",
          "is_beta_estim");
    } else {
      _parameters.push_back(Rcpp::NumericVector(0), "beta");
      _parameters.push_back(false, "has_beta");
      _parameters.push_back(true, "is_beta_estim");
    }
  } else {
    // Rcpp::NumericVector r = Rcpp::runif(X.n_cols); // turnaround mingw bug:
    // https://github.com/msys2/MINGW-packages/issues/5019
    _parameters = Rcpp::List::create(  // Rcpp::Named("sigma2") = Rcpp::runif(1),
                                       // Rcpp::Named("has_sigma2") = true,
        Rcpp::Named("sigma2") = -1,
        Rcpp::Named("has_sigma2") = false,
        Rcpp::Named("is_sigma2_estim") = true,
        // Rcpp::Named("theta") = Rcpp::NumericMatrix(1, X.n_cols, r.begin()),
        // Rcpp::Named("has_theta") = true,
        Rcpp::Named("theta") = Rcpp::NumericMatrix(0, 0),
        Rcpp::Named("has_theta") = false,
        Rcpp::Named("is_theta_estim") = true,
        Rcpp::Named("beta") = Rcpp::NumericVector(0),
        Rcpp::Named("has_beta") = false,
        Rcpp::Named("is_beta_estim") = true);
  }

  Kriging* ok = new Kriging(
      std::move(y),
      std::move(X),
      kernel,
      Trend::fromString(regmodel),
      normalize,
      optim,
      objective,
      Kriging::Parameters{(_parameters["has_sigma2"]) ? make_optional0<double>(_parameters["sigma2"]) : std::nullopt,
                          _parameters["is_sigma2_estim"],
                          (_parameters["has_theta"]) ? make_optional0<arma::mat>(_parameters["theta"]) : std::nullopt,
                          _parameters["is_theta_estim"],
                          (_parameters["has_beta"]) ? make_optional0<arma::vec>(_parameters["beta"]) : std::nullopt,
                          _parameters["is_beta_estim"]});

  Rcpp::XPtr<Kriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "Kriging";
  return obj;
}

// [[Rcpp::export]]
void kriging_fit(Rcpp::List k,
                 arma::vec y,
                 arma::mat X,
                 std::string regmodel = "constant",
                 bool normalize = false,
                 std::string optim = "BFGS",
                 std::string objective = "LL",
                 Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  Rcpp::List _parameters;
  if (parameters.isNotNull()) {
    Rcpp::List params(parameters);
    _parameters = Rcpp::List::create();
    if (params.containsElementNamed("sigma2")) {
      _parameters.push_back(params["sigma2"], "sigma2");
      _parameters.push_back(true, "has_sigma2");
      _parameters.push_back(
          !(params.containsElementNamed("is_sigma2_estim") && !params["is_sigma2_estim"]) && optim != "none",
          "is_sigma2_estim");
    } else {
      //_parameters.push_back(Rcpp::runif(1), "sigma2"); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(true, "has_sigma2");
      _parameters.push_back(-1, "sigma2");
      _parameters.push_back(false, "has_sigma2");
      _parameters.push_back(true, "is_sigma2_estim");
    }
    if (params.containsElementNamed("theta")) {
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(params["theta"]), "theta");
      _parameters.push_back(true, "has_theta");
      _parameters.push_back(
          !(params.containsElementNamed("is_theta_estim") && !params["is_theta_estim"]) && optim != "none",
          "is_theta_estim");
    } else {
      // Rcpp::NumericVector r = Rcpp::runif(X.n_cols); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(Rcpp::NumericMatrix(1, X.n_cols,
      // r.begin()), "theta"); _parameters.push_back(true, "has_theta");
      _parameters.push_back(Rcpp::NumericVector(0), "theta");
      _parameters.push_back(false, "has_theta");
      _parameters.push_back(true, "is_theta_estim");
    }
    if (params.containsElementNamed("beta")) {
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(params["beta"]), "beta");
      _parameters.push_back(true, "has_beta");
      _parameters.push_back(
          !(params.containsElementNamed("is_beta_estim") && !params["is_beta_estim"]) && optim != "none",
          "is_beta_estim");
    } else {
      _parameters.push_back(Rcpp::NumericVector(0), "beta");
      _parameters.push_back(false, "has_beta");
      _parameters.push_back(true, "is_beta_estim");
    }
  } else {
    // Rcpp::NumericVector r = Rcpp::runif(X.n_cols); // turnaround mingw bug:
    // https://github.com/msys2/MINGW-packages/issues/5019
    _parameters = Rcpp::List::create(  // Rcpp::Named("sigma2") = Rcpp::runif(1),
                                       // Rcpp::Named("has_sigma2") = true,
        Rcpp::Named("sigma2") = -1,
        Rcpp::Named("has_sigma2") = false,
        Rcpp::Named("is_sigma2_estim") = true,
        // Rcpp::Named("theta") = Rcpp::NumericMatrix(1, X.n_cols, r.begin()),
        // Rcpp::Named("has_theta") = true,
        Rcpp::Named("theta") = Rcpp::NumericMatrix(0, 0),
        Rcpp::Named("has_theta") = false,
        Rcpp::Named("is_theta_estim") = true,
        Rcpp::Named("beta") = Rcpp::NumericVector(0),
        Rcpp::Named("has_beta") = false,
        Rcpp::Named("is_beta_estim") = true);
  }

  impl_ptr->fit(
      std::move(y),
      std::move(X),
      Trend::fromString(regmodel),
      normalize,
      optim,
      objective,
      Kriging::Parameters{(_parameters["has_sigma2"]) ? make_optional0<double>(_parameters["sigma2"]) : std::nullopt,
                          _parameters["is_sigma2_estim"],
                          (_parameters["has_theta"]) ? make_optional0<arma::mat>(_parameters["theta"]) : std::nullopt,
                          _parameters["is_theta_estim"],
                          (_parameters["has_beta"]) ? make_optional0<arma::vec>(_parameters["beta"]) : std::nullopt,
                          _parameters["is_beta_estim"]});
}

// [[Rcpp::export]]
Rcpp::List kriging_copy(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);

  Rcpp::List obj;
  Rcpp::XPtr<Kriging> impl_copy(new Kriging(*impl_ptr, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "Kriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List kriging_model(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("kernel") = impl_ptr->kernel(),
                                      Rcpp::Named("optim") = impl_ptr->optim(),
                                      Rcpp::Named("objective") = impl_ptr->objective(),
                                      Rcpp::Named("theta") = impl_ptr->theta(),
                                      Rcpp::Named("is_theta_estim") = impl_ptr->is_theta_estim(),
                                      Rcpp::Named("sigma2") = impl_ptr->sigma2(),
                                      Rcpp::Named("is_sigma2_estim") = impl_ptr->is_sigma2_estim(),
                                      Rcpp::Named("X") = impl_ptr->X(),
                                      Rcpp::Named("centerX") = impl_ptr->centerX(),
                                      Rcpp::Named("scaleX") = impl_ptr->scaleX(),
                                      Rcpp::Named("y") = impl_ptr->y(),
                                      Rcpp::Named("centerY") = impl_ptr->centerY(),
                                      Rcpp::Named("scaleY") = impl_ptr->scaleY(),
                                      Rcpp::Named("normalize") = impl_ptr->normalize(),
                                      Rcpp::Named("regmodel") = Trend::toString(impl_ptr->regmodel()),
                                      Rcpp::Named("beta") = impl_ptr->beta(),
                                      Rcpp::Named("is_beta_estim") = impl_ptr->is_beta_estim());

  // because Rcpp::List::create accepts no more than 20 args...
  ret.push_back(impl_ptr->F(), "F");
  ret.push_back(impl_ptr->T(), "T");
  ret.push_back(impl_ptr->M(), "M");
  ret.push_back(impl_ptr->z(), "z");

  return ret;
}

// [[Rcpp::export]]
std::string kriging_summary(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->summary();
}

// [[Rcpp::export]]
Rcpp::List kriging_predict(Rcpp::List k, arma::mat X_n, bool return_stdev = true, bool return_cov = false, bool return_deriv = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  auto pred = impl_ptr->predict(X_n, return_stdev, return_cov, return_deriv);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred));
  if (return_stdev) {
    ret.push_back(std::get<1>(pred), "stdev");
  }
  if (return_cov) {
    ret.push_back(std::get<2>(pred), "cov");
  }
  if (return_deriv) {
    ret.push_back(std::get<3>(pred), "mean_deriv");
    ret.push_back(std::get<4>(pred), "stdev_deriv");
  }

  return ret;
}

// [[Rcpp::export]]
arma::mat kriging_simulate(Rcpp::List k, int nsim, int seed, arma::mat X_n, bool will_update = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->simulate(nsim, seed, X_n, will_update);
}

// [[Rcpp::export]]
arma::mat kriging_update_simulate(Rcpp::List k, arma::vec y_u, arma::mat X_u) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->update_simulate(y_u, X_u);
}

// [[Rcpp::export]]
void kriging_update(Rcpp::List k, arma::vec y_u, arma::mat X_u, bool refit = true) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  impl_ptr->update(y_u, X_u, refit);

  // Rcpp::List obj;
  // obj.attr("object") = impl_ptr;
  // obj.attr("class") = "Kriging";
  // return obj;
}

// [[Rcpp::export]]
void kriging_save(Rcpp::List k, std::string filename) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->save(filename);
}

// [[Rcpp::export]]
arma::mat kriging_covMat(Rcpp::List k,
                          arma::mat X1,
                          arma::mat X2) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->covMat(X1,X2);
}

// [[Rcpp::export]]
Rcpp::List kriging_logLikelihoodFun(Rcpp::List k,
                                    arma::vec theta,
                                    bool return_grad = false,
                                    bool return_hess = false,
                                    bool bench = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<double, arma::vec, arma::mat> ll = impl_ptr->logLikelihoodFun(theta, return_grad, return_hess, bench);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll));
  if (return_grad) {
    ret.push_back(std::get<1>(ll), "logLikelihoodGrad");
  }
  if (return_hess) {
    ret.push_back(std::get<2>(ll), "logLikelihoodHess");
  }

  return ret;
}

// [[Rcpp::export]]
double kriging_logLikelihood(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->logLikelihood();
}

// [[Rcpp::export]]
Rcpp::List kriging_leaveOneOutFun(Rcpp::List k, arma::vec theta, bool return_grad = false, bool bench = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<double, arma::vec> loo = impl_ptr->leaveOneOutFun(theta, return_grad, bench);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("leaveOneOut") = std::get<0>(loo));
  if (return_grad) {
    ret.push_back(std::get<1>(loo), "leaveOneOutGrad");
  }

  return ret;
}

// [[Rcpp::export]]
Rcpp::List kriging_leaveOneOutVec(Rcpp::List k, arma::vec theta) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<arma::vec, arma::vec> yhat = impl_ptr->leaveOneOutVec(theta);

  Rcpp::List ret
      = Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(yhat), Rcpp::Named("stdev") = std::get<1>(yhat));

  return ret;
}

// [[Rcpp::export]]
double kriging_leaveOneOut(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->leaveOneOut();
}

// [[Rcpp::export]]
Rcpp::List kriging_logMargPostFun(Rcpp::List k, arma::vec theta, bool return_grad = false, bool bench = false) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  std::tuple<double, arma::vec> lmp = impl_ptr->logMargPostFun(theta, return_grad, bench);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("logMargPost") = std::get<0>(lmp));
  if (return_grad) {
    ret.push_back(std::get<1>(lmp), "logMargPostGrad");
  }

  return ret;
}

// [[Rcpp::export]]
double kriging_logMargPost(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<Kriging> impl_ptr(impl);

  return impl_ptr->logMargPost();
}

///////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
std::string kriging_kernel(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->kernel();
}

// [[Rcpp::export]]
std::string kriging_optim(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->optim();
}

// [[Rcpp::export]]
std::string kriging_objective(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->objective();
}

// [[Rcpp::export]]
arma::mat kriging_X(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->X();
}

// [[Rcpp::export]]
arma::vec kriging_centerX(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->centerX();
}

// [[Rcpp::export]]
arma::vec kriging_scaleX(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->scaleX();
}

// [[Rcpp::export]]
arma::vec kriging_y(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->y();
}

// [[Rcpp::export]]
double kriging_centerY(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->centerY();
}

// [[Rcpp::export]]
double kriging_scaleY(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->scaleY();
}

// [[Rcpp::export]]
bool kriging_normalize(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->normalize();
}

// [[Rcpp::export]]
std::string kriging_regmodel(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return Trend::toString(impl_ptr->regmodel());
}

// [[Rcpp::export]]
arma::mat kriging_F(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->F();
}

// [[Rcpp::export]]
arma::mat kriging_T(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->T();
}

// [[Rcpp::export]]
arma::mat kriging_M(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->M();
}

// [[Rcpp::export]]
arma::vec kriging_z(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->z();
}

// [[Rcpp::export]]
arma::vec kriging_beta(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->beta();
}

// [[Rcpp::export]]
bool kriging_is_beta_estim(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->is_beta_estim();
}

// [[Rcpp::export]]
arma::vec kriging_theta(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->theta();
}

// [[Rcpp::export]]
bool kriging_is_theta_estim(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->is_theta_estim();
}

// [[Rcpp::export]]
double kriging_sigma2(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->sigma2();
}

// [[Rcpp::export]]
bool kriging_is_sigma2_estim(Rcpp::List k) {
  if (!k.inherits("Kriging"))
    Rcpp::stop("Input must be a Kriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<Kriging> impl_ptr(impl);
  return impl_ptr->is_sigma2_estim();
}
