// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Covariance.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NoiseKriging.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"

#include <optional>
#include "retrofit_utils.hpp"

// [[Rcpp::export]]
Rcpp::List new_NoiseKriging(arma::vec y,
                            arma::vec noise,
                            arma::mat X,
                            std::string kernel,
                            std::string regmodel = "constant",
                            bool normalize = false,
                            std::string optim = "BFGS",
                            std::string objective = "LL",
                            Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  NoiseKriging* ok = new NoiseKriging(kernel);

  Rcpp::List _parameters;
  if (parameters.isNotNull()) {
    Rcpp::List params(parameters);
    _parameters = Rcpp::List::create();
    if (params.containsElementNamed("sigma2")) {
      _parameters.push_back(params["sigma2"], "sigma2");
      _parameters.push_back(true, "has_sigma2");
      _parameters.push_back(!(params.containsElementNamed("is_sigma2_estim") && !params["is_sigma2_estim"]),
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
      _parameters.push_back(!(params.containsElementNamed("is_theta_estim") && !params["is_theta_estim"]),
                            "is_theta_estim");
    } else {
      // Rcpp::NumericVector r = Rcpp::runif(X.n_cols); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(Rcpp::NumericMatrix(1, X.n_cols,
      // r.begin()), "theta"); _parameters.push_back(true, "has_theta");
      _parameters.push_back(Rcpp::NumericMatrix(0, 0), "theta");
      _parameters.push_back(false, "has_theta");
      _parameters.push_back(true, "is_theta_estim");
    }
    if (params.containsElementNamed("beta")) {
      _parameters.push_back(Rcpp::as<Rcpp::NumericMatrix>(params["beta"]), "beta");
      _parameters.push_back(true, "has_beta");
      _parameters.push_back(!(params.containsElementNamed("is_beta_estim") && !params["is_beta_estim"]),
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

  ok->fit(std::move(y),
          std::move(noise),
          std::move(X),
          Trend::fromString(regmodel),
          normalize,
          optim,
          objective,
          NoiseKriging::Parameters{
              (_parameters["has_sigma2"]) ? make_optional0<arma::vec>(_parameters["sigma2"]) : std::nullopt,
              _parameters["is_sigma2_estim"],
              (_parameters["has_theta"]) ? make_optional0<arma::mat>(_parameters["theta"]) : std::nullopt,
              _parameters["is_theta_estim"],
              (_parameters["has_beta"]) ? make_optional0<arma::colvec>(_parameters["beta"]) : std::nullopt,
              _parameters["is_beta_estim"]});

  Rcpp::XPtr<NoiseKriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "NoiseKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List noisekriging_copy(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  Rcpp::List obj;
  Rcpp::XPtr<Kriging> impl_copy(impl_ptr->copy());
  obj.attr("object") = impl_copy;
  obj.attr("class") = "NoiseKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List noisekriging_model(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("kernel") = impl_ptr->kernel(),
                                      Rcpp::Named("optim") = impl_ptr->optim(),
                                      Rcpp::Named("objective") = impl_ptr->objective(),
                                      Rcpp::Named("theta") = impl_ptr->theta(),
                                      Rcpp::Named("is_theta_estim") = impl_ptr->is_theta_estim(),
                                      Rcpp::Named("sigma2") = impl_ptr->sigma2(),
                                      Rcpp::Named("is_sigma2_estim") = impl_ptr->is_sigma2_estim(),
                                      Rcpp::Named("noise") = impl_ptr->noise(),
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
std::string noisekriging_summary(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  return impl_ptr->summary();
}

// [[Rcpp::export]]
Rcpp::List noisekriging_predict(Rcpp::List k, arma::mat X, bool stdev = true, bool cov = false, bool deriv = false) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  auto pred = impl_ptr->predict(X, stdev, cov, deriv);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred));
  if (stdev) {
    ret.push_back(std::get<1>(pred), "stdev");
  }
  if (cov) {
    ret.push_back(std::get<2>(pred), "cov");
  }
  if (deriv) {
    ret.push_back(std::get<3>(pred), "mean_deriv");
    ret.push_back(std::get<4>(pred), "stdev_deriv");
  }

  return ret;
}

// [[Rcpp::export]]
arma::mat noisekriging_simulate(Rcpp::List k, int nsim, int seed, arma::mat X) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  return impl_ptr->simulate(nsim, seed, X);
}

// [[Rcpp::export]]
void noisekriging_update(Rcpp::List k, arma::vec y, arma::vec noise, arma::mat X) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  impl_ptr->update(y, noise, X);

  // Rcpp::List obj;
  // obj.attr("object") = impl_ptr;
  // obj.attr("class") = "NoiseKriging";
  // return obj;
}

// [[Rcpp::export]]
Rcpp::List noisekriging_logLikelihoodFun(Rcpp::List k, arma::vec theta_sigma2, bool grad = false) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  std::tuple<double, arma::vec> ll = impl_ptr->logLikelihoodFun(theta_sigma2, grad);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll));
  if (grad) {
    ret.push_back(std::get<1>(ll), "logLikelihoodGrad");
  }

  return ret;
}

// [[Rcpp::export]]
double noisekriging_logLikelihood(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  return impl_ptr->logLikelihood();
}

///////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
std::string noisekriging_kernel(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->kernel();
}

// [[Rcpp::export]]
std::string noisekriging_optim(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->optim();
}

// [[Rcpp::export]]
std::string noisekriging_objective(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->objective();
}

// [[Rcpp::export]]
arma::mat noisekriging_X(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->X();
}

// [[Rcpp::export]]
arma::vec noisekriging_centerX(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->centerX();
}

// [[Rcpp::export]]
arma::vec noisekriging_scaleX(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->scaleX();
}

// [[Rcpp::export]]
arma::vec noisekriging_y(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->y();
}

// [[Rcpp::export]]
arma::vec noisekriging_noise(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->noise();
}

// [[Rcpp::export]]
double noisekriging_centerY(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->centerY();
}

// [[Rcpp::export]]
double noisekriging_scaleY(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->scaleY();
}

// [[Rcpp::export]]
bool noisekriging_normalize(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->normalize();
}

// [[Rcpp::export]]
std::string noisekriging_regmodel(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return Trend::toString(impl_ptr->regmodel());
}

// [[Rcpp::export]]
arma::mat noisekriging_F(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->F();
}

// [[Rcpp::export]]
arma::mat noisekriging_T(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->T();
}

// [[Rcpp::export]]
arma::mat noisekriging_M(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->M();
}

// [[Rcpp::export]]
arma::vec noisekriging_z(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->z();
}

// [[Rcpp::export]]
arma::vec noisekriging_beta(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->beta();
}

// [[Rcpp::export]]
bool noisekriging_is_beta_estim(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->is_beta_estim();
}

// [[Rcpp::export]]
arma::vec noisekriging_theta(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->theta();
}

// [[Rcpp::export]]
bool noisekriging_is_theta_estim(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->is_theta_estim();
}

// [[Rcpp::export]]
double noisekriging_sigma2(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->sigma2();
}

// [[Rcpp::export]]
bool noisekriging_is_sigma2_estim(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);
  return impl_ptr->is_sigma2_estim();
}
