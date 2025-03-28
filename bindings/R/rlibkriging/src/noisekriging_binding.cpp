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
Rcpp::List new_NoiseKriging(std::string kernel) {
  NoiseKriging* ok = new NoiseKriging(kernel);

  Rcpp::XPtr<NoiseKriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "NoiseKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List new_NoiseKrigingFit(arma::vec y,
                               arma::vec noise,
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
      _parameters.push_back(Rcpp::NumericMatrix(0, 0), "theta");
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

  NoiseKriging* ok = new NoiseKriging(
      std::move(y),
      std::move(noise),
      std::move(X),
      kernel,
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
void noisekriging_fit(Rcpp::List k,
                      arma::vec y,
                      arma::vec noise,
                      arma::mat X,
                      std::string regmodel = "constant",
                      bool normalize = false,
                      std::string optim = "BFGS",
                      std::string objective = "LL",
                      Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

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
      _parameters.push_back(Rcpp::NumericMatrix(0, 0), "theta");
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

  impl_ptr->fit(std::move(y),
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
}

// [[Rcpp::export]]
Rcpp::List noisekriging_copy(Rcpp::List k) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  Rcpp::List obj;
  Rcpp::XPtr<NoiseKriging> impl_copy(new NoiseKriging(*impl_ptr, ExplicitCopySpecifier{}));
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

  Rcpp::List ret = Rcpp::List(21);

  ret["kernel"] = impl_ptr->kernel();
  ret["optim"] = impl_ptr->optim();
  ret["objective"] = impl_ptr->objective();
  ret["theta"] = impl_ptr->theta();
  ret["is_theta_estim"] = impl_ptr->is_theta_estim();
  ret["sigma2"] = impl_ptr->sigma2();
  ret["is_sigma2_estim"] = impl_ptr->is_sigma2_estim();
  ret["noise"] = impl_ptr->noise();
  ret["X"] = impl_ptr->X();
  ret["centerX"] = impl_ptr->centerX();
  ret["scaleX"] = impl_ptr->scaleX();
  ret["y"] = impl_ptr->y();
  ret["centerY"] = impl_ptr->centerY();
  ret["scaleY"] = impl_ptr->scaleY();
  ret["normalize"] = impl_ptr->normalize();
  ret["regmodel"] = Trend::toString(impl_ptr->regmodel());
  ret["beta"] = impl_ptr->beta();
  ret["is_beta_estim"] = impl_ptr->is_beta_estim();
  ret["F"] = impl_ptr->F();
  ret["T"] = impl_ptr->T();
  ret["M"] = impl_ptr->M();
  ret["z"] = impl_ptr->z();

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
Rcpp::List noisekriging_predict(Rcpp::List k,
                                arma::mat X_n,
                                bool return_stdev = true,
                                bool return_cov = false,
                                bool return_deriv = false) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_n.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

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
arma::mat noisekriging_simulate(Rcpp::List k,
                                int nsim,
                                int seed,
                                arma::mat X_n,
                                arma::vec with_noise,
                                bool will_update = false) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_n.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  return impl_ptr->simulate(nsim, seed, X_n, with_noise, will_update);
}

// [[Rcpp::export]]
arma::mat noisekriging_update_simulate(Rcpp::List k, arma::vec y_u, arma::vec noise_u, arma::mat X_u) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_u.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  if (X_u.n_rows != y_u.n_elem)
    Rcpp::stop("Length of arg data should be the same.");

  if (X_u.n_rows != noise_u.n_elem)
    Rcpp::stop("Length of arg data should be the same.");

  return impl_ptr->update_simulate(y_u, noise_u, X_u);
}

// [[Rcpp::export]]
void noisekriging_update(Rcpp::List k, arma::vec y_u, arma::vec noise_u, arma::mat X_u, bool refit = true) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_u.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  if (X_u.n_rows != y_u.n_elem)
    Rcpp::stop("Length of arg data should be the same.");

  if (X_u.n_rows != noise_u.n_elem)
    Rcpp::stop("Length of arg data should be the same.");

  impl_ptr->update(y_u, noise_u, X_u, refit);

  // Rcpp::List obj;
  // obj.attr("object") = impl_ptr;
  // obj.attr("class") = "NoiseKriging";
  // return obj;
}

// [[Rcpp::export]]
void noisekriging_save(Rcpp::List k, std::string filename) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  return impl_ptr->save(filename);
}

// [[Rcpp::export]]
arma::mat noisekriging_covMat(Rcpp::List k, arma::mat X1, arma::mat X2) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X1.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");
  if (d != X2.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  return impl_ptr->covMat(X1, X2);
}

// [[Rcpp::export]]
Rcpp::List noisekriging_logLikelihoodFun(Rcpp::List k,
                                         arma::vec theta_sigma2,
                                         bool return_grad = false,
                                         bool bench = false) {
  if (!k.inherits("NoiseKriging"))
    Rcpp::stop("Input must be a NoiseKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NoiseKriging> impl_ptr(impl);

  if (theta_sigma2.n_elem != impl_ptr->theta().n_elem + 1)
    Rcpp::stop("Length of arg data should be " + std::to_string(impl_ptr->theta().n_elem + 1) + ")");

  std::tuple<double, arma::vec> ll = impl_ptr->logLikelihoodFun(theta_sigma2, return_grad, bench);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll));
  if (return_grad) {
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
