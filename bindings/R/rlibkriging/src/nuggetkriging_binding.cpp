// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Covariance.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NuggetKriging.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"

#include <optional>
#include "retrofit_utils.hpp"

// [[Rcpp::export]]
Rcpp::List new_NuggetKriging(std::string kernel) {
  NuggetKriging* ok = new NuggetKriging(kernel);

  Rcpp::XPtr<NuggetKriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "NuggetKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List new_NuggetKrigingFit(arma::vec y,
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
    if (params.containsElementNamed("nugget")) {
      _parameters.push_back(params["nugget"], "nugget");
      _parameters.push_back(true, "has_nugget");
      _parameters.push_back(
          !(params.containsElementNamed("is_nugget_estim") && !params["is_nugget_estim"]) && optim != "none",
          "is_nugget_estim");
    } else {
      //_parameters.push_back(Rcpp::runif(1), "nugget"); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(true, "has_nugget");
      _parameters.push_back(-1, "nugget");
      _parameters.push_back(false, "has_nugget");
      _parameters.push_back(true, "is_nugget_estim");
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
        // Rcpp::Named("nugget") = Rcpp::runif(1),
        // Rcpp::Named("has_nugget") = true,
        Rcpp::Named("nugget") = -1,
        Rcpp::Named("has_nugget") = false,
        Rcpp::Named("is_nugget_estim") = true,
        // Rcpp::Named("theta") = Rcpp::NumericMatrix(1, X.n_cols, r.begin()),
        // Rcpp::Named("has_theta") = true,
        Rcpp::Named("theta") = Rcpp::NumericMatrix(0, 0),
        Rcpp::Named("has_theta") = false,
        Rcpp::Named("is_theta_estim") = true,
        Rcpp::Named("beta") = Rcpp::NumericVector(0),
        Rcpp::Named("has_beta") = false,
        Rcpp::Named("is_beta_estim") = true);
  }

  NuggetKriging* ok = new NuggetKriging(
      std::move(y),
      std::move(X),
      kernel,
      Trend::fromString(regmodel),
      normalize,
      optim,
      objective,
      NuggetKriging::Parameters{
          (_parameters["has_nugget"]) ? make_optional0<arma::vec>(_parameters["nugget"]) : std::nullopt,
          _parameters["is_nugget_estim"],
          (_parameters["has_sigma2"]) ? make_optional0<arma::vec>(_parameters["sigma2"]) : std::nullopt,
          _parameters["is_sigma2_estim"],
          (_parameters["has_theta"]) ? make_optional0<arma::mat>(_parameters["theta"]) : std::nullopt,
          _parameters["is_theta_estim"],
          (_parameters["has_beta"]) ? make_optional0<arma::colvec>(_parameters["beta"]) : std::nullopt,
          _parameters["is_beta_estim"]});

  Rcpp::XPtr<NuggetKriging> impl_ptr(ok);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "NuggetKriging";
  return obj;
}

// [[Rcpp::export]]
void nuggetkriging_fit(Rcpp::List k,
                       arma::vec y,
                       arma::mat X,
                       std::string regmodel = "constant",
                       bool normalize = false,
                       std::string optim = "BFGS",
                       std::string objective = "LL",
                       Rcpp::Nullable<Rcpp::List> parameters = R_NilValue) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

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
    if (params.containsElementNamed("nugget")) {
      _parameters.push_back(params["nugget"], "nugget");
      _parameters.push_back(true, "has_nugget");
      _parameters.push_back(
          !(params.containsElementNamed("is_nugget_estim") && !params["is_nugget_estim"]) && optim != "none",
          "is_nugget_estim");
    } else {
      //_parameters.push_back(Rcpp::runif(1), "nugget"); // turnaround mingw bug:
      // https://github.com/msys2/MINGW-packages/issues/5019 _parameters.push_back(true, "has_nugget");
      _parameters.push_back(-1, "nugget");
      _parameters.push_back(false, "has_nugget");
      _parameters.push_back(true, "is_nugget_estim");
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
        // Rcpp::Named("nugget") = Rcpp::runif(1),
        // Rcpp::Named("has_nugget") = true,
        Rcpp::Named("nugget") = -1,
        Rcpp::Named("has_nugget") = false,
        Rcpp::Named("is_nugget_estim") = true,
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
                std::move(X),
                Trend::fromString(regmodel),
                normalize,
                optim,
                objective,
                NuggetKriging::Parameters{
                    (_parameters["has_nugget"]) ? make_optional0<arma::vec>(_parameters["nugget"]) : std::nullopt,
                    _parameters["is_nugget_estim"],
                    (_parameters["has_sigma2"]) ? make_optional0<arma::vec>(_parameters["sigma2"]) : std::nullopt,
                    _parameters["is_sigma2_estim"],
                    (_parameters["has_theta"]) ? make_optional0<arma::mat>(_parameters["theta"]) : std::nullopt,
                    _parameters["is_theta_estim"],
                    (_parameters["has_beta"]) ? make_optional0<arma::colvec>(_parameters["beta"]) : std::nullopt,
                    _parameters["is_beta_estim"]});
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_copy(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  Rcpp::List obj;
  Rcpp::XPtr<NuggetKriging> impl_copy(new NuggetKriging(*impl_ptr, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "NuggetKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_model(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  Rcpp::List ret = Rcpp::List(22);

  ret["kernel"] = impl_ptr->kernel();
  ret["optim"] = impl_ptr->optim();
  ret["objective"] = impl_ptr->objective();
  ret["theta"] = impl_ptr->theta();
  ret["is_theta_estim"] = impl_ptr->is_theta_estim();
  ret["sigma2"] = impl_ptr->sigma2();
  ret["is_sigma2_estim"] = impl_ptr->is_sigma2_estim();
  ret["nugget"] = impl_ptr->nugget();
  ret["is_nugget_estim"] = impl_ptr->is_nugget_estim();
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
std::string nuggetkriging_summary(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  return impl_ptr->summary();
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_predict(Rcpp::List k,
                                 arma::mat X_n,
                                 bool return_stdev = true,
                                 bool return_cov = false,
                                 bool return_deriv = false) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

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
arma::mat nuggetkriging_simulate(Rcpp::List k,
                                 int nsim,
                                 int seed,
                                 arma::mat X_n,
                                 bool with_nugget = true,
                                 bool will_update = false) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_n.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  return impl_ptr->simulate(nsim, seed, X_n, with_nugget, will_update);
}

// [[Rcpp::export]]
arma::mat nuggetkriging_update_simulate(Rcpp::List k, arma::vec y_u, arma::mat X_u) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_u.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  if (X_u.n_rows != y_u.n_elem)
    Rcpp::stop("Length of arg data should be the same.");

  return impl_ptr->update_simulate(y_u, X_u);
}

// [[Rcpp::export]]
void nuggetkriging_update(Rcpp::List k, arma::vec y_u, arma::mat X_u, bool refit = true) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X_u.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  if (X_u.n_rows != y_u.n_elem)
    Rcpp::stop("Length of arg data should be the same.");

  impl_ptr->update(y_u, X_u, refit);

  // Rcpp::List obj;
  // obj.attr("object") = impl_ptr;
  // obj.attr("class") = "NuggetKriging";
  // return obj;
}

// [[Rcpp::export]]
void nuggetkriging_save(Rcpp::List k, std::string filename) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  return impl_ptr->save(filename);
}

// [[Rcpp::export]]
arma::mat nuggetkriging_covMat(Rcpp::List k, arma::mat X1, arma::mat X2) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  int d = impl_ptr->X().n_cols;
  if (d != X1.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");
  if (d != X2.n_cols)
    Rcpp::stop("Dimension of arg data should be " + std::to_string(d) + ")");

  return impl_ptr->covMat(X1, X2);
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_logLikelihoodFun(Rcpp::List k,
                                          arma::vec theta_alpha,
                                          bool return_grad = false,
                                          bool bench = false) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  if (theta_alpha.n_elem != impl_ptr->theta().n_elem + 1)
    Rcpp::stop("Length of arg data should be " + std::to_string(impl_ptr->theta().n_elem + 1) + ")");

  std::tuple<double, arma::vec> ll = impl_ptr->logLikelihoodFun(theta_alpha, return_grad, bench);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("logLikelihood") = std::get<0>(ll));
  if (return_grad) {
    ret.push_back(std::get<1>(ll), "logLikelihoodGrad");
  }

  return ret;
}

// [[Rcpp::export]]
double nuggetkriging_logLikelihood(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  return impl_ptr->logLikelihood();
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_logMargPostFun(Rcpp::List k,
                                        arma::vec theta_alpha,
                                        bool return_grad = false,
                                        bool bench = false) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  if (theta_alpha.n_elem != impl_ptr->theta().n_elem + 1)
    Rcpp::stop("Length of arg data should be " + std::to_string(impl_ptr->theta().n_elem + 1) + ")");

  std::tuple<double, arma::vec> lmp = impl_ptr->logMargPostFun(theta_alpha, return_grad, bench);

  Rcpp::List ret = Rcpp::List::create(Rcpp::Named("logMargPost") = std::get<0>(lmp));
  if (return_grad) {
    ret.push_back(std::get<1>(lmp), "logMargPostGrad");
  }

  return ret;
}

// [[Rcpp::export]]
double nuggetkriging_logMargPost(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");

  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);

  return impl_ptr->logMargPost();
}

///////////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
std::string nuggetkriging_kernel(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->kernel();
}

// [[Rcpp::export]]
std::string nuggetkriging_optim(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->optim();
}

// [[Rcpp::export]]
std::string nuggetkriging_objective(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->objective();
}

// [[Rcpp::export]]
arma::mat nuggetkriging_X(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->X();
}

// [[Rcpp::export]]
arma::vec nuggetkriging_centerX(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->centerX();
}

// [[Rcpp::export]]
arma::vec nuggetkriging_scaleX(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->scaleX();
}

// [[Rcpp::export]]
arma::vec nuggetkriging_y(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->y();
}

// [[Rcpp::export]]
double nuggetkriging_centerY(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->centerY();
}

// [[Rcpp::export]]
double nuggetkriging_scaleY(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->scaleY();
}

// [[Rcpp::export]]
bool nuggetkriging_normalize(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->normalize();
}

// [[Rcpp::export]]
std::string nuggetkriging_regmodel(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return Trend::toString(impl_ptr->regmodel());
}

// [[Rcpp::export]]
arma::mat nuggetkriging_F(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->F();
}

// [[Rcpp::export]]
arma::mat nuggetkriging_T(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->T();
}

// [[Rcpp::export]]
arma::mat nuggetkriging_M(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->M();
}

// [[Rcpp::export]]
arma::vec nuggetkriging_z(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->z();
}

// [[Rcpp::export]]
arma::vec nuggetkriging_beta(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->beta();
}

// [[Rcpp::export]]
bool nuggetkriging_is_beta_estim(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->is_beta_estim();
}

// [[Rcpp::export]]
arma::vec nuggetkriging_theta(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->theta();
}

// [[Rcpp::export]]
bool nuggetkriging_is_theta_estim(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->is_theta_estim();
}

// [[Rcpp::export]]
double nuggetkriging_sigma2(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->sigma2();
}

// [[Rcpp::export]]
bool nuggetkriging_is_sigma2_estim(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->is_sigma2_estim();
}

// [[Rcpp::export]]
double nuggetkriging_nugget(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->nugget();
}

// [[Rcpp::export]]
bool nuggetkriging_is_nugget_estim(Rcpp::List k) {
  if (!k.inherits("NuggetKriging"))
    Rcpp::stop("Input must be a NuggetKriging object.");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NuggetKriging> impl_ptr(impl);
  return impl_ptr->is_nugget_estim();
}
