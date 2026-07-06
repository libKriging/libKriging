// clang-format off
// Must before any Rcpp code
#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/NestedKriging.hpp"
#include "libKriging/Trend.hpp"

#include "retrofit_utils.hpp"

// Build Kriging::Parameters from an R named list (subset used by NestedKriging:
// sigma2 / theta / beta and their is_*_estim flags; nugget unused in v1)
static Kriging::Parameters parameters_from_list(Rcpp::Nullable<Rcpp::List> parameters, const std::string& optim) {
  Kriging::Parameters out;
  if (parameters.isNotNull()) {
    Rcpp::List params(parameters);
    if (params.containsElementNamed("sigma2")) {
      out.sigma2 = Rcpp::as<double>(params["sigma2"]);
      out.is_sigma2_estim
          = !(params.containsElementNamed("is_sigma2_estim") && !Rcpp::as<bool>(params["is_sigma2_estim"]))
            && optim != "none";
    }
    if (params.containsElementNamed("theta")) {
      out.theta = Rcpp::as<arma::mat>(params["theta"]);
      out.is_theta_estim = !(params.containsElementNamed("is_theta_estim") && !Rcpp::as<bool>(params["is_theta_estim"]))
                           && optim != "none";
    }
    if (params.containsElementNamed("beta")) {
      out.beta = Rcpp::as<arma::colvec>(params["beta"]);
      out.is_beta_estim = !(params.containsElementNamed("is_beta_estim") && !Rcpp::as<bool>(params["is_beta_estim"]))
                          && optim != "none";
    }
  }
  return out;
}

static NestedKriging::Partition partition_from_string(const std::string& s) {
  if (s == "kmeans")
    return NestedKriging::Partition::KMeans;
  if (s == "random")
    return NestedKriging::Partition::Random;
  Rcpp::stop("Unknown partition '%s'; expected 'kmeans' or 'random'", s.c_str());
}

// [[Rcpp::export]]
Rcpp::List new_NestedKrigingFit(arma::vec y,
                                arma::mat X,
                                std::string kernel,
                                unsigned long nb_groups,
                                std::string aggregation = "NK",
                                std::string partition = "kmeans",
                                int seed = 123,
                                std::string regmodel = "constant",
                                std::string optim = "BFGS",
                                std::string objective = "LL",
                                Rcpp::Nullable<Rcpp::List> parameters = R_NilValue,
                                Rcpp::Nullable<Rcpp::CharacterVector> warping = R_NilValue) {
  NestedKriging* nk
      = new NestedKriging(y,
                          X,
                          kernel,
                          nb_groups,
                          NestedKriging::aggregationFromString(aggregation),
                          partition_from_string(partition),
                          seed,
                          Trend::fromString(regmodel),
                          optim,
                          objective,
                          parameters_from_list(parameters, optim),
                          warping.isNotNull() ? Rcpp::as<std::vector<std::string>>(Rcpp::CharacterVector(warping))
                                              : std::vector<std::string>{});

  Rcpp::XPtr<NestedKriging> impl_ptr(nk);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "NestedKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List nestedkriging_predict(Rcpp::List k, arma::mat X_n, bool return_stdev = true) {
  if (!k.inherits("NestedKriging"))
    Rcpp::stop("Input must be a NestedKriging object");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);

  auto [mean, stdev] = impl_ptr->predict(X_n, return_stdev);
  Rcpp::List out = Rcpp::List::create(Rcpp::Named("mean") = mean);
  if (return_stdev)
    out.push_back(stdev, "stdev");
  return out;
}

// [[Rcpp::export]]
std::string nestedkriging_summary(Rcpp::List k) {
  if (!k.inherits("NestedKriging"))
    Rcpp::stop("Input must be a NestedKriging object");
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->summary();
}

// [[Rcpp::export]]
std::string nestedkriging_kernel(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->kernel();
}

// [[Rcpp::export]]
std::string nestedkriging_aggregation(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return NestedKriging::aggregationToString(impl_ptr->aggregation());
}

// [[Rcpp::export]]
unsigned long nestedkriging_nb_groups(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->nb_groups();
}

// [[Rcpp::export]]
Rcpp::List nestedkriging_groups(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  Rcpp::List out;
  for (const auto& g : impl_ptr->groups())
    out.push_back(arma::conv_to<arma::uvec>::from(g) + 1);  // 1-based for R
  return out;
}

// [[Rcpp::export]]
arma::vec nestedkriging_theta(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->theta();
}

// [[Rcpp::export]]
double nestedkriging_sigma2(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->sigma2();
}

// [[Rcpp::export]]
double nestedkriging_beta0(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->beta0();
}

// [[Rcpp::export]]
Rcpp::CharacterVector nestedkriging_warping(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return Rcpp::wrap(impl_ptr->warping());
}

// [[Rcpp::export]]
arma::mat nestedkriging_X(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->X();
}

// [[Rcpp::export]]
arma::vec nestedkriging_y(Rcpp::List k) {
  SEXP impl = k.attr("object");
  Rcpp::XPtr<NestedKriging> impl_ptr(impl);
  return impl_ptr->y();
}
