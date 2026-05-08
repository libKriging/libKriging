// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/KrigingLoader.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/MLPKriging.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/WarpKriging.hpp"

#include <optional>
#include "retrofit_utils.hpp"

// [[Rcpp::export]]
Rcpp::List kriging_load(std::string filename) {
  Kriging ok = Kriging::load(filename);

  Rcpp::List obj;
  Rcpp::XPtr<Kriging> impl_copy(new Kriging(ok, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "Kriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List noisekriging_load(std::string filename) {
  Kriging ok = Kriging::load(filename);

  Rcpp::List obj;
  Rcpp::XPtr<Kriging> impl_copy(new Kriging(ok, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "NoiseKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_load(std::string filename) {
  Kriging ok = Kriging::load(filename);

  Rcpp::List obj;
  Rcpp::XPtr<Kriging> impl_copy(new Kriging(ok, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "NuggetKriging";
  return obj;
}

// [[Rcpp::export]]
SEXP warpkriging_load(std::string filename) {
  auto* wk = new libKriging::WarpKriging(libKriging::WarpKriging::load(filename));
  Rcpp::XPtr<libKriging::WarpKriging> ptr(wk);
  return ptr;
}

// [[Rcpp::export]]
SEXP mlpkriging_load(std::string filename) {
  auto* mk = new libKriging::MLPKriging(libKriging::MLPKriging::load(filename));
  Rcpp::XPtr<libKriging::MLPKriging> ptr(mk);
  return ptr;
}

// [[Rcpp::export]]
std::string class_saved(std::string filename) {
  switch (KrigingLoader::describe(filename)) {
    case KrigingLoader::KrigingType::Kriging:
      return "Kriging";
      break;
    case KrigingLoader::KrigingType::NuggetKriging:
      return "NuggetKriging";
      break;
    case KrigingLoader::KrigingType::NoiseKriging:
      return "NoiseKriging";
      break;
    case KrigingLoader::KrigingType::Unknown:
      Rcpp::stop("Kriging object type unknown.");
      break;
  }
  Rcpp::stop("Kriging object not identified by loader.");
}