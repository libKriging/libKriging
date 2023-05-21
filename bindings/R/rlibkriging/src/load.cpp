// clang-format off
// Must before any other include
#include "libKriging/utils/lkalloc.hpp"

#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/KrigingLoader.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/NoiseKriging.hpp"
#include "libKriging/NuggetKriging.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"

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
  NoiseKriging ok = NoiseKriging::load(filename);

  Rcpp::List obj;
  Rcpp::XPtr<NoiseKriging> impl_copy(new NoiseKriging(ok, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "NoiseKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List nuggetkriging_load(std::string filename) {
  NuggetKriging ok = NuggetKriging::load(filename);

  Rcpp::List obj;
  Rcpp::XPtr<NuggetKriging> impl_copy(new NuggetKriging(ok, ExplicitCopySpecifier{}));
  obj.attr("object") = impl_copy;
  obj.attr("class") = "NuggetKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List anykriging_load(std::string filename) {
  switch (KrigingLoader::describe(filename)) {
    case KrigingLoader::KrigingType::Kriging:
      return kriging_load(filename);
      break;
    case KrigingLoader::KrigingType::NuggetKriging:
      return nuggetkriging_load(filename);
      break;
    case KrigingLoader::KrigingType::NoiseKriging:
      return noisekriging_load(filename);
      break;
    default:
      Rcpp::stop("Kriging object not identified by loader.");
  }
}