#include <RcppArmadillo.h>

#include "libKriging/OrdinaryKriging.hpp"

// [[Rcpp::export]]
Rcpp::List ordinary_kriging(arma::vec y, arma::mat X) {
  OrdinaryKriging* ok = new OrdinaryKriging();//"gauss"));
  ok->fit(std::move(y), std::move(X));//, OrdinaryKriging::Parameters{0,false,nullptr,false},"ll","bfgs");
  
  Rcpp::XPtr<OrdinaryKriging> impl_ptr(ok);
  
  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "OrdinaryKriging";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List ordinary_kriging_model(Rcpp::List ordinaryKriging) {
  if (! ordinaryKriging.inherits("OrdinaryKriging")) Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");
  
  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);
  
  return Rcpp::List::create(Rcpp::Named("theta") = impl_ptr->theta,
                            Rcpp::Named("sigma2") = impl_ptr->sigma2,
                            Rcpp::Named("X") = impl_ptr->X,
                            Rcpp::Named("y") = impl_ptr->y,
                            Rcpp::Named("T") = impl_ptr->T,
                            Rcpp::Named("z") = impl_ptr->z);
}


// [[Rcpp::export]]
double ordinary_kriging_loglikelihood(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (! ordinaryKriging.inherits("OrdinaryKriging")) Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");
  
  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);
  
  return impl_ptr->logLikelihood(theta);
}

// [[Rcpp::export]]
arma::vec ordinary_kriging_loglikelihoodgrad(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (! ordinaryKriging.inherits("OrdinaryKriging")) Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");
  
  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);
  
  return impl_ptr->logLikelihoodGrad(theta);
}

// [[Rcpp::export]]
Rcpp::List ordinary_kriging_predict(Rcpp::List ordinaryKriging, arma::mat X, bool stdev, bool cov) {
  if (! ordinaryKriging.inherits("OrdinaryKriging")) Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");
  
  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);
  
  if (stdev & cov) {
    auto pred = impl_ptr->predict(X,true,true);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred),
                              Rcpp::Named("stdev") = std::get<1>(pred),
                              Rcpp::Named("cov") = std::get<2>(pred));
  } else if (stdev & !cov) {
    auto pred = impl_ptr->predict(X,true,false);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred),
                              Rcpp::Named("stdev") = std::get<1>(pred));
  } else if (!stdev & cov) {
    auto pred = impl_ptr->predict(X,false,true);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred),
                              Rcpp::Named("cov") = std::get<1>(pred));
  } else if (!stdev & !cov) {
    auto pred = impl_ptr->predict(X,false,false);
    return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred));
  }
}
