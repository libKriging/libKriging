#include <RcppArmadillo.h>

#include "libKriging/LinearRegression.hpp"
#include "libKriging/LinearRegressionOptim.hpp"

// [[Rcpp::export]]
Rcpp::List linear_regression(arma::vec y, arma::mat X) {
  LinearRegression* rl = new LinearRegression();
  rl->fit(std::move(y), std::move(X));

  Rcpp::XPtr<LinearRegression> impl_ptr(rl);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "LinearRegression";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List linear_regression_optim(arma::vec y, arma::mat X) {
  LinearRegressionOptim* rl = new LinearRegressionOptim();
  rl->fit(std::move(y), std::move(X));

  Rcpp::XPtr<LinearRegressionOptim> impl_ptr(rl);

  Rcpp::List obj;
  obj.attr("object") = impl_ptr;
  obj.attr("class") = "LinearRegression";
  return obj;
}

// [[Rcpp::export]]
Rcpp::List linear_regression_predict(Rcpp::List linearRegression, arma::mat X) {
  if (! linearRegression.inherits("LinearRegression")) Rcpp::stop("Input must be a LinearRegression object.");
  SEXP impl = linearRegression.attr("object");
  Rcpp::XPtr<LinearRegression> impl_ptr(impl);
  auto pred = impl_ptr->predict(X);
  return Rcpp::List::create(Rcpp::Named("y") = std::get<0>(pred),
                            Rcpp::Named("stderr") = std::get<1>(pred));
}


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
Rcpp::List ordinary_kriging_predict(Rcpp::List ordinaryKriging, arma::mat X) {
  if (! ordinaryKriging.inherits("OrdinaryKriging")) Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");
  
  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);
  
  auto pred = impl_ptr->predict(X,true,false);
  return Rcpp::List::create(Rcpp::Named("mean") = std::get<0>(pred),
                            Rcpp::Named("stdev") = std::get<1>(pred));
}

