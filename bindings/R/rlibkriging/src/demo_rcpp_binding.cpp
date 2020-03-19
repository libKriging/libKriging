#include <RcppArmadillo.h>

#include "libKriging/demo/DemoArmadilloClass.hpp"

// [[Rcpp::export]]
Rcpp::List buildDemoArmadilloClass(std::string id, arma::mat M) {
  DemoArmadilloClass* s = new DemoArmadilloClass{std::move(id), std::move(M)};

  Rcpp::XPtr<DemoArmadilloClass> impl_ptr(s);

  Rcpp::List obj;
  obj.attr("impl") = impl_ptr;
  obj.attr("class") = "DemoArmadillo";

  return obj;
}

// [[Rcpp::export]]
arma::vec getEigenValues(Rcpp::List obj) {
  if (!obj.inherits("DemoArmadillo"))
    Rcpp::stop("Input must be a DemoArmadillo object");

  SEXP impl = obj.attr("impl");
  Rcpp::XPtr<DemoArmadilloClass> impl_ptr(impl);
  return impl_ptr->getEigenValues();
}

// TODO try to use modul for implicit access : http://www.deanbodenham.com/learn/rcpp-classes-part-2.html