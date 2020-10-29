// clang-format off
// Must before any other include
#include <RcppArmadillo.h>
// clang-format on

#include "libKriging/Bench.hpp"
#include "libKriging/OrdinaryKriging.hpp"
#include "libKriging/LinearRegression.hpp"
#include <chrono>
#include <ctime>

// [[Rcpp::export]]
arma::mat bench_solvetri(int n, arma::mat X, arma::vec y) {
  Bench* b = new Bench(n);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->SolveTri(std::move(X), std::move(y));
}

// [[Rcpp::export]]
arma::mat bench_cholsym(int n, arma::mat X) {
  Bench* b = new Bench(n);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->CholSym(std::move(X));
}

// [[Rcpp::export]]
arma::mat bench_invsympd(int n, arma::mat X) {
  Bench* b = new Bench(n);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->InvSymPD(std::move(X));
}

// [[Rcpp::export]]
Rcpp::List bench_qr(int n, arma::mat X) {
  Bench* b = new Bench(n);
  Rcpp::XPtr<Bench> impl_ptr(b);
  auto ans = impl_ptr->QR(std::move(X));
  return Rcpp::List::create(Rcpp::Named("Q") = std::get<0>(ans), Rcpp::Named("R") = std::get<1>(ans));
}

// [[Rcpp::export]]
double bench_loglik(int n, Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("OrdinaryKriging"))
    Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl_ok = ordinaryKriging.attr("object");
  Rcpp::XPtr<OrdinaryKriging> ok(impl_ok);

  Bench* b = new Bench(n);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->LogLik(*ok, std::move(theta));
}

// [[Rcpp::export]]
Rcpp::List bench_loglikgrad(int n, Rcpp::List ordinaryKriging, arma::vec &theta) {
  auto start = std::chrono::system_clock::now();
  if (!ordinaryKriging.inherits("OrdinaryKriging"))
    Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl_ok = ordinaryKriging.attr("object");
  Rcpp::XPtr<OrdinaryKriging> ok(impl_ok);

  Bench* b = new Bench(n);
  Rcpp::XPtr<Bench> impl_ptr(b);
  auto result = impl_ptr->LogLikGrad(*ok, theta);
  auto end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  elapsed /= 1e6;
  return Rcpp::List::create(Rcpp::Named("result") = std::move(result), Rcpp::Named("time") = elapsed);
}

// [[Rcpp::export]]
arma::mat bench_optim(arma::vec x0) {
  Bench* b = new Bench(1);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->OptimRosenbrock(x0);
}

// [[Rcpp::export]]
double f_optim(arma::vec x) {
  Bench* b = new Bench(1);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->Rosenbrock(x);
}

// [[Rcpp::export]]
arma::vec grad_optim(arma::vec x0) {
  Bench* b = new Bench(1);
  Rcpp::XPtr<Bench> impl_ptr(b);
  return impl_ptr->RosenbrockGrad(x0);
}

// [[Rcpp::export]]
Rcpp::List bench_rcpp_link(const arma::vec &y,
                           const arma::mat &X,
                           int n0step,
                           int n1step) {
  auto start = std::chrono::system_clock::now();

  arma::uword n = X.n_rows;
  arma::mat Xtnorm = trans(X);
  arma::mat R0(n,n);
  for (arma::uword k = 0; k < n0step; k++) {
    for (arma::uword j = 0; j < n; j++) {
      for (arma::uword i = 0; i < j; i++) {
        R0(i, j) = 0;
      }
    }
    for (arma::uword j = 0; j < n; j++) {
      for (arma::uword i = j; i < n; i++) {
        auto&& diff = Xtnorm.col(i) - Xtnorm.col(j);
        const double temp = arma::dot(diff, diff);
        R0(i, j) = exp(-0.5 * temp);
      }
    }
  }

  arma::mat R = arma::symmatl(R0);
  R.diag().ones();
  for (arma::uword k = 0; k < n1step; k++) {
    arma::mat T = chol(R);
  }

  auto end = std::chrono::system_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  elapsed /= 1e6;
  Rcpp::List obj;
  obj.attr("time") = elapsed;
  return obj;
}