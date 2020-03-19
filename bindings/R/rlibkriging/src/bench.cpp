#include "libKriging/Bench.hpp"

#include <RcppArmadillo.h>

#include "libKriging/OrdinaryKriging.hpp"

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
double ordinary_kriging_loglikelihood2(Rcpp::List ordinaryKriging, arma::vec theta) {
  if (!ordinaryKriging.inherits("OrdinaryKriging"))
    Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);

  return impl_ptr->logLikelihood(theta);
}

// [[Rcpp::export]]
arma::vec bench_ordinary_kriging_loglikelihood(Rcpp::List ordinaryKriging, arma::mat theta) {
  if (!ordinaryKriging.inherits("OrdinaryKriging"))
    Rcpp::stop("Input must be a OrdinaryKriging object.");
  SEXP impl = ordinaryKriging.attr("object");

  Rcpp::XPtr<OrdinaryKriging> impl_ptr(impl);

  arma::uword n = theta.n_cols;
  arma::vec res(n);
  for (arma::uword k = 0; k < n; k++) {
    // Rcpp::Rcout << "The value of v : " << impl_ptr->logLikelihood(theta.col(k)) << "\n";
    res.at(k) = impl_ptr->logLikelihood(theta.col(k));
    // res[k] = ordinary_kriging_loglikelihood2(ordinaryKriging, theta.col(k));
  }

  return res;
}

// [[Rcpp::export]]
double benchmatMLE(const arma::mat& X, const arma::vec& y, arma::vec& theta) {
  arma::uword n = X.n_rows, m = X.n_cols;

  // Define regression matrix
  arma::uword nreg = 1;
  arma::mat F(n, nreg);
  F.ones();

  arma::mat R(n, n);
  R.zeros();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      double temp = 1;
      for (int k = 0; k < m; k++) {
        double d = (X(i, k) - X(j, k)) / theta(k);
        temp *= exp(-0.5 * d * d);
      }
      R.at(i, j) = temp;
    }
  }
  R = arma::symmatl(R);  // R + trans(R);
  R.diag().ones();

  // Cholesky decompostion of covariance matrix
  arma::mat T = trans(chol(R));

  // Compute intermediate useful matrices
  arma::mat M = solve(trimatl(T), F, arma::solve_opts::fast);
  arma::mat Q;
  arma::mat G;
  qr_econ(Q, G, M);
  arma::colvec Yt = solve(trimatl(T), y, arma::solve_opts::fast);
  arma::colvec beta = solve(trimatu(G), trans(Q) * Yt, arma::solve_opts::fast);
  arma::colvec z = Yt - M * beta;
  double sigma2_hat = arma::accu(z % z) / n;
  double minus_ll = /*-*/ 0.5 * (n * log(2 * M_PI * sigma2_hat) + 2 * sum(log(T.diag())) + n);

  return (minus_ll);
}