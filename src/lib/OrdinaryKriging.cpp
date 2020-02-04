#include <tuple>

#include "libKriging/OrdinaryKriging.hpp"

LIBKRIGING_EXPORT
OrdinaryKriging::OrdinaryKriging() {}

LIBKRIGING_EXPORT
std::tuple<arma::colvec, arma::colvec> OrdinaryKriging::fit(const arma::vec y, const arma::mat X, const arma::vec theta) {
  int n = X.n_rows, m = X.n_cols;
  int nreg = 1;
  // Define regression matrix
  arma::mat F(n,nreg); F.ones();
  // allocate the matrix we will return (Gauss or Matern 5/2)
  arma::mat CovMatrix(n,n); CovMatrix.zeros();
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < i; j++) {
      double temp = 1;
      for (int k = 0; k < m; k++) {
        // double d = -std::sqrt(5)*std::abs(X(i,k)-X(j,k))/theta(k);
        // temp *= (1+d+std::pow(d,2)/3)*std::exp(d);
        double d = (X(i,k)-X(j,k))/theta(k);
        temp *= exp(-0.5*std::pow(d,2));
      }
      CovMatrix(i,j) = temp;
    }
  }
  CovMatrix = CovMatrix + trans(CovMatrix);
  CovMatrix.diag().ones();
  // Cholesky decompostion of covariance matrix
  arma::mat C = trans(chol(CovMatrix));
  // Compute intermediate useful matrices
  arma::mat Ft = arma::solve(C, F);
  arma::mat Q, G;
  qr_econ(Q,G,Ft);
  arma::colvec Yt = arma::solve(C, y);
  arma::colvec beta = arma::solve(G, trans(Q)*Yt);
  arma::colvec rho = Yt - Ft*beta;
  double  sigma2 = arma::as_scalar(arma::sum(pow(rho,2))/n);
  arma::mat gamma = trans(arma::solve(trans(C), rho));

  return std::make_tuple(std::move(gamma), std::move(theta));
}