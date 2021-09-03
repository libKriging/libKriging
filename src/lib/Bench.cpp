// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Bench.hpp"
#include "libKriging/Kriging.hpp"

#include "libKriging/utils/lk_armadillo.hpp"

// #include <optim.hpp>
#include <tuple>

// #include "libKriging/covariance.h"

LIBKRIGING_EXPORT Bench::Bench(int _n) {
  n = _n;
}

////////////////// LogLik /////////////////////
//' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikFun.R
//  model@covariance <- vect2covparam(model@covariance, param)
//  model@covariance@sd2 <- 1		# to get the correlation matrix
//
//  aux <- covMatrix(model@covariance, model@X)
//
//  R <- aux[[1]]
//  T <- chol(R)
//
//  x <- backsolve(t(T), model@y, upper.tri = FALSE)
//  M <- backsolve(t(T), model@F, upper.tri = FALSE)
//  z <- compute.z(x=x, M=M, beta=beta)
//  sigma2.hat <- compute.sigma2.hat(z)
//  logLik <- -0.5*(model@n * log(2*pi*sigma2.hat) + 2*sum(log(diag(T))) + model@n)

////////////////// LogLikGrad /////////////////////
//' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikGrad.R
//  logLik.derivative <- matrix(0,nparam,1)
//  x <- backsolve(T,z)			# compute x := T^(-1)*z
//  Rinv <- chol2inv(T)			# compute inv(R) by inverting T
//
//  Rinv.upper <- Rinv[upper.tri(Rinv)]
//  xx <- x%*%t(x)
//  xx.upper <- xx[upper.tri(xx)]
//
//  for (k in 1:nparam) {
//    gradR.k <- CovMatrixDerivative(model@covariance, X=model@X, C0=R, k=k)
//    gradR.k.upper <- gradR.k[upper.tri(gradR.k)]
//
//    terme1 <- sum(xx.upper*gradR.k.upper)   / sigma2.hat
//    # quick computation of t(x)%*%gradR.k%*%x /  ...
//    terme2 <- - sum(Rinv.upper*gradR.k.upper)
//    # quick computation of trace(Rinv%*%gradR.k)
//    logLik.derivative[k] <- terme1 + terme2
//  }

LIBKRIGING_EXPORT
arma::mat Bench::SolveTri(const arma::mat& Xtri, const arma::vec& y) {
  arma::mat s;
  for (int i = 0; i < n; i++) {
    s = arma::solve(arma::trimatu(Xtri), y, arma::solve_opts::fast);
  }
  return s;
}

LIBKRIGING_EXPORT
arma::mat Bench::CholSym(const arma::mat& Rsym) {
  arma::mat s;
  for (int i = 0; i < n; i++) {
    s = arma::chol(Rsym);
  }
  return s;
}

LIBKRIGING_EXPORT
std::tuple<arma::mat, arma::mat> Bench::QR(const arma::mat& M) {
  arma::mat Q;
  arma::mat R;
  for (int i = 0; i < n; i++) {
    arma::qr_econ(Q, R, M);
  }
  return std::make_tuple(std::move(Q), std::move(R));
}

LIBKRIGING_EXPORT
arma::mat Bench::InvSymPD(const arma::mat& Rsympd) {
  arma::mat s;
  for (int i = 0; i < n; i++) {
    s = arma::inv_sympd(Rsympd);
  }
  return s;
}

LIBKRIGING_EXPORT
double Bench::LogLik(Kriging& ok, const arma::vec& theta) {
  // arma::vec theta = 0.5*ones(ok->X().n_cols)
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};

  double s = 0;
  for (int i = 0; i < n; i++) {
    s += ok.logLikelihood(theta, nullptr, nullptr, &okm_data);
  }
  return s / n;
}

LIBKRIGING_EXPORT
arma::vec Bench::LogLikGrad(Kriging& ok, const arma::vec& theta) {
  // arma::vec theta = 0.5*ones(ok->X().n_cols)
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};

  arma::vec grad(theta.n_elem);

  arma::vec s = arma::zeros(theta.n_elem);
  for (int i = 0; i < n; i++) {
    s += ok.logLikelihood(theta, &grad, nullptr, &okm_data);
  }
  return s / n;
}
