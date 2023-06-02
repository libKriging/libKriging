// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/utils/lk_armadillo.hpp"

#include "libKriging/Bench.hpp"
#include "libKriging/CacheFunction.hpp"
#include "libKriging/Covariance.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/KrigingException.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include "libKriging/Random.hpp"
#include "libKriging/Trend.hpp"
#include "libKriging/utils/custom_hash_function.hpp"
#include "libKriging/utils/data_from_arma_vec.hpp"
#include "libKriging/utils/hdf5utils.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <tuple>
#include <vector>
#include <map>

/************************************************/
/**      Kriging implementation        **/
/************************************************/

// This will create the dist(xi,xj) function above. Need to parse "covType".
void Kriging::make_Cov(const std::string& covType) {
  m_covType = covType;
  if (covType.compare("gauss") == 0) {
    Cov = Covariance::Cov_gauss;
    DlnCovDtheta = Covariance::DlnCovDtheta_gauss;
    DlnCovDx = Covariance::DlnCovDx_gauss;
    Cov_pow = 2;
  } else if (covType.compare("exp") == 0) {
    Cov = Covariance::Cov_exp;
    DlnCovDtheta = Covariance::DlnCovDtheta_exp;
    DlnCovDx = Covariance::DlnCovDx_exp;
    Cov_pow = 1;
  } else if (covType.compare("matern3_2") == 0) {
    Cov = Covariance::Cov_matern32;
    DlnCovDtheta = Covariance::DlnCovDtheta_matern32;
    DlnCovDx = Covariance::DlnCovDx_matern32;
    Cov_pow = 1.5;
  } else if (covType.compare("matern5_2") == 0) {
    Cov = Covariance::Cov_matern52;
    DlnCovDtheta = Covariance::DlnCovDtheta_matern52;
    DlnCovDx = Covariance::DlnCovDx_matern52;
    Cov_pow = 2.5;
  } else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);

  // arma::cout << "make_Cov done." << arma::endl;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT Kriging::Kriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT Kriging::Kriging(const arma::colvec& y,
                                   const arma::mat& X,
                                   const std::string& covType,
                                   const Trend::RegressionModel& regmodel,
                                   bool normalize,
                                   const std::string& optim,
                                   const std::string& objective,
                                   const Parameters& parameters) {
  if (y.n_elem != X.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X.n_rows) + "x"
                             + std::to_string(X.n_cols) + "), y: (" + std::to_string(y.n_elem) + ")");

  make_Cov(covType);
  fit(y, X, regmodel, normalize, optim, objective, parameters);
}

LIBKRIGING_EXPORT Kriging::Kriging(const Kriging& other, ExplicitCopySpecifier) : Kriging{other} {}

// arma::mat XtX(arma::mat &X) {
//   arma::mat XtX = arma::zeros(X.n_cols,X.n_cols);
//   for (arma::uword i = 0; i < X.n_cols; i++) {
//     for (arma::uword j = 0; j <= i; j++) {
//       for (arma::uword k = 0; k < X.n_rows; k++) {
//         XtX.at(i,j) += X.at(k,i) * X.at(k,j);
//       }
//     }
//   }
//   return(symmatl(XtX));
// }

// Objective function for fit : -logLikelihood

double Kriging::_logLikelihood(const arma::vec& _theta,
                               arma::vec* grad_out,
                               arma::mat* hess_out,
                               Kriging::OKModel* okm_data,
                               std::map<std::string, double>* bench) const {
  // arma::cout << " theta: " << _theta << arma::endl;
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

  Kriging::OKModel* fd = okm_data;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;

  auto t0 = Bench::tic();
  arma::mat R = arma::mat(n, n);
  for (arma::uword i = 0; i < n; i++) {
    R.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = R.at(j, i) = Cov(m_dX.col(i * n + j), _theta);
    }
  }
  t0 = Bench::toc(bench, "R = Cov(dX)", t0);

  // Cholesky decompostion of covariance matrix
  fd->T = LinearAlgebra::safe_chol_lower(R);  // Do NOT trimatl T (slower because copy): trimatl(chol(R, "lower"));
  t0 = Bench::toc(bench, "T = Chol(R)", t0);

  // Compute intermediate useful matrices
  fd->M = solve(fd->T, m_F, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "M = F \\ T", t0);
  arma::mat Q;
  arma::mat G;
  arma::qr_econ(Q, G, fd->M);
  t0 = Bench::toc(bench, "Q,G = QR(M)", t0);

  arma::mat H;
  if (hess_out != nullptr) {  // H is not used otherwise...
    H = Q * Q.t();
    t0 = Bench::toc(bench, "H = Q * tQ", t0);
  }

  arma::colvec Yt = solve(fd->T, m_y, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "Yt = y \\ T", t0);
  if (fd->is_beta_estim) {
    fd->beta = solve(G, Q.t() * Yt, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "B = Qt * Yt \\ G", t0);
  }

  fd->z = Yt - fd->M * fd->beta;
  t0 = Bench::toc(bench, "z = Yt - M * B", t0);

  //' @ref https://github.com/cran/DiceKriging/blob/master/R/computeAuxVariables.R
  if (fd->is_sigma2_estim) {  // means no sigma2 provided
    fd->sigma2 = arma::accu(fd->z % fd->z) / n;
    t0 = Bench::toc(bench, "S2 = Acc(z * z) / n", t0);
  }
  // arma::cout << " sigma2:" << fd->sigma2 << arma::endl;

  double ll = -0.5 * (n * log(2 * M_PI * fd->sigma2) + 2 * sum(log(fd->T.diag())) + n);
  t0 = Bench::toc(bench, "ll = ...log(S2) + Sum(log(Td))...", t0);
  // arma::cout << " ll:" << ll << arma::endl;

  if (grad_out != nullptr) {
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

    t0 = Bench::tic();
    std::vector<arma::mat> gradsR(d);  // if (hess_out != nullptr)
    arma::vec terme1 = arma::vec(d);   // if (hess_out != nullptr)

    arma::mat Linv = solve(fd->T, arma::eye(n, n), LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "Li = I \\ T", t0);
    arma::mat Rinv = (Linv.t() * Linv);  // Do NOT inv_sympd (slower): inv_sympd(R);
    t0 = Bench::toc(bench, "Ri = Lit * Li", t0);

    arma::mat tT = fd->T.t();  // trimatu(trans(fd->T));
    t0 = Bench::toc(bench, "tT = Tt", t0);

    arma::mat x = solve(tT, fd->z, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "x = z \\ tT", t0);

    arma::cube gradR = arma::cube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradR.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = arma::mat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradR_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradR_k.at(i, j) = gradR_k.at(j, i) = gradR.slice(i).col(j)[k];
        }
      }
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      // should make a fast function trace_prod(A,B) -> sum_i(sum_j(Ai,j*Bj,i))
      terme1.at(k)
          = as_scalar((trans(x) * gradR_k) * x) / fd->sigma2;  //; //as_scalar((trans(x) * gradR_k) * x)/ sigma2_hat;
      double terme2 = -arma::trace(Rinv * gradR_k);            //-arma::accu(Rinv % gradR_k_upper)
      (*grad_out).at(k) = (terme1.at(k) + terme2) / 2;
      t0 = Bench::toc(bench, "grad_ll[k] = xt * gradR_k / S2 + tr(Ri * gradR_k)", t0);

      if (hess_out != nullptr) {
        //' @ref O. Roustant
        // for (k in 1:d) {
        //   for (l in 1:k) {
        //     aux <- grad_R[[k]] %*% Rinv %*% grad_R[[l]]
        //     Dkl <- d2_matcor(X, modele_proba$covariance, R, grad_logR, k,l)
        //     xk <- backsolve(t(T),grad_R[[k]]%*%x, upper.tri=FALSE)
        //     xl <- backsolve(t(T),grad_R[[l]]%*%x, upper.tri=FALSE)
        //
        //     hess_A <- - (t(xk) %*% H %*% xl) / sigma2_hat
        //     hess_B <- (t(x) %*% ( -Dkl+2*aux ) %*% x) / sigma2_hat
        //     hess_C <- - grad_A[k] * grad_A[l] / n
        //     hess_D <- - sum(diag( Rinv %*% aux ))
        //     hess_E <- sum(diag( Rinv %*% Dkl ))
        //
        //     hess_log_vrais[k,l] <- 2*hess_A + hess_B + hess_C + hess_D + hess_E
        //     hess_log_vrais[l,k] <- hess_log_vrais[k,l]
        //   }
        // }
        t0 = Bench::tic();

        gradsR[k] = gradR_k;

        for (arma::uword l = 0; l <= k; l++) {
          t0 = Bench::tic();
          arma::mat aux = gradsR[k] * Rinv * gradsR[l];
          t0 = Bench::toc(bench, "aux =  gradR_k[k] * Ri * gradR_k[l]", t0);

          arma::mat hessR_k_l = arma::mat(n, n);
          if (k == l) {
            for (arma::uword i = 0; i < n; i++) {
              hessR_k_l.at(i, i) = 0;
              for (arma::uword j = 0; j < i; j++) {
                double dln_k = gradR.slice(i).col(j)[k];
                hessR_k_l.at(i, j) = hessR_k_l.at(j, i) = dln_k * (dln_k / R.at(i, j) - (Cov_pow + 1) / _theta.at(k));
                // !! NO: it just work for exp type kernels. Matern MUST have a special treatment !!!
              }
            }
          } else {
            for (arma::uword i = 0; i < n; i++) {
              hessR_k_l.at(i, i) = 0;
              for (arma::uword j = 0; j < i; j++) {
                hessR_k_l.at(i, j) = hessR_k_l.at(j, i)
                    = gradR.slice(i).col(j)[k] * gradR.slice(i).col(j)[l] / R.at(i, j);
              }
            }
          }
          // hessR_k_l = arma::symmatu(hessR_k_l);
          t0 = Bench::toc(bench, "hessR_k_l = ...", t0);

          arma::mat xk = solve(fd->T, gradsR[k] * x, LinearAlgebra::default_solve_opts);
          arma::mat xl;
          if (k == l)
            xl = xk;
          else
            xl = solve(fd->T, gradsR[l] * x, LinearAlgebra::default_solve_opts);
          t0 = Bench::toc(bench, "xl = gradR_k[l] * x \\ T", t0);

          // arma::cout << " hess_A:" << -xk.t() * H * xl / sigma2_hat << arma::endl;
          // arma::cout << " hess_B:" << -x.t() * (hessR_k_l - 2*aux) * x / sigma2_hat << arma::endl;
          // arma::cout << " hess_C:" << -terme1.at(k) * terme1.at(l) / n << arma::endl;
          // arma::cout << " hess_D:" << -arma::trace(Rinv * aux)  << arma::endl;
          // arma::cout << " hess_E:" << arma::trace(Rinv * hessR_k_l) << arma::endl;

          (*hess_out).at(l, k) = (*hess_out).at(k, l)
              = (2.0 * xk.t() * H * xl / fd->sigma2 + x.t() * (hessR_k_l - 2 * aux) * x / fd->sigma2
                 + terme1.at(k) * terme1.at(l) / n + arma::trace(Rinv * aux) - arma::trace(Rinv * hessR_k_l))[0]
                / 2;  // should optim there using accu & %
          t0 = Bench::toc(bench, "hess_ll[l,k] = ...", t0);

          // arma::cout << " xk:" << xk << arma::endl;
          // arma::cout << " xl:" << xl << arma::endl;
          // arma::cout << " aux:" << aux << arma::endl;
          // arma::cout << " hessR_k_l:" << hessR_k_l << arma::endl;
        }  // for (arma::uword l = 0; l <= k; l++)
        //// t0 = Bench::toc("  hess_out    ", t0);
        //*hess_out = arma::symmatl(*hess_out);
        //// t0 = Bench::toc("  hess_out_sym", t0);
      }  // if (hess_out != nullptr)
    }    // for (arma::uword k = 0; k < m_X.n_cols; k++)
    // arma::cout << " grad_out:" << *grad_out << arma::endl;
    // if (hess_out != nullptr)
    //  arma::cout << " hess_out:" << *hess_out << arma::endl;
  }  // if (grad_out != nullptr)
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec, arma::mat> Kriging::logLikelihoodFun(const arma::vec& _theta,
                                                                                     const bool _grad,
                                                                                     const bool _hess,
                                                                                     const bool _bench) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};

  double ll = -1;
  arma::vec grad;
  arma::mat hess;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad || _hess) {
      grad = arma::vec(_theta.n_elem);
      if (!_hess) {
        ll = _logLikelihood(_theta, &grad, nullptr, &okm_data, &bench);
      } else {
        hess = arma::mat(_theta.n_elem, _theta.n_elem);
        ll = _logLikelihood(_theta, &grad, &hess, &okm_data, &bench);
      }
    } else
      ll = _logLikelihood(_theta, nullptr, nullptr, &okm_data, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad || _hess) {
      grad = arma::vec(_theta.n_elem);
      if (!_hess) {
        ll = _logLikelihood(_theta, &grad, nullptr, &okm_data, nullptr);
      } else {
        hess = arma::mat(_theta.n_elem, _theta.n_elem);
        ll = _logLikelihood(_theta, &grad, &hess, &okm_data, nullptr);
      }
    } else
      ll = _logLikelihood(_theta, nullptr, nullptr, &okm_data, nullptr);
  }

  return std::make_tuple(ll, std::move(grad), std::move(hess));
}

// Objective function for fit : -LOO

arma::colvec DiagABA(const arma::mat& A, const arma::mat& B) {
  arma::mat D = trimatu(2 * B);
  D.diag() = B.diag();
  D = (A * D) % A;
  arma::colvec c = sum(D, 1);

  return c;
}

double Kriging::_leaveOneOut(const arma::vec& _theta,
                             arma::vec* grad_out,
                             arma::mat* yhat_out,
                             Kriging::OKModel* okm_data,
                             std::map<std::string, double>* bench) const {
  // arma::cout << " theta: " << _theta << arma::endl;
  //' @ref https://github.com/DiceKrigingClub/DiceKriging/blob/master/R/leaveOneOutFun.R
  // model@covariance <- vect2covparam(model@covariance, param)
  // model@covariance@sd2 <- 1		# to get the correlation matrix
  //
  // R <- covMatrix(model@covariance, model@X)[[1]]
  // T <- chol(R)
  //
  // M <- backsolve(t(T), model@F, upper.tri = FALSE)
  //
  // Rinv <- chol2inv(T)             # cost : n*n*n/3
  //...
  //  Rinv.F <- Rinv %*% (model@F)    # cost : 2*n*n*p
  //  T.M <- chol(crossprod(M))       # cost : p*p*p/3, neglected
  //  aux <- backsolve(t(T.M), t(Rinv.F), upper.tri=FALSE)   # cost : p*p*n, neglected
  //  Q <- Rinv - crossprod(aux)      # cost : 2*n*n*(p-1/2)
  //  Q.y <- Q %*% (model@y)          # cost : 2*n*n
  //...
  // sigma2LOO <- 1/diag(Q)
  // errorsLOO <- sigma2LOO * (Q.y)       # cost : n, neglected
  //
  // LOOfun <- as.numeric(crossprod(errorsLOO)/model@n)

  Kriging::OKModel* fd = okm_data;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;

  auto t0 = Bench::tic();
  arma::mat R = arma::mat(n, n);
  for (arma::uword i = 0; i < n; i++) {
    R.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = R.at(j, i) = Cov(m_dX.col(i * n + j), _theta);
    }
  }
  t0 = Bench::toc(bench, "R = Cov(dX)", t0);

  // Cholesky decompostion of covariance matrix
  fd->T = LinearAlgebra::safe_chol_lower(R);
  t0 = Bench::toc(bench, "T = Chol(R)", t0);

  // Compute intermediate useful matrices
  fd->M = solve(fd->T, m_F, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "M = F \\ T", t0);

  arma::mat Rinv = inv_sympd(R);  // didn't find efficient chol2inv equivalent in armadillo
  t0 = Bench::toc(bench, "Ri = inv(R)", t0);

  arma::mat RinvF = Rinv * m_F;
  t0 = Bench::toc(bench, "RiF = Ri * F", t0);

  arma::mat TM = chol(trans(fd->M) * fd->M);  // Can be optimized with a crossprod equivalent in armadillo ?
  // arma::mat aux = solve(trans(TM), trans(RinvF));
  t0 = Bench::toc(bench, "TM = Chol(Mt * M)", t0);

  arma::mat aux = solve(trans(TM), trans(RinvF), LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "aux = RiF \\ TMt", t0);

  arma::mat Q = Rinv - trans(aux) * aux;  // Can be optimized with a crossprod equivalent in armadillo ?
  t0 = Bench::toc(bench, "Q = Ri - auxt*aux", t0);

  arma::mat Qy = Q * m_y;
  t0 = Bench::toc(bench, "Qy = Q * y", t0);

  arma::colvec sigma2LOO = 1 / Q.diag();
  t0 = Bench::toc(bench, "S2l = 1 / diag(Q)", t0);

  arma::colvec errorsLOO = sigma2LOO % Qy;
  t0 = Bench::toc(bench, "E = S2l * Qy", t0);

  if (yhat_out != nullptr) {
    (*yhat_out).col(0) = m_y - errorsLOO;
    (*yhat_out).col(1) = arma::sqrt(sigma2LOO);
  }

  double loo = arma::accu(errorsLOO % errorsLOO) / n;
  t0 = Bench::toc(bench, "loo = Acc(E * E) / n", t0);

  arma::colvec Yt = solve(fd->T, m_y, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "Yt = y \\ T", t0);

  if (fd->is_beta_estim) {
    // fd->beta = solve(fd->M, Yt, LinearAlgebra::default_solve_opts);
    arma::mat Q_qr;
    arma::mat G;
    arma::qr_econ(Q_qr, G, fd->M);
    t0 = Bench::toc(bench, "Q,G = QR(M)", t0);
    fd->beta = solve(G, Q_qr.t() * Yt, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "B = Qt * Yt \\ G", t0);
  }

  fd->z = Yt - fd->M * fd->beta;
  t0 = Bench::toc(bench, "z = Yt - M * B", t0);

  if (fd->is_sigma2_estim) {  // means no sigma2 provided
    fd->sigma2 = arma::mean(errorsLOO % errorsLOO % Q.diag());
    t0 = Bench::toc(bench, "S2 = Mean(E * E * diag(Q))", t0);
  }
  // arma::cout << " sigma2:" << fd->sigma2 << arma::endl;

  if (grad_out != nullptr) {
    //' @ref https://github.com/cran/DiceKriging/blob/master/R/leaveOneOutGrad.R
    // leaveOneOutDer <- matrix(0, nparam, 1)
    // for (k in 1:nparam) {
    //	gradR.k <- covMatrixDerivative(model@covariance, X=model@X, C0=R, k=k)
    //	diagdQ <- - diagABA(A=Q, B=gradR.k)
    //	dsigma2LOO <- - (sigma2LOO^2) * diagdQ
    //	derrorsLOO <- dsigma2LOO * Q.y - sigma2LOO * (Q%*%(gradR.k%*%Q.y))
    //	leaveOneOutDer[k] <- 2*crossprod(errorsLOO, derrorsLOO)/model@n
    //}

    t0 = Bench::tic();
    arma::cube gradR = arma::cube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradR.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < m_X.n_cols; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = arma::mat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradR_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradR_k.at(i, j) = gradR_k.at(j, i) = gradR.slice(i).col(j)[k];
        }
      }
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      arma::colvec diagdQ = -DiagABA(Q, gradR_k);
      t0 = Bench::toc(bench, "diagdQ = DiagABA(Q, gradR_k)", t0);

      arma::colvec dsigma2LOO = -sigma2LOO % sigma2LOO % diagdQ;
      t0 = Bench::toc(bench, "dS2l = -S2l % S2l % diagdQ", t0);

      arma::colvec derrorsLOO = dsigma2LOO % Qy - sigma2LOO % (Q * (gradR_k * Qy));
      t0 = Bench::toc(bench, "dE = dS2l * Qy- S2l * (Q * gradR_k * Qy)", t0);

      (*grad_out)(k) = 2 * dot(errorsLOO, derrorsLOO) / n;
      t0 = Bench::toc(bench, "grad_loo[k] = E * dE / n", t0);
    }
    // arma::cout << "Grad: " << *grad_out <<  arma::endl;
  }
  return loo;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::leaveOneOutFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};

  double loo = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      loo = _leaveOneOut(_theta, &grad, nullptr, &okm_data, &bench);
    } else
      loo = _leaveOneOut(_theta, nullptr, nullptr, &okm_data, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      loo = _leaveOneOut(_theta, &grad, nullptr, &okm_data, nullptr);
    } else
      loo = _leaveOneOut(_theta, nullptr, nullptr, &okm_data, nullptr);
  }

  return std::make_tuple(loo, std::move(grad));
}

LIBKRIGING_EXPORT std::tuple< arma::vec, arma::vec> Kriging::leaveOneOutVec(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};

  double loo = -1;
  arma::mat yhat = arma::mat(_theta.n_elem,2);
  loo = _leaveOneOut(_theta, nullptr, &yhat, &okm_data, nullptr);

  return std::make_tuple(std::move(yhat.col(0)), std::move(yhat.col(1)));
}

// Objective function for fit: bayesian-like approach fromm RobustGaSP

double Kriging::_logMargPost(const arma::vec& _theta,
                             arma::vec* grad_out,
                             Kriging::OKModel* okm_data,
                             std::map<std::string, double>* bench) const {
  // arma::cout << " theta: " << _theta << arma::endl;

  // In RobustGaSP:
  // neg_log_marginal_post_approx_ref <- function(param,nugget,
  // nugget.est,R0,X,zero_mean,output,CL,a,b,kernel_type,alpha) {
  //  lml=log_marginal_lik(param,nugget,nugget.est,R0,X,zero_mean,output,kernel_type,alpha);
  //  lp=log_approx_ref_prior(param,nugget,nugget.est,CL,a,b);
  //  -(lml+lp)
  //}
  // double log_marginal_lik(const Vec param,double nugget, const bool nugget_est, const List R0, const
  // Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output, Eigen::VectorXi
  // kernel_type,const Eigen::VectorXd alpha ){
  //  double nu=nugget;
  //  int param_size=param.size();
  //  Eigen::VectorXd beta= param.array().exp().matrix();
  //  ...beta
  //  R=R+nu*MatrixXd::Identity(num_obs,num_obs);  //not sure
  //
  //  LLT<MatrixXd> lltOfR(R);             // compute the cholesky decomposition of R called lltofR
  //  MatrixXd L = lltOfR.matrixL();   //retrieve factor L  in the decomposition
  //
  //  if(zero_mean=="Yes"){...}else{
  //
  //  int q=X.cols();
  //
  //  MatrixXd R_inv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward
  //  and one backward to compute R.inv%*%X MatrixXd Xt_R_inv_X=X.transpose()*R_inv_X; //Xt%*%R.inv%*%X
  //
  //  LLT<MatrixXd> lltOfXRinvX(Xt_R_inv_X); // cholesky decomposition of Xt_R_inv_X called lltOfXRinvX
  //  MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition
  //  MatrixXd R_inv_X_Xt_R_inv_X_inv_Xt_R_inv=
  //  R_inv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(R_inv_X.transpose())));
  //  //compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward and one backward solve MatrixXd yt_R_inv=
  //  (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); MatrixXd S_2=
  //  (yt_R_inv*output-output.transpose()*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output); double log_S_2=log(S_2(0,0)); return
  //  (-(L.diagonal().array().log().matrix().sum())-(LX.diagonal().array().log().matrix().sum())-(num_obs-q)/2.0*log_S_2);
  //  }
  //}
  // double log_approx_ref_prior(const Vec param,double nugget, bool nugget_est, const Eigen::VectorXd CL,const double
  // a,const double b ){
  //  double nu=nugget;
  //  int param_size=param.size();beta
  //  Eigen::VectorX beta= param.array().exp().matrix();
  //  ...
  //  double t=CL.cwiseProduct(beta).sum()+nu;
  //  return -b*t + a*log(t);
  //}

  Kriging::OKModel* fd = okm_data;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;

  auto t0 = Bench::tic();
  arma::mat R = arma::mat(n, n);
  for (arma::uword i = 0; i < n; i++) {
    R.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = R.at(j, i) = Cov(m_dX.col(i * n + j), _theta);
    }
  }
  t0 = Bench::toc(bench, "R = Cov(dX)", t0);

  // Cholesky decompostion of covariance matrix
  fd->T = LinearAlgebra::safe_chol_lower(R);
  t0 = Bench::toc(bench, "T = Chol(R)", t0);

  //  // Compute intermediate useful matrices
  //  fd->M = solve(fd->T, m_F, LinearAlgebra::default_solve_opts);
  //  // t0 = Bench::toc("M             ", t0);
  //  arma::mat Q;
  //  arma::mat G;
  //  qr_econ(Q, G, fd->M);
  //  // t0 = Bench::toc("QG            ", t0);
  //
  //  arma::mat H = Q * Q.t();  // if (hess_out != nullptr)
  //  // t0 = Bench::toc("H             ", t0);
  //  arma::colvec Yt = solve(fd->T, m_y, LinearAlgebra::default_solve_opts);
  //  // t0 = Bench::toc("Yt            ", t0);
  //  if (fd->is_beta_estim)
  // fd->beta = solve(trimatu(G), Q.t() * Yt, LinearAlgebra::default_solve_opts);
  //  // t0 = Bench::toc("beta          ", t0);
  //  fd->z = Yt - fd->M * fd->beta;
  //  // t0 = Bench::toc("z             ", t0);

  // Keep RobustGaSP naming from now...
  arma::mat X = m_F;
  arma::mat L = fd->T;

  fd->M = solve(L, X, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "M = F \\ T", t0);

  arma::mat R_inv_X = solve(trans(L), fd->M, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "RiF = Ri * F", t0);

  arma::mat Xt_R_inv_X = trans(X) * R_inv_X;  // Xt%*%R.inv%*%X
  t0 = Bench::toc(bench, "FtRiF = Ft * RiF", t0);

  arma::mat LX = chol(Xt_R_inv_X, "lower");  //  retrieve factor LX  in the decomposition
  t0 = Bench::toc(bench, "TF = Chol(FtRiF)", t0);

  arma::mat R_inv_X_Xt_R_inv_X_inv_Xt_R_inv
      = R_inv_X
        * (solve(trans(LX),
                 solve(LX, trans(R_inv_X), LinearAlgebra::default_solve_opts),
                 LinearAlgebra::default_solve_opts));  // compute  R_inv_X_Xt_R_inv_X_inv_Xt_R_inv through one forward
                                                       // and one backward solve
  t0 = Bench::toc(bench, "RiFFtRiFiFtRi = RiF * RiFt \\ M \\ Mt", t0);

  arma::colvec Yt = solve(L, m_y, LinearAlgebra::default_solve_opts);
  t0 = Bench::toc(bench, "Yt = y \\ T", t0);

  if (fd->is_beta_estim) {
    arma::mat Q;
    arma::mat G;
    arma::qr_econ(Q, G, fd->M);
    t0 = Bench::toc(bench, "Q,G = QR(M)", t0);
    fd->beta = solve(G, Q.t() * Yt, LinearAlgebra::default_solve_opts);
    t0 = Bench::toc(bench, "B = Qt * Yt \\ G", t0);
  }

  fd->z = Yt - fd->M * fd->beta;  // required for later predict
  t0 = Bench::toc(bench, "z = Yt - M * B", t0);

  arma::mat yt_R_inv = trans(solve(trans(L), Yt, LinearAlgebra::default_solve_opts));
  t0 = Bench::toc(bench, "YtRi = Yt \\ Tt", t0);

  arma::mat S_2 = (yt_R_inv * m_y - trans(m_y) * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv * m_y);
  t0 = Bench::toc(bench, "S2 = YtRi * y - yt * RiFFtRiFiFtRi * y", t0);

  if (fd->is_sigma2_estim)  // means no sigma2 provided
    fd->sigma2 = S_2(0, 0) / (n - d);

  double log_S_2 = log(S_2(0, 0));
  double log_marginal_lik = -sum(log(L.diag())) - sum(log(LX.diag())) - (m_X.n_rows - m_F.n_cols) / 2.0 * log_S_2;
  t0 = Bench::toc(bench, "lml = -Sum(log(diag(T))) - Sum(log(diag(TF)))...", t0);
  // arma::cout << " log_marginal_lik:" << log_marginal_lik << arma::endl;

  // Default prior params
  double a = 0.2;
  double b = 1.0 / pow(m_X.n_rows, 1.0 / m_X.n_cols) * (a + 1.0);
  // t0 = Bench::toc("b             ", t0);

  arma::vec CL = trans(max(m_X, 0) - min(m_X, 0)) / pow(m_X.n_rows, 1.0 / m_X.n_cols);
  t0 = Bench::toc(bench, "CL = (max(X) - min(X)) / n^1/d", t0);

  double t = arma::accu(CL % pow(_theta, -1.0));
  // arma::cout << " a:" << a << arma::endl;
  // arma::cout << " b:" << b << arma::endl;
  // arma::cout << " t:" << t << arma::endl;

  double log_approx_ref_prior = -b * t + a * log(t);
  // arma::cout << " log_approx_ref_prior:" << log_approx_ref_prior << arma::endl;

  if (grad_out != nullptr) {
    // Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param,double nugget,  bool nugget_est, const List
    // R0, const Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output,
    // Eigen::VectorXi kernel_type,const Eigen::VectorXd alpha){
    // ...
    // VectorXd ans=VectorXd::Ones(param_size);
    // ...
    // MatrixXd Q_output= yt_R_inv.transpose()-R_inv_X_Xt_R_inv_X_inv_Xt_R_inv*output;
    // MatrixXd dev_R_i;
    // MatrixXd Wb_ti;
    // //allow different choices of kernels
    //
    // for(int ti=0;ti<p;ti++){
    //   //kernel_type_ti=kernel_type[ti];
    //   if(kernel_type[ti]==3){
    //     dev_R_i=matern_5_2_deriv( R0[ti],R_ori,beta[ti]);  //now here I have R_ori instead of R
    //   }else {...}
    //   Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
    //   ans[ti]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0);
    // }

    t0 = Bench::tic();
    arma::vec ans = arma::ones(m_X.n_cols);
    arma::mat Q_output = trans(yt_R_inv) - R_inv_X_Xt_R_inv_X_inv_Xt_R_inv * m_y;
    t0 = Bench::toc(bench, "Qo = YtRi - RiFFtRiFiFtRi * y", t0);

    arma::cube gradR = arma::cube(d, n, n);
    for (arma::uword i = 0; i < n; i++) {
      for (arma::uword j = 0; j < i; j++) {
        gradR.slice(i).col(j) = R.at(i, j) * DlnCovDtheta(m_dX.col(i * n + j), _theta);
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    arma::mat Wb_k;
    for (arma::uword k = 0; k < m_X.n_cols; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = arma::mat(n, n);
      for (arma::uword i = 0; i < n; i++) {
        gradR_k.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradR_k.at(i, j) = gradR_k.at(j, i) = gradR.slice(i).col(j)[k];
        }
      }
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      Wb_k = trans(solve(
                 trans(L), solve(L, gradR_k, LinearAlgebra::default_solve_opts), LinearAlgebra::default_solve_opts))
             - gradR_k * R_inv_X_Xt_R_inv_X_inv_Xt_R_inv;
      t0 = Bench::toc(bench, "Wb_k = gradR_k \\ T \\ Tt - gradR_k * RiFFtRiFiFtRi", t0);

      ans[k] = -0.5 * sum(Wb_k.diag())
               + (m_X.n_rows - m_F.n_cols) / 2.0 * (trans(m_y) * trans(Wb_k) * Q_output / S_2(0, 0))[0];
      t0 = Bench::toc(bench, "ans[k] = Sum(diag(Wb_k)) + yt * Wb_kt * Qo / S2...", t0);
    }
    // arma::cout << " log_marginal_lik_deriv:" << ans << arma::endl;
    // arma::cout << " log_approx_ref_prior_deriv:" <<  - (a * CL / t - b * CL) / pow(_theta, 2.0) << arma::endl;

    *grad_out = ans - (a * CL / t - b * CL) / pow(_theta, 2.0);
    // t0 = Bench::toc(" grad_out     ", t0);
    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }

  // arma::cout << " lmp:" << (log_marginal_lik+log_approx_ref_prior) << arma::endl;
  return (log_marginal_lik + log_approx_ref_prior);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logMargPostFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2{};
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};

  double lmp = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      lmp = _logMargPost(_theta, &grad, &okm_data, &bench);
    } else
      lmp = _logMargPost(_theta, nullptr, &okm_data, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      lmp = _logMargPost(_theta, &grad, &okm_data, nullptr);
    } else
      lmp = _logMargPost(_theta, nullptr, &okm_data, nullptr);
  }

  return std::make_tuple(lmp, std::move(grad));
}

LIBKRIGING_EXPORT double Kriging::logLikelihood() {
  return std::get<0>(Kriging::logLikelihoodFun(m_theta, false, false, false));
}

LIBKRIGING_EXPORT double Kriging::leaveOneOut() {
  return std::get<0>(Kriging::leaveOneOutFun(m_theta, false, false));
}

LIBKRIGING_EXPORT double Kriging::logMargPost() {
  return std::get<0>(Kriging::logMargPostFun(m_theta, false, false));
}

double optim_newton(std::function<double(arma::vec& x, arma::vec* grad_out, arma::mat* hess_out)> f,
                    arma::vec& x_0,
                    const arma::vec& x_lower,
                    const arma::vec& x_upper) {
  if (Optim::log_level > 0)
    arma::cout << "x_0: " << x_0 << " ";

  double delta = 0.1;
  double x_tol = 0.01;  // Optim::solution_rel_tolerance;
  double y_tol = Optim::objective_rel_tolerance;
  double g_tol = Optim::gradient_tolerance;
  int max_iteration = Optim::max_iteration;

  arma::vec x_previous(x_0.n_elem);
  arma::vec x_best(x_0.n_elem);
  double f_previous = std::numeric_limits<double>::infinity();
  double f_new = std::numeric_limits<double>::infinity();
  double f_best = std::numeric_limits<double>::infinity();

  arma::vec x = x_0;
  arma::vec grad(x.n_elem);
  arma::mat hess(x.n_elem, x.n_elem);
  int i = 0;
  while (i < max_iteration) {
    if (Optim::log_level > 0) {
      arma::cout << "iteration: " << i << arma::endl;
      arma::cout << "  x: " << x << arma::endl;
    }

    f_previous = f_new;
    f_new = f(x, &grad, &hess);
    if (Optim::log_level > 1)
      arma::cout << "    f_new: " << f_new << arma::endl;
    if (f_best > f_new) {
      f_best = f_new;
      x_best = x;
    }

    if (std::abs((f_new - f_previous) / f_previous) < y_tol) {
      if (Optim::log_level > 0)
        arma::cout << "  X f_new ~ f_previous" << arma::endl;
      break;
    }
    if (arma::abs(grad).max() < g_tol) {
      if (Optim::log_level > 0)
        arma::cout << "  X grad ~ 0" << arma::endl;
      break;
    }

    arma::vec delta_x(x.n_elem);
    if (Optim::log_level > 2)
      arma::cout << "  eig(hess)" << arma::eig_sym(hess) << arma::endl;
    if (arma::all(arma::eig_sym(hess) > 0)) {
      // try to fit a second order polynom to use its minimizer. Otherwise, just iterate with conjugate gradient
      // arma::cout << "!";
      delta_x = arma::solve(hess, grad, arma::solve_opts::likely_sympd);
    } else {
      delta_x = delta * grad / std::sqrt(arma::sum(arma::cdot(grad, grad)));
    }
    if (Optim::log_level > 2)
      arma::cout << "  delta_x: " << delta_x << arma::endl;

    arma::vec x_next = x - delta_x;
    if (Optim::log_level > 1)
      arma::cout << "  x_next: " << x_next << arma::endl;

    for (arma::uword j = 0; j < x_next.n_elem; j++) {
      if (x_next[j] < x_lower[j]) {
        if (Optim::log_level > 2)
          arma::cout << "    <" << x_lower[j] << arma::endl;
        delta_x = delta_x * (x[j] - x_lower[j]) / (x[j] - x_next[j]) / 2;
        x_next = x - delta_x;
      }
      if (x_next[j] > x_upper[j]) {
        if (Optim::log_level > 2)
          arma::cout << "    >" << x_upper[j] << arma::endl;
        delta_x = delta_x * (x_upper[j] - x[j]) / (x_next[j] - x[j]) / 2;
        x_next = x - delta_x;
      }
    }
    if (Optim::log_level > 2)
      arma::cout << "    delta_x: " << delta << arma::endl;
    if (Optim::log_level > 2)
      arma::cout << "    x_next: " << x_next << arma::endl;

    if (arma::abs((x - x_next) / x).max() < x_tol) {
      if (Optim::log_level > 1)
        arma::cout << "  X x_0 ~ x_next" << arma::endl;
      break;
    }

    x_previous = x;
    x = x_next;

    // TODO : keep best result instead of last one
    ++i;
    if (Optim::log_level > 0)
      arma::cout << "  f_best: " << f_best << arma::endl;
  }

  x_0 = x_best;
  if (Optim::log_level > 0)
    arma::cout << " " << x_0 << " " << f_best << arma::endl;

  return f_best;
}

/** Fit the kriging object on (X,y):
 * @param y is n length column vector of output
 * @param X is n*d matrix of input
 * @param regmodel is the regression model to be used for the GP mean (choice between contant, linear, quadratic)
 * @param normalize is a boolean to enforce inputs/output normalization
 * @param optim is an optimizer name from OptimLib, or 'none' to keep parameters unchanged
 * @param objective is 'LOO' or 'LL'. Ignored if optim=='none'.
 * @param parameters starting values for hyper-parameters for optim, or final values if optim=='none'.
 */
LIBKRIGING_EXPORT void Kriging::fit(const arma::colvec& y,
                                    const arma::mat& X,
                                    const Trend::RegressionModel& regmodel,
                                    bool normalize,
                                    const std::string& optim,
                                    const std::string& objective,
                                    const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<double(const arma::vec& _gamma, arma::vec* grad_out, arma::mat* hess_out, Kriging::OKModel* okm_data)>
      fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* hess_out, Kriging::OKModel* okm_data) {
            // Change variable for opt: . -> 1/exp(.)
            // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
            const arma::vec _theta = Optim::reparam_from(_gamma);
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double ll = this->_logLikelihood(_theta, grad_out, hess_out, okm_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > ll: " << ll << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
              *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
            }
            if (hess_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > hess ll: " << hess_out << arma::endl;
              *hess_out = -Optim::reparam_from_deriv2(_theta, *grad_out, *hess_out);
            }
            return -ll;
          });
    } else {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* hess_out, Kriging::OKModel* okm_data) {
            const arma::vec _theta = _gamma;
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double ll = this->_logLikelihood(_theta, grad_out, hess_out, okm_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > ll: " << ll << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
              *grad_out = -*grad_out;
            }
            if (hess_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > hess ll: " << hess_out << arma::endl;
              *hess_out = -*hess_out;
            }
            return -ll;
          });
    }
  } else if (objective.compare("LOO") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::OKModel* okm_data) {
            // Change variable for opt: . -> 1/exp(.)
            // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
            const arma::vec _theta = Optim::reparam_from(_gamma);
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double loo = this->_leaveOneOut(_theta, grad_out, nullptr, okm_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > loo: " << loo << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
              *grad_out = Optim::reparam_from_deriv(_theta, *grad_out);
            }
            return loo;
          });
    } else {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::OKModel* okm_data) {
            const arma::vec _theta = _gamma;
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double loo = this->_leaveOneOut(_theta, grad_out, nullptr, okm_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > loo: " << loo << arma::endl;
            // if (grad_out != nullptr) {
            //   if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
            //   *grad_out = *grad_out; // so not necessary
            //  }
            return loo;
          });
    }
  } else if (objective.compare("LMP") == 0) {
    // Our impl. of https://github.com/cran/RobustGaSP/blob/5cf21658e6a6e327be6779482b93dfee25d24592/R/rgasp.R#L303
    //@see Mengyang Gu, Xiao-jing Wang and Jim Berger, 2018, Annals of Statistics.
    if (Optim::reparametrize) {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::OKModel* okm_data) {
            // Change variable for opt: . -> 1/exp(.)
            // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
            const arma::vec _theta = Optim::reparam_from(_gamma);
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double lmp = this->_logMargPost(_theta, grad_out, okm_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
              *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
            }
            return -lmp;
          });
    } else {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::OKModel* okm_data) {
            const arma::vec _theta = _gamma;
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double lmp = this->_logMargPost(_theta, grad_out, okm_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
              *grad_out = -*grad_out;
            }
            return -lmp;
          });
    }
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective + " (supported are: LL, LOO, LMP)");

  arma::rowvec centerX(d);
  arma::rowvec scaleX(d);
  double centerY;
  double scaleY;
  // Normalization of inputs and output
  m_normalize = normalize;
  if (m_normalize) {
    centerX = min(X, 0);
    scaleX = max(X, 0) - min(X, 0);
    centerY = min(y);
    scaleY = max(y) - min(y);
  } else {
    centerX.zeros();
    scaleX.ones();
    centerY = 0;
    scaleY = 1;
  }
  m_centerX = centerX;
  m_scaleX = scaleX;
  m_centerY = centerY;
  m_scaleY = scaleY;
  {  // FIXME why copies of newX and newy
    arma::mat newX = X;
    newX.each_row() -= centerX;
    newX.each_row() /= scaleX;
    arma::colvec newy = (y - centerY) / scaleY;
    this->m_X = newX;
    this->m_y = newy;
  }

  // Now we compute the distance matrix between points. Will be used to compute R(theta) later (e.g. when fitting)
  m_dX = arma::zeros(d, n * n);
  for (arma::uword ij = 0; ij < m_dX.n_cols; ij++) {
    int i = (int)ij / n;
    int j = ij % n;  // i,j <-> i*n+j
    if (i < j) {
      m_dX.col(ij) = trans(m_X.row(i) - m_X.row(j));
      m_dX.col(j * n + i) = m_dX.col(ij);
    }
  }

  // Define regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);

  arma::mat theta0;
  if (parameters.theta.has_value()) {
    theta0 = parameters.theta.value();
    if (parameters.theta.value().n_cols != d && parameters.theta.value().n_rows == d)
      theta0 = parameters.theta.value().t();
    if (m_normalize)
      theta0.each_row() /= scaleX;
    if (theta0.n_cols != d)
      throw std::runtime_error("Dimension of theta should be nx" + std::to_string(d) + " instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
  }

  if (optim == "none") {  // just keep given theta, no optimisation of ll (but estim sigma2 & beta still possible)
    if (!parameters.theta.has_value())
      throw std::runtime_error("Theta should be given (1x" + std::to_string(d) + ") matrix, when optim=none");

    m_theta = trans(theta0.row(0));
    m_est_theta = false;
    arma::mat T;
    arma::mat M;
    arma::colvec z;
    arma::colvec beta;
    bool is_beta_estim = parameters.is_beta_estim;
    if (parameters.beta.has_value()) {
      beta = parameters.beta.value();
      if (m_normalize)
        beta /= scaleY;
    } else {
      is_beta_estim = true;  // force estim if no value given
    }
    double sigma2 = -1;
    bool is_sigma2_estim = parameters.is_sigma2_estim;
    if (parameters.sigma2.has_value()) {
      sigma2 = parameters.sigma2.value();  // otherwise sigma2 will be re-calculated using given theta
      if (m_normalize)
        sigma2 /= (scaleY * scaleY);
    } else {
      is_sigma2_estim = true;  // force estim if no value given
    }

    Kriging::OKModel okm_data{T, M, z, beta, is_beta_estim, sigma2, is_sigma2_estim};

    arma::vec gamma_tmp = arma::vec(d);
    gamma_tmp = m_theta;
    if (Optim::reparametrize) {
      gamma_tmp = Optim::reparam_to(m_theta);
    }

    /* double min_ofn_tmp = */ fit_ofn(gamma_tmp, nullptr, nullptr, &okm_data);

    m_T = std::move(okm_data.T);
    m_M = std::move(okm_data.M);
    m_z = std::move(okm_data.z);
    m_est_beta = is_beta_estim;
    if (m_est_beta) {
      m_beta = std::move(okm_data.beta);
    } else {
      m_beta = beta;
    }
    m_est_sigma2 = is_sigma2_estim;
    if (m_est_sigma2) {
      m_sigma2 = okm_data.sigma2;
    } else {
      m_sigma2 = sigma2;
    }

  } else {
    arma::vec theta_lower = Optim::theta_lower_factor * trans(max(m_X, 0) - min(m_X, 0));
    arma::vec theta_upper = Optim::theta_upper_factor * trans(max(m_X, 0) - min(m_X, 0));

    if (Optim::variogram_bounds_heuristic) {
      arma::vec dy2 = arma::zeros(n * n);
      for (arma::uword ij = 0; ij < dy2.n_elem; ij++) {
        int i = (int)ij / n;
        int j = ij % n;  // i,j <-> i*n+j
        if (i < j) {
          dy2[ij] = m_y.at(i) - m_y.at(j);
          dy2[ij] *= dy2[ij];
          dy2[j * n + i] = dy2[ij];
        }
      }
      // dy2 /= arma::var(m_y);
      arma::vec dy2dX2_slope = dy2 / arma::sum(m_dX % m_dX, 0).t();
      // arma::cout << "dy2dX_slope:" << dy2dX_slope << arma::endl;
      dy2dX2_slope.replace(arma::datum::nan, 0.0);  // we are not interested in same points where dX=0, and dy=0
      arma::vec w = dy2dX2_slope / sum(dy2dX2_slope);
      arma::mat steepest_dX_mean = arma::abs(m_dX) * w;
      // arma::cout << "steepest_dX_mean:" << steepest_dX_mean << arma::endl;

      theta_lower = arma::max(theta_lower, Optim::theta_lower_factor * steepest_dX_mean);
      // no, only relevant for inf bound: theta_upper = arma::min(theta_upper, Optim::theta_upper_factor *
      // steepest_dX_mean);
      theta_lower = arma::min(theta_lower, theta_upper);
      theta_upper = arma::max(theta_lower, theta_upper);

      // arma::vec steepest_dX = arma::abs(m_dX.col(arma::index_max(dy2dX_slope))); // worst slope dX point
      // arma::cout << "steepest_dX:" << steepest_dX << arma::endl;
      // dy2dX_slope.replace(-1.0, std::numeric_limits<double>::infinity()); // we are not interested in same points
      // where dX=0, and dy=0 arma::vec smoothest_dX = arma::abs(m_dX.col(arma::index_min(dy2dX_slope))); // worst slope
      // dX point arma::cout << "smoothest_dX:" << smoothest_dX << arma::endl; arma::vec weighted_mean_dX = (arma::mean(
      // m_dX % trans(arma::repmat(dy2, 1, m_dX.n_rows)), 1)) / arma::sum(dy2); // weighted mean dX point arma::cout <<
      // "weighted_mean_dX:" << weighted_mean_dX << arma::endl; theta_lower =
      // arma::max(theta_lower,Optim::theta_lower_factor * steepest_dX); theta_upper =
      // arma::min(theta_upper,Optim::theta_upper_factor * steepest_dX);
    }
    // arma::cout << "theta_lower:" << theta_lower << arma::endl;
    // arma::cout << "theta_upper:" << theta_upper << arma::endl;

    if (optim.rfind("BFGS", 0) == 0) {
      Random::init();

      // FIXME parameters.has needs to implemtented (no use case in current code)
      if (!parameters.theta.has_value()) {  // no theta given, so draw 10 random uniform starting values
        int multistart = 1;
        try {
          multistart = std::stoi(optim.substr(4));
        } catch (std::invalid_argument&) {
          // let multistart = 1
        }
        theta0 = arma::repmat(trans(theta_lower), multistart, 1)
                 + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
        // theta0 = arma::abs(0.5 + Random::randn_mat(multistart, d) / 6.0)
        //          % arma::repmat(max(m_X, 0) - min(m_X, 0), multistart, 1);
      } else {  // just use given theta(s) as starting values for multi-bfgs
        theta0 = arma::mat(parameters.theta.value());
        if (m_normalize)
          theta0.each_row() /= scaleX;
      }
      // arma::cout << "theta0:" << theta0 << arma::endl;

      arma::vec gamma_lower = theta_lower;
      arma::vec gamma_upper = theta_upper;
      if (Optim::reparametrize) {
        gamma_lower = Optim::reparam_to(theta_upper);
        gamma_upper = Optim::reparam_to(theta_lower);
      }

      double min_ofn = std::numeric_limits<double>::infinity();

      for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
        arma::vec gamma_tmp = theta0.row(i).t();
        if (Optim::reparametrize)
          gamma_tmp = Optim::reparam_to(theta0.row(i).t());

        gamma_lower = arma::min(gamma_tmp, gamma_lower);
        gamma_upper = arma::max(gamma_tmp, gamma_upper);

        if (Optim::log_level > 0) {
          arma::cout << "BFGS:" << arma::endl;
          arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
          arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
          arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower << " ";
          arma::cout << "  upper_bounds: " << theta_upper << " ";
          arma::cout << "  start_point: " << theta0.row(i) << " ";
        }

        arma::mat T;
        arma::mat M;
        arma::colvec z;
        arma::colvec beta;
        if (parameters.beta.has_value()) {
          beta = parameters.beta.value();
          if (m_normalize)
            beta /= scaleY;
        }
        double sigma2 = -1;
        if (parameters.sigma2.has_value()) {
          sigma2 = parameters.sigma2.value();
          if (m_normalize)
            sigma2 /= (scaleY * scaleY);
        }

        Kriging::OKModel okm_data{T, M, z, beta, parameters.is_beta_estim, sigma2, parameters.is_sigma2_estim};

        lbfgsb::Optimizer optimizer{d};
        optimizer.iprint = Optim::log_level - 2;
        optimizer.max_iter = Optim::max_iteration;
        optimizer.pgtol = Optim::gradient_tolerance;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
        arma::ivec bounds_type{d, arma::fill::value(2)};  // means both upper & lower bounds

        int retry = 0;
        while (retry <= Optim::max_restart) {
          arma::vec gamma_0 = gamma_tmp;
          auto result = optimizer.minimize(
              [&okm_data, &fit_ofn](const arma::vec& vals_inp, arma::vec& grad_out) -> double {
                return fit_ofn(vals_inp, &grad_out, nullptr, &okm_data);
              },
              gamma_tmp,
              gamma_lower.memptr(),
              gamma_upper.memptr(),
              bounds_type.memptr());

          if (Optim::log_level > 0) {
            arma::cout << "     iterations: " << result.num_iters << arma::endl;
            if (Optim::reparametrize) {
              arma::cout << "     start_point: " << Optim::reparam_from(gamma_0).t() << " ";
              arma::cout << "     solution: " << Optim::reparam_from(gamma_tmp).t() << " ";
            } else {
              arma::cout << "     start_point: " << gamma_0.t() << " ";
              arma::cout << "     solution: " << gamma_tmp.t() << " ";
            }
          }

          double sol_to_lb = arma::min(arma::abs(gamma_tmp - gamma_lower));
          double sol_to_ub = arma::min(arma::abs(gamma_tmp - gamma_upper));
          double sol_to_b = Optim::reparametrize ? sol_to_ub : sol_to_lb;  // just consider theta lower bound
          if ((retry < Optim::max_restart)                                 //&& (result.num_iters <= 2 * d)
              && ((sol_to_b < arma::datum::eps)                            // we fastly converged to one bound
                  || (result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0))) {
            gamma_tmp = (theta0.row(i).t() + theta_lower)
                        / pow(2.0, retry + 1);  // so, re-use previous starting point and change it to middle-point

            if (Optim::reparametrize)
              gamma_tmp = Optim::reparam_to(gamma_tmp);

            gamma_lower = arma::min(gamma_tmp, gamma_lower);
            gamma_upper = arma::max(gamma_tmp, gamma_upper);

            retry++;
          } else {
            if (Optim::log_level > 1)
              result.print();
            break;
          }
        }

        // this last call of fit_ofn is to ensure that T and z are up-to-date with solution found.
        double min_ofn_tmp = fit_ofn(gamma_tmp, nullptr, nullptr, &okm_data);

        if (Optim::log_level > 0) {
          arma::cout << "  best objective: " << min_ofn_tmp << arma::endl;
          if (Optim::reparametrize)
            arma::cout << "  best solution: " << Optim::reparam_from(gamma_tmp) << " ";
          else
            arma::cout << "  best solution: " << gamma_tmp << " ";
        }

        if (min_ofn_tmp < min_ofn) {
          m_theta = gamma_tmp;
          if (Optim::reparametrize)
            m_theta = Optim::reparam_from(gamma_tmp);
          m_est_theta = true;
          min_ofn = min_ofn_tmp;
          m_T = std::move(okm_data.T);
          m_M = std::move(okm_data.M);
          m_z = std::move(okm_data.z);
          m_beta = std::move(okm_data.beta);
          m_est_beta = parameters.is_beta_estim;
          m_sigma2 = okm_data.sigma2;
          m_est_sigma2 = parameters.is_sigma2_estim;
        }
      }
    } else if (optim.rfind("Newton", 0) == 0) {
      Random::init();

      // FIXME parameters.has needs to implemtented (no use case in current code)
      if (!parameters.theta.has_value()) {  // no theta given, so draw 10 random uniform starting values
        int multistart = 1;
        try {
          multistart = std::stoi(optim.substr(4));
        } catch (std::invalid_argument&) {
          // let multistart = 1
        }
        theta0 = arma::repmat(trans(theta_lower), multistart, 1)
                 + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
      } else {  // just use given theta(s) as starting values for multi-bfgs
        theta0 = arma::mat(parameters.theta.value());
        if (m_normalize)
          theta0.each_row() /= scaleX;
      }

      // arma::cout << "theta0:" << theta0 << arma::endl;

      arma::vec gamma_lower = theta_lower;
      arma::vec gamma_upper = theta_upper;
      if (Optim::reparametrize) {
        gamma_lower = Optim::reparam_to(theta_upper);
        gamma_upper = Optim::reparam_to(theta_lower);
      }

      double min_ofn = std::numeric_limits<double>::infinity();

      for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
        arma::vec gamma_tmp = theta0.row(i).t();
        if (Optim::reparametrize)
          gamma_tmp = Optim::reparam_to(theta0.row(i).t());

        gamma_lower = arma::min(gamma_tmp, gamma_lower);
        gamma_upper = arma::max(gamma_tmp, gamma_upper);

        if (Optim::log_level > 0) {
          arma::cout << "Newton:" << arma::endl;
          arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
          arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
          arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower << " ";
          arma::cout << "  upper_bounds: " << theta_upper << " ";
          arma::cout << "  start_point: " << theta0.row(i) << " ";
        }

        arma::mat T;
        arma::mat M;
        arma::colvec z;
        arma::colvec beta;
        if (parameters.beta.has_value()) {
          beta = parameters.beta.value();
          if (m_normalize)
            beta /= scaleY;
        }
        double sigma2 = -1;
        if (parameters.sigma2.has_value()) {
          sigma2 = parameters.sigma2.value();
          if (m_normalize)
            sigma2 /= (scaleY * scaleY);
        }

        Kriging::OKModel okm_data{T, M, z, beta, parameters.is_beta_estim, sigma2, parameters.is_sigma2_estim};

        double min_ofn_tmp = optim_newton(
            [&okm_data, &fit_ofn](const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out) -> double {
              return fit_ofn(vals_inp, grad_out, hess_out, &okm_data);
            },
            gamma_tmp,
            gamma_lower,
            gamma_upper);

        if (Optim::log_level > 0) {
          arma::cout << "  best objective: " << min_ofn_tmp << arma::endl;
          if (Optim::reparametrize)
            arma::cout << "  best solution: " << Optim::reparam_from(gamma_tmp) << " ";
          else
            arma::cout << "  best solution: " << gamma_tmp << " ";
        }

        if (min_ofn_tmp < min_ofn) {
          m_theta = gamma_tmp;
          if (Optim::reparametrize)
            m_theta = Optim::reparam_from(gamma_tmp);
          m_est_theta = true;
          min_ofn = min_ofn_tmp;
          m_T = std::move(okm_data.T);
          m_M = std::move(okm_data.M);
          m_z = std::move(okm_data.z);
          m_beta = std::move(okm_data.beta);
          m_est_beta = parameters.is_beta_estim;
          m_sigma2 = okm_data.sigma2;
          m_est_sigma2 = parameters.is_sigma2_estim;
        }
      }
    } else
      throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS[#], Newton[#])");
  }

  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param Xp is m*d matrix of points where to predict output
 * @param std is true if return also stdev column vector
 * @param cov is true if return also cov matrix between Xp
 * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat, arma::mat, arma::mat>
Kriging::predict(const arma::mat& Xp, bool withStd, bool withCov, bool withDeriv) {
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (Xp.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(Xp.n_cols) + " instead of "
                             + std::to_string(d));

  arma::colvec pred_mean(m);
  arma::colvec pred_stdev = arma::zeros(m);
  arma::mat pred_cov = arma::zeros(m, m);
  arma::mat pred_mean_deriv = arma::zeros(m, d);
  arma::mat pred_stdev_deriv = arma::zeros(m, d);

  arma::mat Xtnorm = trans(m_X);  // already normalized if needed
  arma::mat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  /** @ref: https://github.com/cran/DiceKriging/blob/master/R/kmStruct.R#L191 */
  // beta <- object@trend.coef

  // Define regression matrix
  // F.newdata <- model.matrix(object@trend.formula, data = data.frame(newdata))
  arma::mat F_p = Trend::regressionModelMatrix(m_regmodel, Xpnorm);
  Xpnorm = trans(Xpnorm);

  // Compute covariance between training data and new data to predict
  // c.newdata <- covMat1Mat2(object@covariance, X1 = X, X2 = newdata,
  //                          nugget.flag = object@covariance@nugget.flag)
  // ## compute c(x) for x = newdata ; remark that for prediction (or filtering), cov(Yi, Yj)=0
  // ## even if Yi and Yj are the outputs related to the equal points xi and xj.
  arma::mat R_pred = arma::mat(n, m);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      R_pred.at(i, j) = Cov((Xtnorm.col(i) - Xpnorm.col(j)), m_theta);
    }
  }

  // Tinv.c.newdata <- backsolve(t(T), c.newdata, upper.tri=FALSE)
  arma::mat Tinv_pred = solve(m_T, R_pred, arma::solve_opts::fast);
  // y.predict.trend <- F.newdata%*%beta
  // y.predict.complement <- t(Tinv.c.newdata)%*%z
  // y.predict <- y.predict.trend + y.predict.complement
  pred_mean = F_p * m_beta + trans(Tinv_pred) * m_z;
  // Un-normalize predictor
  pred_mean = m_centerY + m_scaleY * pred_mean;

  arma::mat s2_predict_mat;
  arma::mat FinvMtM;
  double total_sd2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n - d) / (n - d - 2) : 1.0);
  if (withStd || withCov) {  // Will use chol(t(M)%*%M) in all these cases
    // Type = "UK"
    // T.M <- chol(t(M)%*%M)
    arma::mat TM = trans(chol(trans(m_M) * m_M));  // same that arma::qr_econ(Q, TM, m_M);
    // s2.predict.mat <- backsolve(t(T.M), t(F.newdata - t(Tinv.c.newdata)%*%M) , upper.tri = FALSE)
    s2_predict_mat = solve(TM, trans(F_p - trans(Tinv_pred) * m_M), arma::solve_opts::fast);

    if (withDeriv) {
      arma::mat m = trans(F_p - trans(Tinv_pred) * m_M);
      arma::mat invMtM = inv_sympd(m_M.t() * m_M);
      FinvMtM = (F_p - trans(Tinv_pred) * m_M) * inv_sympd(m_M.t() * m_M);
    }
  }

  if (withStd) {
    // s2.predict.1 <- apply(Tinv.c.newdata, 2, crossprod)
    arma::colvec s2_predict_1 = trans(sum(Tinv_pred % Tinv_pred, 0));
    s2_predict_1.transform([](double val) {
      return (val > 1.0 ? 1.0 : val);
    });  // constrain this first part to not be negative (rationale: it is the whole stdev for simple kriging)

    // s2.predict.2 <- apply(s2.predict.mat, 2, crossprod)
    arma::colvec s2_predict_2 = trans(sum(s2_predict_mat % s2_predict_mat, 0));
    // s2.predict <- pmax(total.sd2 - s2.predict.1 + s2.predict.2, 0)

    arma::mat s2_predict = total_sd2 * (1.0 - s2_predict_1 + s2_predict_2);
    s2_predict.transform([](double val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
    pred_stdev = sqrt(s2_predict);

    pred_stdev *= m_scaleY;
  }

  if (withCov) {
    arma::mat R_predpred = arma::mat(m, m);
    for (arma::uword i = 0; i < m; i++) {
      R_predpred.at(i, i) = 1;
      for (arma::uword j = 0; j < i; j++) {
        R_predpred.at(i, j) = R_predpred.at(j, i) = Cov((Xpnorm.col(i) - Xpnorm.col(j)), m_theta);
      }
    }

    pred_cov = total_sd2 * (R_predpred - trans(Tinv_pred) * Tinv_pred + trans(s2_predict_mat) * s2_predict_mat);

    pred_cov *= m_scaleY;
  }

  if (withDeriv) {
    /** @ref: https://github.com/cran/DiceOptim/blob/master/R/EI.grad.R#L156 */
    // # Compute derivatives of the covariance and trend functions
    for (arma::uword i = 0; i < m; i++) {  // for each Xp predict point... should be parallel ?

      // dc <- covVector.dx(x=newdata.num, X=X, object=covStruct, c=c)
      arma::mat dc = arma::mat(n, d);
      for (arma::uword j = 0; j < n; j++) {
        dc.row(j) = R_pred.at(j, i) * trans(DlnCovDx(Xpnorm.col(i) - Xtnorm.col(j), m_theta));
      }

      // f.deltax <- trend.deltax(x=newdata.num, model=model)
      /** @ref: https://github.com/cran/DiceKriging/blob/master/R/trend.deltax.R#L69 */
      // // A <- matrix(x, nrow=d, ncol=d, byrow=TRUE)
      // // Apos <- A+h*diag(d)
      // // Aneg <- A-h*diag(d)
      // // newpoints <- data.frame(rbind(Apos, Aneg))
      // // f.newdata <- model.matrix(model@trend.formula, data = newpoints)
      // // f.deltax <- (f.newdata[1:d,]-f.newdata[(d+1):(2*d),])/(2*h)
      const double h = 1.0E-5;  // Value is sensitive only for non linear trends. Otherwise, it gives exact results.
      arma::mat tXpn_i_repd = arma::trans(Xpnorm.col(i) * arma::ones(1, d));  // just duplicate Xp.row(i) d times

      arma::mat F_dx = (Trend::regressionModelMatrix(m_regmodel, tXpn_i_repd + h * arma::eye(d, d))
                        - Trend::regressionModelMatrix(m_regmodel, tXpn_i_repd - h * arma::eye(d, d)))
                       / (2 * h);

      // # Compute gradients of the kriging mean and variance
      //  W <- backsolve(t(T), dc, upper.tri=FALSE)
      arma::mat W = solve(m_T, dc, LinearAlgebra::default_solve_opts);

      // kriging.mean.grad <- t(W)%*%z + t(model@trend.coef%*%f.deltax)
      pred_mean_deriv.row(i) = trans(F_dx * m_beta + trans(W) * m_z);

      //...
      // F.newdata <- model.matrix(model@trend.formula, data=newdata)
      // v <- predx$Tinv.c
      // c <- predx$c
      // ...
      //  if (type=="UK")
      //  { tuuinv <- solve(t(u)%*%u)
      //  kriging.sd2.grad <-  t( -2*t(v)%*%W +
      //                            2*(F.newdata - t(v)%*%u )%*% tuuinv %*%
      //                            (f.deltax - t(t(W)%*%u) ))
      //  kriging.sd.grad <- kriging.sd2.grad / (2*kriging.sd)
      if (withStd) {
        arma::mat pred_stdev_deriv_noTrend = Tinv_pred.t() * W;
        pred_stdev_deriv.row(i) = (-pred_stdev_deriv_noTrend.row(i) + FinvMtM.row(i) * (F_dx.t() - trans(m_M) * W))
                                  * total_sd2 / pred_stdev.at(i);
      }
    }
    pred_mean_deriv *= m_scaleY;
    pred_stdev_deriv *= m_scaleY;
  }

  return std::make_tuple(std::move(pred_mean),
                         std::move(pred_stdev),
                         std::move(pred_cov),
                         std::move(pred_mean_deriv),
                         std::move(pred_stdev_deriv));
  /*if (withStd)
    if (withCov)
      return std::make_tuple(std::move(pred_mean), std::move(pred_stdev), std::move(pred_cov));
    else
      return std::make_tuple(std::move(pred_mean), std::move(pred_stdev), nullptr);
  else if (withCov)
    return std::make_tuple(std::move(pred_mean), std::move(pred_cov), nullptr);
  else
    return std::make_tuple(std::move(pred_mean), nullptr, nullptr);*/
}

/** Draw sample trajectories of kriging at given points X'
 * @param Xp is m*d matrix of points where to simulate output
 * @param nsim is number of simulations to draw
 * @return output is m*nsim matrix of simulations at Xp
 */
LIBKRIGING_EXPORT arma::mat Kriging::simulate(const int nsim, const int seed, const arma::mat& Xp) {
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (Xp.n_cols != d)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(Xp.n_cols) + " instead of "
                             + std::to_string(d));

  arma::mat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_p = Trend::regressionModelMatrix(m_regmodel, Xpnorm);
  Xpnorm = trans(Xpnorm);
  // t0 = Bench::toc("Xpnorm         ", t0);

  // auto t0 = Bench::tic();
  arma::colvec y_trend = F_p * m_beta;  // / std::sqrt(m_sigma2);
  // t0 = Bench::toc("y_trend        ", t0);

  // Compute covariance between new data
  arma::mat Sigma = arma::mat(m, m);
  for (arma::uword i = 0; i < m; i++) {
    Sigma.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      Sigma.at(i, j) = Sigma.at(j, i) = Cov((Xpnorm.col(i) - Xpnorm.col(j)), m_theta);
    }
  }
  // t0 = Bench::toc("Sigma          ", t0);

  // arma::mat T_newdata = chol(Sigma);
  // Compute covariance between training data and new data to predict
  // Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)
  arma::mat Xtnorm = trans(m_X);  // already normalized if needed
  arma::mat Sigma21(n, m);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      Sigma21.at(i, j) = Cov((Xtnorm.col(i) - Xpnorm.col(j)), m_theta);
    }
  }
  // t0 = Bench::toc("Sigma21        ", t0);

  // Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE
  arma::mat Tinv_Sigma21 = solve(m_T, Sigma21, LinearAlgebra::default_solve_opts);
  // t0 = Bench::toc("Tinv_Sigma21   ", t0);

  // y.trend.cond <- y.trend + t(Tinv.Sigma21) %*% object@z
  y_trend += trans(Tinv_Sigma21) * m_z;
  // t0 = Bench::toc("y_trend        ", t0);

  // Sigma.cond <- Sigma11 - t(Tinv.Sigma21) %*% Tinv.Sigma21
  // arma::mat Sigma_cond = Sigma - XtX(Tinv_Sigma21);
  // arma::mat Sigma_cond = Sigma - trans(Tinv_Sigma21) * Tinv_Sigma21;
  arma::mat Sigma_cond = trimatl(Sigma);
  for (arma::uword i = 0; i < Tinv_Sigma21.n_cols; i++) {
    for (arma::uword j = 0; j <= i; j++) {
      Sigma_cond.at(i, j) -= cdot(Tinv_Sigma21.col(i), Tinv_Sigma21.col(j));
      Sigma_cond.at(j, i) = Sigma_cond.at(i, j);
    }
  }
  // t0 = Bench::toc("Sigma_cond     ", t0);

  // T.cond <- chol(Sigma.cond + diag(nugget.sim, m, m))
  arma::mat tT_cond = LinearAlgebra::safe_chol_lower(Sigma_cond);
  // t0 = Bench::toc("T_cond         ", t0);

  // white.noise <- matrix(rnorm(m*nsim), m, nsim)
  // y.rand.cond <- t(T.cond) %*% white.noise
  // y <- matrix(y.trend.cond, m, nsim) + y.rand.cond
  arma::mat yp(m, nsim);
  yp.each_col() = y_trend;

  Random::reset_seed(seed);
  yp += tT_cond * Random::randn_mat(m, nsim) * std::sqrt(m_sigma2)
        * (m_objective.compare("LMP") == 0 ? sqrt((n - d) / (n - d - 2)) : 1.0);
  // t0 = Bench::toc("yp             ", t0);

  // Un-normalize simulations
  yp = m_centerY + m_scaleY * yp;  // * std::sqrt(m_sigma2);
  // t0 = Bench::toc("yp             ", t0);

  return yp;
}

/** Add new conditional data points to previous (X,y)
 * @param newy is m length column vector of new output
 * @param newX is m*d matrix of new input
 * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
 * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
 */
LIBKRIGING_EXPORT void Kriging::update(const arma::vec& newy, const arma::mat& newX) {
  if (newy.n_elem != newX.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(newX.n_rows) + "x"
                             + std::to_string(newX.n_cols) + "), y: (" + std::to_string(newy.n_elem) + ")");

  // rebuild starting parameters
  Parameters parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY),
                        this->m_est_sigma2,
                        std::make_optional(trans(this->m_theta) % this->m_scaleX),
                        this->m_est_theta,
                        std::make_optional(trans(this->m_beta) * this->m_scaleY),
                        this->m_est_beta};
  // re-fit
  // TODO refit() method which will use Shurr forms to fast update matrix (R, ...)
  this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                            newy),  // de-normalize previous data according to suite unnormed new data
            arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, newX),
            m_regmodel,
            m_normalize,
            m_optim,
            m_objective,
            parameters);
}

LIBKRIGING_EXPORT std::string Kriging::summary() const {
  std::ostringstream oss;
  auto colvec_printer = [&oss](const arma::colvec& v) {
    v.for_each([&oss, i = 0](const arma::colvec::elem_type& val) mutable {
      if (i++ > 0)
        oss << ", ";
      oss << val;
    });
  };

  if (m_X.is_empty() || m_X.n_rows == 0) {  // not yet fitted
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
  } else {
    oss << "* data";
    oss << ((m_normalize) ? " (normalized): " : ": ") << m_X.n_rows << "x";
    arma::rowvec Xmins = arma::min(m_X, 0);
    arma::rowvec Xmaxs = arma::max(m_X, 0);
    for (arma::uword i = 0; i < m_X.n_cols; i++) {
      oss << "[" << Xmins[i] << "," << Xmaxs[i] << "]";
      if (i < m_X.n_cols - 1)
        oss << ",";
    }
    oss << " -> " << m_y.n_elem << "x[" << arma::min(m_y) << "," << arma::max(m_y) << "]\n";
    oss << "* trend " << Trend::toString(m_regmodel);
    oss << ((m_est_beta) ? " (est.): " : ": ");
    colvec_printer(m_beta);
    oss << "\n";
    oss << "* variance";
    oss << ((m_est_sigma2) ? " (est.): " : ": ");
    oss << m_sigma2;
    oss << "\n";
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
    oss << "  * range";
    oss << ((m_est_theta) ? " (est.): " : ": ");
    colvec_printer(m_theta);
    oss << "\n";
    oss << "  * fit:\n";
    oss << "    * objective: " << m_objective << "\n";
    oss << "    * optim: " << m_optim << "\n";
  }
  return oss.str();
}

void Kriging::save(const std::string filename) const {
  const auto appflag = arma::hdf5_opts::append;

  uint32_t version = 1;
  saveToHdf5(version, arma::hdf5_name(filename, "version", arma::hdf5_opts::none));
  saveToHdf5(std::string("Kriging"), arma::hdf5_name(filename, "content", appflag));

  // Cov_pow & std::function embedded by make_Cov
  saveToHdf5(m_covType, arma::hdf5_name(filename, "covType", appflag));
  m_X.save(arma::hdf5_name(filename, "X", appflag));
  m_centerX.save(arma::hdf5_name(filename, "centerX", appflag));
  m_scaleX.save(arma::hdf5_name(filename, "scaleX", appflag));
  m_y.save(arma::hdf5_name(filename, "y", appflag));
  saveToHdf5(m_centerY, arma::hdf5_name(filename, "centerY", appflag));
  saveToHdf5(m_scaleY, arma::hdf5_name(filename, "scaleY", appflag));
  saveToHdf5(m_normalize, arma::hdf5_name(filename, "normalize", appflag));

  saveToHdf5(Trend::toString(m_regmodel), arma::hdf5_name(filename, "regmodel", appflag));
  saveToHdf5(m_optim, arma::hdf5_name(filename, "optim", appflag));
  saveToHdf5(m_objective, arma::hdf5_name(filename, "objective", appflag));
  m_dX.save(arma::hdf5_name(filename, "dX", appflag));
  m_F.save(arma::hdf5_name(filename, "F", appflag));
  m_T.save(arma::hdf5_name(filename, "T", appflag));
  m_M.save(arma::hdf5_name(filename, "M", appflag));
  m_z.save(arma::hdf5_name(filename, "z", appflag));
  m_beta.save(arma::hdf5_name(filename, "beta", appflag));
  saveToHdf5(m_est_beta, arma::hdf5_name(filename, "est_beta", appflag));
  m_theta.save(arma::hdf5_name(filename, "theta", appflag));
  saveToHdf5(m_est_theta, arma::hdf5_name(filename, "est_theta", appflag));
  saveToHdf5(m_sigma2, arma::hdf5_name(filename, "sigma2", appflag));
  saveToHdf5(m_est_sigma2, arma::hdf5_name(filename, "est_sigma2", appflag));
}

Kriging Kriging::load(const std::string filename) {
  uint32_t version;
  loadFromHdf5(version, arma::hdf5_name(filename, "version"));
  if (version != 1) {
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 1"));
  }
  std::string content;
  loadFromHdf5(content, arma::hdf5_name(filename, "content"));
  if (content != "Kriging") {
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'Kriging'"));
  }

  std::string covType;
  loadFromHdf5(covType, arma::hdf5_name(filename, "covType"));
  Kriging kr(covType);  // Cov_pow & std::function embedded by make_Cov

  kr.m_X.load(arma::hdf5_name(filename, "X"));
  kr.m_centerX.load(arma::hdf5_name(filename, "centerX"));
  kr.m_scaleX.load(arma::hdf5_name(filename, "scaleX"));
  kr.m_y.load(arma::hdf5_name(filename, "y"));
  loadFromHdf5(kr.m_centerY, arma::hdf5_name(filename, "centerY"));
  loadFromHdf5(kr.m_scaleY, arma::hdf5_name(filename, "scaleY"));
  loadFromHdf5(kr.m_normalize, arma::hdf5_name(filename, "normalize"));

  std::string model;
  loadFromHdf5(model, arma::hdf5_name(filename, "regmodel"));
  kr.m_regmodel = Trend::fromString(model);

  loadFromHdf5(kr.m_optim, arma::hdf5_name(filename, "optim"));
  loadFromHdf5(kr.m_objective, arma::hdf5_name(filename, "objective"));
  kr.m_dX.load(arma::hdf5_name(filename, "dX"));
  kr.m_F.load(arma::hdf5_name(filename, "F"));
  kr.m_T.load(arma::hdf5_name(filename, "T"));
  kr.m_M.load(arma::hdf5_name(filename, "M"));
  kr.m_z.load(arma::hdf5_name(filename, "z"));
  kr.m_beta.load(arma::hdf5_name(filename, "beta"));
  loadFromHdf5(kr.m_est_beta, arma::hdf5_name(filename, "est_beta"));
  kr.m_theta.load(arma::hdf5_name(filename, "theta"));
  loadFromHdf5(kr.m_est_theta, arma::hdf5_name(filename, "est_theta"));
  loadFromHdf5(kr.m_sigma2, arma::hdf5_name(filename, "sigma2"));
  loadFromHdf5(kr.m_est_sigma2, arma::hdf5_name(filename, "est_sigma2"));

  return kr;
}
