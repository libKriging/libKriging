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
#include "libKriging/utils/jsonutils.hpp"
#include "libKriging/utils/nlohmann/json.hpp"
#include "libKriging/utils/utils.hpp"

#include <cassert>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <map>
#include <tuple>
#include <vector>

/************************************************/
/**      Kriging implementation        **/
/************************************************/

// This will create the dist(xi,xj) function above. Need to parse "covType".
void Kriging::make_Cov(const std::string& covType) {
  m_covType = covType;
  if (covType.compare("gauss") == 0) {
    _Cov = Covariance::Cov_gauss;
    _DlnCovDtheta = Covariance::DlnCovDtheta_gauss;
    _DlnCovDx = Covariance::DlnCovDx_gauss;
    _Cov_pow = 2;
  } else if (covType.compare("exp") == 0) {
    _Cov = Covariance::Cov_exp;
    _DlnCovDtheta = Covariance::DlnCovDtheta_exp;
    _DlnCovDx = Covariance::DlnCovDx_exp;
    _Cov_pow = 1;
  } else if (covType.compare("matern3_2") == 0) {
    _Cov = Covariance::Cov_matern32;
    _DlnCovDtheta = Covariance::DlnCovDtheta_matern32;
    _DlnCovDx = Covariance::DlnCovDx_matern32;
    _Cov_pow = 1.5;
  } else if (covType.compare("matern5_2") == 0) {
    _Cov = Covariance::Cov_matern52;
    _DlnCovDtheta = Covariance::DlnCovDtheta_matern52;
    _DlnCovDx = Covariance::DlnCovDx_matern52;
    _Cov_pow = 2.5;
  } else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);

  // arma::cout << "make_Cov done." << arma::endl;
}

LIBKRIGING_EXPORT arma::mat Kriging::covMat(const arma::mat& X1, const arma::mat& X2) {
  arma::mat Xn1 = X1;
  arma::mat Xn2 = X2;
  Xn1.each_row() -= m_centerX;
  Xn1.each_row() /= m_scaleX;
  Xn2.each_row() -= m_centerX;
  Xn2.each_row() /= m_scaleX;

  arma::mat R = arma::mat(X1.n_rows, X2.n_rows, arma::fill::none);
  for (arma::uword i = 0; i < Xn1.n_rows; i++) {
    for (arma::uword j = 0; j < Xn2.n_rows; j++) {
      R.at(i, j) = _Cov((Xn1.row(i)- Xn2.row(j)).t(), m_theta);
    }
  }
  return R * m_sigma2;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT Kriging::Kriging(const std::string& covType) {
  make_Cov(covType);
}

LIBKRIGING_EXPORT Kriging::Kriging(const arma::vec& y,
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

arma::vec Kriging::ones = arma::ones<arma::vec>(0);

Kriging::KModel Kriging::make_Model(const arma::vec& theta,
                               std::map<std::string, double>* bench) const {
    arma::mat R;
    arma::mat L;
    arma::mat Linv;
    arma::mat Fstar;
    arma::vec ystar;
    arma::mat Rstar;
    arma::mat Qstar;
    arma::vec Estar;
    double SSEstar{};
    arma::vec betahat;
    Kriging::KModel m { R, L, Linv, Fstar, ystar, Rstar,Qstar, Estar, SSEstar , betahat };
    
  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::uword p = m_F.n_cols;

  auto t0 = Bench::tic();
  m.R = arma::mat(n,n, arma::fill::none);
  // check if we want to recompute model for same theta, for augmented Xy (using cholesky fast update).
  bool update = (m_theta.size() == theta.size()) && (theta - m_theta).is_zero() && (this->m_T.memptr() != nullptr) && (n > this->m_T.n_rows);
  if (update) { 
    m.L = LinearAlgebra::update_cholCov(&(m.R), m_dX, theta, _Cov, 1, Kriging::ones, m_T, m_R);
  } else 
    m.L = LinearAlgebra::cholCov(&(m.R), m_dX, theta, _Cov, 1, Kriging::ones);
  t0 = Bench::toc(bench, "R = _Cov(dX) & L = Chol(R)", t0);

  // Compute intermediate useful matrices
  arma::mat Fystar = LinearAlgebra::solve(m.L, arma::join_rows(m_F, m_y));
  t0 = Bench::toc(bench, "Fy* = L \\ [F,y]", t0);
  m.Fstar = Fystar.head_cols(p);
  m.ystar = Fystar.tail_cols(1);

  arma::mat Q_qr;
  arma::mat R_qr;
  arma::qr_econ(Q_qr, R_qr, Fystar);
  t0 = Bench::toc(bench, "Q_qr,R_qr = QR(Fy*)", t0);

  m.Rstar = R_qr.head_cols(p);
  m.Qstar = Q_qr.head_cols(p);
  m.Estar = Q_qr.tail_cols(1) * R_qr.at(p,p);
  m.SSEstar = R_qr.at(p, p) * R_qr.at(p, p);

  if (m_est_beta) {
    m.betahat = LinearAlgebra::solve(m.Rstar, R_qr.tail_cols(1));
    t0 = Bench::toc(bench, "^b = R* \\ R_qr[1:p, p+1]", t0);
  } else {
    m.betahat = arma::vec(p, arma::fill::zeros); // whatever: not used
  }

  return m;
}

// Objective function for fit : -logLikelihood

double Kriging::_logLikelihood(const arma::vec& _theta,
                               arma::vec* grad_out,
                               arma::mat* hess_out,
                               Kriging::KModel* model,
                               std::map<std::string, double>* bench) const {
  //arma::cout << " theta: " << _theta << arma::endl;

  Kriging::KModel m = make_Model(_theta, bench);
  if (model != nullptr)
    *model = m;

  arma::uword n = m_X.n_rows;

  double sigma2;
  double ll;
  if (m_est_sigma2) { // DiceKriging: model@case == "LLconcentration_beta_sigma2"
    sigma2 = m.SSEstar / n;
    ll = -0.5 * (n * log(2 * M_PI * sigma2) + 2 * arma::sum(log(m.L.diag())) + n);
  } else { // DiceKriging: model@case == "LLconcentration_beta"
    sigma2 = m_sigma2;
    ll = -0.5 * (n * log(2 * M_PI * sigma2) + 2 * arma::sum(log(m.L.diag())) + as_scalar(LinearAlgebra::crossprod(m.Estar))/sigma2);
  }

  if (grad_out != nullptr) {
    arma::uword d = m_X.n_cols;
    arma::uword p = m_F.n_cols;

    auto t0 = Bench::tic();
    arma::vec terme1 = arma::vec(d);   // useful if (hess_out != nullptr)

    if ((m.Linv.memptr()==nullptr) || (arma::size(m.Linv) != arma::size(m.L))) {
      m.Linv =LinearAlgebra::solve(m.L, arma::mat(n, n,arma::fill::eye));
      t0 = Bench::toc(bench, "L ^-1", t0);
    }

    arma::mat Rinv = LinearAlgebra::crossprod(m.Linv);
    t0 = Bench::toc(bench, "R^-1 = t(L^-1) * L^-1", t0);

    arma::mat x = LinearAlgebra::solve(m.L.t(), m.Estar);
    t0 = Bench::toc(bench, "x = tL \\ z", t0);

    arma::cube gradR = arma::cube(n, n, d, arma::fill::none);
    const arma::vec zeros = arma::vec(d,arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++) {
      gradR.tube(i, i) = zeros;
      for (arma::uword j = 0; j < i; j++) {
        gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
        gradR.tube(j, i) = gradR.tube(i, j); 
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < d; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = gradR.slice(k);
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      // should make a fast function trace_prod(A,B) -> sum_i(sum_j(Ai,j*Bj,i))
      terme1.at(k) = as_scalar(x.t() * gradR_k * x) / sigma2;
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
        arma::mat H = LinearAlgebra::tcrossprod(m.Qstar);
        t0 = Bench::toc(bench, "H =  Q* * t(Q*)", t0);

        for (arma::uword l = 0; l <= k; l++) {
          arma::mat gradR_l = gradR.slice(l); //arma::mat(n, n);

          t0 = Bench::tic();
          arma::mat aux = gradR_k * Rinv * gradR_l;
          t0 = Bench::toc(bench, "aux =  gradR[k] * Ri * gradR[l]", t0);

          arma::mat hessR_k_l = arma::mat(n, n, arma::fill::none);
          if (k == l) {
            for (arma::uword i = 0; i < n; i++) {
              hessR_k_l.at(i, i) = 0;
              for (arma::uword j = 0; j < i; j++) {
                double dln_k = gradR_k.at(i,j);
                hessR_k_l.at(i, j) = hessR_k_l.at(j, i) = dln_k * (dln_k / m.R.at(i, j) - (_Cov_pow + 1) / _theta.at(k));
                // !! NO: it just work for exp type kernels. Matern MUST have a special treatment !!!
              }
            }
          } else {
            for (arma::uword i = 0; i < n; i++) {
              hessR_k_l.at(i, i) = 0;
              for (arma::uword j = 0; j < i; j++) {
                hessR_k_l.at(i, j) = hessR_k_l.at(j, i) = gradR_k.at(i,j) * gradR_l.at(i,j) / m.R.at(i, j);
                    //= gradR.slice(i).col(j)[k] * gradR.slice(i).col(j)[l] / m.R.at(i, j);
              }
            }
          }
          t0 = Bench::toc(bench, "hessR_k_l = ...", t0);

          //arma::mat xk =LinearAlgebra::solve(m.T, gradsR[k] * x);
          arma::mat xk = m.Linv * gradR_k * x;
          t0 = Bench::toc(bench, "xk = L \\ gradR[k] * x", t0);
          arma::mat xl;
          if (k == l)
            xl = xk;
          else
            xl  = m.Linv * gradR_l * x;
          t0 = Bench::toc(bench, "xl = L \\ gradR[l] * x", t0);

          // arma::cout << " hess_A:" << -xk.t() * H * xl / sigma2 << arma::endl;
          // arma::cout << " hess_B:" << -x.t() * (hessR_k_l - 2*aux) * x / sigma2 << arma::endl;
          // arma::cout << " hess_C:" << -terme1.at(k) * terme1.at(l) / n << arma::endl;
          // arma::cout << " hess_D:" << -arma::trace(Rinv * aux)  << arma::endl;
          // arma::cout << " hess_E:" << arma::trace(Rinv * hessR_k_l) << arma::endl;
          
          double h_lk  = 
            (2.0 * xk.t() * H * xl / (sigma2) + 
            x.t() * (hessR_k_l - 2 * aux) * x / (sigma2) +
            arma::trace(Rinv * aux) +
            - arma::trace(Rinv * hessR_k_l))[0];  // should optim there using accu & %
          if (m_est_sigma2)
            h_lk += terme1.at(k) * terme1.at(l) / n;

          (*hess_out).at(l, k) = (*hess_out).at(k, l) = h_lk / 2;
          t0 = Bench::toc(bench, "hess_ll[l,k] = ...", t0);
        }
      }
    }
  }
  return ll;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec, arma::mat> Kriging::logLikelihoodFun(const arma::vec& _theta,
                                                                                     const bool _grad,
                                                                                     const bool _hess,
                                                                                     const bool _bench) {
  double ll = -1;
  arma::vec grad;
  arma::mat hess;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad || _hess) {
      grad = arma::vec(_theta.n_elem);
      if (!_hess) {
        ll = _logLikelihood(_theta, &grad, nullptr, nullptr, &bench);
      } else {
        hess = arma::mat(_theta.n_elem, _theta.n_elem, arma::fill::none);
        ll = _logLikelihood(_theta, &grad, &hess, nullptr, &bench);
      }
    } else
      ll = _logLikelihood(_theta, nullptr, nullptr, nullptr, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad || _hess) {
      grad = arma::vec(_theta.n_elem);
      if (!_hess) {
        ll = _logLikelihood(_theta, &grad, nullptr, nullptr, nullptr);
      } else {
        hess = arma::mat(_theta.n_elem, _theta.n_elem, arma::fill::none);
        ll = _logLikelihood(_theta, &grad, &hess, nullptr, nullptr);
      }
    } else
      ll = _logLikelihood(_theta, nullptr, nullptr, nullptr, nullptr);
  }

  return std::make_tuple(ll, std::move(grad), std::move(hess));
}

// Objective function for fit : -LOO

double Kriging::_leaveOneOut(const arma::vec& _theta,
                             arma::vec* grad_out,
                             arma::mat* yhat_out,
                             Kriging::KModel* model,
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
  //  ## Remark:   Q <- Cinv - Cinv.F %*% solve(t(M)%*%M) %*% t(Cinv.F)
  //...
  // sigma2LOO <- 1/diag(Q)
  // errorsLOO <- sigma2LOO * (Q.y)       # cost : n, neglected
  //
  // LOOfun <- as.numeric(crossprod(errorsLOO)/model@n)

  //arma::cout << " theta: " << _theta << arma::endl;
  Kriging::KModel m = make_Model(_theta, bench);
  if (model != nullptr)
    *model = m;

  arma::uword n = m_X.n_rows;

  auto t0 = Bench::tic();
  if ((m.Linv.memptr()==nullptr) || (arma::size(m.Linv) != arma::size(m.L))) {
    m.Linv =LinearAlgebra::solve(m.L, arma::mat(n, n, arma::fill::eye));
    t0 = Bench::toc(bench, "L ^-1", t0);
  }
  arma::mat By = m.Linv.t() * m.Estar;
  t0 = Bench::toc(bench, "By = L^-1 * E*", t0);
  arma::mat A = m.Qstar.t() * m.Linv;
  t0 = Bench::toc(bench, "A = Q* * L^-1", t0);
  arma::mat B = LinearAlgebra::crossprod(m.Linv) - LinearAlgebra::crossprod(A);
  t0 = Bench::toc(bench, "B = t(L^-1) * L^-1 - t(A) * A", t0);

  arma::vec sigma2LOO = 1 / B.diag();
  t0 = Bench::toc(bench, "S2l = 1 / diag(Q)", t0);

  arma::vec errorsLOO = sigma2LOO % By;
  t0 = Bench::toc(bench, "E = S2l * Qy", t0);

  double loo = arma::accu(errorsLOO % errorsLOO) / n;
  t0 = Bench::toc(bench, "loo = Acc(E * E) / n", t0);

  if (yhat_out != nullptr) {
    (*yhat_out).col(0) = m_y - errorsLOO;
    (*yhat_out).col(1) = arma::sqrt(sigma2LOO);
  }

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

    arma::uword d = m_X.n_cols;

    t0 = Bench::tic();
    arma::cube gradR = arma::cube(n, n, d, arma::fill::none);
    const arma::vec zeros = arma::vec(d, arma::fill::zeros);
    for (arma::uword i = 0; i < n; i++) {
      gradR.tube(i, i) = zeros;
      for (arma::uword j = 0; j < i; j++) {
        gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
        gradR.tube(j, i) = gradR.tube(i, j); 
      }
    }
    t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);

    for (arma::uword k = 0; k < m_X.n_cols; k++) {
      t0 = Bench::tic();
      arma::mat gradR_k = gradR.slice(k);
      t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);

      arma::vec diagdB = - LinearAlgebra::diagABA(B, gradR_k);
      t0 = Bench::toc(bench, "diagdQ = DiagABA(B, gradR_k)", t0);

      arma::vec dsigma2LOO = -sigma2LOO % sigma2LOO % diagdB;
      t0 = Bench::toc(bench, "dS2l = -S2l % S2l % diagdQ", t0);

      arma::vec derrorsLOO = dsigma2LOO % By - sigma2LOO % (B * (gradR_k * By));
      t0 = Bench::toc(bench, "dE = dS2l * By- S2l * (B * gradR_k * By)", t0);

      (*grad_out)(k) = 2 * dot(errorsLOO, derrorsLOO) / n;
      t0 = Bench::toc(bench, "grad_loo[k] = E * dE / n", t0);
    }
  }
  return loo;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::leaveOneOutFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  double loo = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      loo = _leaveOneOut(_theta, &grad, nullptr, nullptr, &bench);
    } else
      loo = _leaveOneOut(_theta, nullptr, nullptr, nullptr, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      loo = _leaveOneOut(_theta, &grad, nullptr, nullptr, nullptr);
    } else
      loo = _leaveOneOut(_theta, nullptr, nullptr, nullptr, nullptr);
  }

  return std::make_tuple(loo, std::move(grad));
}

LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec> Kriging::leaveOneOutVec(const arma::vec& _theta) {
  double loo = -1;
  arma::mat yhat = arma::mat(m_y.n_elem, 2, arma::fill::none);
  loo = _leaveOneOut(_theta, nullptr, &yhat, nullptr, nullptr);

  return std::make_tuple(std::move(yhat.col(0)), std::move(yhat.col(1) * std::sqrt(m_sigma2)));
}

// Objective function for fit: bayesian-like approach fromm RobustGaSP

double Kriging::_logMargPost(const arma::vec& _theta,
                             arma::vec* grad_out,
                             Kriging::KModel* model,
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
  //  MatrixXd Rinv_X=L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(X)); //one forward
  //  and one backward to compute R.inv%*%X MatrixXd Xt_Rinv_X=X.transpose()*Rinv_X; //Xt%*%R.inv%*%X
  //
  //  LLT<MatrixXd> lltOfXRinvX(Xt_Rinv_X); // cholesky decomposition of Xt_Rinv_X called lltOfXRinvX
  //  MatrixXd LX = lltOfXRinvX.matrixL();  //  retrieve factor LX  in the decomposition
  //  MatrixXd Rinv_X_Xt_Rinv_X_inv_Xt_Rinv=
  //  Rinv_X*(LX.transpose().triangularView<Upper>().solve(LX.triangularView<Lower>().solve(Rinv_X.transpose())));
  //  //compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward and one backward solve MatrixXd yt_Rinv=
  //  (L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(output))).transpose(); MatrixXd S_2=
  //  (yt_Rinv*output-output.transpose()*Rinv_X_Xt_Rinv_X_inv_Xt_Rinv*output); double log_S_2=log(S_2(0,0)); return
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

  Kriging::KModel m = make_Model(_theta, bench);
  if (model != nullptr)
    *model = m;

  arma::uword n = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::uword p = m_F.n_cols;

  // RobustGaSP naming...
  //arma::mat X = m_F;
  //arma::mat L = fd->T;

  auto t0 = Bench::tic();
  //m.Fstar : fd->M = solve(L, X, LinearAlgebra::default_solve_opts);

  //arma::mat Rinv_X = solve(trans(L), fd->M, LinearAlgebra::default_solve_opts);
  arma::mat Rinv_X = LinearAlgebra::solve(m.L.t(), m.Fstar);

  //arma::mat Xt_Rinv_X = trans(X) * Rinv_X;  // Xt%*%R.inv%*%X
  arma::mat Xt_Rinv_X = m_F.t() * Rinv_X;  

  //arma::mat LX = chol(Xt_Rinv_X, "lower");  //  retrieve factor LX  in the decomposition
  arma::mat LX = LinearAlgebra::safe_chol_lower(Xt_Rinv_X);

  //arma::mat Rinv_X_Xt_Rinv_X_inv_Xt_Rinv
  //    = Rinv_X
  //      * (solve(trans(LX),
  //               solve(LX, trans(Rinv_X), LinearAlgebra::default_solve_opts),
  //               LinearAlgebra::default_solve_opts));  // compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward
  arma::mat Rinv_X_Xt_Rinv_X_inv_Xt_Rinv =
  Rinv_X * (LinearAlgebra::solve(trans(LX),
                 LinearAlgebra::solve(LX, trans(Rinv_X))));  // compute  Rinv_X_Xt_Rinv_X_inv_Xt_Rinv through one forward

  arma::mat yt_Rinv = trans(solve(trans(m.L), m.ystar));
  t0 = Bench::toc(bench, "YtRi = Yt \\ Tt", t0);

  arma::mat S_2 = (yt_Rinv * m_y - trans(m_y) * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv * m_y);
  t0 = Bench::toc(bench, "S2 = YtRi * y - yt * RiFFtRiFiFtRi * y", t0);

  double sigma2;
  if (m_est_sigma2) {
    sigma2 = S_2(0, 0) / (n - p);
  } else {
    sigma2 = m_sigma2;
  }
  double log_S_2 = log(sigma2 * (n-p));

  double log_marginal_lik = -sum(log(m.L.diag())) - sum(log(LX.diag())) - (n - p) / 2.0 * log_S_2;
  t0 = Bench::toc(bench, "lml = -Sum(log(diag(T))) - Sum(log(diag(TF)))...", t0);
  //arma::cout << " log_marginal_lik:" << log_marginal_lik << arma::endl;

  // Default prior params
  double a = 0.2;
  double b = 1.0 / pow(n, 1.0 / d) * (a + d);
  // t0 = Bench::toc(bench,"b             ", t0);

  arma::vec CL = trans(max(m_X, 0) - min(m_X, 0)) / pow(n, 1.0 / d);
  t0 = Bench::toc(bench, "CL = (max(X) - min(X)) / n^1/d", t0);

  double t = arma::accu(CL / _theta);
  // arma::cout << " a:" << a << arma::endl;
  // arma::cout << " b:" << b << arma::endl;
  // arma::cout << " t:" << t << arma::endl;

  double log_approx_ref_prior = -b * t + a * log(t);
  //arma::cout << " log_approx_ref_prior:" << log_approx_ref_prior << arma::endl;

  if (grad_out != nullptr) {
    // Eigen::VectorXd log_marginal_lik_deriv(const Eigen::VectorXd param,double nugget,  bool nugget_est, const List
    // R0, const Eigen::Map<Eigen::MatrixXd> & X,const String zero_mean,const Eigen::Map<Eigen::MatrixXd> & output,
    // Eigen::VectorXi kernel_type,const Eigen::VectorXd alpha){
    // ...
    // VectorXd ans=VectorXd::Ones(param_size);
    // ...
    // MatrixXd Q_output= yt_Rinv.transpose()-Rinv_X_Xt_Rinv_X_inv_Xt_Rinv*output;
    // MatrixXd dev_R_i;
    // MatrixXd Wb_ti;
    // //allow different choices of kernels
    //
    // for(int ti=0;ti<p;ti++){
    //   //kernel_type_ti=kernel_type[ti];
    //   if(kernel_type[ti]==3){
    //     dev_R_i=matern_5_2_deriv( R0[ti],R_ori,beta[ti]);  //now here I have R_ori instead of R
    //   }else {...}
    //   Wb_ti=(L.transpose().triangularView<Upper>().solve(L.triangularView<Lower>().solve(dev_R_i))).transpose()-dev_R_i*Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
    //   ans[ti]=-0.5*Wb_ti.diagonal().sum()+(num_obs-q)/2.0*(output.transpose()*Wb_ti.transpose()*Q_output/S_2(0,0))(0,0);
    // }

    if (m_est_sigma2) {
      t0 = Bench::tic();
      arma::vec ans = arma::vec(d,arma::fill::none);
      arma::mat Q_output = trans(yt_Rinv) - Rinv_X_Xt_Rinv_X_inv_Xt_Rinv * m_y;
      t0 = Bench::toc(bench, "Qo = YtRi - RiFFtRiFiFtRi * y", t0);
  
      arma::cube gradR = arma::cube(n, n, d, arma::fill::zeros);
      //const arma::vec zeros = arma::vec(d,arma::fill::zeros);
      for (arma::uword i = 0; i < n; i++) {
        //gradR.tube(i, i) = zeros;
        for (arma::uword j = 0; j < i; j++) {
          gradR.tube(i, j) = m.R.at(i, j) * _DlnCovDtheta(m_dX.col(i * n + j), _theta);
          gradR.tube(j, i) = gradR.tube(i, j); 
        }
      }
      t0 = Bench::toc(bench, "gradR = R * dlnCov(dX)", t0);
  
      arma::mat Wb_k;
      for (arma::uword k = 0; k < d; k++) {
        t0 = Bench::tic();
        arma::mat gradR_k = gradR.slice(k);
        t0 = Bench::toc(bench, "gradR_k = gradR[k]", t0);
  
        Wb_k = trans( LinearAlgebra::solve(
                   trans(m.L),LinearAlgebra::solve(m.L, gradR_k)))
               - gradR_k * Rinv_X_Xt_Rinv_X_inv_Xt_Rinv;
        t0 = Bench::toc(bench, "Wb_k = gradR_k \\ L \\ Tt - gradR_k * RiFFtRiFiFtRi", t0);
  
        ans[k] = -sum(Wb_k.diag()) / 2.0 + as_scalar(trans(m_y) * trans(Wb_k) * Q_output) / (2.0 * sigma2);
        t0 = Bench::toc(bench, "ans[k] = Sum(diag(Wb_k)) + yt * Wb_kt * Qo / S2...", t0);
      }
      //arma::cout << " log_marginal_lik_deriv:" << ans << arma::endl;
      //arma::cout << " log_approx_ref_prior_deriv:" <<  - (a * CL / t - b * CL) / pow(_theta, 2.0) << arma::endl;
  
      *grad_out = ans - (a * CL / t - b * CL) / square(_theta);
      // t0 = Bench::toc(bench," grad_out     ", t0);
    } else { // TODO: we do not have (yet) formula when sigma2 is fixed... :(
      *grad_out = arma::vec(d, arma::fill::zeros);
      double _eps = 1e-6;
      for (arma::uword k = 0; k < d; k++) {
        arma::vec theta_eps = _theta;
        theta_eps[k] += _eps;
        (*grad_out)[k] = (_logMargPost(theta_eps, nullptr, nullptr, nullptr) - (log_marginal_lik + log_approx_ref_prior))/_eps;
      }
    }
    // arma::cout << " grad_out:" << *grad_out << arma::endl;
  }

  // arma::cout << " lmp:" << (log_marginal_lik+log_approx_ref_prior) << arma::endl;
  return (log_marginal_lik + log_approx_ref_prior);
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::logMargPostFun(const arma::vec& _theta,
                                                                        const bool _grad,
                                                                        const bool _bench) {
  double lmp = -1;
  arma::vec grad;

  if (_bench) {
    std::map<std::string, double> bench;
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      lmp = _logMargPost(_theta, &grad, nullptr, &bench);
    } else
      lmp = _logMargPost(_theta, nullptr, nullptr, &bench);

    size_t num = 0;
    for (auto& kv : bench)
      num = std::max(kv.first.size(), num);
    for (auto& kv : bench)
      arma::cout << "| " << Bench::pad(kv.first, num, ' ') << " | " << kv.second << " |" << arma::endl;

  } else {
    if (_grad) {
      grad = arma::vec(_theta.n_elem);
      lmp = _logMargPost(_theta, &grad, nullptr, nullptr);
    } else
      lmp = _logMargPost(_theta, nullptr, nullptr, nullptr);
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
      delta_x = delta * grad / std::sqrt(arma::sum(arma::square(grad)));
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
LIBKRIGING_EXPORT void Kriging::fit(const arma::vec& y,
                                    const arma::mat& X,
                                    const Trend::RegressionModel& regmodel,
                                    bool normalize,
                                    const std::string& optim,
                                    const std::string& objective,
                                    const Parameters& parameters) {
  const arma::uword n = X.n_rows;
  const arma::uword d = X.n_cols;

  std::function<double(const arma::vec& _gamma, arma::vec* grad_out, arma::mat* hess_out, Kriging::KModel* km_data)> fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    if (Optim::reparametrize) {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* hess_out, Kriging::KModel* km_data) {
            // Change variable for opt: . -> 1/exp(.)
            // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
            const arma::vec _theta = Optim::reparam_from(_gamma);
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double ll = this->_logLikelihood(_theta, grad_out, hess_out, km_data, nullptr);
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
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* hess_out, Kriging::KModel* km_data) {
            const arma::vec _theta = _gamma;
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double ll = this->_logLikelihood(_theta, grad_out, hess_out, km_data, nullptr);
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
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::KModel* km_data) {
            // Change variable for opt: . -> 1/exp(.)
            // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
            const arma::vec _theta = Optim::reparam_from(_gamma);
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double loo = this->_leaveOneOut(_theta, grad_out, nullptr, km_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > loo: " << loo << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad ll: " << grad_out << arma::endl;
              *grad_out = Optim::reparam_from_deriv(_theta, *grad_out);
            }
            return loo;
          });
    } else {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::KModel* km_data) {
            const arma::vec _theta = _gamma;
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double loo = this->_leaveOneOut(_theta, grad_out, nullptr, km_data, nullptr);
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
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::KModel* km_data) {
            // Change variable for opt: . -> 1/exp(.)
            // DEBUG: if (Optim::log_level>3) arma::cout << "> gamma: " << _gamma << arma::endl;
            const arma::vec _theta = Optim::reparam_from(_gamma);
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double lmp = this->_logMargPost(_theta, grad_out, km_data, nullptr);
            // DEBUG: if (Optim::log_level>3) arma::cout << "  > lmp: " << lmp << arma::endl;
            if (grad_out != nullptr) {
              // DEBUG: if (Optim::log_level>3) arma::cout << "  > grad lmp: " << grad_out << arma::endl;
              *grad_out = -Optim::reparam_from_deriv(_theta, *grad_out);
            }
            return -lmp;
          });
    } else {
      fit_ofn = CacheFunction(
          [this](const arma::vec& _gamma, arma::vec* grad_out, arma::mat* /*hess_out*/, Kriging::KModel* km_data) {
            const arma::vec _theta = _gamma;
            // DEBUG: if (Optim::log_level>3) arma::cout << "> theta: " << _theta << arma::endl;
            double lmp = this->_logMargPost(_theta, grad_out, km_data, nullptr);
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

  arma::rowvec centerX;
  arma::rowvec scaleX;
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
    centerX = arma::rowvec(d, arma::fill::zeros);
    scaleX = arma::rowvec(d, arma::fill::ones);
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
    arma::vec newy = (y - centerY) / scaleY;
    this->m_X = newX;
    this->m_y = newy;
  }

  // Now we compute the distance matrix between points. Will be used to compute R(theta) later (e.g. when fitting)
  // Note: m_dX is transposed compared to m_X
  m_dX = arma::mat(d, n * n, arma::fill::zeros);
  for (arma::uword ij = 0; ij < m_dX.n_cols; ij++) {
    int i = (int)ij / n;
    int j = ij % n;  // i,j <-> i*n+j
    if (i < j) {
      m_dX.col(ij) = trans(m_X.row(i) - m_X.row(j));
      m_dX.col(j * n + i) = m_dX.col(ij);
    }
  }
  m_maxdX = arma::max(arma::abs(m_dX), 1);

  // Define regression matrix
  m_regmodel = regmodel;
  m_F = Trend::regressionModelMatrix(regmodel, m_X);
  m_est_beta = (m_regmodel != Trend::RegressionModel::None);
  if ((parameters.beta.has_value()) && parameters.beta.value().n_elem>0) { // Then force beta to be fixed (not estimated, no variance)
    m_est_beta = false;
    m_beta = parameters.beta.value();
    if (m_normalize)
      m_beta /= scaleY;
  } else m_est_beta = true;

  arma::mat theta0;
  if (parameters.theta.has_value()) {
    theta0 = parameters.theta.value();
    if ((parameters.theta.value().n_cols != d) && (parameters.theta.value().n_rows == d))
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

    double sigma2 = -1;
    m_est_sigma2 = parameters.is_sigma2_estim;
    if (parameters.sigma2.has_value()) {
      sigma2 = parameters.sigma2.value();  // otherwise sigma2 will be re-calculated using given theta
      if (m_normalize)
        sigma2 /= (scaleY * scaleY);
    } else m_est_sigma2 = true;

    Kriging::KModel m = make_Model(m_theta, nullptr);

    m_T = std::move(m.L);
    m_R = std::move(m.R);
    m_M = std::move(m.Fstar);
    m_circ = std::move(m.Rstar);
    m_star = std::move(m.Qstar);
    if (m_est_beta) {
      m_beta = std::move(m.betahat);
      m_z = std::move(m.Estar);
    } else {
      // m_beta = parameters.beta.value(); already done above
      m_z = std::move(m.ystar) - m_M * m_beta;
    }
    if (m_est_sigma2) {
      m_sigma2 = m.SSEstar / n;
    } else {
      m_sigma2 = sigma2;
    }

  } else {
    arma::vec theta_lower = Optim::theta_lower_factor * m_maxdX;
    arma::vec theta_upper = Optim::theta_upper_factor * m_maxdX;

    if (Optim::variogram_bounds_heuristic) {
      arma::vec dy2 = arma::vec(n * n,arma::fill::zeros);
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
    }
    // arma::cout << "theta_lower:" << theta_lower << arma::endl;
    // arma::cout << "theta_upper:" << theta_upper << arma::endl;

    if (optim.rfind("BFGS", 0) == 0) {
      Random::init();

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

      if (parameters.theta.has_value()) {  // just use given theta(s) as starting values for multi-bfgs
        arma::mat theta0_tmp = arma::mat(parameters.theta.value());
        if (m_normalize)
          theta0_tmp.each_row() /= scaleX;
        theta0 = arma::join_cols(theta0, theta0_tmp);
      }
      // arma::cout << "theta0:" << theta0 << arma::endl;

      arma::vec gamma_lower = theta_lower;
      arma::vec gamma_upper = theta_upper;
      if (Optim::reparametrize) {
        gamma_lower = Optim::reparam_to(theta_upper);
        gamma_upper = Optim::reparam_to(theta_lower);
      }

      double min_ofn = std::numeric_limits<double>::infinity();

      for (arma::uword i = 0; i < multistart; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
        arma::vec gamma_tmp = theta0.row(i % multistart).t();
        if (Optim::reparametrize)
          gamma_tmp = Optim::reparam_to(theta0.row(i % multistart).t());

        gamma_lower = arma::min(gamma_tmp, gamma_lower);
        gamma_upper = arma::max(gamma_tmp, gamma_upper);

        if (Optim::log_level > 0) {
          arma::cout << "BFGS:" << arma::endl;
          arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
          arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
          arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower.t() << " ";
          arma::cout << "  upper_bounds: " << theta_upper.t() << " ";
          arma::cout << "  start_point: " << theta0.row(i % multistart) << " ";
        }

        m_est_sigma2 = parameters.is_sigma2_estim;
        if ((!m_est_sigma2) && (parameters.sigma2.has_value())) {
          m_sigma2 = parameters.sigma2.value();
          if (m_normalize)
            m_sigma2 /= (scaleY * scaleY);
        } else {
          m_est_sigma2 = true;  // force estim if no value given
        }

        lbfgsb::Optimizer optimizer{d};
        optimizer.iprint = Optim::log_level - 2;
        optimizer.max_iter = Optim::max_iteration;
        optimizer.pgtol = Optim::gradient_tolerance;
        optimizer.factr = Optim::objective_rel_tolerance / 1E-13;
        arma::ivec bounds_type{d, arma::fill::value(2)};  // means both upper & lower bounds

        int retry = 0;
        double best_f_opt = std::numeric_limits<double>::infinity();
        arma::vec best_gamma = gamma_tmp;
        Kriging::KModel m = make_Model(theta0.row(i % multistart).t(),nullptr);

        while (retry <= Optim::max_restart) {
          arma::vec gamma_0 = gamma_tmp;
          auto result = optimizer.minimize(
              [&m, &fit_ofn](const arma::vec& vals_inp, arma::vec& grad_out) -> double {
                return fit_ofn(vals_inp, &grad_out, nullptr, &m);
              },
              gamma_tmp,
              gamma_lower.memptr(),
              gamma_upper.memptr(),
              bounds_type.memptr());

          if (Optim::log_level > 0) {
            arma::cout << "     iterations: " << result.num_iters << arma::endl;
            arma::cout << "     status: " << result.task << arma::endl;
            if (Optim::reparametrize) {
              arma::cout << "     start_point: " << Optim::reparam_from(gamma_0).t() << " ";
              arma::cout << "     solution: " << Optim::reparam_from(gamma_tmp).t() << " ";
            } else {
              arma::cout << "     start_point: " << gamma_0.t() << " ";
              arma::cout << "     solution: " << gamma_tmp.t() << " ";
            }
          }

          if (result.f_opt < best_f_opt) {
            best_f_opt = result.f_opt;
            best_gamma = gamma_tmp;
          }

          double sol_to_lb = arma::min(arma::abs(gamma_tmp - gamma_lower));
          double sol_to_ub = arma::min(arma::abs(gamma_tmp - gamma_upper));
          double sol_to_b = std::min(sol_to_ub, sol_to_lb); 
          //Optim::reparametrize ? sol_to_ub : sol_to_lb;  // just consider theta lower bound
          if ((retry < Optim::max_restart)
              && (
                (result.task.rfind("ABNORMAL_TERMINATION_IN_LNSRCH", 0) == 0)  // error in algorithm
                || ((sol_to_b < arma::datum::eps) && (result.num_iters <= 2))  // we are stuck on a bound   
                || (result.f_opt > best_f_opt) // maybe still better start point available
              )) {
            gamma_tmp = (theta0.row(i % multistart).t() + theta_lower)
                        / pow(2.0, retry + 1);  // so move starting point to middle-point

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

        // last call to ensure that T and z are up-to-date with solution.
        double min_ofn_tmp = fit_ofn(best_gamma, nullptr, nullptr, &m);

        if (Optim::log_level > 0) {
          arma::cout << "  best objective: " << min_ofn_tmp << arma::endl;
          if (Optim::reparametrize)
            arma::cout << "  best solution: " << Optim::reparam_from(best_gamma).t() << " ";
          else
            arma::cout << "  best solution: " << best_gamma.t() << " ";
        }

        if (min_ofn_tmp < min_ofn) {
          m_theta = best_gamma;
          if (Optim::reparametrize)
            m_theta = Optim::reparam_from(best_gamma);
          m_est_theta = true;
          min_ofn = min_ofn_tmp;

          m_T = std::move(m.L);
          m_R = std::move(m.R);
          m_M = std::move(m.Fstar);
          m_circ = std::move(m.Rstar);
          m_star = std::move(m.Qstar);
          if (m_est_beta) {
            m_beta = std::move(m.betahat);
            m_z = std::move(m.Estar);
          }  else {
            // m_beta = parameters.beta.value(); already done above
            m_z = std::move(m.ystar) - m_M * m_beta;
          }
          if (m_est_sigma2) {
            m_sigma2 = m.SSEstar / n;
            if (m_objective.compare("LMP") == 0) {
              m_sigma2 = m.SSEstar / (n - m_F.n_cols);
            }
          }
        }
      }
    } else if (optim.rfind("Newton", 0) == 0) {
      Random::init();

      int multistart = 1;
      try {
        multistart = std::stoi(optim.substr(6));
      } catch (std::invalid_argument&) {
        // let multistart = 1
      }

      theta0 = arma::repmat(trans(theta_lower), multistart, 1)
               + Random::randu_mat(multistart, d) % arma::repmat(trans(theta_upper - theta_lower), multistart, 1);
      // theta0 = arma::abs(0.5 + Random::randn_mat(multistart, d) / 6.0)
      //          % arma::repmat(max(m_X, 0) - min(m_X, 0), multistart, 1);

      if (parameters.theta.has_value()) {  // just use given theta(s) as starting values for multi-bfgs
        arma::mat theta0_tmp = arma::mat(parameters.theta.value());
        if (m_normalize)
          theta0_tmp.each_row() /= scaleX;
        theta0 = arma::join_cols(theta0, theta0_tmp);
      }      // arma::cout << "theta0:" << theta0 << arma::endl;

      arma::vec gamma_lower = theta_lower;
      arma::vec gamma_upper = theta_upper;
      if (Optim::reparametrize) {
        gamma_lower = Optim::reparam_to(theta_upper);
        gamma_upper = Optim::reparam_to(theta_lower);
      }

      double min_ofn = std::numeric_limits<double>::infinity();

      for (arma::uword i = 0; i < multistart; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
        arma::vec gamma_tmp = theta0.row(i % multistart).t();
        if (Optim::reparametrize)
          gamma_tmp = Optim::reparam_to(theta0.row(i % multistart).t());

        gamma_lower = arma::min(gamma_tmp, gamma_lower);
        gamma_upper = arma::max(gamma_tmp, gamma_upper);

        if (Optim::log_level > 0) {
          arma::cout << "Newton:" << arma::endl;
          arma::cout << "  max iterations: " << Optim::max_iteration << arma::endl;
          arma::cout << "  null gradient tolerance: " << Optim::gradient_tolerance << arma::endl;
          arma::cout << "  constant objective tolerance: " << Optim::objective_rel_tolerance << arma::endl;
          arma::cout << "  reparametrize: " << Optim::reparametrize << arma::endl;
          arma::cout << "  normalize: " << m_normalize << arma::endl;
          arma::cout << "  lower_bounds: " << theta_lower.t() << " ";
          arma::cout << "  upper_bounds: " << theta_upper.t() << " ";
          arma::cout << "  start_point: " << theta0.row(i % multistart) << " ";
        }

        m_est_sigma2 = parameters.is_sigma2_estim;
        if ((!m_est_sigma2) && (parameters.sigma2.has_value())) {
          m_sigma2 = parameters.sigma2.value();
          if (m_normalize)
            m_sigma2 /= (scaleY * scaleY);
        } else {
          m_est_sigma2 = true;  // force estim if no value given
        }        

        Kriging::KModel m = make_Model(theta0.row(i % multistart).t(),nullptr);
        double min_ofn_tmp = optim_newton(
            [&m, &fit_ofn](const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out) -> double {
              return fit_ofn(vals_inp, grad_out, hess_out, &m);
            },
            gamma_tmp,
            gamma_lower,
            gamma_upper);

        if (Optim::log_level > 0) {
          arma::cout << "  best objective: " << min_ofn_tmp << arma::endl;
          if (Optim::reparametrize)
            arma::cout << "  best solution: " << Optim::reparam_from(gamma_tmp).t() << " ";
          else
            arma::cout << "  best solution: " << gamma_tmp.t() << " ";
        }

        if (min_ofn_tmp < min_ofn) {
          m_theta = gamma_tmp;
          if (Optim::reparametrize)
            m_theta = Optim::reparam_from(gamma_tmp);
          m_est_theta = true;
          min_ofn = min_ofn_tmp;

          m_T = std::move(m.L);
          m_R = std::move(m.R);
          m_M = std::move(m.Fstar);
          m_circ = std::move(m.Rstar);
          m_star = std::move(m.Qstar);
          m_z = std::move(m.Estar);
          if (m_est_beta) {
            m_beta = std::move(m.betahat);
            m_z = std::move(m.Estar);
          }  else {
            // m_beta = parameters.beta.value(); already done above
            m_z = std::move(m.ystar) - m_M * m_beta;
          }
          if (m_est_sigma2) {
            m_sigma2 = m.SSEstar / n;
            if (m_objective.compare("LMP") == 0) {
              m_sigma2 = m.SSEstar / (n - m_F.n_cols);
            }
          }
        }
      }
    } else
      throw std::runtime_error("Unsupported optim: " + optim + " (supported are: none, BFGS[#], Newton[#])");
  }

  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param X_n is n_n*d matrix of points where to predict output
 * @param return_stdev is true if return also stdev column vector
 * @param return_cov is true if return also cov matrix between X_n
 * @param return_deriv is true if return also derivative of prediction wrt x
 * @return output prediction: n_n means, [n_n standard deviations], [n_n*n_n full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::vec, arma::vec, arma::mat, arma::mat, arma::mat>
Kriging::predict(const arma::mat& X_n, bool return_stdev, bool return_cov, bool return_deriv) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::runtime_error("Predict locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  arma::vec yhat_n = arma::vec(n_n,arma::fill::none);
  arma::vec ysd2_n = arma::vec(n_n,arma::fill::zeros);
  arma::mat Sigma_n = arma::mat(n_n, n_n,arma::fill::zeros);
  arma::mat Dyhat_n = arma::mat(n_n, d,arma::fill::zeros);
  arma::mat Dysd2_n = arma::mat(n_n, d,arma::fill::zeros);

  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = X_n;
  // Normalize X_n
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  double sigma2 = m_sigma2 * (m_objective.compare("LMP") == 0 ? (n_o - m_F.n_cols) / (n_o - m_F.n_cols - 2) : 1.0);

  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  Xn_n = trans(Xn_n);

  auto t0 = Bench::tic();
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  for (arma::uword i = 0; i < n_o; i++) {
    for (arma::uword j = 0; j < n_n; j++) {
      arma::vec dij = Xn_o.col(i) - Xn_n.col(j);
      if (dij.is_zero(arma::datum::eps))
        R_on.at(i, j) = 1.0;
      else
        R_on.at(i, j) = _Cov(dij, m_theta);
    }
  }
  t0 = Bench::toc(nullptr, "R_on       ", t0);

  arma::mat Rstar_on =LinearAlgebra::solve(m_T, R_on);
  t0 = Bench::toc(nullptr, "Rstar_on   ", t0);

  yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr, "yhat_n     ", t0);

  // Un-normalize predictor
  yhat_n = m_centerY + m_scaleY * yhat_n;

  arma::mat  Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve(m_circ, E_n);
  t0 = Bench::toc(nullptr, "Ecirc_n    ", t0);

  if (return_stdev) {
  ysd2_n = 1.0 - sum(Rstar_on % Rstar_on,0).as_col() +  sum(Ecirc_n % Ecirc_n, 1).as_col();
  ysd2_n.transform([](double val) { return (std::isnan(val) || val < 0 ? 0.0 : val); });
  ysd2_n *= sigma2 * m_scaleY * m_scaleY;
  t0 = Bench::toc(nullptr, "ysd2_n     ", t0);
  }

  if (return_cov) {
  // Compute the covariance matrix between new data points
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  for (arma::uword i = 0; i < n_n; i++) {
    //R_nn.at(i, i) = 1;
    for (arma::uword j = 0; j < i; j++) {
      R_nn.at(i, j) = R_nn.at(j, i) = _Cov((Xn_n.col(i) - Xn_n.col(j)), m_theta);
    }
  }
  R_nn.diag().ones();
  t0 = Bench::toc(nullptr, "R_nn       ", t0);

  Sigma_n = R_nn - trans(Rstar_on) * Rstar_on + Ecirc_n * trans(Ecirc_n);
  Sigma_n *= sigma2 * m_scaleY * m_scaleY;
  t0 = Bench::toc(nullptr, "Sigma_n    ", t0);
  }

  if (return_deriv) {
  //// https://github.com/libKriging/dolka/blob/bb1dbf0656117756165bdcff0bf5e0a1f963fbef/R/kmStuff.R#L322C1-L363C10
  //for (i in 1:n_n) {
  //  
  //  ## =================================================================
  //  ## 'DF_n_i' and 'DR_on_i' are matrices with
  //  ## dimension c(n_n, d)
  //  ## =================================================================
  //  
  //  DF_n_i <- trend.deltax(x = XNew[i, ], model = object)
  //  KOldNewi <- as.vector(KOldNew[ , i])
  //  DR_on_i <- covVector.dx(x = as.vector(XNew[i, ]),
  //                              X = X,
  //                              object = object@covariance,
  //                              c = KOldNewi)
  //  
  //  KOldNewStarDer[ , i, i, ] <-
  //      backsolve(L, DR_on_i, upper.tri = FALSE)
  //  
  //  ## Gradient of the kriging trend and mean
  //  muNewHatDer[i, i, ] <- crossprod(DF_n_i, betaHat)
  //  ## dim in product c(d, n) and NULL(length d)
  //  mean_nHatDer[i, i, ] <- muNewHatDer[i, i, ] +
  //      crossprod(KOldNewStarDer[ , i, i,  ], zStar) 
  //  ## dim in product c(d, n) and NULL(length n)
  //  s2Der[i, i,  ] <-
  //      - 2 * crossprod(KOldNewStarDer[ , i, i, ],
  //                      drop(KOldNewStar[ , i]))
  //  
  //  ## dim in product c(d, n) and c(n, p)
  //  
  //  if (type == "UK") {
  //      ENewDagDer[i, i, , ] <-
  //          backsolve(t(RStar),
  //                    DF_n_i -
  //                    t(crossprod(KOldNewStarDer[ , i, i, ], FStar)),
  //                    upper.tri = FALSE)
  //      ## dim in product NULL (length p) and c(p, d) because of 'drop'
  //      s2Der[i, i, ] <- s2Der[i, i, ] + 2 * drop(ENewDagT[ , i]) %*%
  //          drop(ENewDagDer[i, i, , ])
  //  }
  // numerical derivative step : value is sensitive only for non linear trends. Otherwise, it gives exact results.
  const double h = 1.0E-5; 
  arma::mat h_eye_d = h * arma::mat(d, d, arma::fill::eye);

  // Compute the derivatives of the covariance and trend functions
  for (arma::uword i = 0; i < n_n; i++) {  // for each predict point... should be parallel ?
      arma::mat DR_on_i = arma::mat(n_o, d, arma::fill::none);
      for (arma::uword j = 0; j < n_o; j++) {
        DR_on_i.row(j) = R_on.at(j, i) * trans(_DlnCovDx(Xn_n.col(i) - Xn_o.col(j), m_theta));
      }
      t0 = Bench::toc(nullptr, "DR_on_i    ", t0);

      arma::mat tXn_n_repd_i = arma::trans(Xn_n.col(i) * arma::mat(1, d, arma::fill::ones));  // just duplicate X_n.row(i) d times
      arma::mat DF_n_i = (Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i + h_eye_d)
                        - Trend::regressionModelMatrix(m_regmodel, tXn_n_repd_i - h_eye_d))
                       / (2 * h);
      t0 = Bench::toc(nullptr, "DF_n_i     ", t0);

      arma::mat W_i = LinearAlgebra::solve(m_T, DR_on_i);
      t0 = Bench::toc(nullptr, "W_i        ", t0);
      Dyhat_n.row(i) = trans(DF_n_i * m_beta + trans(W_i) * m_z);
      t0 = Bench::toc(nullptr, "Dyhat_n    ", t0);

      if (return_stdev) {
        arma::mat DEcirc_n_i = LinearAlgebra::solve(m_circ.t(), trans(DF_n_i - W_i.t() * m_M));
        Dysd2_n.row(i) = -2 * Rstar_on.col(i).t() * W_i + 2 * Ecirc_n.row(i) * DEcirc_n_i;
        t0 = Bench::toc(nullptr, "Dysd2_n    ", t0);
      }
  }
  Dyhat_n *= m_scaleY;
  Dysd2_n *= sigma2 * m_scaleY * m_scaleY;
  }

  return std::make_tuple(std::move(yhat_n),
                         std::move(arma::sqrt(ysd2_n)),
                         std::move(Sigma_n),
                         std::move(Dyhat_n),
                         std::move(Dysd2_n / (2 * arma::sqrt(ysd2_n) * arma::mat(1, d, arma::fill::ones))));
  /*if (return_stdev)
    if (return_cov)
      return std::make_tuple(std::move(yhat_n), std::move(pred_stdev), std::move(pred_cov));
    else
      return std::make_tuple(std::move(yhat_n), std::move(pred_stdev), nullptr);
  else if (return_cov)
    return std::make_tuple(std::move(yhat_n), std::move(pred_cov), nullptr);
  else
    return std::make_tuple(std::move(yhat_n), nullptr, nullptr);*/
}

/** Draw sample trajectories of kriging at given points X'
 * @param X_n is n_n*d matrix of points where to simulate output
 * @param seed is seed for random number generator
 * @param nsim is number of simulations to draw
 * @param will_update is true if we want to keep simulations data for future update
 * @return output is n_n*nsim matrix of simulations at X_n
 */
LIBKRIGING_EXPORT arma::mat Kriging::simulate(const int nsim, const int seed, const arma::mat& X_n, const bool will_update) {
  arma::uword n_n = X_n.n_rows;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  if (X_n.n_cols != d)
    throw std::runtime_error("Simulate locations have wrong dimension: " + std::to_string(X_n.n_cols) + " instead of "
                             + std::to_string(d));

  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = X_n;
  // Normalize X_n
  Xn_n.each_row() -= m_centerX;
  Xn_n.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_n = Trend::regressionModelMatrix(m_regmodel, Xn_n);
  //if (will_update) sim_Fp = F_n;
  Xn_n = trans(Xn_n);

  auto t0 = Bench::tic();
  // Compute covariance between new data
  arma::mat R_nn = arma::mat(n_n, n_n, arma::fill::none);
  for (arma::uword i = 0; i < n_n; i++) {
    //R_nn.at(i, i) = 1.0;
    for (arma::uword j = 0; j < i; j++) {
      R_nn.at(i, j) = R_nn.at(j, i) = _Cov((Xn_n.col(i) - Xn_n.col(j)), m_theta);
    }
  }
  R_nn.diag().ones();
  t0 = Bench::toc(nullptr,"R_nn          ", t0);

  // Compute covariance between training data and new data to predict
  arma::mat R_on = arma::mat(n_o, n_n, arma::fill::none);
  for (arma::uword i = 0; i < n_o; i++) {
    for (arma::uword j = 0; j < n_n; j++) {
      R_on.at(i, j) = _Cov((Xn_o.col(i) - Xn_n.col(j)), m_theta);
    }
  }
  t0 = Bench::toc(nullptr,"R_on        ", t0);

  arma::mat Rstar_on =LinearAlgebra::solve(m_T, R_on);
  t0 = Bench::toc(nullptr,"Rstar_on   ", t0);
  //arma::cout << "Rstar_on:" << Rstar_on << arma::endl;

  arma::vec yhat_n = F_n * m_beta + trans(Rstar_on) * m_z;
  t0 = Bench::toc(nullptr,"yhat_n        ", t0);

  arma::mat Fhat_n = trans(Rstar_on) * m_M;
  arma::mat E_n = F_n - Fhat_n;
  arma::mat Ecirc_n = LinearAlgebra::rsolve(m_circ, E_n);
  t0 = Bench::toc(nullptr,"Ecirc_n       ", t0);
  //arma::cout << "eig(R_nn):" << arma::eig_sym(R_nn) << arma::endl;

//arma::cout << "t(Rstar_on)*Rstar_on:" <<  trans(Rstar_on) * Rstar_on << arma::endl;

  arma::mat SigmaNoTrend_nKo = R_nn - trans(Rstar_on) * Rstar_on ;
  //arma::cout << "eig(SigmaNoTrend_nKo):" << arma::eig_sym(SigmaNoTrend_nKo) << arma::endl;

  arma::mat Sigma_nKo = SigmaNoTrend_nKo + Ecirc_n * trans(Ecirc_n);
  t0 = Bench::toc(nullptr,"Sigma_nKo     ", t0);

  //arma::cout << "eig(Sigma_nKo):" << arma::eig_sym(Sigma_nKo) << arma::endl;

  arma::mat LSigma_nKo = LinearAlgebra::safe_chol_lower(Sigma_nKo);
  t0 = Bench::toc(nullptr,"LSigma_nKo     ", t0);

  arma::mat y_n = arma::mat(n_n, nsim, arma::fill::none);
  y_n.each_col() = yhat_n;
  Random::reset_seed(seed);
  y_n += LSigma_nKo * Random::randn_mat(n_n, nsim) * std::sqrt(m_sigma2);

  // Un-normalize simulations
  y_n = m_centerY + m_scaleY * y_n;

  if (will_update) {
    lastsimup_Xn_u.clear(); // force reset to force update_simulate consider new data
    lastsim_y_n = y_n;

    lastsim_Xn_n = Xn_n;
    lastsim_seed = seed;
    lastsim_nsim = nsim;

    lastsim_R_nn = R_nn;
    lastsim_F_n = F_n;

    lastsim_L_oCn = Rstar_on;
    lastsim_L_nCn = LinearAlgebra::safe_chol_lower(SigmaNoTrend_nKo);
    t0 = Bench::toc(nullptr,"L_nCn     ", t0);

    lastsim_L_on = arma::join_rows(
          arma::join_cols(m_T, lastsim_L_oCn.t()),
          arma::join_cols(arma::zeros(n_o, n_n), lastsim_L_nCn));


    arma::mat Linv_on = LinearAlgebra::solve(lastsim_L_on, arma::mat(n_o+n_n, n_o+n_n,arma::fill::eye));
    t0 = Bench::toc(nullptr,"Linv_on     ", t0);
    lastsim_Rinv_on = Linv_on.t() * Linv_on;
    t0 = Bench::toc(nullptr,"Rinv_on     ", t0);

    lastsim_F_on = arma::join_cols(m_F, lastsim_F_n);
    lastsim_Fstar_on = LinearAlgebra::solve(lastsim_L_on, lastsim_F_on);
    t0 = Bench::toc(nullptr,"Fstar_on     ", t0);
    arma::mat Q_Fstar_on;
    arma::qr(Q_Fstar_on, lastsim_circ_on, lastsim_Fstar_on);
    lastsim_Fcirc_on = LinearAlgebra::rsolve(lastsim_circ_on, lastsim_F_on);
    t0 = Bench::toc(nullptr,"Fcirc_on     ", t0);

    lastsim_Fhat_nKo = lastsim_L_oCn.t() * m_M;
    t0 = Bench::toc(nullptr,"Fhat_nKo     ", t0);
    lastsim_Ecirc_nKo = LinearAlgebra::rsolve(m_circ, F_n - lastsim_Fhat_nKo);
    t0 = Bench::toc(nullptr,"Ecirc_nKo     ", t0);
  }

  // Un-normalize simulations
  return y_n;
}

LIBKRIGING_EXPORT arma::mat Kriging::rand(const int nsim, const int seed, const arma::mat& X_n, const bool will_update) {
  return simulate(nsim, seed, X_n, will_update);
}

LIBKRIGING_EXPORT arma::mat Kriging::update_simulate(const arma::vec& y_u, const arma::mat& X_u) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x"
                             + std::to_string(m_X.n_cols) + "), new X: (...x"
                             + std::to_string(X_u.n_cols) + ")");

  if (lastsim_y_n.is_empty() || lastsim_y_n.n_rows == 0)
    throw std::runtime_error("No previous simulation data available");

  if (lastsim_Xn_n.n_rows != X_u.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x"
                             + std::to_string(X_u.n_cols) + "), last sim X: (...x"
                             + std::to_string(lastsim_Xn_n.n_rows) + ")");

  arma::uword n_n = lastsim_Xn_n.n_cols;
  arma::uword n_o = m_X.n_rows;
  arma::uword d = m_X.n_cols;
  arma::mat Xn_o = trans(m_X);  // already normalized if needed
  arma::mat Xn_n = lastsim_Xn_n; // already normalized 

  arma::uword n_on = n_o + n_n;
  //arma::mat Xn_on = arma::join_cols(Xn_o, Xn_n);
  arma::mat F_on = arma::join_cols(m_F, lastsim_F_n);

  arma::uword n_u = X_u.n_rows;
  // Normalize X_u
  arma::mat Xn_u = X_u;
  Xn_u.each_row() -= m_centerX;
  Xn_u.each_row() /= m_scaleX;

  // Define regression matrix
  arma::mat F_u = Trend::regressionModelMatrix(m_regmodel, Xn_u);

  auto t0 = Bench::tic();
  Xn_u = trans(Xn_u);
  t0 = Bench::toc(nullptr,"Xn_u.t()      ", t0);

  bool use_lastsimup = (!lastsimup_Xn_u.is_empty()) && arma::approx_equal(lastsimup_Xn_u, Xn_u, "absdiff", arma::datum::eps);
  if (! use_lastsimup) {
    lastsimup_Xn_u = Xn_u;
  
    // Compute covariance between updated data
    lastsimup_R_uu = arma::mat(n_u, n_u, arma::fill::none);
    for (arma::uword i = 0; i < n_u; i++) {
      //lastsimup_R_uu.at(i, i) = 1.0;
      for (arma::uword j = 0; j < i; j++) {
        lastsimup_R_uu.at(i, j) = lastsimup_R_uu.at(j, i) = _Cov((Xn_u.col(i) - Xn_u.col(j)), m_theta);
      }
    }
    lastsimup_R_uu.diag().ones();
    t0 = Bench::toc(nullptr,"R_uu          ", t0);
  
    // Compute covariance between updated/old data
    lastsimup_R_uo = arma::mat(n_u, n_o, arma::fill::none);
    for (arma::uword i = 0; i < n_u; i++) {
      for (arma::uword j = 0; j < n_o; j++) {
        lastsimup_R_uo.at(i, j) = _Cov((Xn_u.col(i) - Xn_o.col(j)), m_theta);
      }
    }
    t0 = Bench::toc(nullptr,"R_uo          ", t0);
  
    // Compute covariance between updated/new data
    lastsimup_R_un = arma::mat(n_u, n_n, arma::fill::none);
    for (arma::uword i = 0; i < n_u; i++) {
      for (arma::uword j = 0; j < n_n; j++) {
        lastsimup_R_un.at(i, j) = _Cov((Xn_u.col(i) - Xn_n.col(j)), m_theta);
      }
    }
    t0 = Bench::toc(nullptr,"R_un          ", t0);
  }

  // ======================================================================
  // FOXY step #1 Extend the simulation to the design 'X_u' IF
  // NECESSARY. Remind that the simulation number j is
  // conditional on the given 'y_o' and on 'y_n = Y_sim[ , j]'
  //
  // CAUTION. To avoid unnecessary re-computations we here use
  // auxiliary variables that where computed in the creation of
  // the KM0 step AND later in the simulation. The first ones are
  // in 'theKM0$Extra' and the second ones are in 'Ex'
  //
  // In indices 'C' means comma and 'K' means pipe '|'
  //
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat R_onCu = arma::join_rows(lastsimup_R_uo, lastsimup_R_un).t();
    arma::mat Rstar_onCu = LinearAlgebra::solve(lastsim_L_on, R_onCu);
    t0 = Bench::toc(nullptr,"Rstar_onCu          ", t0);

    arma::mat Ecirc_uKon = LinearAlgebra::rsolve(lastsim_circ_on, F_u - Rstar_onCu.t() * lastsim_Fstar_on);
    t0 = Bench::toc(nullptr,"Ecirc_uKon          ", t0);
  
    arma::mat Sigma_uKon = lastsimup_R_uu - Rstar_onCu.t() * Rstar_onCu + Ecirc_uKon * Ecirc_uKon.t();
    t0 = Bench::toc(nullptr,"Sigma_uKon          ", t0);
    
    arma::mat LSigma_uKon = LinearAlgebra::safe_chol_lower(Sigma_uKon);
    t0 = Bench::toc(nullptr,"LSigma_uKon          ", t0);
  
    arma::mat W_uCon = (R_onCu.t() + Ecirc_uKon * lastsim_Fcirc_on.t()) * lastsim_Rinv_on;
    t0 = Bench::toc(nullptr,"W_uCon          ", t0);

    arma::mat m_u = W_uCon.head_cols(n_o) * m_y;
    arma::mat M_u = arma::repmat(m_u,1,lastsim_nsim) +  W_uCon.tail_cols(n_n) * lastsim_y_n;
    
    Random::reset_seed(lastsim_seed);
    lastsimup_y_u = M_u + LSigma_uKon * Random::randn_mat(n_u, lastsim_nsim) * std::sqrt(m_sigma2);    
    t0 = Bench::toc(nullptr,"y_u          ", t0);
  }

  // ======================================================================
  // FOXY step #2   Update the simulated paths on 'X_n'
  // ======================================================================

  if (!use_lastsimup) {
    arma::mat Rstar_ou = LinearAlgebra::solve(m_T, lastsimup_R_uo.t());
    t0 = Bench::toc(nullptr,"Rstar_ou          ", t0);

    arma::mat Fhat_uKo = Rstar_ou.t() * m_M;
    arma::mat Ecirc_uKo = LinearAlgebra::rsolve(m_circ,F_u - Fhat_uKo);
    t0 = Bench::toc(nullptr,"Ecirc_uKo          ", t0);

    arma::mat Rtild_uCu = lastsimup_R_uu - Rstar_ou.t() * Rstar_ou + Ecirc_uKo * Ecirc_uKo.t();
    t0 = Bench::toc(nullptr,"Rtild_uCu          ", t0);

    arma::mat Rtild_nCu = lastsimup_R_un - Rstar_ou.t() * lastsim_L_oCn + Ecirc_uKo * lastsim_Ecirc_nKo.t(); 
    t0 = Bench::toc(nullptr,"Rtild_nCu          ", t0);

    lastsimup_Wtild_nKu = LinearAlgebra::solve(Rtild_uCu, Rtild_nCu).t();
    t0 = Bench::toc(nullptr,"Wtild_nKu          ", t0);
  }

  return lastsim_y_n + lastsimup_Wtild_nKu * (arma::repmat(y_u,1,lastsim_nsim) - lastsimup_y_u);  
}

LIBKRIGING_EXPORT arma::mat Kriging::update_rand(const arma::vec& y_u, const arma::mat& X_u) {
  return update_simulate(y_u, X_u);
}

/** Add new conditional data points to previous (X,y), then perform new fit.
 * @param y_u is n_u length column vector of new output
 * @param X_u is n_u*d matrix of new input
 * @param refit is true if we want to re-fit the model
 */
LIBKRIGING_EXPORT void Kriging::update(const arma::vec& y_u, const arma::mat& X_u, const bool refit) {
  if (y_u.n_elem != X_u.n_rows)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (" + std::to_string(X_u.n_rows) + "x"
                             + std::to_string(X_u.n_cols) + "), y: (" + std::to_string(y_u.n_elem) + ")");

  if (X_u.n_cols != m_X.n_cols)
    throw std::runtime_error("Dimension of new data should be the same:\n X: (...x"
                             + std::to_string(m_X.n_cols) + "), new X: (...x"
                             + std::to_string(X_u.n_cols) + ")");

  // rebuild starting parameters
  Parameters parameters;
  if (refit) {// re-fit
    if (m_est_beta)
    parameters = Parameters{std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY),
                        this->m_est_sigma2,
                        std::make_optional(trans(this->m_theta) % this->m_scaleX),
                        this->m_est_theta,
                        std::make_optional(arma::ones<arma::vec>(0)),
                        true};
    else 
    parameters = Parameters{
                        std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY),
                        this->m_est_sigma2,
                        std::make_optional(trans(this->m_theta) % this->m_scaleX),
                        this->m_est_theta,
                        std::make_optional(trans(this->m_beta) * this->m_scaleY),
                        false};
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                            y_u),  // de-normalize previous data according to suite unnormed new data
            arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
            m_regmodel,
            m_normalize,
            m_optim,
            m_objective,
            parameters);
  } else {// just update
    parameters = Parameters{
                        std::make_optional(this->m_sigma2 * this->m_scaleY * this->m_scaleY),
                        false,
                        std::make_optional(trans(arma::mat(this->m_theta)) % this->m_scaleX),
                        false,
                        std::make_optional(arma::vec(this->m_beta) * this->m_scaleY),
                        false};
    this->fit(arma::join_cols(m_y * this->m_scaleY + this->m_centerY,
                            y_u),  // de-normalize previous data according to suite unnormed new data
            arma::join_cols((m_X.each_row() % this->m_scaleX).each_row() + this->m_centerX, X_u),
            m_regmodel,
            m_normalize,
            "none",
            m_objective,
            parameters);
  }
}

LIBKRIGING_EXPORT std::string Kriging::summary() const {
  std::ostringstream oss;
  auto vec_printer = [&oss](const arma::vec& v) {
    v.for_each([&oss, i = 0](const arma::vec::elem_type& val) mutable {
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
    vec_printer(m_beta);
    oss << "\n";
    oss << "* variance";
    oss << ((m_est_sigma2) ? " (est.): " : ": ");
    oss << m_sigma2;
    oss << "\n";
    oss << "* covariance:\n";
    oss << "  * kernel: " << m_covType << "\n";
    oss << "  * range";
    oss << ((m_est_theta) ? " (est.): " : ": ");
    vec_printer(m_theta);
    oss << "\n";
    oss << "  * fit:\n";
    oss << "    * objective: " << m_objective << "\n";
    oss << "    * optim: " << m_optim << "\n";
  }
  return oss.str();
}

void Kriging::save(const std::string filename) const {
  nlohmann::json j;

  j["version"] = 2;
  j["content"] = "Kriging";

  // _Cov_pow & std::function embedded by make_Cov
  j["covType"] = m_covType;
  j["X"] = to_json(m_X);
  j["centerX"] = to_json(m_centerX);
  j["scaleX"] = to_json(m_scaleX);
  j["y"] = to_json(m_y);
  j["centerY"] = m_centerY;
  j["scaleY"] = m_scaleY;
  j["normalize"] = m_normalize;
  j["regmodel"] = Trend::toString(m_regmodel);
  j["optim"] = m_optim;
  j["objective"] = m_objective;
  // Auxiliary data
  j["dX"] = to_json(m_dX);
  j["maxdX"] = to_json(m_maxdX);
  j["F"] = to_json(m_F);
  j["T"] = to_json(m_T);
  j["R"] = to_json(m_R);
  j["M"] = to_json(m_M);
  j["star"] = to_json(m_star);
  j["circ"] = to_json(m_circ);
  j["z"] = to_json(m_z);
  j["beta"] = to_json(m_beta);
  j["est_beta"] = m_est_beta;
  j["theta"] = to_json(m_theta);
  j["est_theta"] = m_est_theta;
  j["sigma2"] = m_sigma2;
  j["est_sigma2"] = m_est_sigma2;

  std::ofstream f(filename);
  f << std::setw(4) << j;
}

Kriging Kriging::load(const std::string filename) {
  std::ifstream f(filename);
  nlohmann::json j = nlohmann::json::parse(f);

  uint32_t version = j["version"].template get<uint32_t>();
  if (version != 2) {
    throw std::runtime_error(asString("Bad version to load from '", filename, "'; found ", version, ", requires 2"));
  }
  std::string content = j["content"].template get<std::string>();
  if (content != "Kriging") {
    throw std::runtime_error(
        asString("Bad content to load from '", filename, "'; found '", content, "', requires 'Kriging'"));
  }

  std::string covType = j["covType"].template get<std::string>();
  Kriging kr(covType);  // _Cov_pow & std::function embedded by make_Cov

  kr.m_X = mat_from_json(j["X"]);
  kr.m_centerX = rowvec_from_json(j["centerX"]);
  kr.m_scaleX = rowvec_from_json(j["scaleX"]);
  kr.m_y = colvec_from_json(j["y"]);
  kr.m_centerY = j["centerY"].template get<decltype(kr.m_centerY)>();
  kr.m_scaleY = j["scaleY"].template get<decltype(kr.m_scaleY)>();
  kr.m_normalize = j["normalize"].template get<decltype(kr.m_normalize)>();

  std::string model = j["regmodel"].template get<std::string>();
  kr.m_regmodel = Trend::fromString(model);

  kr.m_optim = j["optim"].template get<decltype(kr.m_optim)>();
  kr.m_objective = j["objective"].template get<decltype(kr.m_objective)>();
  // Auxiliary data
  kr.m_dX = mat_from_json(j["dX"]);
  kr.m_maxdX = colvec_from_json(j["maxdX"]);
  kr.m_F = mat_from_json(j["F"]);
  kr.m_T = mat_from_json(j["T"]);
  kr.m_R = mat_from_json(j["R"]);
  kr.m_M = mat_from_json(j["M"]);
  kr.m_star = mat_from_json(j["star"]);
  kr.m_circ = mat_from_json(j["circ"]);
  kr.m_z = colvec_from_json(j["z"]);
  kr.m_beta = colvec_from_json(j["beta"]);
  kr.m_est_beta = j["est_beta"].template get<decltype(kr.m_est_beta)>();
  kr.m_theta = colvec_from_json(j["theta"]);
  kr.m_est_theta = j["est_theta"].template get<decltype(kr.m_est_theta)>();
  kr.m_sigma2 = j["sigma2"].template get<decltype(kr.m_sigma2)>();
  kr.m_est_sigma2 = j["est_sigma2"].template get<decltype(kr.m_est_sigma2)>();

  return kr;
}
