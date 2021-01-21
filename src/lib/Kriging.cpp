// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Kriging.hpp"

#include <armadillo>
#include <optim.hpp>
#include <tuple>
#include <cassert>

std::chrono::high_resolution_clock::time_point tic() {
  return std::chrono::high_resolution_clock::now();
}
std::chrono::high_resolution_clock::time_point toc(std::string what,
                                                   std::chrono::high_resolution_clock::time_point t0) {
  const auto t = std::chrono::high_resolution_clock::now();
  arma::cout << what << ":     " << (std::chrono::duration<double>(t - t0)).count() * 1000 << arma::endl;
  return t;
}

//' @ref: https://github.com/psbiomech/dace-toolbox-source/blob/master/dace.pdf
//'  (where CovMatrix<-R, Ft<-M, C<-T, rho<-z)
//' @ref: https://github.com/cran/DiceKriging/blob/master/R/kmEstimate.R (same variables names)

//' @ref https://github.com/cran/DiceKriging/blob/master/src/CovFuns.c
// Covariance function on normalized data
std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&)> CovNorm_fun_gauss
    = [](arma::subview_col<double>&& xi, arma::subview_col<double>&& xj) {
        // double temp = 0;
        // for (arma::uword k = 0; k < xi.n_elem; k++) {
        //  double d = xi[k] - xj[k];
        //  temp += d * d;
        //}
        // Not better in perf. So:
        auto&& diff = (xi - xj);
        const double temp = arma::dot(diff, diff);

        return exp(-0.5 * temp);
      };

std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&, int)> Dln_CovNorm_gauss
    = [](arma::subview_col<double>&& xi, arma::subview_col<double>&& xj, int dim) {
        double d = (xi[dim] - xj[dim]);
        return d * d;
      };

std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&)> CovNorm_fun_exp
    = [](arma::subview_col<double>&& xi, arma::subview_col<double>&& xj) {
        auto&& diff = (xi - xj);
        return exp(-arma::sum(arma::abs(diff)));
      };

std::function<double(arma::subview_col<double>&&, arma::subview_col<double>&&, int)> Dln_CovNorm_exp
    = [](arma::subview_col<double>&& xi, arma::subview_col<double>&& xj, int dim) { return fabs(xi[dim] - xj[dim]); };

/************************************************/
/** implementation details forward declaration **/
/************************************************/

namespace {  // anonymous namespace for local implementation details
auto regressionModelMatrix(const Kriging::RegressionModel& regmodel,
                           const arma::mat& newX,
                           arma::uword n,
                           arma::uword d) -> arma::mat;
}  // namespace

/************************************************/
/**      Kriging implementation        **/
/************************************************/

// returns distance matrix form Xp to X
LIBKRIGING_EXPORT
arma::mat Kriging::Cov(const arma::mat& X, const arma::mat& Xp) {
  arma::mat Xtnorm = trans(X);
  Xtnorm.each_col() /= m_theta;
  arma::mat Xptnorm = trans(Xp);
  Xptnorm.each_col() /= m_theta;

  arma::uword n = X.n_rows;
  arma::uword np = Xp.n_rows;

  // Should bre replaced by for_each
  arma::mat R(n, np);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < np; j++) {
      R.at(i, j) = Kriging::CovNorm_fun(Xtnorm.col(i), Xptnorm.col(j));
    }
  }
  return R;
}

// Optimized version when Xp=X
LIBKRIGING_EXPORT
arma::mat Kriging::Cov(const arma::mat& X) {
  // Should be tyaken from covariance.h from nestedKriging ?
  // return getCrossCorrMatrix(X,Xp,parameters,covType);

  arma::mat Xtnorm = trans(X);
  Xtnorm.each_col() /= m_theta;
  arma::uword n = X.n_rows;

  // Should bre replaced by for_each
  arma::mat R(n, n);
  R.zeros();
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = Kriging::CovNorm_fun(Xtnorm.col(i), Xtnorm.col(j));
    }
  }

  R = arma::symmatl(R);  // R + trans(R);
  R.diag().ones();
  return R;
}
//// same for one point
// LIBKRIGING_EXPORT arma::colvec Kriging::Cov(const arma::mat& X,
//                                                    const arma::rowvec& x,
//                                                    const arma::colvec& theta) {
//  // FIXME mat(x) : an arma::mat from a arma::rowvec ?
//  return Kriging::Cov(&X, arma::mat(&x), &theta).col(1);  // TODO to be optimized...
//}

// This will create the dist(xi,xj) function above. Need to parse "covType".
void Kriging::make_Cov(const std::string& covType) {
  m_covType = covType;
  if (covType.compare("gauss") == 0) {
    CovNorm_fun = CovNorm_fun_gauss;
    Dln_CovNorm = Dln_CovNorm_gauss;
  } else if (covType.compare("exp") == 0) {
    CovNorm_fun = CovNorm_fun_exp;
    Dln_CovNorm = Dln_CovNorm_exp;
  } else
    throw std::invalid_argument("Unsupported covariance kernel: " + covType);

  // arma::cout << "make_Cov done." << arma::endl;
}

// at least, just call make_Cov(kernel)
LIBKRIGING_EXPORT Kriging::Kriging(const std::string& covType) {
  make_Cov(covType);
}

auto solve_opts
    = arma::solve_opts::fast + arma::solve_opts::no_approx + arma::solve_opts::no_band + arma::solve_opts::no_sympd;

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
double Kriging::logLikelihood(const arma::vec& _theta,
                              arma::vec* grad_out,
                              arma::mat* hess_out,
                              Kriging::OKModel* okm_data) const {
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

//  // auto t0 = tic();
//  arma::mat Xtnorm = trans(m_X);
//  Xtnorm.each_col() /= _theta;
//  // t0 = toc("Xtnorm        ", t0);
//
//  arma::uword n = m_X.n_rows;

//  // Allocate the matrix // arma::mat R = Cov(fd->X, _theta);
//  // Should be replaced by for_each ?
//  arma::mat R = arma::ones(n, n);
//  for (arma::uword i = 0; i < n; i++) {
//    for (arma::uword j = 0; j < i; j++) {
//      R.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xtnorm.col(j));
//    }
//  }
//  R = arma::symmatl(R);  // (R + trans(R))/2;
//  // t0 = toc("R             ", t0);

  // TIMEOUT 
  
//  // Cholesky decompostion of covariance matrix
//  fd->T = trimatl(chol(R, "lower"));
//  // t0 = toc("T             ", t0);
//
//  // Compute intermediate useful matrices
//  fd->M = solve(fd->T, m_F, solve_opts);
//  // t0 = toc("M             ", t0);
//  arma::mat Q;
//  arma::mat G;
//  qr_econ(Q, G, fd->M);
//  // t0 = toc("QG            ", t0);
//
//  arma::mat H = Q * Q.t();  // if (hess_out != nullptr)
//  // t0 = toc("H             ", t0);
//  arma::colvec Yt = solve(fd->T, m_y, solve_opts);
//  // t0 = toc("Yt            ", t0);
//  if (fd->estim_beta)
//    fd->beta = solve(trimatu(G), Q.t() * Yt, solve_opts);
//  //t0 = toc("beta          ", t0);
//  fd->z = Yt - fd->M * fd->beta;
//  // t0 = toc("z             ", t0);
//
//  //' @ref https://github.com/cran/DiceKriging/blob/master/R/computeAuxVariables.R
//  if (fd->estim_sigma2) // means no sigma2 provided
//    fd->sigma2 = arma::accu(fd->z % fd->z) / n;
//  // t0 = toc("sigma2_hat    ", t0);
//
//  double ll = -0.5 * (n * log(2 * M_PI * fd->sigma2) + 2 * sum(log(fd->T.diag())) + n);
//  // arma::cout << " ll:" << ll << arma::endl;
//  
//  if (grad_out != nullptr) {
//    //' @ref https://github.com/cran/DiceKriging/blob/master/R/logLikGrad.R
//    //  logLik.derivative <- matrix(0,nparam,1)
//    //  x <- backsolve(T,z)			# compute x := T^(-1)*z
//    //  Rinv <- chol2inv(T)			# compute inv(R) by inverting T
//    //
//    //  Rinv.upper <- Rinv[upper.tri(Rinv)]
//    //  xx <- x%*%t(x)
//    //  xx.upper <- xx[upper.tri(xx)]
//    //
//    //  for (k in 1:nparam) {
//    //    gradR.k <- CovMatrixDerivative(model@covariance, X=model@X, C0=R, k=k)
//    //    gradR.k.upper <- gradR.k[upper.tri(gradR.k)]
//    //
//    //    terme1 <- sum(xx.upper*gradR.k.upper)   / sigma2.hat
//    //    # quick computation of t(x)%*%gradR.k%*%x /  ...
//    //    terme2 <- - sum(Rinv.upper*gradR.k.upper)
//    //    # quick computation of trace(Rinv%*%gradR.k)
//    //    logLik.derivative[k] <- terme1 + terme2
//    //  }
//
//    // t0 = tic();
//    arma::cube gradsR = arma::cube(n, n, m_X.n_cols);  // if (hess_out != nullptr)
//    arma::vec terme1 = arma::vec(m_X.n_cols);                   // if (hess_out != nullptr)
//    // t0 = toc(" +gradsR         ", t0);
//
//    // arma::mat Linv = solve(fd->T, arma::eye(n, n), solve_opts);
//    // t0 = toc(" Linv            ",t0);
//    arma::mat Rinv = inv_sympd(R);  // trans(Linv) * Linv;
//    // t0 = toc(" Rinv            ", t0);
//
//    arma::mat tT = trimatu(trans(fd->T));
//    // t0 = toc(" tT              ", t0);
//    arma::mat x = solve(tT, fd->z, solve_opts);
//    // t0 = toc(" x               ", t0);
//    arma::mat xx = x * x.t();
//    // t0 = toc(" xx              ", t0);
//
//    for (arma::uword k = 0; k < m_X.n_cols; k++) {
//      arma::mat gradR_k_upper = R;
//      for (arma::uword i = 0; i < n; i++) {
//        gradR_k_upper.at(i, i) = 0;
//        for (arma::uword j = 0; j < i; j++) {
//          gradR_k_upper.at(j, i) *= Dln_CovNorm(Xtnorm.col(i), Xtnorm.col(j), k);
//        }
//      }
//      gradR_k_upper /= _theta(k);
//      // t0 = toc(" gradR_k_upper", t0);
//      arma::mat gradR_k = symmatu(gradR_k_upper);
//      //t0 = toc(" gradR_k      ", t0);
//      
//      // should make a fast function trace_prod(A,B) -> sum_i(sum_j(Ai,j*Bj,i))
//      terme1.at(k)
//          = as_scalar((trans(x) * gradR_k) * x) / fd->sigma2;  //; //as_scalar((trans(x) * gradR_k) * x)/ sigma2_hat;
//      double terme2 = -arma::trace(Rinv * gradR_k);            //-arma::accu(Rinv % gradR_k_upper)
//      (*grad_out).at(k) = (terme1.at(k) + terme2) / 2;
//      // t0 = toc(" grad_out     ", t0);
//      // arma::cout << " grad_out:" << *grad_out << arma::endl;
//
//      if (hess_out != nullptr) {
//        //' @ref O. Roustant
//        // for (k in 1:d) {
//        //   for (l in 1:k) {
//        //     aux <- grad_R[[k]] %*% Rinv %*% grad_R[[l]]
//        //     Dkl <- d2_matcor(X, modele_proba$covariance, R, grad_logR, k,l)
//        //     xk <- backsolve(t(T),grad_R[[k]]%*%x, upper.tri=FALSE)
//        //     xl <- backsolve(t(T),grad_R[[l]]%*%x, upper.tri=FALSE)
//        //
//        //     hess_A <- - (t(xk) %*% H %*% xl) / sigma2_hat
//        //     hess_B <- (t(x) %*% ( -Dkl+2*aux ) %*% x) / sigma2_hat
//        //     hess_C <- - grad_A[k] * grad_A[l] / n
//        //     hess_D <- - sum(diag( Rinv %*% aux ))
//        //     hess_E <- sum(diag( Rinv %*% Dkl ))
//        //
//        //     hess_log_vrais[k,l] <- 2*hess_A + hess_B + hess_C + hess_D + hess_E
//        //     hess_log_vrais[l,k] <- hess_log_vrais[k,l]
//        //   }
//        // }
//
//        gradsR.slice(k) = gradR_k;
//
//        for (arma::uword l = 0; l <= k; l++) {
//          // t0 = tic();
//          arma::mat aux = gradsR.slice(k) * Rinv * gradsR.slice(l);
//          // t0 = toc("  aux         ", t0);
//
//          arma::mat hessR_k_l = R;
//          if (k == l) {
//            for (arma::uword i = 0; i < n; i++) {
//              hessR_k_l.at(i, i) = 0;
//              for (arma::uword j = 0; j < i; j++) {
//                double dln_k = Dln_CovNorm(Xtnorm.col(i), Xtnorm.col(j), k) / _theta(k);
//                hessR_k_l.at(j, i) *= dln_k * (dln_k - 3 / _theta(k));
//              }
//            }
//          } else {
//            for (arma::uword i = 0; i < n; i++) {
//              hessR_k_l.at(i, i) = 0;
//              for (arma::uword j = 0; j < i; j++) {
//                hessR_k_l.at(j, i) *= Dln_CovNorm(Xtnorm.col(i), Xtnorm.col(j), k) / _theta(k)
//                                      * Dln_CovNorm(Xtnorm.col(i), Xtnorm.col(j), l) / _theta(l);
//              }
//            }
//          }
//          hessR_k_l = arma::symmatu(hessR_k_l);
//          // t0 = toc("  hessR_k_l   ", t0);
//
//          arma::mat xk = solve(fd->T, gradsR.slice(k) * x, solve_opts);
//          arma::mat xl;
//          if (k == l)
//            xl = xk;
//          else
//            xl = solve(fd->T, gradsR.slice(l) * x, solve_opts);
//          // t0 = toc("  xk xl       ", t0);
//
//          // arma::cout << " hess_A:" << -xk.t() * H * xl / sigma2_hat << arma::endl;
//          // arma::cout << " hess_B:" << -x.t() * (hessR_k_l - 2*aux) * x / sigma2_hat << arma::endl;
//          // arma::cout << " hess_C:" << -terme1.at(k) * terme1.at(l) / n << arma::endl;
//          // arma::cout << " hess_D:" << -arma::trace(Rinv * aux)  << arma::endl;
//          // arma::cout << " hess_E:" << arma::trace(Rinv * hessR_k_l) << arma::endl;
//
//          (*hess_out).at(k, l)
//              = (2.0 * xk.t() * H * xl / fd->sigma2 + x.t() * (hessR_k_l - 2 * aux) * x / fd->sigma2
//                 + terme1.at(k) * terme1.at(l) / n + arma::trace(Rinv * aux) - arma::trace(Rinv * hessR_k_l))[0]
//                / 2;  // should optim there using accu & %
//
//          // arma::cout << " xk:" << xk << arma::endl;
//          // arma::cout << " xl:" << xl << arma::endl;
//          // arma::cout << " aux:" << aux << arma::endl;
//          // arma::cout << " hessR_k_l:" << hessR_k_l << arma::endl;
//        }  // for (arma::uword l = 0; l <= k; l++)
//        // t0 = toc("  hess_out    ", t0);
//        *hess_out = arma::symmatl(*hess_out);
//        // t0 = toc("  hess_out_sym", t0);
//      }  // if (hess_out != nullptr)
//    }    // for (arma::uword k = 0; k < m_X.n_cols; k++)
//    // arma::cout << " grad_out:" << *grad_out << arma::endl;
//    // if (hess_out != nullptr) {
//    //   arma::cout << " hess_out:" << *hess_out << arma::endl;
//    //}
//  }  // if (grad_out != nullptr)
//
//  // Trace & check calcs
//  // arma::cout << " theta:" << _theta << arma::endl;
//  // arma::cout << " ll:" << ll << arma::endl;
//  // arma::cout << " sigma2:" << sigma2_hat << arma::endl;
//  // arma::cout << " T:" << fd->T << arma::endl;
//  // arma::cout << " z:" << fd->z << arma::endl;
//  // arma::cout << " H:" << H << arma::endl;
//
//  // if (grad_out != nullptr) {
//  //  arma::cout << "[DEBUG]    logLikelihood: " << _theta << arma::endl;
//  //  arma::cout << "[DEBUG]      sigma2_hat: " << sigma2_hat << arma::endl;
//  //  arma::cout << "[DEBUG]      ll: " << ll << arma::endl;
//  //
//  //  // numerical diff :
//  //  double eps = 0.0001;
//  //  arma::vec g = arma::vec(m_X.n_cols);
//  //  for (arma::uword i = 0; i < m_X.n_cols; i++) {
//  //    arma::vec eps_i = arma::zeros(m_X.n_cols);
//  //    eps_i.at(i) = eps;
//  //
//  //    arma::mat T;
//  //    arma::mat M;
//  //    arma::colvec z;
//  //    arma::colvec beta;
//  //    Kriging::OKModel okm_data_eps{T, M, z, beta};
//  //    arma::cout << "[DEBUG]    <<<<<<<<<<<<<<<<<<<<<<<<<<<" << arma::endl;
//  //    double ll_eps = logLikelihood(_theta + eps_i, nullptr, nullptr, &okm_data_eps);
//  //    g.at(i) = (ll_eps - ll) / eps;
//  //    arma::cout << "[DEBUG]    >>>>>>>>>>>>>>>>>>>>>>>>>>>" << arma::endl;
//  //  }
//  //  arma::cout << "[DEBUG]      Grad (num.): " << g << arma::endl;
//  //  arma::cout << "[DEBUG]      Grad: " << *grad_out << arma::endl;
//  //
//  //  if (hess_out != nullptr) {
//  //    // numerical diff :
//  //    double eps = 0.0001;
//  //    arma::mat h = arma::mat(m_X.n_cols, m_X.n_cols);
//  //    for (arma::uword i = 0; i < m_X.n_cols; i++) {
//  //      arma::vec eps_i = arma::zeros(m_X.n_cols);
//  //      eps_i.at(i) = eps;
//  //
//  //      arma::mat T;
//  //      arma::mat M;
//  //      arma::colvec z;
//  //      arma::colvec beta;
//  //      Kriging::OKModel okm_data_eps2{T, M, z, beta};
//  //      arma::vec grad_eps(_theta.n_elem);
//  //      arma::cout << "[DEBUG]    <<<<<<<<<<<<<<<<<<<<<<<<<<<" << arma::endl;
//  //      double ll_eps = logLikelihood(_theta + eps_i, &grad_eps, nullptr, &okm_data_eps2);
//  //      h.col(i) = (grad_eps - *grad_out) / eps;
//  //      arma::cout << "[DEBUG]    >>>>>>>>>>>>>>>>>>>>>>>>>>>" << arma::endl;
//  //    }
//  //    arma::cout << "[DEBUG]      Hess (num.): " << ((h + h.t()) / 2) << arma::endl;
//  //    arma::cout << "[DEBUG]      Hess: " << *hess_out << arma::endl;
//  //  }
//  //  arma::cout << arma::endl;
//  //}
//  return ll;
  return 0;
}

LIBKRIGING_EXPORT double Kriging::logLikelihoodFun(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  double sigma2;
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};
  
  return logLikelihood(_theta, nullptr, nullptr, &okm_data);
}

LIBKRIGING_EXPORT arma::vec Kriging::logLikelihoodGrad(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  Kriging::OKModel okm_data{T, M, z, beta};
  
  arma::vec grad; // (_theta.n_elem);
//  double ll = logLikelihood(_theta, &grad, nullptr, &okm_data);
  return grad;
}

LIBKRIGING_EXPORT arma::mat Kriging::logLikelihoodHess(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  double sigma2;
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};
  
  arma::vec grad(_theta.n_elem);
  arma::mat hess(_theta.n_elem, _theta.n_elem);
  double ll = logLikelihood(_theta, &grad, &hess, &okm_data);

  return hess;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec, arma::mat> Kriging::logLikelihoodEval(const arma::vec& _theta, const bool _grad, const bool _hess) {
  arma::mat T;
  arma::mat M;
  arma::mat z;
  arma::colvec beta;
  double sigma2;
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};
  
  double ll = -1;
  arma::vec grad;
  arma::mat hess;
  if (_grad || _hess) {
    grad = arma::vec(_theta.n_elem);
    if (!_hess) {
      ll = logLikelihood(_theta, &grad, nullptr, &okm_data);
    } else {
      hess = arma::mat(_theta.n_elem, _theta.n_elem); 
      ll = logLikelihood(_theta, &grad, &hess, &okm_data);
    }
  } else 
    ll = logLikelihood(_theta, nullptr, nullptr, &okm_data);
  
  return std::make_tuple(std::move(ll), std::move(grad), std::move(hess));
}

// Utility function for LOO
arma::colvec DiagABA(const arma::mat& A, const arma::mat& B) {
  arma::mat D = trimatu(2 * B);
  D.diag() = B.diag();
  D = (A * D) % A;
  arma::colvec c = sum(D, 1);

  return c;
}

// Objective function for fit : -LOO
double Kriging::leaveOneOut(const arma::vec& _theta, arma::vec* grad_out, Kriging::OKModel* okm_data) const {
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

  // auto t0 = tic();
  arma::mat Xtnorm = trans(m_X);
  Xtnorm.each_col() /= _theta;
  // t0 = toc("Xtnorm        ", t0);

  arma::uword n = m_X.n_rows;

  // Allocate the matrix // arma::mat R = Cov(fd->X, _theta);
  // Should be replaced by for_each
  arma::mat R = arma::ones(n, n);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < i; j++) {
      R.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xtnorm.col(j));
    }
  }
  R = arma::symmatl(R);  // (R + trans(R))/2;
  // t0 = toc("R             ", t0);

  // Cholesky decompostion of covariance matrix
  fd->T = trans(chol(R));
  // t0 = toc("T             ", t0);

  // Compute intermediate useful matrices
  fd->M = solve(trimatl(fd->T), m_F, arma::solve_opts::fast);
  // t0 = toc("M             ", t0);
  arma::mat Rinv = inv_sympd(R);  // didn't find efficient chol2inv equivalent in armadillo
  // t0 = toc("Rinv          ", t0);
  arma::mat RinvF = Rinv * m_F;
  // t0 = toc("RinvF         ", t0);
  arma::mat TM = chol(trans(fd->M) * fd->M);  // Can be optimized with a crossprod equivalent in armadillo ?
  // arma::mat aux = solve(trans(TM), trans(RinvF));
  // t0 = toc("TM            ", t0);
  arma::mat aux = solve(trimatl(trans(TM)), trans(RinvF), arma::solve_opts::fast);
  // t0 = toc("aux           ", t0);
  arma::mat Q = Rinv - trans(aux) * aux;  // Can be optimized with a crossprod equivalent in armadillo ?
  // t0 = toc("Q             ", t0);
  arma::mat Qy = Q * m_y;
  // t0 = toc("Qy            ", t0);

  arma::colvec sigma2LOO = 1 / Q.diag();
  // t0 = toc("sigma2LOO     ", t0);

  arma::colvec errorsLOO = sigma2LOO % Qy;
  // t0 = toc("errorsLOO     ", t0);

  // TODO:
  //if (fd->beta == nullptr)
  //  fd->beta = ???
  
  double loo = arma::accu(errorsLOO % errorsLOO) / n;

  if (fd->estim_sigma2) // means no sigma2 provided
    fd->sigma2 = arma::mean(errorsLOO % errorsLOO % Q.diag());
  
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

    for (arma::uword k = 0; k < m_X.n_cols; k++) {
      arma::mat gradR_k_upper = R;
      for (arma::uword i = 0; i < n; i++) {
        gradR_k_upper.at(i, i) = 0;
        for (arma::uword j = 0; j < i; j++) {
          gradR_k_upper.at(j, i) *= Dln_CovNorm(Xtnorm.col(i), Xtnorm.col(j), k);
        }
      }
      gradR_k_upper /= _theta(k);
      // t0 = toc(" gradR_k_upper", t0);
      arma::mat gradR_k = symmatu(gradR_k_upper);
      // t0 = toc(" gradR_k      ", t0);

      arma::colvec diagdQ = -DiagABA(Q, gradR_k);
      // t0 = toc(" diagdQ       ", t0);
      arma::colvec dsigma2LOO = -sigma2LOO % sigma2LOO % diagdQ;
      // t0 = toc(" dsigma2LOO   ", t0);
      arma::colvec derrorsLOO = dsigma2LOO % Qy - sigma2LOO % (Q * (gradR_k * Qy));
      // t0 = toc(" derrorsLOO   ", t0);
      (*grad_out)(k) = 2 * dot(errorsLOO, derrorsLOO) / n;
      // t0 = toc(" grad_out      ", t0);
    }
    // arma::cout << "Grad: " << *grad_out <<  arma::endl;
  }
  return loo;
}

LIBKRIGING_EXPORT double Kriging::leaveOneOutFun(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2;
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};
  
  return leaveOneOut(_theta, nullptr, &okm_data);
}

LIBKRIGING_EXPORT arma::vec Kriging::leaveOneOutGrad(const arma::vec& _theta) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2;
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};
  
  arma::vec grad(_theta.n_elem);

  double loo = leaveOneOut(_theta, &grad, &okm_data);

  return grad;
}

LIBKRIGING_EXPORT std::tuple<double, arma::vec> Kriging::leaveOneOutEval(const arma::vec& _theta, const bool _grad) {
  arma::mat T;
  arma::mat M;
  arma::colvec z;
  arma::colvec beta;
  double sigma2;
  Kriging::OKModel okm_data{T, M, z, beta, true, sigma2, true};
  
  double loo = -1;
  arma::vec grad;
  if (_grad) {
    grad = arma::vec(_theta.n_elem);
    loo = leaveOneOut(_theta, &grad, &okm_data);
  } else 
    loo = leaveOneOut(_theta, nullptr, &okm_data);
  
  return std::make_tuple(std::move(loo), std::move(grad));
}

double optim_newton(std::function<double(arma::vec& x, arma::vec* grad_out, arma::mat* hess_out)> f,
                    arma::vec& x_0,
                    const arma::vec& x_lower,
                    const arma::vec& x_upper) {
  // arma::cout << "x_0: " << x_0 << " ";

  double delta = 0.1;
  double x_tol = 0.001;
  double y_tol = 1;
  double g_tol = 0.01;
  int max_iteration = 10;

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
    // arma::cout << "." ;
    // arma::cout << "iteration: " << i << arma::endl;
    // arma::cout << "  x: " << x << arma::endl;

    f_previous = f_new;
    f_new = f(x, &grad, &hess);
    // arma::cout << "    f_new: " << f_new << arma::endl;
    if (f_best > f_new) {
      f_best = f_new;
      x_best = x;
    }

    if (std::abs(f_new - f_previous) < y_tol) {
      // arma::cout << "  X f_new ~ f_previous" << arma::endl;
      break;
    }
    if (arma::abs(grad).max() < g_tol) {
      // arma::cout << "  X grad ~ 0" << arma::endl;
      break;
    }

    arma::vec delta_x(x.n_elem);
    // arma::cout << "  eig(hess)" << arma::eig_sym(hess) << arma::endl;
    if (arma::all(arma::eig_sym(hess) > 0)) {
      // try to fit a second order polynom to use its minimizer. Otherwise, just iterate with conjugate gradient
      // arma::cout << "!";
      delta_x = arma::solve(hess, grad, arma::solve_opts::likely_sympd);
    } else {
      delta_x = delta * grad / std::sqrt(arma::sum(arma::cdot(grad, grad)));
    }
    // arma::cout << "  delta_x: " << delta_x << arma::endl;

    arma::vec x_next = x - delta_x;
    // arma::cout << "  x_next: " << x_next << arma::endl;

    for (int j = 0; j < x_next.n_elem; j++) {
      if (x_next[j] < x_lower[j]) {
        // arma::cout << "    <" << x_lower[j] << arma::endl;
        delta_x = delta_x * (x[j] - x_lower[j]) / (x[j] - x_next[j]) / 2;
        x_next = x - delta_x;
      }
      if (x_next[j] > x_upper[j]) {
        // arma::cout << "    >" << x_upper[j] << arma::endl;
        delta_x = delta_x * (x_upper[j] - x[j]) / (x_next[j] - x[j]) / 2;
        x_next = x - delta_x;
      }
    }
    // arma::cout << "    delta_x: " << delta << arma::endl;
    // arma::cout << "    x_next: " << x_next << arma::endl;

    if (arma::abs(x - x_next).max() < x_tol) {
      // arma::cout << "  X x_0 ~ x_next" << arma::endl;
      break;
    }

    x_previous = x;
    x = x_next;

    // TODO : keep best result instead of last one
    ++i;
    // arma::cout << "  f_best: " << f_best << arma::endl;
  }

  x_0 = x_best;
  // arma::cout << " " << x_0 << " " << f_best << arma::endl;

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
                                    const RegressionModel& regmodel,
                                    bool normalize,
                                    const std::string& optim,
                                    const std::string& objective,
                                    const Parameters& parameters) {
  std::function<double(const arma::vec& _theta, arma::vec* grad_out, arma::mat* hess_out, Kriging::OKModel* okm_data)>
      fit_ofn;
  m_optim = optim;
  m_objective = objective;
  if (objective.compare("LL") == 0) {
    fit_ofn = [this](const arma::vec& _theta, arma::vec* grad_out, arma::mat* hess_out, Kriging::OKModel* okm_data) {
      double ll = this->logLikelihood(_theta, grad_out, hess_out, okm_data);
      if (grad_out != nullptr)
        *grad_out = -*grad_out;
      if (hess_out != nullptr)
        *hess_out = -*hess_out;
      return -ll;
    };
  } else if (objective.compare("LOO") == 0) {
    fit_ofn = [this](const arma::vec& _theta, arma::vec* grad_out, arma::mat* hess_out, Kriging::OKModel* okm_data) {
      return this->leaveOneOut(_theta, grad_out, okm_data);
    };
  } else
    throw std::invalid_argument("Unsupported fit objective: " + objective);

  arma::uword n = X.n_rows;
  arma::uword d = X.n_cols;
  arma::rowvec centerX(d);
  arma::rowvec scaleX(d);
  double centerY;
  double scaleY;
  // Normalization of inputs and output
  if (normalize) {
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

  // Define regression matrix
  m_regmodel = regmodel;
  m_F = regressionModelMatrix(regmodel, m_X, n, d);

  arma::mat theta0 = parameters.theta;
  if (parameters.has_theta) {
    if (parameters.theta.n_cols != m_X.n_cols && parameters.theta.n_rows == m_X.n_cols)
      theta0 = parameters.theta.t();
    if (theta0.n_cols != m_X.n_cols)
      throw std::runtime_error("Dimension of theta should be nx" + std::to_string(m_X.n_cols) + " instead of "
                               + std::to_string(theta0.n_rows) + "x" + std::to_string(theta0.n_cols));
  }

  if (optim == "none") {  // just keep given theta, no optimisation of ll
    if (!parameters.has_theta)
      throw std::runtime_error("Theta should be given (1x" + std::to_string(m_X.n_cols) + ") matrix, when optim=none");
    m_theta = trans(theta0.row(0));
    m_est_theta = false;
    arma::mat T;
    arma::mat M;
    arma::mat z;
    arma::colvec beta;
    if (parameters.has_beta) beta = parameters.beta;
    double sigma2 = -1;
    if (parameters.has_sigma2) sigma2 = parameters.sigma2;
    Kriging::OKModel okm_data{T, M, z, beta, !parameters.has_beta, sigma2, !parameters.has_sigma2};
    double minus_ll_tmp = fit_ofn(m_theta, nullptr, nullptr, &okm_data);
    m_T = std::move(okm_data.T);
    m_M = std::move(okm_data.M);
    m_z = std::move(okm_data.z);
    m_beta = std::move(okm_data.beta);
    m_est_beta = !parameters.has_beta;
    m_sigma2 = std::move(okm_data.sigma2);
    m_est_sigma2 = !parameters.has_sigma2;
  } else if (optim.rfind("BFGS", 0) == 0) {
    // FIXME parameters.has needs to implemtented (no use case in current code)
    if (!parameters.has_theta) {      // no theta given, so draw 10 random uniform starting values
      int multistart = 1;             // TODO? stoi(substr(optim_method,)) to hold 'bfgs10' as a 10 multistart bfgs
      arma::arma_rng::set_seed(123);  // FIXME arbitrary seed for reproducible random sequences
      theta0 = arma::randu(multistart, X.n_cols) % (max(m_X, 0) - min(m_X, 0));
    } else {  // just use given theta(s) as starting values for multi-bfgs
      theta0 = arma::mat(parameters.theta);
    }

    // arma::cout << "theta0:" << theta0 << arma::endl;

    optim::algo_settings_t algo_settings;
    algo_settings.print_level = 0;
    algo_settings.iter_max = 10;  // TODO change by default?
    algo_settings.rel_sol_change_tol = 0.1;
    algo_settings.grad_err_tol = 1e-8;
    algo_settings.vals_bound = true;
    algo_settings.lower_bounds = 1e-10 * arma::ones<arma::vec>(X.n_cols);
    algo_settings.upper_bounds = 2 * trans(max(m_X, 0) - min(m_X, 0));
    double minus_ll = std::numeric_limits<double>::infinity();
    for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
      arma::colvec theta_tmp = trans(theta0.row(i));
      arma::mat T;
      arma::mat M;
      arma::mat z;
      arma::colvec beta;
      if (parameters.has_beta) beta = parameters.beta;
      double sigma2 = -1;
      if (parameters.has_sigma2) sigma2 = parameters.sigma2;
      Kriging::OKModel okm_data{T, M, z, beta, !parameters.has_beta, sigma2, !parameters.has_sigma2};
      bool bfgs_ok = optim::lbfgs(
          theta_tmp,
          [&okm_data, this, fit_ofn](const arma::vec& vals_inp, arma::vec* grad_out, void*) -> double {
            return fit_ofn(vals_inp, grad_out, nullptr, &okm_data);
          },
          nullptr,
          algo_settings);

      // if (bfgs_ok) { // FIXME always succeeds ?
      double minus_ll_tmp
          = fit_ofn(theta_tmp,
                    nullptr,
                    nullptr,
                    &okm_data);  // this last call also ensure that T and z are up-to-date with solution found.
      if (minus_ll_tmp < minus_ll) {
        m_theta = std::move(theta_tmp);
        m_est_theta = true;
        minus_ll = minus_ll_tmp;
        m_T = std::move(okm_data.T);
        m_M = std::move(okm_data.M);
        m_z = std::move(okm_data.z);
        m_beta = std::move(okm_data.beta);
        m_est_beta = !parameters.has_beta;
        m_sigma2 = std::move(okm_data.sigma2);
        m_est_sigma2 = !parameters.has_sigma2;
      }
      // }
    }
  } else if (optim.rfind("Newton", 0) == 0) {
    // FIXME parameters.has needs to implemtented (no use case in current code)
    if (!parameters.has_theta) {      // no theta given, so draw 10 random uniform starting values
      int multistart = 1;             // TODO? stoi(substr(optim_method,)) to hold 'bfgs10' as a 10 multistart bfgs
      arma::arma_rng::set_seed(123);  // FIXME arbitrary seed for reproducible random sequences
      theta0 = arma::randu(multistart, X.n_cols) % (max(m_X, 0) - min(m_X, 0));
    } else {  // just use given theta(s) as starting values for multi-bfgs
      theta0 = arma::mat(parameters.theta);
    }

    // arma::cout << "theta0:" << theta0 << arma::endl;

    double minus_ll = std::numeric_limits<double>::infinity();
    for (arma::uword i = 0; i < theta0.n_rows; i++) {  // TODO: use some foreach/pragma to let OpenMP work.
      arma::colvec theta_tmp = trans(theta0.row(i));
      arma::mat T;
      arma::mat M;
      arma::mat z;
      arma::colvec beta;
      if (parameters.has_beta) beta = parameters.beta;
      double sigma2 = -1;
      if (parameters.has_sigma2) sigma2 = parameters.sigma2;
      Kriging::OKModel okm_data{T, M, z, beta, !parameters.has_beta, sigma2, !parameters.has_sigma2};
      double minus_ll_tmp = optim_newton(
          [&okm_data, this, fit_ofn](const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out) -> double {
            return fit_ofn(vals_inp, grad_out, hess_out, &okm_data);
          },
          theta_tmp,
          1e-10 * arma::ones<arma::vec>(X.n_cols),
          2 * trans(max(m_X, 0) - min(m_X, 0)));

      if (minus_ll_tmp < minus_ll) {
        m_theta = std::move(theta_tmp);
        m_est_theta = true;
        minus_ll = minus_ll_tmp;
        m_T = std::move(okm_data.T);
        m_M = std::move(okm_data.M);
        m_z = std::move(okm_data.z);
        m_beta = std::move(okm_data.beta);
        m_est_beta = !parameters.has_beta;
        m_sigma2 = std::move(okm_data.sigma2);
        m_est_sigma2 = !parameters.has_sigma2;
      }
    }
  } else
    throw std::runtime_error("Not a suitable optim: " + optim);

  if (!parameters.has_sigma2)
    m_sigma2 *= scaleY * scaleY;
  
  // arma::cout << "theta:" << m_theta << arma::endl;
}

/** Compute the prediction for given points X'
 * @param Xp is m*d matrix of points where to predict output
 * @param std is true if return also stdev column vector
 * @param cov is true if return also cov matrix between Xp
 * @return output prediction: m means, [m standard deviations], [m*m full covariance matrix]
 */
LIBKRIGING_EXPORT std::tuple<arma::colvec, arma::colvec, arma::mat> Kriging::predict(const arma::mat& Xp,
                                                                                     bool withStd,
                                                                                     bool withCov) {
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;
  arma::colvec pred_mean(m);
  arma::colvec pred_stdev(m);
  arma::mat pred_cov(m, m);
  pred_stdev.zeros();
  pred_cov.zeros();

  arma::mat Xtnorm = trans(m_X);
  Xtnorm.each_col() /= m_theta;
  arma::mat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::uword d = m_X.n_cols;
  arma::mat Ftest = regressionModelMatrix(m_regmodel, Xpnorm, m, d);

  // Compute covariance between training data and new data to predict
  arma::mat R(n, m);
  Xpnorm = trans(Xpnorm);
  Xpnorm.each_col() /= m_theta;
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      R.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xpnorm.col(j));
    }
  }
  arma::mat Tinv_newdata = solve(trimatl(m_T), R, arma::solve_opts::fast);
  pred_mean = Ftest * m_beta + trans(Tinv_newdata) * m_z;
  // Un-normalize predictor
  pred_mean = m_centerY + m_scaleY * pred_mean;

  if (withStd) {
    double total_sd2 = m_sigma2;
    // s2.predict.1 <- apply(Tinv.c.newdata, 2, crossprod)
    arma::colvec s2_predict_1 = m_sigma2 * trans(sum(Tinv_newdata % Tinv_newdata, 0));
    // Type = "UK"
    // T.M <- chol(t(M)%*%M)
    arma::mat TM = trans(chol(trans(m_M) * m_M));
    // s2.predict.mat <- backsolve(t(T.M), t(F.newdata - t(Tinv.c.newdata)%*%M) , upper.tri = FALSE)
    arma::mat s2_predict_mat = solve(trimatl(TM), trans(Ftest - trans(Tinv_newdata) * m_M), arma::solve_opts::fast);
    // s2.predict.2 <- apply(s2.predict.mat, 2, crossprod)
    arma::colvec s2_predict_2 = m_sigma2 * trans(sum(s2_predict_mat % s2_predict_mat, 0));
    // s2.predict <- pmax(total.sd2 - s2.predict.1 + s2.predict.2, 0)
    arma::mat s2_predict = total_sd2 - s2_predict_1 + s2_predict_2;
    s2_predict.elem(find(pred_stdev < 0)).zeros();
    pred_stdev = sqrt(s2_predict);
    if (withCov) {
      // C.newdata <- covMatrix(object@covariance, newdata)[[1]]
      arma::mat C_newdata(m, m);
      for (arma::uword i = 0; i < m; i++) {
        for (arma::uword j = 0; j < m; j++) {
          C_newdata.at(i, j) = CovNorm_fun(Xpnorm.col(i), Xpnorm.col(j));
        }
      }
      // cond.cov <- C.newdata - crossprod(Tinv.c.newdata)
      // cond.cov <- cond.cov + crossprod(s2.predict.mat)
      pred_cov = m_sigma2 * (C_newdata - trans(Tinv_newdata) * Tinv_newdata + trans(s2_predict_mat) * s2_predict_mat);
    }
  } else if (withCov) {
    arma::mat C_newdata(m, m);
    for (arma::uword i = 0; i < m; i++) {
      for (arma::uword j = 0; j < m; j++) {
        C_newdata.at(i, j) = CovNorm_fun(Xpnorm.col(i), Xpnorm.col(j));
      }
    }
    // Need to compute matrices computed in withStd case
    arma::mat TM = trans(chol(trans(m_M) * m_M));
    arma::mat s2_predict_mat = solve(trimatl(TM), trans(Ftest - trans(Tinv_newdata) * m_M), arma::solve_opts::fast);
    pred_cov = m_sigma2 * (C_newdata - trans(Tinv_newdata) * Tinv_newdata + trans(s2_predict_mat) * s2_predict_mat);
  }

  return std::make_tuple(std::move(pred_mean), std::move(pred_stdev), std::move(pred_cov));
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
  // Here nugget.sim = 1e-10 to avoid chol failures of Sigma_cond)
  double nugget_sim = 1e-10;
  arma::uword m = Xp.n_rows;
  arma::uword n = m_X.n_rows;

  arma::mat Xpnorm = Xp;
  // Normalize Xp
  Xpnorm.each_row() -= m_centerX;
  Xpnorm.each_row() /= m_scaleX;

  // Define regression matrix
  arma::uword d = m_X.n_cols;
  arma::mat F_newdata = regressionModelMatrix(m_regmodel, Xpnorm, m, d);

  // auto t0 = tic();
  arma::colvec y_trend = F_newdata * m_beta;// / std::sqrt(m_sigma2);
  // t0 = toc("y_trend        ", t0);

  Xpnorm = trans(Xpnorm);
  Xpnorm.each_col() /= m_theta;
  // t0 = toc("Xpnorm         ", t0);

  // Compute covariance between new data
  arma::mat Sigma = arma::ones(m, m);
  for (arma::uword i = 0; i < m; i++) {
    for (arma::uword j = 0; j < i; j++) {
      Sigma.at(i, j) = CovNorm_fun(Xpnorm.col(i), Xpnorm.col(j));
    }
  }
  Sigma = arma::symmatl(Sigma);  // R + trans(R);
  // t0 = toc("Sigma          ", t0);

  // arma::mat T_newdata = chol(Sigma);
  // Compute covariance between training data and new data to predict
  // Sigma21 <- covMat1Mat2(object@covariance, X1 = object@X, X2 = newdata, nugget.flag = FALSE)
  arma::mat Xtnorm = trans(m_X);
  Xtnorm.each_col() /= m_theta;
  arma::mat Sigma21(n, m);
  for (arma::uword i = 0; i < n; i++) {
    for (arma::uword j = 0; j < m; j++) {
      Sigma21.at(i, j) = CovNorm_fun(Xtnorm.col(i), Xpnorm.col(j));
    }
  }
  // t0 = toc("Sigma21        ", t0);

  // Tinv.Sigma21 <- backsolve(t(object@T), Sigma21, upper.tri = FALSE
  arma::mat Tinv_Sigma21 = solve(trimatl(m_T), Sigma21, solve_opts);
  // t0 = toc("Tinv_Sigma21   ", t0);

  // y.trend.cond <- y.trend + t(Tinv.Sigma21) %*% object@z
  y_trend += trans(Tinv_Sigma21) * m_z;
  // t0 = toc("y_trend        ", t0);

  // Sigma.cond <- Sigma11 - t(Tinv.Sigma21) %*% Tinv.Sigma21
  // arma::mat Sigma_cond = Sigma - XtX(Tinv_Sigma21);
  // arma::mat Sigma_cond = Sigma - trans(Tinv_Sigma21) * Tinv_Sigma21;
  arma::mat Sigma_cond = trimatl(Sigma);
  for (arma::uword i = 0; i < Tinv_Sigma21.n_cols; i++) {
    for (arma::uword j = 0; j <= i; j++) {
      for (arma::uword k = 0; k < Tinv_Sigma21.n_rows; k++) {
        Sigma_cond.at(i, j) -= Tinv_Sigma21.at(k, i) * Tinv_Sigma21.at(k, j);
      }
    }
  }
  Sigma_cond = arma::symmatl(Sigma_cond);  // R + trans(R);
  // t0 = toc("Sigma_cond     ", t0);

  // T.cond <- chol(Sigma.cond + diag(nugget.sim, m, m))
  Sigma_cond.diag() += nugget_sim;
  arma::mat tT_cond = chol(Sigma_cond, "lower");
  // t0 = toc("T_cond         ", t0);

  // white.noise <- matrix(rnorm(m*nsim), m, nsim)
  // y.rand.cond <- t(T.cond) %*% white.noise
  // y <- matrix(y.trend.cond, m, nsim) + y.rand.cond
  arma::mat yp(m, nsim);
  yp.each_col() = y_trend;
  
  arma::arma_rng::set_seed(seed); 
  yp += tT_cond * arma::randn(m, nsim) * std::sqrt(m_sigma2);
  // t0 = toc("yp             ", t0);

  // Un-normalize simulations
  yp = m_centerY + m_scaleY * yp;// * std::sqrt(m_sigma2);
  // t0 = toc("yp             ", t0);

  return yp;
}

/** Add new conditional data points to previous (X,y)
 * @param newy is m length column vector of new output
 * @param newX is m*d matrix of new input
 * @param optim_method is an optimizer name from OptimLib, or 'none' to keep previously estimated parameters unchanged
 * @param optim_objective is 'loo' or 'loglik'. Ignored if optim_method=='none'.
 */
LIBKRIGING_EXPORT void Kriging::update(const arma::vec& newy, const arma::mat& newX, bool normalize = false) {
  // rebuild data
  m_X = join_cols(m_X, newX);
  m_y = join_cols(m_y, newy);

  // rebuild starting parameters
  Parameters parameters{this->m_sigma2, false, trans(this->m_theta), true};
  // re-fit
  // TODO refit() method which will use Shurr forms to fast update matrix (R, ...)
  this->fit(m_y, m_X, m_regmodel, normalize, m_optim, m_objective, parameters);
}

/************************************************/
/**          implementation details            **/
/************************************************/

namespace {  // anonymous namespace for local implementation details

auto regressionModelMatrix(const Kriging::RegressionModel& regmodel,
                           const arma::mat& newX,
                           arma::uword n,
                           arma::uword d) -> arma::mat {
  arma::mat F;  // uses modern RTO to avoid returned object copy
  switch (regmodel) {
    case Kriging::RegressionModel::Constant: {
      F.set_size(n, 1);
      F = arma::ones(n, 1);
      return F;
    } break;

    case Kriging::RegressionModel::Linear: {
      F.set_size(n, 1 + d);
      F.col(0) = arma::ones(n, 1);
      for (arma::uword i = 0; i < d; i++) {
        F.col(i + 1) = newX.col(i);
      }
      return F;
    } break;
      
  case Kriging::RegressionModel::Interactive: {
    F.set_size(n, 1 + d + d * (d - 1) / 2);
    F.col(0) = arma::ones(n, 1);
    arma::uword count = 1;
    for (arma::uword i = 0; i < d; i++) {
      F.col(count) = newX.col(i);
      count += 1;
      for (arma::uword j = 0; j < i; j++) {
        F.col(count) = newX.col(i) % newX.col(j);
        count += 1;
      }
    }
    return F;
  } break;
    
  case Kriging::RegressionModel::Quadratic: {
    F.set_size(n, 1 + 2 * d + d * (d - 1) / 2);
    F.col(0) = arma::ones(n, 1);
    arma::uword count = 1;
    for (arma::uword i = 0; i < d; i++) {
      F.col(count) = newX.col(i);
      count += 1;
      for (arma::uword j = 0; j <= i; j++) {
        F.col(count) = newX.col(i) % newX.col(j);
        count += 1;
      }
    }
    return F;
  } break;
  }
}

static char const* enum_RegressionModel_strings[] = {"constant", "linear", "interactive", "quadratic"};

}  // namespace

Kriging::RegressionModel Kriging::RegressionModelUtils::fromString(const std::string& value) {
  static auto begin = std::begin(enum_RegressionModel_strings);
  static auto end = std::end(enum_RegressionModel_strings);

  auto find = std::find(begin, end, value);
  if (find != end) {
    return static_cast<RegressionModel>(std::distance(begin, find));
  } else {
    // FIXME use std::optional as returned type
    throw std::exception();
  }
}

std::string Kriging::RegressionModelUtils::toString(const Kriging::RegressionModel& e) {
  assert(static_cast<std::size_t>(e) < sizeof(enum_RegressionModel_strings));
  return enum_RegressionModel_strings[static_cast<int>(e)];
}
