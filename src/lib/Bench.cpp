// clang-format off
// MUST BE at the beginning before any other <cmath> include (e.g. in armadillo's headers)
#define _USE_MATH_DEFINES // required for Visual Studio
#include <cmath>
// clang-format on

#include "libKriging/Bench.hpp"
#include "libKriging/OrdinaryKriging.hpp"

#include <armadillo>
// #include <optim.hpp>
#include <ensmallen.hpp>
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
double Bench::LogLik(OrdinaryKriging& ok, const arma::vec& theta) {
  // arma::vec theta = 0.5*ones(ok->X().n_cols)
  double s = 0;
  for (int i = 0; i < n; i++) {
    s += ok.logLikelihoodFun(theta);
  }
  return s / n;
}

LIBKRIGING_EXPORT
arma::vec Bench::LogLikGrad(OrdinaryKriging& ok, const arma::vec& theta) {
  // arma::vec theta = 0.5*ones(ok->X().n_cols)
  arma::vec s = arma::zeros(theta.n_elem);
  for (int i = 0; i < n; i++) {
    s += ok.logLikelihoodGrad(theta);
  }
  return s / n;
  }


double a = .5;
double b = 50;
inline double rosenbrock_fun(arma::vec X) noexcept {
  return (a-X(0))*(a-X(0))+b*(X(1)-X(0)*X(0))*(X(1)-X(0)*X(0));
};
inline arma::vec rosenbrock_grad(arma::vec X) noexcept {
  arma::vec g = arma::zeros(2);
  g(0) = -2*(a-X(0)) + 4*b*(X(0)*X(0)*X(0) - X(1)*X(0));
  g(1) = 2*b*(X(1) - X(0)*X(0));
  return g;
};

  // double ofn_rosenbrock(const arma::vec& x,arma::vec* grad_out,void* ofn_data) {
  //   if (ofn_data) {
  //   Bench::OFNData* fd = reinterpret_cast<Bench::OFNData*>(ofn_data);
  //   if (fd->histx.n_cols != x.n_elem)
  //     fd->histx = arma::reshape(x,1,x.n_elem);
  //   else
  //     fd->histx.insert_rows(fd->histx.n_rows,trans(arma::mat(x)));
  //   }
  //   arma::cout << "x "<< x << arma::endl;
  //   if (grad_out) {
  //     arma::cout << "g";
  //     *grad_out = rosenbrock_grad(x);
  //   } else     arma::cout << "o";
  //   
  //   return rosenbrock_fun(x);
  // }
  
  class Rosenbrock_DifferentiableFunction {
  public:
    arma::mat histx;
    arma::mat histg;
    
    // Given parameters x, return the value of f(x).
    double Evaluate(const arma::mat& x) {
      arma::cout << x << arma::endl;
      if (!histx.is_empty())
        histx.insert_rows(histx.n_rows,x.t());
      else
        histx = x.t();
      double y = rosenbrock_fun(x);
      arma::cout << y << arma::endl;
      return y;
    }
    
    // Given parameters x and a matrix g, store f'(x) in the provided matrix g.
    // g should have the same size (rows, columns) as x.
    void Gradient(const arma::mat& x, arma::mat& gradient) {
      arma::cout << x << arma::endl;
      if (!histg.is_empty())
        histg.insert_rows(histg.n_rows,x.t());
      else
        histg = x.t();
      gradient = arma::mat(rosenbrock_grad(x));
      return;
    }
    
    // OPTIONAL: this may be implemented in addition to---or instead
    // of---Evaluate() and Gradient().  If this is the only function implemented,
    // implementations of Evaluate() and Gradient() will be automatically
    // generated using template metaprogramming.  Often, implementing
    // EvaluateWithGradient() can result in more efficient optimizations.
    //
    // Given parameters x and a matrix g, return the value of f(x) and store
    // f'(x) in the provided matrix g.  g should have the same size (rows,
    // columns) as x.
    //double EvaluateWithGradient(const arma::mat& x, arma::mat& g);
  };
  
  
  
  
  LIBKRIGING_EXPORT
    double Bench::Rosenbrock(arma::vec& x) {
      return rosenbrock_fun(x);
    } 
  
  LIBKRIGING_EXPORT
    arma::vec Bench::RosenbrockGrad(arma::vec& x) { 
      return rosenbrock_grad(x);
    } 
  
  LIBKRIGING_EXPORT
      arma::mat Bench::OptimRosenbrock(arma::vec& x0) {

        ens::L_BFGS lbfgs;
        lbfgs.MaxIterations() = 10;
        
        arma::mat coords = arma::mat(x0);
        Rosenbrock_DifferentiableFunction f;
        lbfgs.Optimize(f, coords);
        
        return f.histx;        

    // optim::algo_settings_t algo_settings;
    //     algo_settings.vals_bound = true;
    //     algo_settings.lower_bounds = arma::zeros<arma::vec>(2);
    //     algo_settings.upper_bounds = arma::ones<arma::vec>(2);
    //     
    // algo_settings.iter_max = 10;  // TODO change by default?
    // algo_settings.err_tol = 1e-9;
    // 
    // algo_settings.gd_method = 5;
    // algo_settings.gd_settings.step_size=0.01;
    // algo_settings.gd_settings.norm_term=1e-7;
    // algo_settings.gd_settings.ada_rho=0.9;
    // 
    // algo_settings.cg_method = 5; 
    // 
    // Bench::OFNData ofn_data; // FIXME AFTER
    // arma::cout << "> gd 5 ";
    // ofn_data.histx = arma::zeros(1,2);
    //   bool bfgs_ok = optim::gd(
    //     x0,
    //     ofn_rosenbrock,
    //     (void*)(&ofn_data),
    //     algo_settings);
    //   arma::cout << " <" << arma::endl;
    // 
    // return ofn_data.histx;
}
