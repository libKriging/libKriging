#include <sstream>
#include <iostream>
#include <functional>
#include <optional>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include "libKriging/utils/lk_armadillo.hpp"
#define private public
#define protected public
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#undef private
#undef protected
#include <cstdio>
static double f2d(double x,double y){return std::sin(3.0*x)+std::cos(5.0*y)+x*y;}
int main(int argc,char**argv){
  arma::uword n=std::atoi(argv[1]); double th=std::atof(argv[2]); arma::uword nu=std::atoi(argv[3]); std::string spec=argv[4];
  LinearAlgebra::set_cg_warning(false);
  arma::arma_rng::set_seed(123); arma::mat X(n+nu,2,arma::fill::randu); arma::vec y(n+nu); for(arma::uword i=0;i<n+nu;++i) y(i)=f2d(X(i,0),X(i,1));
  Kriging::Parameters p; p.theta=arma::mat(1,2,arma::fill::value(th)); p.is_theta_estim=false;
  Kriging k(y.head(n),X.head_rows(n),"matern5_2",Trend::RegressionModel::Constant,false,"none",spec,p);
  k.update(y.tail(nu),X.tail_rows(nu),false);
  arma::uword N=k.m_X.n_rows; arma::mat R(N,N); LinearAlgebra::covMat_sym_X(&R,k.m_X.t(),k.m_theta,k._Cov,1.0); R.diag().ones();
  arma::mat FY=arma::join_rows(k.m_F,k.m_y); arma::mat ex=arma::solve(R,FY,arma::solve_opts::likely_sympd);
  const arma::mat& c=k.m_iterative_RinvFY_cache;
  // erreur en norme A (énergie), pertinente pour CG
  arma::mat e=c-ex; double eA=std::sqrt(arma::accu(e%(R*e))), xA=std::sqrt(arma::accu(ex%(R*ex)));
  std::printf("relerr=%.2e  Anorm_relerr=%.2e  relres=%.2e\n",arma::norm(e)/arma::norm(ex),eA/xA,arma::norm(R*c-FY)/arma::norm(FY));
}
