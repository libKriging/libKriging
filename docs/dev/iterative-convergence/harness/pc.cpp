#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include <chrono>
#include <cstdio>
static double f2d(double x,double y){return std::sin(3.0*x)+std::cos(5.0*y)+x*y;}
int main(int argc,char**argv){
  arma::uword n=std::atoi(argv[1]); double th=std::atof(argv[2]); arma::uword rank=argc>3?std::atoi(argv[3]):50;
  LinearAlgebra::set_cg_warning(false);
  arma::arma_rng::set_seed(123); arma::mat X(n,2,arma::fill::randu); arma::vec y(n); for(arma::uword i=0;i<n;++i) y(i)=f2d(X(i,0),X(i,1));
  arma::arma_rng::set_seed(456); arma::mat Xt(50,2,arma::fill::randu);
  Kriging::Parameters p; p.theta=arma::mat(1,2,arma::fill::value(th)); p.is_theta_estim=false;
  Kriging k(y,X,"matern5_2",Trend::RegressionModel::Constant,false,"none","LL",p);
  auto [me,se,c,dm,ds]=k.predict(Xt,true,false,false);
  for(int pc=0;pc<2;++pc){
    auto t0=std::chrono::steady_clock::now();
    auto [m,s]=k.predictIterative(Xt,true,0,1e-8,pc==1,rank);
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("n=%5u th=%.2f %s  t=%7.3fs  err_mean=%.1e err_sd=%.1e\n",(unsigned)n,th,pc?"nystrom":"plain  ",dt,
      arma::abs(m-me).max()/arma::stddev(y),arma::abs(s-se).max()/arma::stddev(y));
    std::fflush(stdout);
  }
}
