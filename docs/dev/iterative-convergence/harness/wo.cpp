#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include "libKriging/Optim.hpp"
#include <chrono>
#include <cstdio>
static double f2d(double x,double y){ if(std::getenv("ROUGH")) return std::sin(12.0*x)*std::cos(10.0*y)+std::sin(20.0*x*y); return std::sin(3.0*x)+std::cos(5.0*y)+x*y;}
int main(int argc,char**argv){
  std::string mode=argv[1]; arma::uword n=std::atoi(argv[2]); std::string spec=argv[3];
  LinearAlgebra::set_cg_warning(false); if(std::getenv("THUP")) Optim::set_theta_upper_factor(std::atof(std::getenv("THUP")));
  arma::arma_rng::set_seed(123); arma::mat X(n+40,2,arma::fill::randu); arma::vec y(n+40); for(arma::uword i=0;i<n+40;++i) y(i)=f2d(X(i,0),X(i,1));
  auto t0=std::chrono::steady_clock::now();
  if(mode=="fit"){
    Kriging k(y.head(n),X.head_rows(n),"matern5_2",Trend::RegressionModel::Constant,false,argv[4],spec);
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("FIT t=%.2fs theta=(%.4f,%.4f) sigma2=%.4g\n", dt, k.theta()(0), k.theta()(1), k.sigma2());
  } else {
    Kriging::Parameters p; p.theta=arma::mat(1,2,arma::fill::value(std::atof(argv[4]))); p.is_theta_estim=false;
    Kriging k(y.head(n),X.head_rows(n),"matern5_2",Trend::RegressionModel::Constant,false,"none",spec,p);
    std::fprintf(stderr,"--- update\n");
    arma::uword nu=std::atoi(argv[5]);
    t0=std::chrono::steady_clock::now();
    k.update(y.subvec(n,n+nu-1),X.rows(n,n+nu-1),false);
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("UPDATE t=%.3fs\n",dt);
  }
}
