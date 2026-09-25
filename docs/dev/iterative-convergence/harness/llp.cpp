#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include <chrono>
#include <cstdio>
static double f2d(double x,double y){return std::sin(3.0*x)+std::cos(5.0*y)+x*y;}
int main(int argc,char**argv){
  arma::uword n=std::atoi(argv[1]); double th=std::atof(argv[2]);
  LinearAlgebra::set_cg_warning(false);
  arma::arma_rng::set_seed(123); arma::mat X(n,2,arma::fill::randu); arma::vec y(n); for(arma::uword i=0;i<n;++i) y(i)=f2d(X(i,0),X(i,1));
  const arma::vec theta(2,arma::fill::value(th));
  Kriging::Parameters p; p.theta=arma::mat(1,2,arma::fill::value(th)); p.is_theta_estim=false;
  Kriging ke(y,X,"matern5_2",Trend::RegressionModel::Constant,false,"none","LL",p);
  auto [lle,ge]=ke.logLikelihoodFun(theta,true,false);
  for(const char* spec: {"LLIterative(30,0,24,2,1e-8)","LLIterative(30,50,24,2,1e-8)"}){
    Kriging k(y,X,"matern5_2",Trend::RegressionModel::Constant,false,"none",spec,p);
    auto t0=std::chrono::steady_clock::now();
    auto [ll,g]=k.logLikelihoodIterativeFun(theta,true);
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    std::printf("n=%u th=%.2f %-30s t=%.2fs ll=%.3f (exact %.3f, err %.2e rel)  |g-g_ex|/|g_ex|=%.2e\n",(unsigned)n,th,spec,dt,ll,lle,std::abs(ll-lle)/std::abs(lle),arma::norm(g-ge)/arma::norm(ge));
  }
}
