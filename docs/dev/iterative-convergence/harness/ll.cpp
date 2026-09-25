#include "libKriging/utils/lk_armadillo.hpp"
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#include <cstdio>
#include <cstdlib>
static double f2d(double x,double y){return std::sin(3.0*x)+std::cos(5.0*y)+x*y;}
int main(int argc,char**argv){
  arma::uword n=160; arma::arma_rng::set_seed(123); arma::mat X(n,2,arma::fill::randu); arma::vec y(n);
  for(arma::uword i=0;i<n;++i) y(i)=f2d(X(i,0),X(i,1));
  std::string obj=std::string("LLIterative(30,0,24,")+argv[1]+",1e-10)";
  LinearAlgebra::set_cg_warning(false);
  const arma::vec theta{0.25,0.3}; double ll[2]; arma::vec g[2];
  for(int d=0;d<2;++d){ setenv("LK_ITERATIVE_DENSE_MAX_MB",d?"4096":"0",1);
    Kriging::Parameters p; p.theta=arma::mat(1,2,arma::fill::value(0.3)); p.is_theta_estim=false;
    Kriging k(y,X,"matern5_2",Trend::RegressionModel::Constant,false,"none",obj,p);
    auto r=k.logLikelihoodIterativeFun(theta,true); ll[d]=std::get<0>(r); g[d]=std::get<1>(r);
    if(d){auto e=k.logLikelihoodFun(theta,false,false); std::printf("  ll exact(Cholesky)=%.6f\n",std::get<0>(e));}
  }
  std::printf("  %s: ll_mf=%.6f ll_de=%.6f |dll|=%.2e  |dg|max=%.2e (|g|max=%.2e)\n",obj.c_str(),ll[0],ll[1],std::abs(ll[0]-ll[1]),arma::abs(g[0]-g[1]).max(),arma::abs(g[0]).max());
}
