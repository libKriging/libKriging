#include <sstream>
#include <iostream>
#include <functional>
#include <optional>
#include <map>
#include <vector>
#include <memory>
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
// variante: 0=pas de reprise, 1=reprise complète (p=r, comme la lib), 2=remplacement de résidu (p conservé)
static arma::vec cg(const arma::mat& A,const arma::vec& b,const arma::vec& x0,int maxit,int variant,const arma::vec& xex,std::vector<int> marks){
  arma::vec x=x0, r=b-A*x, p=r; double rz=arma::dot(r,r); const double bn=arma::norm(b);
  size_t mi=0;
  for(int it=1;it<=maxit;++it){
    arma::vec Ap=A*p; double a=rz/arma::dot(p,Ap); x+=a*p;
    bool rs=(variant>0)&&(it%50==0);
    if(rs) r=b-A*x; else r-=a*Ap;
    double rzn=arma::dot(r,r);
    if(variant==1&&rs) p=r; else p=r+(rzn/rz)*p;
    rz=rzn;
    if(mi<marks.size()&&it==marks[mi]){ std::printf(" it=%4d res=%.1e err=%.1e |",it,arma::norm(b-A*x)/bn,arma::norm(x-xex)/arma::norm(xex)); ++mi; }
  }
  std::printf("\n"); return x;
}
int main(int argc,char**argv){
  arma::uword n=std::atoi(argv[1]); double th=std::atof(argv[2]);
  arma::arma_rng::set_seed(123); arma::mat X(n,2,arma::fill::randu); arma::vec y(n); for(arma::uword i=0;i<n;++i) y(i)=f2d(X(i,0),X(i,1));
  Kriging::Parameters pr; pr.theta=arma::mat(1,2,arma::fill::value(th)); pr.is_theta_estim=false;
  setenv("LK_ITERATIVE_DENSE_MAX_MB","4096",1);
  Kriging k(y,X,"matern5_2",Trend::RegressionModel::Constant,false,"none","LLIterative(30,0,24,2,1e-10)",pr);
  arma::mat R(n,n); LinearAlgebra::covMat_sym_X(&R,k.m_X.t(),k.m_theta,k._Cov,1.0); R.diag().ones();
  arma::vec b=k.m_y-k.m_F*k.m_beta; arma::vec xex=arma::solve(R,b,arma::solve_opts::likely_sympd);
  arma::vec x0=k.m_iterative_RinvFY_cache.col(1)-k.m_iterative_RinvFY_cache.col(0)*k.m_beta(0);
  std::printf("n=%u theta=%.2f cond=%.2e  |x0-xex|/|xex|=%.2e\n",(unsigned)n,th,arma::cond(R),arma::norm(x0-xex)/arma::norm(xex));
  std::vector<int> marks={int(n),int(2*n),1000,5000};
  const char* nm[]={"sans reprise     ","reprise p=r (lib)","remplacement r   "};
  arma::vec z(n,arma::fill::zeros);
  for(int v=0;v<3;++v){ std::printf("COLD %s",nm[v]); cg(R,b,z,5000,v,xex,marks); std::printf("WARM %s",nm[v]); cg(R,b,x0,5000,v,xex,marks);} 
  // bibliothèque réelle, 2n itérations
  auto Am=[&](const arma::mat& V){return arma::mat(R*V);};
  for(int w=0;w<2;++w){ arma::mat X0=x0; arma::mat s=LinearAlgebra::conjugateGradientBatched(Am,b,2*n,arma::vec{1e-8},{},nullptr,w?&X0:nullptr);
    std::printf("LIB %s 2n: res=%.1e err=%.1e\n",w?"WARM":"COLD",arma::norm(b-R*s)/arma::norm(b),arma::norm(s-xex)/arma::norm(xex)); }
}
