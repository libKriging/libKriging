#include <sstream>
#include <iostream>
#include <fstream>
#include <functional>
#include <optional>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <mutex>
#include <thread>
#include "libKriging/utils/lk_armadillo.hpp"
#define private public
#define protected public
#include "libKriging/Kriging.hpp"
#include "libKriging/LinearAlgebra.hpp"
#undef private
#undef protected
#include <cstdio>
#include <cstdlib>
#include <cmath>

static double f2d(double x, double y){ // copie de tests (Branin-like) -- remplacé ci-dessous si différent
  return std::sin(3.0*x)+std::cos(5.0*y)+x*y;
}
static void make_data(arma::uword n, arma::mat& X, arma::vec& y, unsigned seed=123){
  arma::arma_rng::set_seed(seed); X=arma::mat(n,2,arma::fill::randu); y=arma::vec(n);
  for(arma::uword i=0;i<n;++i) y(i)=f2d(X(i,0),X(i,1));
}
static Kriging mk(const arma::vec& y,const arma::mat& X,const std::string& obj,double th){
  Kriging::Parameters p; p.theta=arma::mat(1,X.n_cols,arma::fill::value(th)); p.is_theta_estim=false;
  return Kriging(y,X,"matern5_2",Trend::RegressionModel::Constant,false,"none",obj,p);
}
// référence exacte (Cholesky) avec les beta/sigma2/theta DU modèle
static void exact(const Kriging& k,const arma::mat& Xt,arma::vec& m,arma::vec& s,arma::vec& w){
  arma::mat R(k.m_X.n_rows,k.m_X.n_rows); LinearAlgebra::covMat_sym_X(&R,k.m_X.t(),k.m_theta,k._Cov,1.0); R.diag().ones();
  arma::vec resid=k.m_y-k.m_F*k.m_beta; w=arma::solve(R,resid,arma::solve_opts::likely_sympd);
  arma::mat Xn=Xt; Xn.each_row()-=k.m_centerX; Xn.each_row()/=k.m_scaleX;
  arma::mat Ron(k.m_X.n_rows,Xt.n_rows); LinearAlgebra::covMat_rect(&Ron,k.m_X.t(),Xn.t(),k.m_theta,k._Cov,1.0);
  arma::mat Fn=Trend::regressionModelMatrix(k.m_regmodel,Xn);
  m=(Fn*k.m_beta+Ron.t()*w)*k.m_scaleY+k.m_centerY;
  arma::mat V=arma::solve(R,Ron,arma::solve_opts::likely_sympd); arma::vec quad=arma::sum(Ron%V,0).t();
  arma::mat WF=arma::solve(R,k.m_F); arma::mat E=Fn-Ron.t()*WF; arma::vec g=arma::sum(E.t()%arma::solve(k.m_F.t()*WF,E.t()),0).t();
  s=arma::sqrt(arma::clamp(k.m_sigma2*(1-quad+g),0.0,arma::datum::inf))*k.m_scaleY;
  std::fprintf(stderr,"  cond(R)=%.3e\n",arma::cond(R));
}
int main(int argc,char**argv){
  arma::uword n=std::atoi(argv[1]); double th=std::atof(argv[2]); std::string obj=argv[3];
  const char* mode=argv[4]; // "mf" ou "de"
  setenv("LK_ITERATIVE_DENSE_MAX_MB", std::string(mode)=="mf"?"0":"4096",1);
  arma::mat X; arma::vec y; make_data(n,X,y); arma::mat Xt; arma::vec yt; make_data(15,Xt,yt,456);
  Kriging k=mk(y,X,obj,th);
  arma::vec me,se,we; exact(k,Xt,me,se,we);
  // erreur du cache lui-même (R^-1 y colonne p)
  if(k.m_iterative_RinvFY_cache.n_cols>0){
    arma::mat R(k.m_X.n_rows,k.m_X.n_rows); LinearAlgebra::covMat_sym_X(&R,k.m_X.t(),k.m_theta,k._Cov,1.0); R.diag().ones();
    arma::mat ex=arma::solve(R,arma::join_rows(k.m_F,k.m_y),arma::solve_opts::likely_sympd);
    std::fprintf(stderr,"  cache relerr=%.3e  relres=%.3e\n",arma::norm(k.m_iterative_RinvFY_cache-ex)/arma::norm(ex),
      arma::norm(R*k.m_iterative_RinvFY_cache-arma::join_rows(k.m_F,k.m_y))/arma::norm(arma::join_rows(k.m_F,k.m_y)));
  } else std::fprintf(stderr,"  pas de cache\n");
  for(int warm=1;warm>=0;--warm){
    Kriging kk=Kriging(k,ExplicitCopySpecifier{});
    if(!warm) kk.m_iterative_RinvFY_cache.reset();
    std::fprintf(stderr,"--- %s\n",warm?"WARM":"COLD");
    auto [m,s]=kk.predictIterative(Xt,true);
    std::fprintf(stderr,"  |mean-exact|max=%.3e  |sd-exact|max=%.3e  sd(y)=%.3e\n",arma::abs(m-me).max(),arma::abs(s-se).max(),arma::stddev(y));
  }
}
