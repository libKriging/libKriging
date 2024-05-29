library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)
kernel="gauss"

for (kernel in c("exp","matern3_2","matern5_2","gauss")) {
  context(paste0("Check logMargPost for kernel ",kernel))
  
  f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X)
  points(X,y)
  
  #library(rlibkriging)
  r <- NuggetKriging(y, X, kernel, objective="LMP", parameters=list(nugget=0,is_nugget_estim=TRUE))
  
eps = 0.00001 # used for comparison bw computed grad and num grad
precision = 0.01

  alpha0=r$sigma2()/(r$sigma2()+r$nugget())
  ll2_theta = function(theta) logMargPostFun(r,c(theta,alpha0))$logMargPost
  # second arg is alpha=1 for nugget=0
  plot(Vectorize(ll2_theta),ylab="LMP",xlab="theta",xlim=c(0.01,1))
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    ll2x = logMargPostFun(r,c(x,alpha0))$logMargPost[1]
    gll2x = logMargPostFun(r,c(x,alpha0),return_grad = T)$logMargPostGrad[,1]
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')

    ll2x_eps = logMargPostFun(r,c(x+eps,alpha0))$logMargPost[1]
    test_that(desc=paste0("Numerical and analytical gradient are (almost) the same at theta=",x),
          expect_equal((ll2x_eps-ll2x)/eps,gll2x,tol = precision))
  }
  
  theta0 = r$theta()
  ll2_alpha = function(alpha) logMargPostFun(r,c(theta0,alpha))$logMargPost
  plot(Vectorize(ll2_alpha),ylab="LMP",xlab="alpha",xlim=c(0.01,1))
  for (x in seq(0.01,0.99,,11)){
    envx = new.env()
    ll2x = logMargPostFun(r,c(theta0,x))$logMargPost[1]
    gll2x = logMargPostFun(r,c(theta0,x),return_grad = T)$logMargPostGrad[,2]
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')

    ll2x_eps = logMargPostFun(r,c(theta0,x+eps))$logMargPost[1]
    test_that(desc=paste0("Numerical and analytical gradient are (almost) the same at alpha=",x),
          expect_equal((ll2x_eps-ll2x)/eps,gll2x,tol = precision))
  }


}
