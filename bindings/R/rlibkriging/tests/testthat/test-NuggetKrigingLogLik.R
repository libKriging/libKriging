#install.packages("bindings/R/rlibkriging_0.3-2_R_x86_64-pc-linux-gnu.tar.gz")

library(testthat)

kernel="gauss"
for (kernel in c("exp","matern3_2","matern5_2","gauss")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
  f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X)
  points(X,y)

  k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F),nugget=0, nugget.estim=TRUE)
  alpha0 = k@covariance@sd2/(k@covariance@sd2+k@covariance@nugget)
  ll_theta = function(theta) DiceKriging::logLikFun(c(theta,alpha0),k)
  plot(Vectorize(ll_theta),ylab="LL",xlab="theta",xlim=c(0.01,1))
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    llx = DiceKriging::logLikFun(c(x,alpha0),k,envx)
    gllx = DiceKriging::logLikGrad(c(x,alpha0),k,envx)[1,]
    arrows(x,llx,x+.1,llx+.1*gllx)
  }
  
  library(rlibkriging)
  r <- NuggetKriging(y, X, kernel, parameters=list(nugget=0,estim_nugget=TRUE))
  ll2_theta = function(theta) logLikelihood(r,c(theta,alpha0))$logLikelihood
  # second arg is alpha=1 for nugget=0
  # plot(Vectorize(ll2),col='red'), add=T) 
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    ll2x = logLikelihood(r,c(x,alpha0))$logLikelihood
    gll2x = logLikelihood(r,c(x,alpha0),grad = T)$logLikelihoodGrad[,1]
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
  }
  
  theta0 = k@covariance@range.val
  ll_alpha = function(alpha) DiceKriging::logLikFun(c(theta0,alpha),k)
  plot(Vectorize(ll_alpha),ylab="LL",xlab="alpha",xlim=c(0.01,1))
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    llx = DiceKriging::logLikFun(c(theta0,x),k,envx)
    gllx = DiceKriging::logLikGrad(c(theta0,x),k,envx)[2,]
    arrows(x,llx,x+.1,llx+.1*gllx)
  }
  ll2_alpha = function(alpha) logLikelihood(r,c(theta0,alpha))$logLikelihood
  #plot(Vectorize(ll2_alpha),col='red',add=T)
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    ll2x = logLikelihood(r,c(theta0,x))$logLikelihood
    gll2x = logLikelihood(r,c(theta0,x),grad = T)$logLikelihoodGrad[,2]
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
  }

  precision <- 1e-8  # the following tests should work with it, since the computations are analytical
  x=.25
  xenv=new.env()
  test_that(desc="logLik is the same that DiceKriging one", 
            expect_equal(
              logLikelihood(r,c(theta0,x))$logLikelihood[1]
              ,
              DiceKriging::logLikFun(c(theta0,x),k,xenv)
              ,tolerance = precision))
  
  test_that(desc="logLik Grad is the same that DiceKriging one", 
            expect_equal(
              logLikelihood(r,c(theta0,x),grad=T)$logLikelihoodGrad[,2]
              ,
              DiceKriging::logLikGrad(c(theta0,x),k,xenv)[2,]
              ,tolerance= precision))
  xenv=new.env()
  test_that(desc="logLik is the same that DiceKriging one", 
            expect_equal(
              logLikelihood(r,c(x,alpha0))$logLikelihood[1]
              ,
              DiceKriging::logLikFun(c(x,alpha0),k,xenv)
              ,tolerance = precision))
  
  test_that(desc="logLik Grad is the same that DiceKriging one", 
            expect_equal(
              logLikelihood(r,c(x,alpha0),grad=T)$logLikelihoodGrad[,1]
              ,
              DiceKriging::logLikGrad(c(x,alpha0),k,xenv)[1,]
              ,tolerance= precision))

}


########################## 2D



for (kernel in c("matern3_2","matern5_2","gauss","exp")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
  f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
  n <- 100
  set.seed(123)
  X <- cbind(runif(n),runif(n),runif(n))
  y <- f(X)

  k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F),nugget=0, nugget.estim=TRUE)
  
  #library(rlibkriging)
  r <- NuggetKriging(y, X, kernel, parameters=list(nugget=0,estim_nugget=TRUE))
  
  precision <- 1e-8  # the following tests should work with it, since the computations are analytical
  x=c(.2,.5,.7,0.01)
  xenv=new.env()
  test_that(desc="logLik is the same that DiceKriging one", 
            expect_equal(
              logLikelihood(r,x)$logLikelihood[1]
              ,
              DiceKriging::logLikFun(x,k,xenv)
              ,tolerance = precision))
  
  test_that(desc="logLik Grad is the same that DiceKriging one", 
            expect_equal(
              logLikelihood(r,x,grad=T)$logLikelihoodGrad[1,]
              ,
              t(DiceKriging::logLikGrad(x,k,xenv))[1,]
              ,tolerance= precision))
}


