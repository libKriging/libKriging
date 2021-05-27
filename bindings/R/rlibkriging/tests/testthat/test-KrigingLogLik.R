#install.packages("bindings/R/rlibkriging_0.3-2_R_x86_64-pc-linux-gnu.tar.gz")

library(testthat)

for (kernel in c("exp","matern3_2","matern5_2","gauss")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
  f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X)
  points(X,y)

  k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F))
  ll = function(theta) DiceKriging::logLikFun(theta,k)
  plot(Vectorize(ll),ylab="LL",xlab="theta",xlim=c(0.01,1))
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    llx = DiceKriging::logLikFun(x,k,envx)
    gllx = DiceKriging::logLikGrad(x,k,envx)
    arrows(x,llx,x+.1,llx+.1*gllx)
  }
  
  library(rlibkriging)
  r <- Kriging(y, X, kernel)
  ll2 = function(theta) logLikelihood(r,theta)$logLikelihood
  # plot(Vectorize(ll2),col='red',add=T) # FIXME fails with "error: chol(): decomposition failed"
  for (x in seq(0.01,1,,11)){
    envx = new.env()
    ll2x = logLikelihood(r,x)$logLikelihood
    gll2x = logLikelihood(r,x,grad = T)$logLikelihoodGrad
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
  }
  
  precision <- 1e-8  # the following tests should work with it, since the computations are analytical
  x=.5
  xenv=new.env()
  test_that(desc="logLik is the same that DiceKriging one", 
            expect_equal(logLikelihood(r,x)$logLikelihood[1],DiceKriging::logLikFun(x,k,xenv),tolerance = precision))
  
  test_that(desc="logLik Grad is the same that DiceKriging one", 
            expect_equal(logLikelihood(r,x,grad=T)$logLikelihoodGrad,DiceKriging::logLikGrad(x,k,xenv),tolerance= precision))
}


########################## 2D



for (kernel in c("matern3_2","matern5_2","gauss","exp")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
  f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
  n <- 100
  set.seed(123)
  X <- cbind(runif(n),runif(n),runif(n))
  y <- f(X)

  k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F))
  
  #library(rlibkriging)
  r <- Kriging(y, X, kernel)
  
  precision <- 1e-8  # the following tests should work with it, since the computations are analytical
  x=c(.2,.5,.7)
  xenv=new.env()
  test_that(desc="logLik is the same that DiceKriging one", 
            expect_equal(logLikelihood(r,x)$logLikelihood[1],DiceKriging::logLikFun(x,k,xenv),tolerance = precision))
  
  test_that(desc="logLik Grad is the same that DiceKriging one", 
            expect_equal(logLikelihood(r,x,grad=T)$logLikelihoodGrad[1,],t(DiceKriging::logLikGrad(x,k,xenv))[1,],tolerance= precision))
}


