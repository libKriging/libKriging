for (kernel in c("exp","matern3_2","matern5_2","gauss")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))
  

#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#rlibkriging:::optim_log(3)
#kernel="exp"

  f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X) + 0.1*rnorm(nrow(X))

tmin=0.01
tmax=1

  k = DiceKriging::km(design=X,response=y,noise.var=rep(0.1^2,nrow(X)),covtype = kernel,control = list(trace=F))
  ll_k = function(theta_sigma2) apply(theta_sigma2,1,function(...)DiceKriging::logLikFun(...,k))
  x=seq(tmin,tmax,,51)
  contour(x,x,matrix(ll_k(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
  for (x1 in seq(tmin,tmax,,11)){
  for (x2 in seq(tmin,tmax,,11)){
    envx = new.env()
    llx = DiceKriging::logLikFun(c(x1,x2),k,envx)
    gllx = DiceKriging::logLikGrad(c(x1,x2),k,envx)
    arrows(x1,x2,x1+0.1*gllx[1],x2+.01*gllx[2])
  }}
  
  library(rlibkriging)
  r <- NoiseKriging(y,noise=rep(0.1^2,nrow(X)), X, kernel)
  ll_r = function(theta_sigma2) logLikelihoodFun(r,theta_sigma2)$logLikelihood
  x=seq(tmin,tmax,,51)
  contour(x,x,matrix(ll_r(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
  for (x1 in seq(tmin,tmax,,11)){
  for (x2 in seq(tmin,tmax,,11)){
    envx = new.env()
    llx = logLikelihoodFun(r,c(x1,x2))$logLikelihood
    gllx = logLikelihoodFun(r,c(x1,x2),grad = T)$logLikelihoodGrad
    arrows(x1,x2,x1+0.1*gllx[1],x2+.01*gllx[2])
  }}
  

  precision <- 1e-8  # the following tests should work with it, since the computations are analytical
  x=c(.5,.5)
  xenv=new.env()
  test_that(desc="logLik is the same that DiceKriging one", 
            expect_equal(logLikelihoodFun(r,x)$logLikelihood,DiceKriging::logLikFun(x,k,xenv),tolerance = precision))
  
  test_that(desc="logLik Grad is the same that DiceKriging one", 
            expect_equal(logLikelihoodFun(r,x,grad=T)$logLikelihoodGrad,t(DiceKriging::logLikGrad(x,k,xenv)),tolerance= precision))
}

