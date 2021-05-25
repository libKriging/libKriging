library(testthat)

#kernel="matern3_2" #
for (kernel in c("gauss","exp")){ # NOT YET WORKING: ,"matern3_2","matern5_2")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))

logn <- seq(1.1, 2, by=.1)

#i=1 #
for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- 1+floor(log(n)) #floor(2+i/3)
  
  # print(n)

  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y <- f(X)
  

k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F))
ll = function(theta) DiceKriging::logLikFun(theta,k)
gll = function(theta) {e = new.env(); ll=DiceKriging::logLikFun(theta,k,e); DiceKriging::logLikGrad(theta,k,e)}
hll = function(theta,eps=0.0001) {
  d = length(theta)
  h = matrix(NA,d,d)
  g = gll(theta)
  for (i in 1:d) {
    eps_i = matrix(0, nrow=d,ncol=1);
    eps_i[i] = eps;
    h[i,] = (gll(as.numeric(theta + eps_i))-g)/eps
  }
  (h+t(h))/2
}

library(rlibkriging)
r <- Kriging(y, X, kernel)
ll_C = function(theta) logLikelihood(r,theta)$logLikelihood[1]
gll_C = function(theta) t(logLikelihood(r,theta,grad=T)$logLikelihoodGrad)
hll_C = function(theta) logLikelihood(r,theta,hess=T)$logLikelihoodHess[,,]


x=runif(d)

#print(ll(x))
#print(ll_C(x))
#print(gll(x))
#print(gll_C(x))
#print(hll(x))
#print(hll_C(x))

precision <- 0.1
test_that(desc="logLik is the same that DiceKriging one", 
         expect_true(abs(ll(x)-ll_C(x)) < precision))

test_that(desc="logLik Grad is the same that DiceKriging one", 
         expect_true(max(abs(gll(x)-gll_C(x))/abs(gll(x))) < precision))

test_that(desc="logLik Hess is the same that DiceKriging one", 
         expect_true(max(abs(hll(x)-hll_C(x))/abs(hll(x))) < precision))

}
         
}
