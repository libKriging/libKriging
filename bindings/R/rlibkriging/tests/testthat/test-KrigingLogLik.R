library(testthat)

for (kernel in c("gauss","exp")) {
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = kernel)
ll = function(theta) DiceKriging::logLikFun(theta,k)
plot(Vectorize(ll),ylab="LL",xlab="theta")
for (x in seq(0.01,1,,11)){
  envx = new.env()
  llx = DiceKriging::logLikFun(x,k,envx)
  gllx = DiceKriging::logLikGrad(x,k,envx)
  arrows(x,llx,x+.1,llx+.1*gllx)
}

r <- kriging(y, X, kernel)
ll2 = function(theta) kriging_loglikelihood(r,theta)
# plot(Vectorize(ll2),col='red',add=T) # FIXME fails with "error: chol(): decomposition failed"
for (x in seq(0.01,1,,11)){
  envx = new.env()
  ll2x = kriging_logLikelihood(r,x)
  gll2x = kriging_logLikelihoodGrad(r,x)
  arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
}

precision <- 1e-8  # the following tests should work with it, since the computations are analytical
x=.5
xenv=new.env()
test_that(desc="logLik is the same that DiceKriging one", 
         expect_equal(kriging_logLikelihood(r,x),DiceKriging::logLikFun(x,k,xenv),tolerance = precision))

test_that(desc="logLik Grad is the same that DiceKriging one", 
          expect_equal(kriging_logLikelihoodGrad(r,x),DiceKriging::logLikGrad(x,k,xenv),tolerance= precision))
}
         

