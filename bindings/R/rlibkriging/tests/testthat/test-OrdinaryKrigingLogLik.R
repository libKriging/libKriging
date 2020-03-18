library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = "gauss")
ll = function(theta) DiceKriging::logLikFun(theta,k)
plot(Vectorize(ll))
for (x in seq(0.01,1,,11)){
  envx = new.env()
  llx = DiceKriging::logLikFun(x,k,envx)
  gllx = DiceKriging::logLikGrad(x,k,envx)
  arrows(x,llx,x+.1,llx+.1*gllx)
}

r <- ordinary_kriging(y, X)
ll2 = function(theta) ordinary_kriging_loglikelihood(r,theta)
# plot(Vectorize(ll2),col='red',add=T) # FIXME fails with "error: chol(): decomposition failed"
for (x in seq(0.01,1,,11)){
  envx = new.env()
  ll2x = ordinary_kriging_loglikelihood(r,x)
  gll2x = ordinary_kriging_loglikelihoodgrad(r,x)
  arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
}

precision <- 1e-8  # the following tests should work with it, since the computations are analytical
x=.5
xenv=new.env()
test_that(desc="logLik is the same that DiceKriging one", 
         expect_true(relative_error(ordinary_kriging_loglikelihood(r,x),DiceKriging::logLikFun(x,k,xenv)) < precision))

test_that(desc="logLik Grad is the same that DiceKriging one", 
         expect_true(relative_error(ordinary_kriging_loglikelihoodgrad(r,x),DiceKriging::logLikGrad(x,k,xenv)) < precision))
         