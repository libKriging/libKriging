library(testthat)

for (kernel in c("gauss","exp")) {
# kernel = "gauss"
  context(paste0("Check LogLikelihood for kernel ",kernel))
  
  f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
  n <- 100
  set.seed(123)
  X <- cbind(runif(n),runif(n),runif(n))
  y <- f(X)
  d = ncol(X)
  
k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F))
ll = function(theta) DiceKriging::leaveOneOutFun(theta,k)

r <- Kriging(y, X, kernel)
ll2 = function(theta) leaveOneOut(r,theta)

precision <- 1e-8  # the following tests should work with it, since the computations are analytical
x=runif(d)
xenv=new.env()
test_that(desc="leaveOneOut is the same that DiceKriging one",
         expect_equal(leaveOneOut(r,x)$leaveOneOut[1],DiceKriging::leaveOneOutFun(x,k,xenv),tolerance = precision))

test_that(desc="leaveOneOut Grad is the same that DiceKriging one",
          expect_equal(t(leaveOneOut(r,x,grad=T)$leaveOneOutGrad),DiceKriging::leaveOneOutGrad(x,k,xenv),tolerance= precision))
}


