library(rlibkriging, lib.loc="bindings/R/Rlibs")
library(testthat)

context("increment Kriging")

f <- function(X) apply(X, 1,
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
n <- 1000
d <- 3
set.seed(1234)
X <- matrix(runif(n*d),ncol=d)
y <- f(X)+rnorm(1000,0,0.1)
r = NULL
try( r <- NuggetKriging(y, X, "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), beta = matrix(0.01), nugget=0.1^2, sigma2=0.09)) )

no = floor(n*0.7)
try( ro <- NuggetKriging(y[1:no], X[1:no,], "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), beta = matrix(0.01), nugget=0.1^2, sigma2=0.09)) )
# update with new points, compute LL but no fit (since optim=none)
ro$update(y[(no+1):n],X[(no+1):n,], refit=TRUE)

test_that(desc="Updated (no refit) Kriging equals Kriging with all data",
          expect_equal(ro$T(), r$T(), tol=1e-5))

# m1 = microbenchmark::microbenchmark(
#   r <- Kriging(y, X, "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), beta = matrix(0.01))),
#   times=100
# )
# 
# m2 = microbenchmark::microbenchmark(
#   {
#     ro <- Kriging(y[1:no], X[1:no,], "gauss","constant",FALSE,"none","LL", parameters=list(theta = matrix(.5,ncol=3), beta = matrix(0.01)))
#     ro$update(y[(no+1):n], X[(no+1):n,], refit=FALSE)
#   },
#   times=100
# )

