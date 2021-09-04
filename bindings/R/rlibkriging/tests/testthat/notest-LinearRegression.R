library(testthat)

# loading package using wildcard
# pack=list.files(".",pattern = "rlibkriging_",full.names = T)
# install.packages(pack[1],repos=NULL)

install.packages(pkgs="rlibkriging_0.4-6.tgz", type="source", repos=NULL)
library(rlibkriging)
n <- 10
X <- as.matrix(runif(n))
y = 4*X+rnorm(n,0,.1)
r <- linear_regression(y, X)
plot(X,y)
x=as.matrix(seq(0,1,,100))
px = linear_regression_predict(r,x)

lines(x,px$y)
lines(x,px$y-2*px$stderr,col='red')
lines(x,px$y+2*px$stderr,col='red')

precision <- 1e-1  # the following tests should work with it, since the computations are analytical
test_that(desc="Predict linear reg. is exact on the design points", 
          expect_equal(linear_regression_predict(r,X)$y,y, tol= precision))
