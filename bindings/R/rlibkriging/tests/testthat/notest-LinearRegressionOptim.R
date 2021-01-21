library(testthat)

n <- 10
X <- as.matrix(runif(n))
eps = 0.1
set.seed(123)
y = 4*X+rnorm(n,0,eps)
r <- linear_regression_optim(y, X)
plot(X,y)
x=as.matrix(seq(0,1,,100))
px = linear_regression_predict(r,x)

lines(x,px$y)
lines(x,px$y-2*px$stderr,col='red')
lines(x,px$y+2*px$stderr,col='red')

precision <- 3*eps #will work for 99.9% cases, so should be ok for 100 points
test_that(desc="Predict linear reg. is exact on the design points", 
          expect_equal(linear_regression_predict(r,X)$y,y,tol= precision))
