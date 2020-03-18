library(testthat)

n <- 10
X <- as.matrix(runif(n))
y = 4*X+rnorm(n,0,.1)
r <- linear_regression(y, X)
plot(X,y)
x=as.matrix(seq(0,1,,100))
px = predict.LinearRegression(r,x)

lines(x,px$y)
lines(x,px$y-2*px$stderr,col='red')
lines(x,px$y+2*px$stderr,col='red')

precision <- 1e-1  # the following tests should work with it, since the computations are analytical
test_that(desc="Predict linear reg. is exact on the design points", 
          expect_true(relative_error(predict.LinearRegression(r,X)$y,y) < precision))
