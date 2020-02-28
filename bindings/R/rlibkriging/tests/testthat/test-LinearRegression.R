library(testthat)

set.seed(123)

precision <- 1e-14

ncol <- 3
nrow <- 4
X <- matrix(nrow=nrow, ncol=ncol)
X[,2:ncol] = matrix(rexp(nrow*(ncol-1)))
X[,1] = 1
sol <- rexp(ncol)
y <- X %*% sol
r <- linear_regression(y, X)
# the following tests should work with it, since the computations are analytical
test_that(desc="Predict linear reg. is exact on analytic solution",
          expect_true(relative_error(predict.LinearRegression(r,X)$y,y) < precision))

noise_amplitude <- 0.1

n <- 10
X <- as.matrix(runif(n))
y = 4*X+rnorm(n,0,noise_amplitude)
r <- linear_regression(y, X)
plot(X,y)
x=as.matrix(seq(0,1,,100))
px = predict.LinearRegression(r,x)

lines(x,px$y)
lines(x,px$y-2*px$stderr,col='red')
lines(x,px$y+2*px$stderr,col='red')

test_that(desc="Predict linear reg. is exact on the design points",
          expect_true(relative_error(predict.LinearRegression(r,X)$y,y) < 10 * precision + 10 * noise_amplitude))


