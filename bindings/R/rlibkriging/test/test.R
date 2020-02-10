library(rlibkriging)
demo_binding1()
demo_binding2()


set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
getEigenValues(Z)


ncol <- 3
nrow <- 4
X <- matrix(nrow=nrow, ncol=ncol)
X[,2:ncol] = matrix(rexp(nrow*(ncol-1)))
X[,1] = 1
sol <- rexp(ncol)
y <- X %*% sol
r <- linear_regression(y, X)
(y - predict.LinearRegression(r,X)$y)/predict.LinearRegression(r,X)$stderr
