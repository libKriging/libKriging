library(rlibkriging)
demo_binding1()
demo_binding2()


set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
getEigenValues(Z)


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


ncol <- 3
nrow <- 4
X <- matrix(nrow=nrow, ncol=ncol)
X[,2:ncol] = matrix(rexp(nrow*(ncol-1)))
X[,1] = 1
sol <- rexp(ncol)
y <- X %*% sol
r <- linear_regression(y, X)
(y - predict.LinearRegression(r,X)$y)/predict.LinearRegression(r,X)$stderr


