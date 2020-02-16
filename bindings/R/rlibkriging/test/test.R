library(rlibkriging)
demo_binding1()
demo_binding2()


set.seed(42)
X <- matrix(rnorm(4*4), 4, 4)
Z <- X %*% t(X)
obj <- buildDemoArmadilloClass("Z",Z)
getEigenValues(obj)

# Remove all internal references
rm(obj)
gc()

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
predict.LinearRegression(r,X)
ro <- linear_regression_optim(y, X)
predict.LinearRegression(ro,X)
# (y - predict.LinearRegression(r,X)$y)/predict.LinearRegression(r,X)$stderr


n <- 10
X <- as.matrix(runif(n))
y = 4*X+rnorm(n,0,.1)
r <- linear_regression_optim(y, X)
plot(X,y)
x=as.matrix(seq(0,1,,100))
px = predict.LinearRegression(r,x)
lines(x,px$y)
lines(x,px$y-2*px$stderr,col='red')
lines(x,px$y+2*px$stderr,col='red')


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
for (x in seq(0,1,,11)){
  envx = new.env()
  llx = DiceKriging::logLikFun(x,k,envx)
  gllx = DiceKriging::logLikGrad(x,k,envx)
  arrows(x,llx,x+.1,llx+.1*gllx)
}

r <- ordinary_kriging(y, X)
ll2 = function(theta) ordinary_kriging_loglikelihood(r,theta)
plot(Vectorize(ll2),col='red')
for (x in seq(0,1,,11)){
  envx = new.env()
  ll2x = ordinary_kriging_loglikelihood(r,x)
  gll2x = ordinary_kriging_loglikelihoodgrad(r,x)
  arrows(x,ll2x,x+.1,ll2x+.1*gll2x)
}

# x=as.matrix(seq(0,1,,100))
# px = predict.OrdinariKriging(r,x)
# lines(x,px$y)
# lines(x,px$y-2*px$stderr,col='red')
# lines(x,px$y+2*px$stderr,col='red')



