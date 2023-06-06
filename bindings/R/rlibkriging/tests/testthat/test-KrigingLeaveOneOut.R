for (kernel in c("gauss","exp","matern3_2","matern5_2")) {
# kernel = "gauss"
  context(paste0("Check leaveOneOut for kernel ",kernel))
  
f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
points(X,y)
k = DiceKriging::km(design=X,response=y,covtype = kernel,control = list(trace=F), formula = as.formula("~."))
ll = function(theta) DiceKriging::leaveOneOutFun(theta,k)

plot(Vectorize(ll),ylab="LL",xlab="theta",xlim=c(0.01,1))
for (x in seq(0.01,1,,11)){
  envx = new.env()
  llx = DiceKriging::leaveOneOutFun(x,k,envx)
  gllx = DiceKriging::leaveOneOutGrad(x,k,envx)
  arrows(x,llx,x+.1,llx+.1*gllx)
}

r <- Kriging(y, X, kernel, regmodel="linear")
ll2 = function(theta) leaveOneOutFun(r,theta)$leaveOneOut[1]
# plot(Vectorize(ll2),col='red',add=T) # FIXME fails with "error: chol(): decomposition failed"
for (x in seq(0.01,1,,11)){
  envx = new.env()
  ll2x = leaveOneOutFun(r,x)$leaveOneOut[1]
  gll2x = leaveOneOutFun(r,x,grad=T)$leaveOneOutGrad
  arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
}

precision <- 1e-8  # the following tests should work with it, since the computations are analytical
x=.5
xenv=new.env()
test_that(desc="leaveOneOut is the same that DiceKriging one",
         expect_equal(leaveOneOutFun(r,x)$leaveOneOut[1],DiceKriging::leaveOneOutFun(x,k,xenv),tolerance = precision))

test_that(desc="leaveOneOut Grad is the same that DiceKriging one",
          expect_equal(leaveOneOutFun(r,x,grad=T)$leaveOneOutGrad,DiceKriging::leaveOneOutGrad(x,k,xenv),tolerance= precision))
}



context(paste0("Check leaveOneOutVec"))
  
f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
r <- Kriging(y, X, kernel="gauss", regmodel="linear")

loo = list(mean=matrix(NA,n,1),stdev=matrix(NA,n,1))
for (i in 1:n) {
  ri <- Kriging(y[-i], X[-i,,drop=FALSE], kernel="gauss", regmodel="linear", optim="none", parameters=list(theta=r$theta(),sigma2=r$sigma2()))
  predi = predict(ri,X[i])
  loo$mean[i,1] = predi$mean
  loo$stdev[i,1] = predi$stdev
}

loovec = r$leaveOneOutVec(r$theta())

test_that(desc="leaveOneOut vector mean is equal to k-fold loo",
          expect_equal(loovec$mean,loo$mean,tolerance= precision))
test_that(desc="leaveOneOut vector stdev is equal to k-fold loo",
          expect_equal(loovec$stdev,loo$stdev,tolerance= precision))

k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F), formula = as.formula("~."), 
                    coef.var=r$sigma2(), coef.cov=r$theta()[1])
loovec_km = DiceKriging::leaveOneOut.km(k,type="UK",trend.reestim=TRUE)

test_that(desc="leaveOneOut vector mean is equal to DiceKriging one",
          expect_equal(loovec_km$mean,loovec$mean[,1],tolerance= precision))
test_that(desc="leaveOneOut vector stdev is equal to DiceKriging one",
          expect_equal(loovec_km$sd,loovec$stdev[,1],tolerance= precision))