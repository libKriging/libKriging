library(RobustGaSP)

context("RobustGaSP-Nugget / Fit: 1D")

f = function(x) 10*(1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7))
#plot(f)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
#points(X,y)
k = RobustGaSP::rgasp(design=X,response=y,nugget.est = TRUE)
library(rlibkriging)
r <- NuggetKriging(y, X,
  kernel="matern5_2",
  regmodel = "constant", normalize = FALSE,
  optim = "none",
  objective = "LMP", 
  parameters=list(
    theta=matrix(1/k@beta_hat),is_theta_estim=F,
    nugget=k@nugget*k@sigma2_hat,is_nugget_estim=F,
    sigma2=k@sigma2_hat,is_sigma2_estim=F))
# m = as.list(r)

# Check predict

ntest <- 100
Xtest <- seq(0,1,,ntest)
Ytest_rgasp <- predict(k,matrix(Xtest,ncol=1))
Ytest_libK <- predict(r,Xtest)

plot(f)
points(X,y)
lines(Xtest,Ytest_rgasp$mean,col='blue')
polygon(c(Xtest,rev(Xtest)),
        c(Ytest_rgasp$mean+2*Ytest_rgasp$sd,rev(Ytest_rgasp$mean-2*Ytest_rgasp$sd)),
        col=rgb(0,0,1,0.1), border=NA)

lines(Xtest,Ytest_libK$mean,col='red')
polygon(c(Xtest,rev(Xtest)),
        c(Ytest_libK$mean+2*Ytest_libK$stdev,rev(Ytest_libK$mean-2*Ytest_libK$stdev)),
        col=rgb(1,0,0,0.1), border=NA)

precision <- 1e-3
test_that(desc=paste0("pred mean is the same that RobustGaSP one"),
          expect_equal(predict(r,0.7)$mean[1],predict(k,matrix(0.7))$mean,tol = precision))
test_that(desc=paste0("pred sd is the same that RobustGaSP one"),
          expect_equal(predict(r,0.7)$stdev[1],predict(k,matrix(0.7))$sd,tol = precision))



context("RobustGaSP-Nugget / Fit: 2D (Branin)")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)
model = NULL
r = NULL
library(RobustGaSP)
k = rgasp(design=X,response=y, nugget.est = T)
library(rlibkriging)
r <- NuggetKriging(y, X,
  kernel="matern5_2",
  regmodel = "constant", normalize = FALSE,
  optim = "none",
  objective = "LMP",
  parameters=list(
    theta=matrix(1/k@beta_hat,nrow=2),is_theta_estim=F,
    nugget=k@nugget*k@sigma2_hat,is_nugget_estim=F,
    sigma2=k@sigma2_hat,is_sigma2_estim=F))
  

# Check predict
x2=0.8
f1 = function(x) f(cbind(x,x2))

ntest <- 100
Xtest <- seq(0,1,,ntest)
Ytest_rgasp <- predict(k,matrix(cbind(Xtest,x2),ncol=2))
Ytest_libK <- predict(r,cbind(Xtest,x2))

plot(f1)
lines(Xtest,Ytest_rgasp$mean,col='blue')
polygon(c(Xtest,rev(Xtest)),
        c(Ytest_rgasp$mean+2*Ytest_rgasp$sd,rev(Ytest_rgasp$mean-2*Ytest_rgasp$sd)),
        col=rgb(0,0,1,0.1), border=NA)

lines(Xtest,Ytest_libK$mean,col='red')
polygon(c(Xtest,rev(Xtest)),
        c(Ytest_libK$mean+2*Ytest_libK$stdev,rev(Ytest_libK$mean-2*Ytest_libK$stdev)),
        col=rgb(1,0,0,0.1), border=NA)

precision <- 1e-1
test_that(desc=paste0("pred mean is the same that RobustGaSP one"),
          expect_equal(predict(r,c(0.7,x2))$mean[1],predict(k,matrix(c(0.7,x2),ncol=2))$mean,tol = precision))
test_that(desc=paste0("pred sd is the same that RobustGaSP one"),
          expect_equal(predict(r,c(0.7,x2))$stdev[1],predict(k,matrix(c(0.7,x2),ncol=2))$sd,tol = precision))

