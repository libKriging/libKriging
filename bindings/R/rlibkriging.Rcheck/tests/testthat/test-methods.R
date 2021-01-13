library(testthat)

# install.packages("../rlibkriging_0.1-10_R_x86_64-pc-linux-gnu.tar.gz",repos=NULL)
# library(rlibkriging)

# f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
f <- function(X) apply(X, 1, function(x)
  prod(sin(2*pi*( x * (seq(0,1,l=1+length(x))[-1])^2 )))
)
n <- 20
set.seed(123)
X <- cbind(runif(n),runif(n))
y <- f(X)
d = ncol(X)

x=seq(0,1,,51)
contour(x,x,matrix(f(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
points(X)

r <- Kriging(y, X,"gauss",parameters = list(theta=matrix(.5,ncol=2)))


context("print")

p = capture.output(print(r))


context("logLikelihood")

ll = function(Theta){apply(Theta,1,function(theta) kriging_logLikelihood(r,theta)$logLikelihood)}
t=seq(0.01,2,,51)
contour(t,t,matrix(ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30)
points(kriging_model(r)$theta[1],kriging_model(r)$theta[2])
# logLikelihood(r,t(kriging_model(r)$theta))

test_that("logLikelihood returned",
          expect_equal(names(logLikelihood(r,runif(d))),c("logLikelihood")))
test_that("logLikelihood logLikelihoodGrad returned",
          expect_equal(names(logLikelihood(r,runif(d),grad=T)),c("logLikelihood","logLikelihoodGrad")))
test_that("logLikelihood logLikelihoodGrad logLikelihoodHessreturned",
          expect_equal(names(logLikelihood(r,runif(d),grad=T,hess=T)),c("logLikelihood","logLikelihoodGrad","logLikelihoodHess")))

test_that("logLikelihood dim",
          expect_equal(dim(logLikelihood(r,rbind(runif(d),runif(d)))$logLikelihood),c(2,1)))
test_that("logLikelihoodGrad dim",
          expect_equal(dim(logLikelihood(r,rbind(runif(d),runif(d)),grad=T)$logLikelihoodGrad),c(2,d)))
test_that("logLikelihoodHess dim",
          expect_equal(dim(logLikelihood(r,rbind(runif(d),runif(d)),grad=T,hess=T)$logLikelihoodHess),c(2,d,d)))


context("leaveOneOut")

loo = function(Theta){apply(Theta,1,function(theta) kriging_leaveOneOut(r,theta)$leaveOneOut)}
t=seq(0.01,2,,51)
contour(t,t,matrix(loo(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30)
points(kriging_model(r)$theta[1],kriging_model(r)$theta[2])
# leaveOneOut(r,t(kriging_model(r)$theta))

test_that("leaveOneOut returned",
          expect_equal(names(leaveOneOut(r,runif(d))),c("leaveOneOut")))
test_that("leaveOneOut leaveOneOutGrad returned",
          expect_equal(names(leaveOneOut(r,runif(d),grad=T)),c("leaveOneOut","leaveOneOutGrad")))

test_that("leaveOneOut dim",
          expect_equal(dim(leaveOneOut(r,rbind(runif(d),runif(d)))$leaveOneOut),c(2,1)))
test_that("leaveOneOutGrad dim",
          expect_equal(dim(leaveOneOut(r,rbind(runif(d),runif(d)),grad=T)$leaveOneOutGrad),c(2,d)))


context("predict")

test_that("predict mean stdev returned",
          expect_equal(names(predict(r,runif(d))),c("mean","stdev")))
test_that("predict mean returned",
          expect_equal(names(predict(r,runif(d),stdev=F)),c("mean")))
test_that("predict mean stdev cov returned",
          expect_equal(names(predict(r,runif(d),cov=T)),c("mean","stdev","cov")))

test_that("predict mean dim",
          expect_equal(dim(predict(r,rbind(runif(d),runif(d)))$mean),c(2,1)))
test_that("predict stdev dim",
          expect_equal(dim(predict(r,rbind(runif(d),runif(d)))$stdev),c(2,1)))
test_that("predict cov dim",
          expect_equal(dim(predict(r,rbind(runif(d),runif(d)),cov=T)$cov),c(2,2)))


context("simulate")

test_that("simulate dim",
          expect_equal(dim(simulate(r,x=rbind(runif(d),runif(d)))),c(2,1)))
test_that("simulate nsim dim",
          expect_equal(dim(simulate(r,x=rbind(runif(d),runif(d)),nsim=10)),c(2,10)))

test_that("simulate mean X",
          expect_equal(mean(simulate(r,nsim = 100, x=X[1,]+0.00005)),y[1],tolerance = 0.01))
x = runif(d)
test_that("simulate mean",
          expect_equal(mean(simulate(r,nsim = 100, x=x)),predict(r,x)$mean[1],tolerance = 0.01))
test_that("simulate sd",
          expect_equal(sd(simulate(r,nsim = 100, x=x)),predict(r,x)$stdev[1],tolerance = 0.01))


context("update")

X2 = matrix(runif(d*10),ncol=d)
y2 = f(X2)
x=seq(0,1,,51)
contour(x,x,matrix(f(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
points(X)
points(X2,col='red')

r2 <- Kriging(c(y,y2), rbind(X,X2),"gauss", parameters = list(theta=matrix(kriging_model(r)$theta,ncol=2)))
ll2 = function(Theta){apply(Theta,1,function(theta) kriging_logLikelihood(r2,theta)$logLikelihood)}
t=seq(0.01,2,,51)
contour(t,t,matrix(ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30)
points(kriging_model(r)$theta[1],kriging_model(r)$theta[2])
contour(t,t,matrix(ll2(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30,add=T,col='red')
points(kriging_model(r2)$theta[1],kriging_model(r2)$theta[2],col='red')

p2 = capture.output(print(r2))

update(object=r,y2,X2)

pu = capture.output(print(r))

test_that("update",
          expect_false(all(p == pu)))
test_that("update",
          expect_true(all(pu == p2)))
