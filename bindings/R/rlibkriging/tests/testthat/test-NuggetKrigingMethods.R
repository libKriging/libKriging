#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#library(testthat)

# f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
f <- function(X) apply(X, 1, function(x)
  prod(sin(2*pi*( x * (seq(0,1,l=1+length(x))[-1])^2 )))
)
n <- 20
set.seed(123)
X <- cbind(runif(n),runif(n))
y <- f(X) + rnorm(n,0,0.01)
d = ncol(X)

x=seq(0,1,,51)
contour(x,x,matrix(f(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
points(X)

r <- NuggetKriging(y, X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2),nugget=0,sigma2=1))
l= as.list(r)
# ll = function(X) {
#   logLikelihoodFun(r,X,return_grad=F)$logLikelihood
# }
# contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
# gll = function(X) {
#   logLikelihoodFun(r,X,return_grad=TRUE)$logLikelihoodGrad
# }
# for (ix in 1:21) {
# for (iy in 1:21) {
#   xx = c(ix/21,iy/21)
#   g = gll(xx)
#   arrows(xx[1],xx[2],xx[1]+g[1]/1000,xx[2]+g[2]/1000,col='grey',length=0.05)
# }
# }

context("nugget / print")

p = capture.output(print(r))


context("nugget / logLikelihood")

ll = function(Theta){apply(Theta,1,function(theta) logLikelihoodFun(r,c(theta,l$sigma2/(l$nugget+l$sigma2)))$logLikelihood)}
t=seq(0.01,2,,51)
contour(t,t,matrix(ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30)
points(as.list(r)$theta[1],as.list(r)$theta[2])
# logLikelihoodFun(r,t(as.list(r)$theta))

test_that("nugget / logLikelihoodFun returned",
          expect_equal(names(logLikelihoodFun(r,runif(d+1))),c("logLikelihood")))
test_that("nugget / logLikelihoodFun logLikelihoodGrad returned",
          expect_equal(names(logLikelihoodFun(r,runif(d+1),return_grad=TRUE)),c("logLikelihood","logLikelihoodGrad")))

test_that("nugget / logLikelihoodFun dim",
          expect_equal(dim(logLikelihoodFun(r,rbind(runif(d+1),runif(d+1)))$logLikelihood),c(2,1)))
test_that("nugget / logLikelihoodGrad dim",
          expect_equal(dim(logLikelihoodFun(r,rbind(runif(d+1),runif(d+1)),return_grad=TRUE)$logLikelihoodGrad),c(2,d+1)))


context("nugget / predict")

test_that("nugget / predict mean stdev returned",
          expect_equal(names(predict(r,runif(d))),c("mean","stdev")))
test_that("nugget / predict mean returned",
          expect_equal(names(predict(r,runif(d),return_stdev=F)),c("mean")))
test_that("nugget / predict mean stdev cov returned",
          expect_equal(names(predict(r,runif(d),return_cov=TRUE)),c("mean","stdev","cov")))

test_that("nugget / predict mean dim",
          expect_equal(dim(predict(r,rbind(runif(d),runif(d)))$mean),c(2,1)))
test_that("nugget / predict stdev dim",
          expect_equal(dim(predict(r,rbind(runif(d),runif(d)))$stdev),c(2,1)))
test_that("nugget / predict cov dim",
          expect_equal(dim(predict(r,rbind(runif(d),runif(d)),return_cov=TRUE)$cov),c(2,2)))


context("nugget / simulate")

test_that("nugget / simulate dim",
          expect_equal(dim(simulate(r,x=rbind(runif(d),runif(d)))),c(2,1)))
test_that("nugget / simulate nsim dim",
          expect_equal(dim(simulate(r,x=rbind(runif(d),runif(d)),nsim=10)),c(2,10)))

# offset is not legitimate because nugget kriging is interpolating
#test_that("nugget / simulate mean X",
#          expect_equal(mean(simulate(r,nsim = 100, x=X[1,]+0.00005)),y[1],tolerance = 0.01))
test_that("nugget / simulate mean X",
          expect_equal(mean(simulate(r,nsim = 100, x=X[1,])),y[1],tolerance = 0.01))
set.seed(12345)
x = runif(d)
test_that("nugget / simulate mean",
          expect_equal(mean(simulate(r,nsim = 10000, x=x)),predict(r,x)$mean[1],tolerance = 0.01))
test_that("nugget / simulate sd",
          expect_equal(sd(simulate(r,nsim = 10000, x=x)),predict(r,x)$stdev[1],tolerance = 0.01))


context("nugget / update")

set.seed(1234)
X2 = matrix(runif(d*10),ncol=d)
y2 = f(X2) + rnorm(nrow(X2),0,0.01)
x=seq(0,1,,51)
contour(x,x,matrix(f(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
points(X)
points(X2,col='red')

#r20 <- Kriging(c(y,y2), rbind(X,X2),"gauss")
#ll = function(X) {
#  logLikelihoodFun(r20,X,return_grad=F)$logLikelihood
#}
#contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
#gll = function(X) {
#  logLikelihoodFun(r20,X,return_grad=TRUE)$logLikelihoodGrad
#}
#for (ix in 1:21) {
#for (iy in 1:21) {
#  xx = c(ix/21,iy/21)
#  g = gll(xx)
#  arrows(xx[1],xx[2],xx[1]+g[1]/1000,xx[2]+g[2]/1000,col='grey',length=0.05)
#}
#}



t=seq(0.01,10,,51)
contour(t,t,matrix(ll(as.matrix(expand.grid(t,t))),nrow=length(t)),xlim=c(0,10),ylim=c(0,1),nlevels = 30)
points(as.list(r)$theta[1],as.list(r)$theta[2],pch=20)

#cat("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!","\n")
#cat(paste0(collapse="\n",p),"\n")
#cat(r$logLikelihood(),"\n")
#
#rlibkriging:::optim_log(1)

set.seed(1234)
r2 <- NuggetKriging(c(y,y2), rbind(X,X2),"gauss", parameters = list(theta=matrix(as.list(r)$theta,ncol=2),nugget=0,sigma2=1))
ll2 = function(Theta){apply(Theta,1,function(theta) logLikelihoodFun(r2,c(theta,as.list(r2)$sigma2/(as.list(r2)$sigma2+as.list(r2)$nugget)))$logLikelihood)}
contour(t,t,matrix(ll2(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30,add=TRUE,col='red')
points(as.list(r2)$theta[1],as.list(r2)$theta[2],col='red',pch=20)

p2 = capture.output(print(r2))

update(object=r,y2,X2)
llu = function(Theta){apply(Theta,1,function(theta) logLikelihoodFun(r,  c(theta,as.list(r2)$sigma2/(as.list(r2)$sigma2+as.list(r2)$nugget)) )$logLikelihood)}
contour(t,t,matrix(llu(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30,add=TRUE,col='blue')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='blue',pch=20)

pu = capture.output(print(r))

test_that("nugget / update",
          expect_false(all(p == pu)))

# cat(paste0(collapse="\n",p2),"\n")
# cat(r2$logLikelihood(),"\n")
# cat(paste0(collapse="\n",capture.output(print( r2$logLikelihoodFun(c(r2$theta(),r2$sigma2()/(r2$sigma2()+r2$nugget())),TRUE) ))),"\n")
# cat(paste0(collapse="\n",pu),"\n")
# cat(r$logLikelihood(),"\n")
# cat(paste0(collapse="\n",capture.output(print( r$logLikelihoodFun(c(r$theta(),r$sigma2()/(r$sigma2()+r$nugget())),TRUE) ))),"\n")

test_that("nugget / update almost converge",
          expect_equal(as.list(r2)$theta, as.list(r)$theta, tolerance = 2E-2))
