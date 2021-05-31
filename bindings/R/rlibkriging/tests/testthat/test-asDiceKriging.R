library(testthat)

# install.packages("../rlibkriging_0.1-10_R_x86_64-pc-linux-gnu.tar.gz",repos=NULL)
# library(rlibkriging)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
# f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 5
set.seed(123)
X <- cbind(runif(n))
y <- f(X)
d = ncol(X)

library(DiceKriging)
# kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
m1 <- km(design=X, response=y,covtype = "gauss",formula=~1,estim.method="LOO",parinit = c(.15),control = list(trace=F))
library(rlibkriging)
as_m1 <- as_km(design=X, response=y,covtype = "gauss",formula=~1,estim.method="LOO",parinit = c(.15))

test_that("m1.leaveOneOutFun == as_m1.leaveOneOutFun",
          expect_true(leaveOneOutFun(m1@covariance@range.val,m1) == leaveOneOutFun(m1@covariance@range.val,as_m1)))

test_that("m1.argmax(loo) == as_m1.argmax(loo)", 
          expect_equal(m1@covariance@range.val,as_m1@covariance@range.val,tol=0.1))

plot(Vectorize(function(.t) leaveOneOutFun(param=as.numeric(.t),model=m1)))
abline(v=m1@covariance@range.val)
plot(Vectorize(function(.t) leaveOneOut(as_m1@Kriging,as.numeric(.t))),add=T,col='red')
abline(v=as_m1@covariance@range.val,col='red')



##########################################################################

library(testthat)

# install.packages("../rlibkriging_0.1-10_R_x86_64-pc-linux-gnu.tar.gz",repos=NULL)
# library(rlibkriging)

context("# A 2D example - Branin-Hoo function")

branin <- function (x) {
  x1 <- x[1] * 15 - 5
  x2 <- x[2] * 15
  (x2 - 5/(4 * pi^2) * (x1^2) + 5/pi * x1 - 6)^2 + 10 * (1 - 1/(8 * pi)) * cos(x1) + 10
}

# a 16-points factorial design, and the corresponding response
d <- 2; n <- 16
design.fact <- expand.grid(x1=seq(0,1,length=4), x2=seq(0,1,length=4))
y <- apply(design.fact, 1, branin) 

library(DiceKriging)
# kriging model 1 : matern5_2 covariance structure, no trend, no nugget effect
m1 <- km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1),control = list(trace=F))
as_m1 <- as_km(design=design.fact, response=y,covtype = "gauss",parinit = c(.5,1))

test_that("m1.logLikFun == as_m1.logLikFun",
          expect_true(logLikFun(m1@covariance@range.val,m1) == logLikFun(m1@covariance@range.val,as_m1)))

test_that("m1.argmax(logLig) == as_m1.argmax(logLig)", 
          expect_equal(m1@covariance@range.val,as_m1@covariance@range.val,tol=0.1))

ll = function(Theta){apply(Theta,1,function(theta) logLikFun(theta,m1))}
as_ll = function(Theta){apply(Theta,1,function(theta) logLikelihood(as_m1@Kriging,theta)$logLikelihood[1])}
t=seq(0.01,2,,51)
contour(t,t,matrix(ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30)
contour(t,t,matrix(as_ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30,add=T,col='red')
points(m1@covariance@range.val[1],m1@covariance@range.val[2])
points(as_m1@covariance@range.val[1],as_m1@covariance@range.val[2],col='red')

p =    predict(   m1,newdata=matrix(.5,ncol=2),type="UK",checkNames=F,light.return=T)
as_p = predict(as_m1,newdata=matrix(.5,ncol=2),type="UK",checkNames=F,light.return=T)

test_that("p$mean,as_p$mean",
          expect_equal(p$mean[1],as_p$mean[1],tol=0.1))
test_that("p$sd,as_p$sd",
          expect_equal(p$sd[1],as_p$sd[1],tol=0.1))


################################################################################

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
  plot(f)
set.seed(123)
X <- as.matrix(runif(5))
y <- f(X)
  points(X,y,col='blue')
r <- Kriging(y, X, "gauss")
x = seq(0,1,,101)
s_x = simulate(r, nsim=3, x=x)
  lines(x,s_x[,1],col='blue')
  lines(x,s_x[,2],col='blue')
  lines(x,s_x[,3],col='blue')
  
# sk_x = simulate(as_km(r), nsim=3, newdata=x)
#   lines(x,sk_x[,1],col='red')
#   lines(x,sk_x[,2],col='red')
#   lines(x,sk_x[,3],col='red')
  
################################################################################
f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
# f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 5
set.seed(123)
X <- cbind(runif(n))
y <- f(X)
d = ncol(X)
plot(X,y)

formula=~1
design=X
response=y
covtype="gauss"

# k <<-       DiceKriging::km(formula = formula,design = design, response = response, covtype = covtype,
#                coef.cov = 0.5, coef.var=0.5, coef.trend = 0.5, 
#                control = list(trace=F))
# NOT working for logLikFun, because @method is not available (bug in DiceKriging ?)
# as_k <<- rlibkriging::as_km(formula = formula,design = design, response = response, covtype = covtype,
#                             coef.cov = 0.5, coef.var=0.5, coef.trend = 0.5)

k <<-       DiceKriging::km(formula = formula,design = design, response = response, covtype = covtype,
                            #coef.cov = 0.5, coef.var=0.5, coef.trend = 0.5, 
                            control = list(trace=F))

as_k <<- rlibkriging::as_km(formula = formula,design = design, response = response, covtype = covtype,
                           coef.cov = k@covariance@range.val, coef.var=k@covariance@sd2, coef.trend = k@trend.coef)

# plot(Vectorize(function(.t)DiceKriging::logLikFun(.t,k)[1]),xlim=c(0.000001,1))
# plot(Vectorize(function(.t)rlibkriging::logLikelihood(as_k@Kriging,.t)$logLikelihood[1]),xlim=c(0.000001,1))

x = runif(ncol(X))
test_that("DiceKriging::logLik == rlibkriging::logLikelihood",
          expect_equal(DiceKriging::logLikFun(x,k)[1],rlibkriging::logLikelihood(as_k@Kriging,x)$logLikelihood[1]))
test_that("DiceKriging::leaveOneOut == rlibkriging::leaveOneOut",
          expect_equal(DiceKriging::leaveOneOutFun(x,k)[1],rlibkriging::leaveOneOut(as_k@Kriging,x)$leaveOneOut[1]))

x = matrix(x,ncol=d)
test_that("DiceKriging::predict == rlibkriging::predict",
          expect_equal(DiceKriging::predict(k,newdata=x,type = "UK",checkNames=F)$mean[1],rlibkriging::predict(as_k,newdata = x,type = "UK")$mean[1],tol=0.01))

x = matrix(X[2,],ncol=d)+0.001
n = 1000
set.seed(123)
sims_k = DiceKriging::simulate(k,nsim = n,newdata = x,checkNames=F,cond=TRUE, nugget.sim=1e-10)
sims_as_k = rlibkriging::simulate(as_k,nsim = n,newdata = x,checkNames=F,cond=TRUE)
t = t.test(sims_k,sims_as_k,var.equal=F)

if (t$p.value<0.05) {
  plot(f)
  points(X,y)
  xx = seq(0,1,,101)
  for (i in 1:100) {
    lines(xx,DiceKriging::simulate(k,nsim = 1,newdata = xx,checkNames=F,cond=TRUE, nugget.sim=1e-10),col=rgb(0,0,1,0.02))
    lines(xx,rlibkriging::simulate(as_k,nsim = 1,newdata = xx,checkNames=F,cond=TRUE, nugget.sim=0),col=rgb(1,0,0,0.02))
  }
}
print(t)
# issue #100
# test_that("DiceKriging::simulate ~= rlibkriging::simulate",
#           expect_true(t$p.value>0.05))
################################################################################


f <- function(X) apply(X, 1, function(x) prod(sin((x*pi-.5)^2)))
n <- 5#100
set.seed(123)
X <- cbind(runif(n))#,runif(n),runif(n))
y <- f(X)
d = ncol(X)
#plot(function(x)f(as.matrix(x)))
#points(X,y)

test_args = function(formula ,design ,response ,covtype, estim.method ) {
  context(paste0("asDiceKriging: ",paste0(sep=", ",formula,
                                        paste0("design ",nrow(design),"x",ncol(design)),
                                        paste0("response ",nrow(response),"x",ncol(response)),
                                        covtype)))
  
  set.seed(123)
  
  parinit = runif(ncol(design))
  k <<-       DiceKriging::km(formula = formula,design = design, response = response, covtype = covtype, estim.method = estim.method, parinit = parinit, control = list(trace=F))
  as_k <<- rlibkriging::as_km(formula = formula,design = design, response = response, covtype = covtype, estim.method = estim.method, parinit = parinit)
  
  #print(k)
  #print(as_k)
  #if (e=="MLE") {
  #  plot(Vectorize(function(t)DiceKriging::logLikFun(t,k)[1]),xlim=c(0.0001,2))
  #} else {
  #  plot(Vectorize(function(t)DiceKriging::leaveOneOutFun(t,k)[1]),xlim=c(0.0001,2))
  #}
  #abline(v=k@covariance@range.val)
  #if (e=="MLE") {
  #  plot(Vectorize(function(t)rlibkriging::logLikelihood(as_k@Kriging,t)$logLikelihood[1]),xlim=c(0.0001,2),add=T,col='red')
  #} else {
  #  plot(Vectorize(function(t)rlibkriging::leaveOneOut(as_k@Kriging,t)$leaveOneOut[1]),xlim=c(0.0001,2),add=T,col='red')
  #}
  #abline(v=as_k@covariance@range.val,col='red')

  t = runif(ncol(X))
  test_that("DiceKriging::logLikFun == rlibkriging::logLikelihood",
            expect_equal(DiceKriging::logLikFun(t,k)[1],rlibkriging::logLikelihood(as_k@Kriging,t)$logLikelihood[1]))
  test_that("DiceKriging::leaveOneOutFun == rlibkriging::leaveOneOut",
            expect_equal(DiceKriging::leaveOneOutFun(t,k)[1],rlibkriging::leaveOneOut(as_k@Kriging,t)$leaveOneOut[1]))
  
  x = matrix(runif(d),ncol=d)
  test_that("DiceKriging::predict == rlibkriging::predict",
            expect_equal(DiceKriging::predict(k,newdata=x,type = "UK",checkNames=F)$mean[1],rlibkriging::predict(as_k,newdata = x,type = "UK")$mean[1],tol=0.01))
  
  n = 1000
  set.seed(123)
  sims_k <<-       DiceKriging::simulate(k,nsim = n,newdata = x,checkNames=F,cond=TRUE, nugget.sim=1e-10)
  sims_as_k <<- rlibkriging::simulate(as_k,nsim = n,newdata = x,checkNames=F,cond=TRUE)
  t = t.test(t(sims_k),sims_as_k,var.equal=F,paired=F)
  print(t)
  # issue #100 
  # test_that("DiceKriging::simulate ~= rlibkriging::simulate",
  #           expect_true(t$p.value>0.05))
}

#### Test the whole matrix of km features already available
for (f in c( ~1 , ~. , ~.^2 ))
  for (co in c("gauss","exp","matern3_2","matern5_2"))
    for (e in c("MLE","LOO")) {
      print(paste0("kernel:",co," objective:",e," trend:",paste0(f,collapse="")))
      test_args(formula = f,design = X,response = y,covtype = co, estim.method = e)
    }
