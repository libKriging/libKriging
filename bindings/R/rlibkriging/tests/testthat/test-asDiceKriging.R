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
          expect_equal(m1@covariance@range.val,as_m1@covariance@range.val,tol=0.01))

ll = function(Theta){apply(Theta,1,function(theta) logLikFun(theta,m1))}
as_ll = function(Theta){apply(Theta,1,function(theta) kriging_logLikelihood(as_m1@Kriging,theta)$logLikelihood)}
t=seq(0.01,2,,51)
contour(t,t,matrix(ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30)
contour(t,t,matrix(as_ll(as.matrix(expand.grid(t,t))),nrow=length(t)),nlevels = 30,add=T,col='red')
points(m1@covariance@range.val[1],m1@covariance@range.val[2])
points(as_m1@covariance@range.val[1],as_m1@covariance@range.val[2],col='red')

p = predict(m1,newdata=matrix(.5,ncol=2),type="UK",checkNames=F,light.return=T)
as_p =predict(as_m1,newdata=matrix(.5,ncol=2),type="UK",checkNames=F,light.return=T)

test_that("p$mean,as_p$mean",
          expect_equal(p$mean[1],as_p$mean[1],tol=0.1))
test_that("p$sd,as_p$sd",
          expect_equal(p$sd[1],as_p$sd[1],tol=0.1))



################################################################################


f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 100
set.seed(123)
X <- cbind(runif(n),runif(n),runif(n))
y <- f(X)
d = ncol(X)

test_args = function(formula ,design ,response ,covtype, estim.method ) {
  context(paste0("asDiceKriging: ",paste0(sep=", ",formula,
                                        paste0("design ",nrow(design),"x",ncol(design)),
                                        paste0("response ",nrow(response),"x",ncol(response)),
                                        covtype)))
  
  set.seed(123)
  
  parinit = runif(ncol(design))
  k <<- DiceKriging::km(formula = formula,design = design, response = response, covtype = covtype, estim.method = estim.method, parinit = parinit,control = list(trace=F))
  as_k <<- as_km(formula = formula,design = design, response = response, covtype = covtype, estim.method = estim.method, parinit = parinit)
    
  x = runif(ncol(X))
  test_that("DiceKriging::logLikFun == rlibkriging::logLikelihood",expect_equal(DiceKriging::logLikFun(x,k)[1],rlibkriging::logLikelihood(as_k@Kriging,x)$logLikelihood[1]))
  test_that("DiceKriging::leaveOneOutFun == rlibkriging::leaveOneOut",expect_equal(DiceKriging::leaveOneOutFun(x,k)[1],rlibkriging::leaveOneOut(as_k@Kriging,x)$leaveOneOut[1]))
  
  x = matrix(x,ncol=d)
  test_that("DiceKriging::predict == rlibkriging::predict",expect_equal(DiceKriging::predict(k,newdata=x,type = "UK",checkNames=F)$mean[1],rlibkriging::predict(as_k,newdata = x,type = "UK")$mean[1],tol=0.01))
  
  n = 1000
  set.seed(123)
  sims_k = DiceKriging::simulate(k,nsim = n,newdata = x,checkNames=F,cond=TRUE)
  sims_as_k = rlibkriging::simulate(as_k,nsim = n,newdata = x,checkNames=F,cond=TRUE)
  t = t.test(sims_k,sims_as_k,var.equal=F)
  test_that("DiceKriging::simulate ~= rlibkriging::simulate",expect_true(t$p.value<0.001))
}

#### Test the whole matrix of km features already available
for (f in c( ~1 , ~. , ~.^2 ))
  for (co in c("gauss","exp"))
    for (e in c("MLE","LOO"))
      test_args(formula = f,design = X,response = y,covtype = co, estim.method = e)
