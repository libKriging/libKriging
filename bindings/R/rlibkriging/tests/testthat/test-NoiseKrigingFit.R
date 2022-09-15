library(testthat)
#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#rlibkriging:::optim_log(3)

context("Fit: 1D")

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X) + 0.1*rnorm(nrow(X))
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,noise.var=rep(0.1^2,nrow(X)),covtype = "gauss",control = list(trace=F),nugget.estim=F,optim.method='BFGS',multistart = 20)
r <- NoiseKriging(y,rep(0.1^2,nrow(X)), X, "gauss", optim = "BFGS")
l = as.list(r)

ll = Vectorize(function(x) logLikelihoodFun(r,c(x,k@covariance@sd2))$logLikelihood)
plot(ll,xlim=c(0.000001,1))
  for (x in seq(0.000001,1,,11)){
    envx = new.env()
    ll2x = logLikelihoodFun(r,c(x,k@covariance@sd2))$logLikelihood
    gll2x = logLikelihoodFun(r,c(x,k@covariance@sd2),grad = T)$logLikelihoodGrad[1]
    arrows(x,ll2x,x+.1,ll2x+.1*gll2x,col='red')
  }

theta_ref = optimize(ll,interval=c(0.001,1),maximum=T)$maximum
abline(v=theta_ref,col='black')
abline(v=as.list(r)$theta,col='red')
abline(v=k@covariance@range.val,col='blue')

test_that(desc="Noise / Fit: 1D / fit of theta by DiceKriging is right",
          expect_equal(theta_ref, k@covariance@range.val, tol= 1e-3))

test_that(desc="Noise / Fit: 1D / fit of theta by libKriging is right",
          expect_equal(array(theta_ref), array(as.list(r)$theta), tol= 0.01))

#############################################################

context("Fit: 2D (Branin)")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)+ 10*rnorm(nrow(X))
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,noise.var=rep(10^2,nrow(X)),covtype = "gauss",control = list(trace=F),nugget.estim=F,optim.method='BFGS',multistart = 20)
r <- NoiseKriging(y, noise=rep(10^2,nrow(X)),X, "gauss", optim = "BFGS")
#plot(Vectorize(function(a) r$logLikelihoodFun(c(r$theta(),a))$logLikelihood))
l = as.list(r)

# save(list=ls(),file="fit-nugget-2d.Rdata")

sigma2_k = k@covariance@sd2
sigma2_r = as.list(r)$sigma2
test_that(desc="Noise / Fit: 2D (Branin) / fit of LL by DiceKriging is same that libKriging",
          expect_equal(k@logLik,r$logLikelihood(), tol= 1e-2))

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      y=-logLikelihoodFun(r,c(unlist(x),sigma2_k))$logLikelihood
                      #print(y);
                      y})}
#DiceView::contourview(ll,xlim=c(0.1,2),ylim=c(0.1,2))
x=seq(0.1,1,,51)
contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.1,0.1),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="Noise / Fit: 2D (Branin) / fit of theta 2D is _quite_ the same that DiceKriging one",
          expect_equal(ll(array(as.list(r)$theta)), ll(k@covariance@range.val), tol=1e-1))



#############################################################

context("Fit: 2D (Branin) multistart")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X) + 10*rnorm(nrow(X))
k = NULL
r = NULL

parinit = matrix(runif(10*ncol(X)),ncol=ncol(X))
k <- tryCatch( # needed to catch warning due to %dopar% usage when using multistart
    withCallingHandlers(
      {
        error_text <- "No error."
        DiceKriging::km(design=X,response=y,noise.var=rep(10^2,nrow(X)),covtype = "gauss", parinit=parinit,control = list(trace=F),nugget.estim=F,optim.method='BFGS',multistart = 20)
      },
      warning = function(e) {
        error_text <<- trimws(paste0("WARNING: ", e))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      return(list(value = NA, error_text = trimws(paste0("ERROR: ", e))))
    },
    finally = {
    }
  )
r <- NoiseKriging(y,noise=rep(10^2,nrow(X)), X, "gauss", parameters=list(theta=parinit))
l = as.list(r)

# save(list=ls(),file="fit-nugget-multistart.Rdata")

sigma2_k = k@covariance@sd2
sigma2_r = as.list(r)$sigma2
test_that(desc="Noise / Fit: 2D (Branin) multistart / fit of LL by DiceKriging is same that libKriging",
          expect_equal(k@logLik,r$logLikelihood(), tol= 0.01))

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
# print(dim(X));
apply(X,1,
      function(x) {
        # print(dim(x))
        #print(matrix(unlist(x),ncol=2));
        y=-logLikelihoodFun(r,c(unlist(x),sigma2_k))$logLikelihood
        #print(y);
        y})}
#DiceView::contourview(ll,xlim=c(0.1,2),ylim=c(0.1,2))
x=seq(0.1,2,,51)
contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.1,0.1),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="Noise / Fit: 2D (Branin) multistart / fit of theta 2D is _quite_ the same that DiceKriging one",
          expect_equal(ll(array(as.list(r)$theta)), ll(k@covariance@range.val), tol= 1e-1))


################################################################################

context("Fit: 2D _not_ in [0,1]^2")

# "unnormed" version of Branin: [0,1]x[0,15] -> ...
branin_15 <- function (x) {
  x1 <- x[1] * 15 - 5
  x2 <- x[2] #* 15
  (x2 - 5/(4 * pi^2) * (x1^2) + 5/pi * x1 - 6)^2 + 10 * (1 - 1/(8 * pi)) * cos(x1) + 10
}

f = function(X) apply(X,1,branin_15)
n <- 15
set.seed(1234)
X <- cbind(runif(n,0,1),runif(n,0,15))
y = f(X) + 10*rnorm(nrow(X))
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,noise.var=rep(10^2,nrow(X)),covtype = "gauss",control = list(trace=F),nugget.estim=FALSE,optim="BFGS",multistart=20)#,parinit = c(0.5,5))
r <- NoiseKriging(y,noise=rep(10^2,nrow(X)), X, "gauss",, optim = "BFGS")#, parameters=list(theta=matrix(c(0.5,5),ncol=2)))
l = as.list(r)

# save(list=ls(),file="fit-nugget-2d-not01.Rdata")

sigma2_k = k@covariance@sd2
sigma2_r = as.list(r)$sigma2
test_that(desc="Noise / Fit: 2D _not_ in [0,1]^2 / fit of LL by DiceKriging is same that libKriging",
          expect_equal(k@logLik,r$logLikelihood(), tol= 0.01))

ll_r = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
# print(dim(X));
apply(X,1,
      function(x) {
        # print(dim(x))
        #print(matrix(unlist(x),ncol=2));
        -logLikelihoodFun(r,c(unlist(x),sigma2_k))$logLikelihood
        #print(y);
        })}
#DiceView::contourview(ll,xlim=c(0.1,2),ylim=c(0.1,2))
x1=seq(0.001,2,,51)
x2=seq(0.001,30,,51)
contour(x1,x2,matrix(ll_r(as.matrix(expand.grid(x1,x2))),nrow=length(x1)),nlevels = 30,col='red')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
ll_r(t(as.list(r)$theta))

ll_k = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
apply(X,1,function(x) {-DiceKriging::logLikFun(c(x,sigma2_k),k)})}
contour(x1,x2,matrix(ll_k(as.matrix(expand.grid(x1,x2))),nrow=length(x1)),nlevels = 30,add=T)
points(k@covariance@range.val[1],k@covariance@range.val[2])
ll_k(k@covariance@range.val)

theta_ref = optim(par=matrix(c(.2,10),ncol=2),ll_r,lower=c(0.001,0.001),upper=c(2,30),method="L-BFGS-B")$par
points(theta_ref,col='black')

test_that(desc="Noise / Fit: 2D _not_ in [0,1]^2 / fit of theta 2D is _quite_ the same that DiceKriging one",
          expect_equal(ll_r(array(as.list(r)$theta)), ll_k(k@covariance@range.val), tol=1e-1))

