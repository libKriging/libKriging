library(testthat)
library(rlibkriging)

context("Fit: 1D")

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(1234)
X <- as.matrix(runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F))
r <- Kriging(y, X, "gauss")

ll = Vectorize(function(x) logLikelihood(r,x)$logLikelihood)
plot(ll,xlim=c(0.001,1))
theta_ref = optimize(ll,interval=c(0.001,1),maximum=T)$maximum
abline(v=theta_ref,col='black')
abline(v=as.list(r)$theta,col='red')
abline(v=k@covariance@range.val,col='blue')

test_that(desc="fit of theta by DiceKriging is right", 
          expect_equal(theta_ref, k@covariance@range.val, tol= 1e-3))

test_that(desc="fit of theta by libKriging is right", 
          expect_equal(array(theta_ref), array(as.list(r)$theta), tol= 0.01))
         
#############################################################

context("Fit: 2D (Branin)")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F))
r <- Kriging(y, X, "gauss")

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2); 
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      # print(dim(x))
                      #print(matrix(unlist(x),ncol=2));
                      y=-logLikelihood(r,matrix(unlist(x),ncol=2))$logLikelihood
                      #print(y);
                      y})}
#DiceView::contourview(ll,xlim=c(0.01,2),ylim=c(0.01,2))
x=seq(0.01,2,,51)
contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.01,0.01),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="fit of theta 2D is _quite_ the same that DiceKriging one", 
          expect_equal(ll(array(as.list(r)$theta)), ll(k@covariance@range.val), tol=1e-1))



#############################################################

context("Fit: 2D (Branin) multistart")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)
k = NULL
r = NULL

parinit = matrix(runif(10*ncol(X)),ncol=ncol(X))
k <- tryCatch( # needed to catch warning due to %dopar% usage when using multistart
    withCallingHandlers(
      {
        error_text <- "No error."
        DiceKriging::km(design=X,response=y,covtype = "gauss",multistart = 10, parinit=parinit,control = list(trace=F))
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
r <- Kriging(y, X, "gauss", parameters=list(theta=parinit))

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2); 
# print(dim(X));
apply(X,1,
      function(x) {
        # print(dim(x))
        #print(matrix(unlist(x),ncol=2));
        y=-logLikelihood(r,matrix(unlist(x),ncol=2))$logLikelihood
        #print(y);
        y})}
#DiceView::contourview(ll,xlim=c(0.01,2),ylim=c(0.01,2))
x=seq(0.01,2,,51)
contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.01,0.01),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="fit of theta 2D is _quite_ the same that DiceKriging one", 
          expect_equal(ll(array(as.list(r)$theta)), ll(k@covariance@range.val), tol= 1e-3))



#############################################################

context("Fit: 2D")

f <- function(X) apply(X, 1, 
                       function(x)
                         prod(
                           sin(2*pi*
                                 ( x * (seq(0,1,l=1+length(x))[-1])^2 )
                           )))
logn <- 1 #seq(1, 2.5, by=.1)
n <- floor(10^logn)
d <- 2
set.seed(1234)
X <- matrix(runif(n*d),ncol=d)
y <- f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F))

x=seq(0,2,,51)
mll_fun <- function(x) -apply(x,1,
                              function(theta) 
                                DiceKriging::logLikFun(theta,k)
)
contour(x,x,matrix(mll_fun(expand.grid(x,x)),nrow=length(x)),nlevels = 30)
 
# use same startup point for convergence
r <- Kriging(y, X, "gauss","constant",FALSE,"BFGS","LL",
             parameters=list(theta=matrix(k@parinit,ncol=2)))

points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="fit of theta 2D is the same that DiceKriging one", 
          expect_equal(array(as.list(r)$theta),array(k@covariance@range.val),tol=  5e-2))

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
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F),parinit = c(0.25,10))
r <- Kriging(y, X, "gauss",parameters=list(theta=matrix(c(0.25,10),ncol=2)))

ll_r = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2); 
# print(dim(X));
apply(X,1,
      function(x) {
        # print(dim(x))
        #print(matrix(unlist(x),ncol=2));
        -logLikelihood(r,matrix(unlist(x),ncol=2))$logLikelihood
        #print(y);
        })}
#DiceView::contourview(ll,xlim=c(0.01,2),ylim=c(0.01,2))
x1=seq(0.001,2,,51)
x2=seq(0.001,30,,51)
contour(x1,x2,matrix(ll_r(as.matrix(expand.grid(x1,x2))),nrow=length(x1)),nlevels = 30,col='red')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
ll_r(t(as.list(r)$theta))

ll_k = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2); 
apply(X,1,function(x) {-DiceKriging::logLikFun(x,k)})}
contour(x1,x2,matrix(ll_k(as.matrix(expand.grid(x1,x2))),nrow=length(x1)),nlevels = 30,add=T)
points(k@covariance@range.val[1],k@covariance@range.val[2])
ll_k(k@covariance@range.val)

theta_ref = optim(par=matrix(c(.25,10),ncol=2),ll_r,lower=c(0.001,0.001),upper=c(2,30),method="L-BFGS-B")$par
points(theta_ref,col='black')

test_that(desc="fit of theta 2D is _quite_ the same that DiceKriging one", 
          expect_equal(ll_r(array(as.list(r)$theta)), ll_k(k@covariance@range.val), tol=1e-1))

