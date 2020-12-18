library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss")
r <- kriging(y, X, "gauss")

precision <- 1e-3
test_that(desc="fit of theta is the same that DiceKriging one", 
          expect_true(abs(kriging_model(r)$theta - k@covariance@range.val) < precision))
         
#############################################################

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(123)
X <- cbind(runif(n),runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F))
r <- kriging(y, X, "gauss","constant",FALSE,"Newton","LL")

kriging_model(r)$theta
k@covariance@range.val

precision <- 1e-3
test_that(desc="fit of theta 2D is the same that DiceKriging one", 
          expect_true(abs(kriging_model(r)$theta - k@covariance@range.val) < precision))

#############################################################

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
 
r <- kriging(y, X, "gauss","constant",FALSE,"Newton","LL",parameters=list(sigma2=0,has_sigma2=FALSE,theta=matrix(k@parinit,ncol=2),has_theta=TRUE))

kriging_model(r)$theta
k@covariance@range.val

precision <- 1e-3
test_that(desc="fit of theta 2D is the same that DiceKriging one", 
          expect_true(abs(kriging_model(r)$theta - k@covariance@range.val) < precision))
