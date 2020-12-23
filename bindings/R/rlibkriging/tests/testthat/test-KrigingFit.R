library(testthat)

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(1234)
X <- as.matrix(runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss")
r <- kriging(y, X, "gauss")

ll = Vectorize(function(x) kriging_logLikelihood(r,x))
plot(ll,xlim=c(0.001,1))
theta_ref = optimize(ll,interval=c(0.001,1),maximum=T)$maximum
abline(v=theta_ref,col='black')
abline(v=kriging_model(r)$theta,col='red')
abline(v=k@covariance@range.val,col='blue')

precision <- 1e-3
test_that(desc="fit of theta by DiceKriging is right", 
          expect_equal(theta_ref, k@covariance@range.val, tol= precision))

test_that(desc="fit of theta by libKriging is right", 
          expect_equal(array(theta_ref), array(kriging_model(r)$theta), tol= precision))
         
#############################################################

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F))
r <- kriging(y, X, "gauss")

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2); 
                  #print(X);
                  apply(X,1,
                    function(x) {#print(matrix(unlist(x),ncol=2));
                      y=-kriging_logLikelihood(r,matrix(unlist(x),ncol=2))
                      #print(y);
                      y})}
#DiceView::contourview(ll,xlim=c(0.01,2),ylim=c(0.01,2))
x=seq(0.01,2,,51)
contour(x,x,matrix(ll(expand.grid(x,x)),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.01,0.01),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(kriging_model(r)$theta[1],kriging_model(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')


precision <- 1e-1
test_that(desc="fit of theta 2D is _quite_ the same that DiceKriging one", 
          expect_equal(ll(array(kriging_model(r)$theta)), ll(k@covariance@range.val), tol=precision))

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
 
# use same startup point for convergence
r <- kriging(y, X, "gauss","constant",FALSE,"BFGS","LL",parameters=list(sigma2=0,has_sigma2=FALSE,theta=matrix(k@parinit,ncol=2),has_theta=TRUE))

points(kriging_model(r)$theta[1],kriging_model(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

precision <- 5e-2
test_that(desc="fit of theta 2D is the same that DiceKriging one", 
          expect_equal(array(kriging_model(r)$theta),array(k@covariance@range.val),tol= precision))


