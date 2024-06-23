#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#library(testthat)

context("Fit: 1D")

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F),nugget.estim=T,optim.method='BFGS',multistart = 20)
r <- NuggetKriging(y, X, "gauss", optim = "BFGS20")
l = as.list(r)

# save(list=ls(),file="fit-nugget-1d.Rdata")


alpha_k = k@covariance@sd2/(k@covariance@sd2+k@covariance@nugget)
alpha_r = as.list(r)$sigma2/(as.list(r)$sigma2+as.list(r)$nugget)
test_that(desc="Nugget / Fit: 1D / fit of alpha by DiceKriging is same that libKriging",
          expect_equal(alpha_k,alpha_r, tol= 1e-4))

ll_a = Vectorize(function(a) logLikelihoodFun(r,c(k@covariance@range.val,a))$logLikelihood)
plot(ll_a,xlim=c(0.001,1),lwd=3)
llk_a = Vectorize(function(a) DiceKriging::logLikFun(model=k,c(k@covariance@range.val,a)))
curve(llk_a, add=TRUE, col='blue')
for (a in seq(0.01,0.99,,21)){
  envx = new.env()
  ll2x = logLikelihoodFun(r,c(k@covariance@range.val,a))$logLikelihood
  gll2x = logLikelihoodFun(r,c(k@covariance@range.val,a),return_grad = T)$logLikelihoodGrad[,2]
  arrows(a,ll2x,a+.1,ll2x+.1*gll2x,col='red')
}
abline(v=alpha_k,col='blue')
abline(v=alpha_r,col='red')

ll_t = Vectorize(function(x) logLikelihoodFun(r,c(x,alpha_k))$logLikelihood)
plot(ll_t,xlim=c(0.001,1))
#ll = Vectorize(function(x) logLikelihoodFun(r,c(x,alpha_r))$logLikelihood)
#plot(ll_,xlim=c(0.001,1))

theta_ref = optimize(ll_t,interval=c(0.001,1),maximum=T)$maximum
abline(v=theta_ref,col='black')
abline(v=as.list(r)$theta,col='red')
abline(v=k@covariance@range.val,col='blue')

test_that(desc="Nugget / Fit: 1D / fit of theta by DiceKriging is right",
          expect_equal(theta_ref, k@covariance@range.val, tol= 1e-3))

test_that(desc="Nugget / Fit: 1D / fit of theta by libKriging is right",
          expect_equal(array(theta_ref), array(as.list(r)$theta), tol= 0.01))

# see joint ll over theta & alpha
# ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
# apply(X,1,
#      function(x) {
#        y=-logLikelihoodFun(r,c(unlist(x)))$logLikelihood
#        #print(y);
#        y})}
# x=seq(0.01,0.99,,51)
# without reparam: 
# contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 50)
# abline(v=(theta_ref),col='black')
# abline(v=(as.list(r)$theta),col='red')
# abline(v=(k@covariance@range.val),col='blue')
# abline(h=(alpha_k),col='blue')
# abline(h=(alpha_r),col='red')
# with reparam:
# contour(log(x),-log(1-x),matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 50)
# abline(v=log(theta_ref),col='black')
# abline(v=log(as.list(r)$theta),col='red')
# abline(v=log(k@covariance@range.val),col='blue')
# abline(h=-log(1-alpha_k),col='blue')
# abline(h=-log(1-alpha_r),col='red')

#############################################################

context("Fit: 1D, nugget preset")

f = function(x) 1-1/2*(sin(12*x)/(1+x)+2*cos(7*x)*x^5+0.7)
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)
nu=0.1
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F),nugget.estim=FALSE, nugget = nu,optim.method='BFGS',multistart = 20)
#equivalent to NoiseKriging, not NuggetKriging: 
rr <- NoiseKriging(y, rep(0.1,nrow(y)), X, "gauss", optim = "BFGS20")
r <- NuggetKriging(y, X, "gauss", optim = "BFGS20", parameters=list(nugget=nu, is_nugget_estim=FALSE ))
l = as.list(r)

# save(list=ls(),file="fit-nuggetpreset-1d.Rdata")

alpha_k = k@covariance@sd2/(k@covariance@sd2+k@covariance@nugget)
alpha_r = as.list(r)$sigma2/(as.list(r)$sigma2+as.list(r)$nugget)
test_that(desc="Nugget / Fit: 1D / fit of alpha by DiceKriging is same that libKriging",
          expect_equal(alpha_k,alpha_r, tol= 1e-4))

theta=k@covariance@range.val #r$theta()
ll_a = Vectorize(function(a) r$logLikelihoodFun(c(theta,a))$logLikelihood)
plot(ll_a,xlim=c(0.1,1),lwd=3)
llk_a = Vectorize(function(a) {s2 = nu*a/(1-a); DiceKriging::logLikFun(model=k,c(theta,s2))})
curve(llk_a, add=TRUE, col='blue',xlim=c(0.1,0.999))
for (a in seq(0.01,0.99,,21)){
  ll2x = r$logLikelihoodFun(c(theta,a))$logLikelihood
  gll2x = r$logLikelihoodFun(c(theta,a),return_grad = T)$logLikelihoodGrad[,2]  
  arrows(a,ll2x,a+.1,ll2x+.1*gll2x,col='red', lwd=5)

  envx = new.env()
  
  s2 = nu*a/(1-a)
  ll2x_k = DiceKriging::logLikFun(c(theta,s2),k, envir=envx)
  gll2x_k = DiceKriging::logLikGrad(c(theta,s2),k, envir=envx)[2] * nu/(1-a)^2 # chain rule
  arrows(a,ll2x_k,a+.1,ll2x_k+.1*gll2x_k,col='blue',lwd=3)

  ll2x = rr$logLikelihoodFun(c(theta,s2))$logLikelihood
  gll2x = rr$logLikelihoodFun(c(theta,s2),return_grad = T)$logLikelihoodGrad[,2]* nu/(1-a)^2
  arrows(a,ll2x,a+.1,ll2x+.1*gll2x,col='green')
}
abline(v=alpha_k,col='blue')
abline(v=alpha_r,col='red')

ll_t = Vectorize(function(x) r$logLikelihoodFun(c(x,alpha_k))$logLikelihood)
plot(ll_t,xlim=c(0.001,1))
llk_t = Vectorize(function(x) DiceKriging::logLikFun(model=k,c(x,alpha_k)))
curve(llk_t, add=TRUE, col='blue')
#ll = Vectorize(function(x) logLikelihoodFun(r,c(x,alpha_r))$logLikelihood)
#plot(ll_,xlim=c(0.001,1))

theta_ref = optimize(ll_t,interval=c(0.001,1),maximum=T)$maximum
abline(v=theta_ref,col='black')
abline(v=as.list(r)$theta,col='red')
abline(v=k@covariance@range.val,col='blue')

test_that(desc="Nugget / Fit: 1D / fit of theta by DiceKriging is right",
          expect_equal(theta_ref, k@covariance@range.val, tol= 1e-3))

test_that(desc="Nugget / Fit: 1D / fit of theta by libKriging is right",
          expect_equal(array(theta_ref), array(as.list(r)$theta), tol= 0.01))

# see joint ll over theta & alpha
# ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
# apply(X,1,
#      function(x) {
#        y=-logLikelihoodFun(r,c(unlist(x)))$logLikelihood
#        #print(y);
#        y})}
# x=seq(0.01,0.99,,51)
# without reparam: 
# contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 50)
# abline(v=(theta_ref),col='black')
# abline(v=(as.list(r)$theta),col='red')
# abline(v=(k@covariance@range.val),col='blue')
# abline(h=(alpha_k),col='blue')
# abline(h=(alpha_r),col='red')
# with reparam:
# contour(log(x),-log(1-x),matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 50)
# abline(v=log(theta_ref),col='black')
# abline(v=log(as.list(r)$theta),col='red')
# abline(v=log(k@covariance@range.val),col='blue')
# abline(h=-log(1-alpha_k),col='blue')
# abline(h=-log(1-alpha_r),col='red')

#############################################################

context("Fit: 2D (Branin)")

f = function(X) apply(X,1,DiceKriging::branin)
n <- 15
set.seed(1234)
X <- cbind(runif(n),runif(n))
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F),nugget.estim=T,optim.method='BFGS',multistart = 1)
#rlibkriging:::optim_log(4)
#rlibkriging:::optim_use_variogram_bounds_heuristic(T)
#rlibkriging:::optim_set_max_iteration(100)
r <- NuggetKriging(y, X, "gauss", optim = "BFGS")
#plot(Vectorize(function(a) r$logLikelihoodFun(c(r$theta(),a))$logLikelihood))
#sectionview(function(ta)r$logLikelihoodFun(ta)$logLikelihood,center=c(r$theta(),r$sigma2()/(r$sigma2()+r$nugget())))
l = as.list(r)

# save(list=ls(),file="fit-nugget-2d.Rdata")

alpha_k = k@covariance@sd2/(k@covariance@sd2+k@covariance@nugget)
alpha_r = as.list(r)$sigma2/(as.list(r)$sigma2+as.list(r)$nugget)
test_that(desc="Nugget / Fit: 2D (Branin) / fit of alpha by DiceKriging is same that libKriging",
          expect_equal(alpha_k,alpha_r, tol= 1e-3))

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
                  # print(dim(X));
                  apply(X,1,
                    function(x) {
                      y=-logLikelihoodFun(r,c(unlist(x),alpha_k))$logLikelihood
                      #print(y);
                      y})}
#DiceView::contourview(ll,dim=2,Xlim=c(0.01,2))
x=seq(0.01,2,,51)
contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.01,0.01),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="Nugget / Fit: 2D (Branin) / fit of theta 2D is _quite_ the same that DiceKriging one",
          expect_equal(ll(array(as.list(r)$theta)), ll(k@covariance@range.val), tol=1e-1))

#lll = function(ta) r$logLikelihoodFun(ta)$logLikelihood
#DiceView::sectionview(lll,vectorized=T,center=c(r$theta(),r$sigma2()/(r$sigma2()+r$nugget())))

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
        DiceKriging::km(design=X,response=y,covtype = "gauss", parinit=parinit,control = list(trace=F),nugget.estim=T,optim.method='BFGS',multistart = 20)
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
r <- NuggetKriging(y, X, "gauss", optim = "BFGS20", parameters=list(theta=parinit))
l = as.list(r)

# save(list=ls(),file="fit-nugget-multistart.Rdata")

alpha_k = k@covariance@sd2/(k@covariance@sd2+k@covariance@nugget)
alpha_r = as.list(r)$sigma2/(as.list(r)$sigma2+as.list(r)$nugget)
test_that(desc="Nugget / Fit: 2D (Branin) multistart / fit of alpha by DiceKriging is same that libKriging",
          expect_equal(alpha_k,alpha_r, tol= 1e-4))

ll = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
# print(dim(X));
apply(X,1,
      function(x) {
        # print(dim(x))
        #print(matrix(unlist(x),ncol=2));
        y=-logLikelihoodFun(r,c(unlist(x),alpha_k))$logLikelihood
        #print(y);
        y})}
#DiceView::contourview(ll,xlim=c(0.01,2),ylim=c(0.01,2))
x=seq(0.01,2,,51)
contour(x,x,matrix(ll(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)

theta_ref = optim(par=matrix(c(.2,.5),ncol=2),ll,lower=c(0.01,0.01),upper=c(2,2),method="L-BFGS-B")$par
points(theta_ref,col='black')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
points(k@covariance@range.val[1],k@covariance@range.val[2],col='blue')

test_that(desc="Nugget / Fit: 2D (Branin) multistart / fit of theta 2D is _quite_ the same that DiceKriging one",
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
y = f(X)
k = NULL
r = NULL
k = DiceKriging::km(design=X,response=y,covtype = "gauss",control = list(trace=F),nugget.estim=TRUE,optim="BFGS",multistart=20)#,parinit = c(0.5,5))
r <- NuggetKriging(y, X, "gauss",, optim = "BFGS")#, parameters=list(theta=matrix(c(0.5,5),ncol=2)))
l = as.list(r)

# save(list=ls(),file="fit-nugget-2d-not01.Rdata")

alpha_k = k@covariance@sd2/(k@covariance@sd2+k@covariance@nugget)
alpha_r = as.list(r)$sigma2/(as.list(r)$sigma2+as.list(r)$nugget)
test_that(desc="Nugget / Fit: 2D _not_ in [0,1]^2 / fit of alpha by DiceKriging is same that libKriging",
          expect_equal(alpha_k,alpha_r, tol= 1e-4))

ll_r = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
# print(dim(X));
apply(X,1,
      function(x) {
        # print(dim(x))
        #print(matrix(unlist(x),ncol=2));
        -logLikelihoodFun(r,c(unlist(x),alpha_k))$logLikelihood
        #print(y);
        })}
#DiceView::contourview(ll,xlim=c(0.01,2),ylim=c(0.01,2))
x1=seq(0.001,2,,51)
x2=seq(0.001,30,,51)
contour(x1,x2,matrix(ll_r(as.matrix(expand.grid(x1,x2))),nrow=length(x1)),nlevels = 30,col='red')
points(as.list(r)$theta[1],as.list(r)$theta[2],col='red')
ll_r(t(as.list(r)$theta))

ll_k = function(X) {if (!is.matrix(X)) X = matrix(X,ncol=2);
apply(X,1,function(x) {-DiceKriging::logLikFun(c(x,alpha_k),k)})}
contour(x1,x2,matrix(ll_k(as.matrix(expand.grid(x1,x2))),nrow=length(x1)),nlevels = 30,add=T)
points(k@covariance@range.val[1],k@covariance@range.val[2])
ll_k(k@covariance@range.val)

theta_ref = optim(par=matrix(c(.2,10),ncol=2),ll_r,lower=c(0.001,0.001),upper=c(2,30),method="L-BFGS-B")$par
points(theta_ref,col='black')

test_that(desc="Nugget / Fit: 2D _not_ in [0,1]^2 / fit of theta 2D is _quite_ the same that DiceKriging one",
          expect_equal(ll_r(array(as.list(r)$theta)), ll_k(k@covariance@range.val), tol=1e-1))

