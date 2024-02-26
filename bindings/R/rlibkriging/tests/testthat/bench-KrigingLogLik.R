library(testthat)

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 100
set.seed(123)
X <- cbind(runif(n),runif(n),runif(n),runif(n))
y <- f(X)

xx <- seq(0.01, 4,, 101)
times <- list(R_ll=rep(NA, length(xx)), R_gll=rep(NA, length(xx)), 
              cpp_ll=rep(NA, length(xx)), cpp_gll=rep(NA, length(xx)),
              cpp_nf_ll=rep(NA, length(xx)), cpp_nf_gll=rep(NA, length(xx)))
times.n = 100

k = NULL # ensure stop if km fails
k <- DiceKriging::km(design=X, response=y, covtype = "gauss")
i <- 1
for (x in xx){
  times$R_ll[i]=system.time(for (j in 1:times.n) llx <- DiceKriging::logLikFun(rep(x,4),k))
  times$R_gll[i]=system.time(for (j in 1:times.n) { envx <- new.env(); llx <- DiceKriging::logLikFun(rep(x,4),k,envx); gllx <- DiceKriging::logLikGrad(rep(x,4),k,envx)})
  i <- i+1
}

library(rlibkriging, lib.loc="bindings/R/Rlibs")

rlibkriging:::covariance_use_approx_singular(TRUE)

r = NULL # ensure stop if Kriging fails
r <- Kriging(y, X, "gauss")
i <- 1
for (x in xx){
  times$cpp_ll[i]=system.time(for (j in 1:times.n) ll2x <- logLikelihoodFun(r,rep(x,4)))
  times$cpp_gll[i]=system.time(for (j in 1:times.n) gll2x <- logLikelihoodFun(r,rep(x,4),grad=T))
  i <- i+1
}

rlibkriging:::covariance_use_approx_singular(FALSE)

r_nf <- Kriging(y, X, "gauss")
i <- 1
for (x in xx){
  times$cpp_nf_ll[i]=system.time(for (j in 1:times.n) ll2x_f <- logLikelihoodFun(r_nf,rep(x,4)))
  times$cpp_nf_gll[i]=system.time(for (j in 1:times.n) gll2x_f <- logLikelihoodFun(r_nf,rep(x,4),grad=T))
  i <- i+1
}

rlibkriging:::covariance_use_approx_singular(TRUE)


plot(xx,Vectorize(function(x)DiceKriging::logLikFun(rep(x,4),k))(xx))
rlibkriging:::covariance_use_approx_singular(TRUE)
points(xx,Vectorize(function(x)logLikelihoodFun(r,rep(x,4))$logLikelihood)(xx),col='red')
rlibkriging:::covariance_use_approx_singular(FALSE)
points(xx,Vectorize(function(x)logLikelihoodFun(r_nf,rep(x,4))$logLikelihood)(xx),col='orange')
rlibkriging:::covariance_use_approx_singular(TRUE)

# plot rcond
y0 = min(Vectorize(function(x)logLikelihoodFun(r,rep(x,4))$logLikelihood)(xx))
y1 = max(Vectorize(function(x)logLikelihoodFun(r,rep(x,4))$logLikelihood)(xx))-y0
rc_xx = array(NA,length(xx))
for (i in 1:length(xx)) {
  rx <- Kriging(y, X, "gauss", optim="none", parameters=list(theta=matrix(rep(xx[i],4)),sigma2=1))
  rc_xx[i] = log(rlibkriging:::linalg_rcond_chol(rx$T()))
}
lines(xx,y0+y1*(rc_xx-min(rc_xx))/(max(rc_xx)-min(rc_xx)),col='red',lty=2)
abline(h=y0+y1*(log(1e-15*nrow(X))-min(rc_xx))/(max(rc_xx)-min(rc_xx)),col='red',lty=2)


plot(xx,log(times$R_ll),ylim=c(log(min(min(times$R_ll,na.rm = T),min(times$cpp_ll,na.rm = T))),log(max(max(times$R_ll,na.rm = T),max(times$cpp_ll,na.rm = T)))),xlab="x",ylab="log(user_time (s))", panel.first=grid())
text(20,-1,"DiceKriging")
points(xx,log(times$cpp_ll),col='red')
text(80,-1,"libKriging",col = 'red')
points(xx,log(times$cpp_nf_ll),col='orange')
text(80,-1,"libKriging fast (no ll quad failover)",col = 'orange')

par(mfrow=c(1,2))
plot(times$R_ll,times$cpp_ll,main="times R/cpp",col='blue')
points(times$R_ll,times$cpp_nf_ll,main="times R/cpp approx singular")
abline(a = 0,b=1,col='red')
abline(a = 0,b=2,col='orange')
abline(a = 0,b=.5,col='orange')
plot(times$R_gll,times$cpp_gll,main="times R/cpp - grad",col='blue')
points(times$R_gll,times$cpp_nf_gll,main="times R/cpp approx singular - grad")
abline(a = 0,b=1,col='red')
abline(a = 0,b=2,col='orange')
abline(a = 0,b=.5,col='orange')
par(mfrow=c(1,1))

plot(times$cpp_nf_ll,times$cpp_ll,main="times R/cpp",col='blue')
abline(a = 0,b=1,col='red')
abline(a = 0,b=2,col='orange')
abline(a = 0,b=.5,col='orange')