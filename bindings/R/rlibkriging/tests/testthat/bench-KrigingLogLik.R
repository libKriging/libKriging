library(testthat)

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 100
set.seed(123)
X <- cbind(runif(n),runif(n),runif(n))
y <- f(X)

xx <- seq(0.01, 1,, 21)
times <- list(R_ll=rep(NA, length(xx)), R_gll=rep(NA, length(xx)), cpp_ll=rep(NA, length(xx)), cpp_gll=rep(NA, length(xx)))
times.n = 100

k <- DiceKriging::km(design=X, response=y, covtype = "gauss")
i <- 1
for (x in xx){
  times$R_ll[i]=system.time(for (j in 1:times.n) llx <- DiceKriging::logLikFun(rep(x,3),k))
  times$R_gll[i]=system.time(for (j in 1:times.n) { envx <- new.env(); llx <- DiceKriging::logLikFun(rep(x,3),k,envx); gllx <- DiceKriging::logLikGrad(rep(x,3),k,envx)})
  i <- i+1
}

 pack=list.files(file.path("bindings","R"),pattern = ".tar.gz",full.names = T)
 install.packages(pack,repos=NULL)
 library(rlibkriging)
r <- Kriging(y, X, "gauss")
i <- 1
for (x in xx){
  times$cpp_ll[i]=system.time(for (j in 1:times.n) ll2x <- logLikelihood(r,rep(x,3)))
  times$cpp_gll[i]=system.time(for (j in 1:times.n) gll2x <- logLikelihood(r,rep(x,3),grad=T))
  i <- i+1
}

plot(xx,Vectorize(function(x)DiceKriging::logLikFun(rep(x,3),k))(xx))
points(xx,Vectorize(function(x)logLikelihood(r,rep(x,3))$logLikelihood)(xx),col='red')

plot(xx,log(times$R_ll),ylim=c(log(min(min(times$R_ll,na.rm = T),min(times$cpp_ll,na.rm = T))),log(max(max(times$R_ll,na.rm = T),max(times$cpp_ll,na.rm = T)))),xlab="x",ylab="log(user_time (s))", panel.first=grid())
text(20,-1,"DiceKriging")
points(xx,log(times$cpp_ll),col='red')
text(80,-1,"libKriging",col = 'red')

hist(times$R_ll/times$cpp_ll,main="times R/cpp")
abline(v=1.0,col='red')

hist(times$R_gll/times$cpp_gll,main="times R/cpp")
abline(v=1.0,col='red')
