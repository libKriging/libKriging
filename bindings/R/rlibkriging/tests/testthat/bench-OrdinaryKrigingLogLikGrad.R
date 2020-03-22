library(testthat)

f = function(X) apply(X,1,function(x) sum(x^2))

logn = seq(1,2.5,by=.1)
times = list(R=rep(NA,length(logn)),cpp=rep(NA,length(logn)))
N = 1000

theta=0.5

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- floor(2+i/3)
  
  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y = f(X)
  
  k = DiceKriging::km(design=X,response=y,covtype = "gauss")
  gllx = rep(0,ncol(X))
  times$R[i] = system.time(
    try({for (t in 1:N) {envx = new.env();DiceKriging::logLikFun(rep(theta,ncol(X)),k,envx);gllx <- gllx+ DiceKriging::logLikGrad(rep(theta,ncol(X)),k,envx)}})
  )[1]
  gllx=gllx/N
  
  r <- ordinary_kriging(y, X)
  times$cpp[i] = system.time(
    try({gll2x <- bench_loglikgrad(N,r,rep(theta,ncol(X)))})
    # try({for (t in 1:N) ll2x <- ordinary_kriging_loglikelihoodgrad(r,rep(theta,ncol(X)))}) # Loop should be done inside c++, not from R...
  )[1]
  
  if (max(abs((gllx-gll2x)/gllx))>1E-3) stop("gradLL is not identical betw C++/R")
}

plot(main = "1000 logLikGrad",floor(10^logn),log(times$R),ylim=c(log(min(min(times$R),min(times$cpp))),log(max(max(times$R),max(times$cpp)))),xlab="nb points",ylab="log(user_time (s))")
text(20,0,"R")
points(floor(10^logn),log(times$cpp),col='red')
text(80,0,"C++",col = 'red')
