library(foreach)
registerDoSEQ()

f = function(X) apply(X,1,function(x) sum((x-.5)^2))

logn = seq(1,2,by=.1)
times = list(R=rep(NA,length(logn)),cpp=rep(NA,length(logn)))

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- floor(2+i/3)
  
  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y = f(X)
  
  times$R[i] = system.time(
    try(k <- DiceKriging::km(design=X,response=y,covtype = "gauss", multistart = 10,control = list(trace=T,maxit=10), lower=rep(0.001,d),upper=rep(2*sqrt(d),d)))
  )[1]
  
  times$cpp[i] = system.time(
    try(r <- ordinary_kriging(y, X))
  )[1]
  
  ll_cpp = ordinary_kriging_loglikelihood(r,ordinary_kriging_model(r)$theta)
  e = new.env()
  ll_R = DiceKriging::logLikFun(k@covariance@range.val,k,e)
  
  gll_cpp = ordinary_kriging_loglikelihoodgrad(r,ordinary_kriging_model(r)$theta)
  gll_R = DiceKriging::logLikGrad(k@covariance@range.val,k,e)
  
  if (abs(ll_cpp - DiceKriging::logLikFun(param=as.numeric(ordinary_kriging_model(r)$theta),model=k))/ll_cpp>.1)
    stop("LL function is not the same bw DiceKriging/libKriging: ",ordinary_kriging_loglikelihood(r,ordinary_kriging_model(r)$theta)," vs. ",DiceKriging::logLikFun(param=as.numeric(ordinary_kriging_model(r)$theta),model=k))  
  
  if ((ll_cpp - ll_R)/ll_R < -.01 )
    warning("libKriging LL ",ll_cpp," << DiceKriging LL ",ll_R)
  
}

plot(floor(10^logn),log(times$R),ylim=c(log(min(min(times$R),min(times$cpp))),log(max(max(times$R),max(times$cpp)))),xlab="nb points",ylab="log(temps (s))")
text(20,0,"DiceKriging")
points(floor(10^logn),log(times$cpp),col='red')
text(80,0,"libKriging",col = 'red')
