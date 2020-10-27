library(foreach)
registerDoSEQ()

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))

logn <- 2.5 #seq(1, 2.5, by=.1)
times <- list(R=rep(NA, length(logn)), cpp=rep(NA, length(logn)))
times.n = 10

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- 1+floor(log(n))
  
  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y <- f(X)
  
  times$R[i] = system.time(
    # try(
      # for (j in 1:times.n) 
        k <- DiceKriging::km(design=X,response=y,covtype = "gauss", multistart = 1,control = list(trace=F,maxit=10))
      # ) #, lower=rep(0.001,d),upper=rep(2*sqrt(d),d)))
  )[1]
  
  times$cpp[i] = system.time(
    # try(
      # for (j in 1:times.n) 
        r <- ordinary_kriging(y, X,"gauss")
      # )
  )[1]
  
  ll_cpp <- ordinary_kriging_logLikelihood(r, ordinary_kriging_model(r)$theta)
  e <- new.env()
  ll_R <- DiceKriging::logLikFun(k@covariance@range.val, k, e)
  
  gll_cpp <- ordinary_kriging_logLikelihoodGrad(r, ordinary_kriging_model(r)$theta)
  gll_R <- DiceKriging::logLikGrad(k@covariance@range.val, k, e)
  
  if (abs(ll_cpp - DiceKriging::logLikFun(param=as.numeric(ordinary_kriging_model(r)$theta),model=k))/ll_cpp>.1)
    stop("LL function is not the same bw DiceKriging/libKriging: ",ordinary_kriging_logLikelihood(r,ordinary_kriging_model(r)$theta)," vs. ",DiceKriging::logLikFun(param=as.numeric(ordinary_kriging_model(r)$theta),model=k))  
  
  if ((ll_cpp - ll_R)/ll_R < -.01 )
    warning("libKriging LL ",ll_cpp," << DiceKriging LL ",ll_R)
  
}

plot(floor(10^logn),log(times$R),ylim=c(log(min(min(times$R,na.rm = T),min(times$cpp,na.rm = T))),log(max(max(times$R,na.rm = T),max(times$cpp,na.rm = T)))),xlab="nb points",ylab="log(user_time (s))", panel.first=grid())
text(20,-1,"DiceKriging")
points(floor(10^logn),log(times$cpp),col='red')
text(80,-1,"libKriging",col = 'red')
