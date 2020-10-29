library(foreach)
library(rlibkriging)
registerDoSEQ()

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))

logn <- seq(1.1, 2.5, by=.1)
times <- list(R_user=rep(NA, length(logn)),
              R_elapsed=rep(NA, length(logn)),
              cpp_user=rep(NA, length(logn)), 
              cpp_elapsed=rep(NA, length(logn)))
times.n = 1

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- 1+floor(log(n)) #floor(2+i/3)
  
  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y <- f(X)

  t = system.time(
    try(for (j in 1:times.n) k <- DiceKriging::km(design=X,response=y,covtype = "gauss", multistart = 1,control = list(trace=F,maxit=10), lower=rep(0.001,d),upper=rep(2*sqrt(d),d)))
  ) # 1: user time, 2: system time, 3:elapsed time
  times$R_user[i] <- t[1]
  times$R_elapsed[i] <- t[3]

  t = system.time(
    try(for (j in 1:times.n) r <- ordinary_kriging(y, X,"gauss"))
  )
  times$cpp_user[i] <- t[1]
  times$cpp_elapsed[i] <- t[3]
  
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

plot(floor(10^logn),log(times$R_elapsed), col='grey', pch=3,
     ylim=c(max(-7, log(min(min(times$R_user,na.rm = T),min(times$cpp_user,na.rm = T)))),
                    log(max(max(times$R_user,na.rm = T),max(times$cpp_user,na.rm = T)))),
     xlab="nb points",ylab="log(user_time (s))", panel.first=grid())
points(floor(10^logn),log(times$R_user),col='black')
points(floor(10^logn),log(times$cpp_elapsed),col='orange', pch=3)
points(floor(10^logn),log(times$cpp_user),col='red')
text(200,-4.5,"DiceKriging", col= 'black', pos = 2)
text(200,-4.5,"+ elapsed", col= 'grey', pos = 4)
text(200,-5,"libKriging",col = 'red', pos = 2)
text(200,-5,"+ elapsed", col= 'orange', pos = 4)
