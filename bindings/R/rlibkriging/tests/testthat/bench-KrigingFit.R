library(foreach)
registerDoSEQ()

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))

logn <- seq(1.1, 3, by=.1)
times <- list(R=rep(NA, length(logn)), cpp=rep(NA, length(logn)))
N = 10

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- 1+floor(log(n)) #floor(2+i/3)
  
  print(n)
  set.seed(123)
  X <- DiceDesign::lhsDesign(n,dimension=d)$design #matrix(runif(n*d),ncol=d)
  y <- f(X)
  
  k <- NULL
  times$R[i] = system.time(
    try(for (j in 1:N) k <- DiceKriging::km(design=X,response=y,covtype = "gauss", multistart = 1,control = list(trace=F,maxit=10), 
                                                  lower=rep(0.001,d),upper=rep(2*sqrt(d),d)))
  )[1]
  
  r <- NULL
  times$cpp[i] = system.time(
    try(for (j in 1:N) r <- kriging(y, X,"gauss","constant",FALSE,"BFGS","LL",
                      # to let start optim at same initial point
                      parameters=list(sigma2=0,has_sigma2=FALSE,theta=matrix(k@parinit,ncol=d),has_theta=TRUE)))
  )[1]
  
  ll_cpp <- kriging_logLikelihood(r, kriging_model(r)$theta)
  e <- new.env()
  ll_R <- DiceKriging::logLikFun(k@covariance@range.val, k, e)
  
  gll_cpp <- kriging_logLikelihoodGrad(r, kriging_model(r)$theta)
  gll_R <- DiceKriging::logLikGrad(k@covariance@range.val, k, e)
    
  if (abs(ll_cpp - DiceKriging::logLikFun(param=as.numeric(kriging_model(r)$theta),model=k))/ll_cpp>.1)
    stop("LL function is not the same bw DiceKriging/libKriging: ",kriging_logLikelihood(r,kriging_model(r)$theta)," vs. ",DiceKriging::logLikFun(param=as.numeric(kriging_model(r)$theta),model=k))  
  
  if ((ll_cpp - ll_R)/ll_R < -.01 )
    warning("libKriging LL ",ll_cpp," << DiceKriging LL ",ll_R)

  if ((ll_R - ll_cpp)/ll_R < -.01 )
    warning("DiceKriging LL ",ll_R," << libKriging LL ",ll_cpp)
}

plot(floor(10^logn),log(times$R),ylim=c(log(min(min(times$R,na.rm = T),min(times$cpp,na.rm = T))),log(max(max(times$R,na.rm = T),max(times$cpp,na.rm = T)))),xlab="nb points",ylab="log(user_time (s))", panel.first=grid())
text(20,-1,"DiceKriging")
points(floor(10^logn),log(times$cpp),col='red')
text(80,-1,"libKriging",col = 'red')
