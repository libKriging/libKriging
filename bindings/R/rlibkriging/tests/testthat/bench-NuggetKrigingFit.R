 pack=list.files(file.path("bindings","R"),pattern = ".tar.gz",full.names = T)
 install.packages(pack,repos=NULL)
 library(rlibkriging)

library(testthat)

f <- function(X) apply(X, 1, function(x) rnorm(1,0,0.01)+sum(sin(2*pi*(x-.5)^2)))
n <- 1000
set.seed(123)
X <- DiceDesign::lhsDesign(n,dimension=6)$design #cbind(runif(n),runif(n))
y <- f(X)
d = ncol(X)

N = 20
times <- list(R=rep(NA, N), cpp=rep(NA, N))

for (i in 1:N) {
    set.seed(i)
    try(times$R[i] <- system.time(
                        k <- DiceKriging::km(design=X,response=y,covtype = "gauss",nugget.estim = T,
                         multistart = 1,control = list(trace=T,maxit=10)) #,lower=rep(0.001,d),upper=rep(2*sqrt(d),d))
                    ))
    try(times$cpp[i] <- system.time(
                        r <- NuggetKriging(y, X,"gauss","constant",FALSE,"BFGS","LL",
                            # to let start optim at same initial point
                            parameters=list(theta=matrix(k@parinit,ncol=d), nugget=0.001))
                    ))

r_alpha = r$sigma2()/(r$sigma2()+r$nugget())
  ll_cpp <- logLikelihoodFun(r, t(c(as.list(r)$theta,r_alpha)))$logLikelihood
  e <- new.env()
  ll_R <- DiceKriging::logLikFun(k@covariance@range.val, k, e)

  gll_cpp <- logLikelihoodFun(r, t(c(as.list(r)$theta,r_alpha)))$logLikelihoodGrad
  gll_R <- DiceKriging::logLikGrad(k@covariance@range.val, k, e)

  # if (abs(ll_cpp - DiceKriging::logLikFun(param=as.numeric(t(as.list(r)$theta)),model=k))/ll_cpp>.1)
  #   stop("LL function is not the same bw DiceKriging/libKriging: ",logLikelihoodFun(r,t(c(as.list(r)$theta,r_alpha)))," vs. ",DiceKriging::logLikFun(param=as.numeric(t(as.list(r)$theta)),model=k))

  if ((ll_cpp - ll_R)/ll_R < -.01 )
    warning("libKriging LL ",ll_cpp," << DiceKriging LL ",ll_R)

  if ((ll_R - ll_cpp)/ll_R < -.01 )
    warning("DiceKriging LL ",ll_R," << libKriging LL ",ll_cpp)
}

plot(times$R,times$cpp, xlim=range(c(times$R,times$cpp)),ylim=range(c(times$R,times$cpp)))
abline(a=0,b=1,col='red')
abline(a=0,b=2,col='red')
abline(a=0,b=0.5,col='red')


