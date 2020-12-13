library(testthat)

f <- function(X) apply(X, 1, function(x) sum(x^2))

logn <- seq(1, 2.5, by=.1)
times <- list(R=rep(NA, length(logn)), cpp=rep(NA, length(logn)))
N <- 1000

theta <- 0.5

for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- floor(2+i/3)

  print(n)
  set.seed(123)
  X <- matrix(runif(n*d),ncol=d)
  y <- f(X)

  k <- DiceKriging::km(design=X, response=y, covtype = "gauss")
  llx <- 0
  times$R[i] = system.time(
    try({for (t in 1:N) llx <- llx+DiceKriging::logLikFun(rep(theta,ncol(X)),k)})
  )[1]
  llx <- llx/N

  r <- ordinary_kriging(y, X,"gauss")
  times$cpp[i] = system.time(
    try({ll2x <- bench_LogLik(N,r,rep(theta,ncol(X)))})
    # try({for (t in 1:N) ll2x <- ordinary_kriging_loglikelihood(r,rep(theta,ncol(X)))}) # Loop should be done inside c++, not from R...
  )[1]

  if (abs((llx-ll2x)/llx)>1E-3) stop("LL is not identical betw C++/R")
}

plot(main = "1000 logLik",floor(10^logn),log(times$R),col='black',ylim=c(log(min(min(times$R),min(times$cpp))),log(max(max(times$R),max(times$cpp)))),xlab="nb points",ylab="log(user_time (s))", panel.first=grid())
text(20,0,"DiceKriging",col='black')
points(floor(10^logn),log(times$cpp),col='red')
text(80,0,"C++",col = 'red')
