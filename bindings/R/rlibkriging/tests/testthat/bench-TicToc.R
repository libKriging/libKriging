library(rlibkriging, lib.loc="bindings/R/Rlibs")

f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
n <- 200
set.seed(123)
X <- cbind(runif(n),runif(n),runif(n),runif(n))
y <- f(X)

times.n = 10
x = .5

r = NULL
r <- Kriging(y, X, "gauss")
for (j in 1:times.n) {
  invisible(logLikelihoodFun(r,rep(x,4),           bench=TRUE ))
}

