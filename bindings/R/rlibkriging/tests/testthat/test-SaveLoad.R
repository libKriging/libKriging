library(testthat)

#library(rlibkriging, lib.loc="bindings/R/Rlibs")
#library(rlibkriging)

# f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
f <- function(X) apply(X, 1, function(x)
  prod(sin(2*pi*( x * (seq(0,1,l=1+length(x))[-1])^2 )))
)
n <- 20
set.seed(123)
X <- cbind(runif(n),runif(n))
y <- f(X)
d = ncol(X)

x=seq(0,1,,51)
contour(x,x,matrix(f(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
points(X)

r <- Kriging(y, X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2)))
print(r)

rlibkriging::save(r, filename="r.h5")

r2 <- rlibkriging::load.Kriging(filename="r.h5")
print(r2)
