#library(rlibkriging)
library(rlibkriging, lib.loc="bindings/R/Rlibs")

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

k <- Kriging(y, X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2)))
print(k)

rlibkriging::save(k, filename="k.h5")

k2 <- rlibkriging::load(filename="k.h5")
print(k2)

t0 = matrix(c(0.41764678,0.48861303),ncol=2)

nuk <- NuggetKriging(y, X,"gauss",parameters = list(theta=t0))