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

k <- Kriging(y, X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2)))
print(k)

rlibkriging::save(k, filename="k.json")

k2 <- rlibkriging::load(filename="k.json")
print(k2)

test_that("Save/Load NuggetKriging", expect_true( print(k) == print(k2)))

nuk <- NuggetKriging(y, X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2)))
print(nuk)

rlibkriging::save(nuk, filename="nuk.json")

nuk2 <- rlibkriging::load(filename="nuk.json")
print(nuk2)

test_that("Save/Load NuggetKriging", expect_true( print(nuk) == print(nuk2)))

nok <- NoiseKriging(y, rep(0.1^2,nrow(X)), X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2)))
print(nok)

rlibkriging::save(nok, filename="nok.json")

nok2 <- rlibkriging::load(filename="nok.json")
print(nok2)

test_that("Save/Load NoiseKriging", expect_true( print(nok) == print(nok2)))
