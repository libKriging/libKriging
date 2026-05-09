##library(rlibkriging, lib.loc="bindings/R/Rlibs")
##library(testthat)

# f <- function(X) apply(X, 1, function(x) prod(sin((x-.5)^2)))
f <- function(X) apply(X, 1, function(x)
  prod(sin(2*pi*( x * (seq(0,1,l=1+length(x))[-1])^2 )))
)
n <- 20
set.seed(123)
X <- cbind(runif(n),runif(n))
y <- f(X)
d = ncol(X)

x=seq(0,1,,5)
contour(x,x,matrix(f(as.matrix(expand.grid(x,x))),nrow=length(x)),nlevels = 30)
points(X)

context("Test rlibkriging::save/load explicitely")

k <- Kriging(y, X,"gauss",parameters = list(theta=matrix(runif(40),ncol=2)))
print(k)

unlink("k.json")
save(k, filename="k.json")

k2 <- load(filename="k.json")
print(k2)

test_that("Save/Load NuggetKriging", expect_true( print(k) == print(k2)))

nuk <- Kriging(y, X,"gauss", noise = "nugget",parameters = list(theta=matrix(runif(40),ncol=2)))
print(nuk)

unlink("nuk.json")
save(nuk, filename="nuk.json")

nuk2 <- load(filename="nuk.json")
print(nuk2)

test_that("Save/Load NuggetKriging", expect_true( print(nuk) == print(nuk2)))

nok <- Kriging(y, X, "gauss", noise=rep(0.1^2,nrow(X)), parameters = list(theta=matrix(runif(40),ncol=2)))
print(nok)

unlink("nok.json")
save(nok, filename="nok.json")

nok2 <- load(filename="nok.json")
print(nok2)

test_that("Save/Load Kriging with noise", expect_true( print(nok) == print(nok2)))

context("Test rlibkriging::save/load versus base::save/load")

save(k, filename="k.json")
k2=load(filename="k.json")
test_that("Save/Load Kriging", expect_true( print(k) == print(k2)))

a = "abcd"
save(a,file="test.Rdata")
rm("a")
load("test.Rdata",verbose=TRUE, envir=.GlobalEnv)
test_that("Save/Load Rdata", expect_true( a == "abcd"))
