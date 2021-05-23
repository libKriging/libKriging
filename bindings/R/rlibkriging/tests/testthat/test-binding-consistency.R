# Title     : API and result regression test
# Objective : create reproductible data for all binding tests
# Created by: pascal
# Created on: 23/05/2021

do_write = TRUE
tolerance <- 1e-14

relative_error = function(x, y) {
  x_norm <- norm(x)
  y_norm <- norm(y)
  if (x_norm > 0 || y_norm > 0) {
    diff_norm <- norm(x - y)
    return(diff_norm / max(x_norm, y_norm))
  } else {
    return(0)
  }
}

library(testthat)
library(rlibkriging, lib = "/Users/pascal/haveneer/libKriging.perso/bindings/R/Rlibs")

prefix = "data1-scal"
# @formatter:off
f = function(x) { 1 - 1/2 * (sin(12*x) / (1+x) + 2*cos(7*x) * x^5 + 0.7) }
# @formatter:on
n <- 5
set.seed(123)
X <- as.matrix(runif(n))
y = f(X)

filex <- sprintf("%s-X.csv", prefix)
filey <- sprintf("%s-y.csv", prefix)
if (do_write) {
  write.table(X, file = filex, row.names = FALSE, col.names = FALSE, sep = ',')
  write.table(y, file = filey, row.names = FALSE, col.names = FALSE, sep = ',')
}
X = as.matrix(read.table(file = filex, header = FALSE, sep = ',')) # a matrix, not a dataframe
y = as.matrix(read.table(file = filey, header = FALSE, sep = ',')) # a matrix, not a dataframe

kernel <- "gauss"
r <- Kriging(y, X, kernel)
x <- 0.3 # extracted from 'for (x in seq(0.01,1,,11))' 
llo = rlibkriging::leaveOneOut(r, x, grad = TRUE)
loglik = rlibkriging::logLikelihood(r, x, grad = TRUE)

for (n in names(llo)) {
  filer <- sprintf("%s-result-%s.csv", prefix, n)
  if (do_write) {
    write.table(llo[[n]], file = filer, row.names = FALSE, col.names = FALSE, sep = ',')
  }
  testme <- as.matrix(read.table(file = filer, header = FALSE, sep = ','))
  expect_equal(dim(llo[[n]]), dim(testme))
  expect_true(relative_error(llo[[n]], testme) < tolerance)
}
for (n in names(loglik)) {
  filer <- sprintf("%s-result-%s.csv", prefix, n)
  if (do_write) {
    write.table(loglik[[n]], file = filer, row.names = FALSE, col.names = FALSE, sep = ',')
  }
  testme <- as.matrix(read.table(file = filer, header = FALSE, sep = ','))
  expect_equal(dim(loglik[[n]]), dim(testme))
  expect_true(relative_error(loglik[[n]], testme) < tolerance)
}


prefix = "data2-grad"
f <- function(X) apply(X, 1, function(x) prod(sin((x - .5)^2)))
logn <- seq(1.1, 2, by = .1)
for (i in 1:length(logn)) {
  n <- floor(10^logn[i])
  d <- 1 + floor(log(n))
  set.seed(123)
  X <- matrix(runif(n * d), ncol = d)
  y <- f(X)

  filex <- sprintf("%s-%i-X.csv", prefix, i)
  filey <- sprintf("%s-%i-y.csv", prefix, i)
  if (do_write) {
    write.table(X, file = filex, row.names = FALSE, col.names = FALSE, sep = ',')
    write.table(y, file = filey, row.names = FALSE, col.names = FALSE, sep = ',')
  }
  X = as.matrix(read.table(file = filex, header = FALSE, sep = ',')) # a matrix, not a dataframe
  y = as.matrix(read.table(file = filey, header = FALSE, sep = ',')) # a matrix, not a dataframe

  kernel <- "gauss"
  r <- Kriging(y, X, kernel)
  loglik = rlibkriging::logLikelihood(r, x, grad = TRUE) # TODO needs also Hessian ?
  for (n in names(loglik)) {
    filer <- sprintf("%s-%i-result-%s.csv", prefix, i, n)
    if (do_write) {
      write.table(loglik[[n]], file = filer, row.names = FALSE, col.names = FALSE, sep = ',')
    }
    testme <- as.matrix(read.table(file = filer, header = FALSE, sep = ','))
    expect_equal(dim(loglik[[n]]), dim(testme))
    expect_true(relative_error(loglik[[n]], testme) < tolerance)
  }
}