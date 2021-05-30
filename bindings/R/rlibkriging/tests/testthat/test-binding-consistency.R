# Title     : API and result regression test
# Objective : create reproductible data for all binding tests
# Created by: pascal
# Created on: 23/05/2021

do_write = FALSE # set to TRUE to regenerate a set of values
tolerance <- 1e-12

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

find_dir = function() {
  path = getwd()
  found <- FALSE
  while (!is.null(path) && !found) {
    testpath <- file.path(path, ".git", "..", "tests", "references")
    if (dir.exists(testpath)) {
      return(testpath)
    } else {
      parent = dirname(path)
      if (parent == path || is.null(parent)) {
        stop("Cannot find reference test directory")
      }
      path = parent
    }
  }
}

#library(testthat)
#library(rlibkriging, lib = "/Users/pascal/haveneer/libKriging.perso/bindings/R/Rlibs")

refpath = find_dir()
print(paste0("Reference directory=", refpath))
prefix = "data1-scal"

filex <- file.path(refpath, sprintf("%s-X.csv", prefix))
filey <- file.path(refpath, sprintf("%s-y.csv", prefix))
if (do_write) {
  # @formatter:off
  f = function(x) { 1 - 1/2 * (sin(12*x) / (1+x) + 2*cos(7*x) * x^5 + 0.7) }
  # @formatter:on
  n <- 5
  set.seed(123)
  X <- as.matrix(runif(n))
  y = f(X)
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
  filer <- file.path(refpath, sprintf("%s-result-%s.csv", prefix, n))
  if (do_write) {
    write.table(llo[[n]], file = filer, row.names = FALSE, col.names = FALSE, sep = ',')
  }
  testme <- as.matrix(read.table(file = filer, header = FALSE, sep = ','))
  test_that(desc = paste0("Computed result should be  close to reference value on ", prefix, " llo$", n), {
    expect_equal(dim(llo[[n]]), dim(testme))
    expect_lt(relative_error(llo[[n]], testme), tolerance) })
}
for (n in names(loglik)) {
  filer <- file.path(refpath, sprintf("%s-result-%s.csv", prefix, n))
  if (do_write) {
    write.table(loglik[[n]], file = filer, row.names = FALSE, col.names = FALSE, sep = ',')
  }
  testme <- as.matrix(read.table(file = filer, header = FALSE, sep = ','))
  test_that(desc = paste0("Computed result should be close to reference value on ", prefix, " loglik$", n), {
    expect_equal(dim(loglik[[n]]), dim(testme))
    expect_lt(relative_error(loglik[[n]], testme), tolerance) })
}


prefix = "data2-grad"
logn <- seq(1.1, 2, by = .1)
for (i in 1:length(logn)) {
  filex <- file.path(refpath, sprintf("%s-%i-X.csv", prefix, i))
  filey <- file.path(refpath, sprintf("%s-%i-y.csv", prefix, i))
  if (do_write) {
    f <- function(X) apply(X, 1, function(x) prod(sin((x - .5)^2)))
    n <- floor(10^logn[i])
    d <- 1 + floor(log(n))
    set.seed(123)
    X <- matrix(runif(n * d), ncol = d)
    y <- f(X)
    write.table(X, file = filex, row.names = FALSE, col.names = FALSE, sep = ',')
    write.table(y, file = filey, row.names = FALSE, col.names = FALSE, sep = ',')
  }
  X = as.matrix(read.table(file = filex, header = FALSE, sep = ',')) # a matrix, not a dataframe
  y = as.matrix(read.table(file = filey, header = FALSE, sep = ',')) # a matrix, not a dataframe

  kernel <- "gauss"
  r <- Kriging(y, X, kernel)
  loglik = rlibkriging::logLikelihood(r, x, grad = TRUE) # TODO needs also Hessian ?
  for (n in names(loglik)) {
    filer <- file.path(refpath, sprintf("%s-%i-result-%s.csv", prefix, i, n))
    if (do_write) {
      write.table(loglik[[n]], file = filer, row.names = FALSE, col.names = FALSE, sep = ',')
    }
    testme <- as.matrix(read.table(file = filer, header = FALSE, sep = ','))
    test_that(desc = paste0("Computed result should be close to reference value on ", prefix, " loglik$", n), {
      expect_equal(dim(loglik[[n]]), dim(testme))
      expect_lt(relative_error(loglik[[n]], testme), tolerance) })
  }
}