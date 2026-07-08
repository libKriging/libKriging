## Constructor argument consistency across bindings (see PR).
library(testthat)
library(rlibkriging)

context("Kriging constructor argument consistency")

set.seed(123)
X <- as.matrix(runif(12))
f <- function(x) 1 - 1 / 2 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
y <- f(X)

test_that("objective = 'VLL(m)' is accepted (Vecchia log-likelihood)", {
  expect_error(Kriging(y, X, "matern5_2", objective = "VLL(3)"), NA)
  expect_error(Kriging(y, X, "matern5_2", objective = "VLL"), NA)
})

test_that("objective still accepts LL / LOO / LMP and rejects unknown values", {
  expect_error(Kriging(y, X, "gauss", objective = "LOO"), NA)
  expect_error(Kriging(y, X, "gauss", objective = "LMP"), NA)
  expect_error(Kriging(y, X, "gauss", objective = "NOPE"))
})

test_that("regmodel = 'quadratic' is available", {
  expect_error(Kriging(y, X, "matern3_2", regmodel = "quadratic"), NA)
})

test_that("noise is the last argument (aligned with Python / WarpKriging)", {
  # named form (recommended) keeps working
  expect_error(Kriging(y, X, "gauss", noise = "nugget"), NA)
  # positional form: noise is now after parameters
  expect_error(
    Kriging(y, X, "gauss", "constant", FALSE, "BFGS", "LL", NULL, "nugget"),
    NA
  )
})
