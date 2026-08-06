# Test suite for gradient-enhanced kriging: Kriging(..., dydX=) / fit(..., dydX=)
# Run with: testthat::test_file("test-GradientKriging.R")

library(testthat)
library(rlibkriging)

context("Gradient-enhanced kriging (dydX)")

f2d <- function(x) sin(3 * x[1]) + cos(5 * x[2])
df2d <- function(x) c(3 * cos(3 * x[1]), -5 * sin(5 * x[2]))

make_design <- function(n, seed = 123) {
  set.seed(seed)
  X <- matrix(runif(2 * n), n, 2)
  y <- apply(X, 1, f2d)
  dy <- t(apply(X, 1, df2d))
  list(X = X, y = y, dy = dy)
}

test_that("dydX interpolates values and gradients", {
  d <- make_design(20)
  k <- Kriging(d$y, d$X, "gauss", dydX = d$dy)

  expect_equal(dim(k$dy()), c(20, 2))

  p <- predict(k, d$X, return_stdev = TRUE, return_deriv = TRUE)
  expect_true(max(abs(p$mean - d$y)) < 1e-4)
  expect_true(max(abs(p$mean_deriv - d$dy)) < 1e-3)
})

test_that("dydX=NULL is a value-only fit", {
  d <- make_design(20)
  k <- Kriging(d$y, d$X, "gauss")
  expect_equal(length(k$dy()), 0)
})

test_that("dydX beats a value-only fit out of sample", {
  d <- make_design(15, seed = 72)
  dtest <- make_design(200, seed = 720)

  k_plain <- Kriging(d$y, d$X, "gauss")
  k_grad <- Kriging(d$y, d$X, "gauss", dydX = d$dy)

  mean_plain <- predict(k_plain, dtest$X, return_stdev = FALSE)$mean
  mean_grad <- predict(k_grad, dtest$X, return_stdev = FALSE)$mean

  rmse_plain <- sqrt(mean((mean_plain - dtest$y)^2))
  rmse_grad <- sqrt(mean((mean_grad - dtest$y)^2))
  expect_true(rmse_grad < rmse_plain)
})

test_that("fit(..., dydX=) clears gradient observations on a later fit without dydX", {
  d <- make_design(20)
  k <- Kriging("gauss")
  fit(k, d$y, d$X, dydX = d$dy)
  expect_true(length(k$dy()) > 0)

  fit(k, d$y, d$X)
  expect_equal(length(k$dy()), 0)
})

test_that("dydX rejects a non-differentiable kernel", {
  d <- make_design(10)
  expect_error(Kriging(d$y, d$X, "exp", dydX = d$dy))
})

test_that("dydX rejects a wrongly shaped matrix", {
  d <- make_design(10)
  expect_error(Kriging(d$y, d$X, "gauss", dydX = d$dy[, 1, drop = FALSE]))
})
