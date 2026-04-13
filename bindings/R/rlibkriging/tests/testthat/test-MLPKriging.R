## *************************************************************************
##  test-MLPKriging.R — testthat tests for MLPKriging R binding
##
##  Mirrors tests/MLPKrigingTest.cpp (Deep Kernel Learning).
## *************************************************************************

library(testthat)
library(rlibkriging)

f1d <- function(x) {
  1 - 0.5 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}

branin <- function(x1, x2) {
  a <- 1; b <- 5.1 / (4 * pi^2); cc <- 5 / pi
  r <- 6; s <- 10; t <- 1 / (8 * pi)
  a * (x2 - b * x1^2 + cc * x1 - r)^2 + s * (1 - t) * cos(x1) + s
}

# ===========================================================================
#  Test 1: 1D fit / predict
# ===========================================================================
test_that("MLPKriging works (1D, gauss)", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 10))
  y <- f1d(X)

  k <- MLPKriging(y, X,
                  hidden_dims = c(16, 8), d_out = 2,
                  activation = "selu", kernel = "gauss",
                  normalize = TRUE,
                  parameters = list(max_iter_adam = "300"))

  expect_s3_class(k, "MLPKriging")
  cat(summary(k))

  p <- predict(k, X, stdev = TRUE)
  expect_equal(length(p$mean), 10)
  expect_true(all(is.finite(p$mean)))
  cat("  [1D] Max train error:", max(abs(p$mean - y)), "\n")

  x_pred <- as.matrix(seq(0.01, 0.99, length.out = 50))
  p2 <- predict(k, x_pred, stdev = TRUE)
  rmse <- sqrt(mean((p2$mean - f1d(x_pred))^2))
  cat("  [1D] RMSE on dense grid:", rmse, "\n")

  expect_equal(kernel(k), "gauss")
  expect_equal(activation(k), "selu")
  expect_equal(hidden_dims(k), c(16L, 8L))
  expect_equal(feature_dim(k), 2L)
  expect_true(sigma2(k) > 0)
  expect_true(is.finite(logLikelihood(k)))
})

# ===========================================================================
#  Test 2: Branin 2D fit / predict / simulate / update
# ===========================================================================
test_that("MLPKriging works (Branin 2D, matern5_2)", {
  set.seed(77)
  n <- 20
  X <- matrix(runif(n * 2), ncol = 2)
  y <- mapply(function(i) branin(X[i, 1] * 15 - 5, X[i, 2] * 15), 1:n)

  k <- MLPKriging(y, X,
                  hidden_dims = c(32, 16), d_out = 3,
                  activation = "selu", kernel = "matern5_2",
                  normalize = TRUE,
                  parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  X_test <- matrix(runif(10 * 2), ncol = 2)
  p <- predict(k, X_test, stdev = TRUE)
  expect_equal(length(p$mean), 10)
  expect_true(all(is.finite(p$mean)))
  expect_true(all(is.finite(p$stdev)))

  sims <- simulate(k, nsim = 20, seed = 42, x = X_test)
  expect_equal(dim(sims), c(10, 20))
  expect_true(all(is.finite(sims)))

  X_new <- matrix(runif(3 * 2), ncol = 2)
  y_new <- mapply(function(i) branin(X_new[i, 1] * 15 - 5, X_new[i, 2] * 15), 1:3)
  update(k, y_new, X_new)

  p2 <- predict(k, X_test, stdev = TRUE)
  expect_equal(length(p2$mean), 10)
  cat("  [2D] predictions OK after update\n")
})

# ===========================================================================
#  Test 3: predict derivative
# ===========================================================================
test_that("MLPKriging predict with deriv returns finite derivatives", {
  set.seed(88)
  n <- 20
  X <- matrix(runif(n * 2), ncol = 2)
  f2 <- function(x1, x2) sin(3 * x1) + cos(4 * x2) + 0.5 * x1 * x2
  y <- mapply(function(i) f2(X[i, 1], X[i, 2]), 1:n)

  k <- MLPKriging(y, X,
                  hidden_dims = c(16, 8), d_out = 2,
                  activation = "selu", kernel = "gauss",
                  parameters = list(max_iter_adam = "100"))

  X_new <- matrix(runif(5 * 2), ncol = 2)
  p <- predict(k, X_new, return_stdev = TRUE, return_deriv = TRUE)
  expect_true(!is.null(p$mean_deriv))
  expect_true(!is.null(p$stdev_deriv))
  expect_equal(dim(p$mean_deriv), c(5, 2))
  expect_equal(dim(p$stdev_deriv), c(5, 2))
  expect_true(all(is.finite(p$mean_deriv)))
  expect_true(all(is.finite(p$stdev_deriv)))
})

# ===========================================================================
#  Test 4: logLikelihoodFun
# ===========================================================================
test_that("MLPKriging logLikelihoodFun works with gradient", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 10))
  y <- f1d(X)

  k <- MLPKriging(y, X,
                  hidden_dims = c(16, 8), d_out = 2,
                  activation = "selu", kernel = "gauss",
                  parameters = list(max_iter_adam = "200"))

  th <- theta(k)
  ll <- logLikelihoodFun(k, th, return_grad = TRUE)
  expect_true(is.finite(ll$logLikelihood))
  expect_true(all(is.finite(ll$gradient)))

  ll2 <- logLikelihoodFun(k, th, return_grad = FALSE)
  expect_true(is.finite(ll2$logLikelihood))
})

# ===========================================================================
#  Test 5: is_fitted and getters
# ===========================================================================
test_that("MLPKriging is_fitted and getters work", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 8))
  y <- f1d(X)

  k <- MLPKriging(y, X,
                  hidden_dims = c(16, 8), d_out = 2,
                  activation = "selu", kernel = "gauss",
                  parameters = list(max_iter_adam = "100"))

  expect_true(is_fitted(k))
  expect_true(all(theta(k) > 0))
  expect_true(sigma2(k) > 0)
})
