library(testthat)
library(rlibkriging)

context("Kriging predictCG (matrix-free conjugate-gradient prediction)")

f2d <- function(x1, x2) sin(3 * x1) + cos(5 * x2) + x1 * x2

make_data <- function(n, seed = 123) {
  set.seed(seed)
  X <- matrix(runif(2 * n), ncol = 2)
  y <- f2d(X[, 1], X[, 2])
  list(X = X, y = y)
}

# Fixed, moderate theta (optim = "none"): on this noise-free deterministic
# test function, a free BFGS fit is known to drift theta toward a
# near-singular correlation matrix (unrelated to predictCG itself -- see
# KrigingNystromTest.cpp / docs/math/PredictCG.md history for the same
# issue), which would make predict/predictCG's agreement noisy rather than
# a clean correctness signal.
make_fixed_theta_model <- function(y, X, theta_val = 0.3) {
  Kriging(y, X, "matern5_2",
          optim = "none",
          parameters = list(theta = matrix(theta_val, 1, ncol(X)), sigma2 = 1))
}

test_that("predictCG mean/stdev match exact predict at a moderate theta", {
  d <- make_data(60)
  k <- make_fixed_theta_model(d$y, d$X)
  Xt <- matrix(runif(2 * 20), ncol = 2)

  p_exact <- predict(k, Xt, return_stdev = TRUE)
  p_cg <- predictCG(k, Xt, return_stdev = TRUE)

  expect_lt(max(abs(p_exact$mean - p_cg$mean)), 0.05 * sd(d$y))
  expect_lt(max(abs(p_exact$stdev - p_cg$stdev)), 0.05 * sd(d$y))
})

test_that("predictCG defaults to mean only (no stdev element)", {
  d <- make_data(40)
  k <- make_fixed_theta_model(d$y, d$X)
  Xt <- matrix(runif(2 * 5), ncol = 2)

  p <- predictCG(k, Xt)
  expect_equal(names(p), "mean")
  expect_equal(length(p$mean), 5)
})

test_that("predictCG interpolates the training data", {
  d <- make_data(30)
  k <- make_fixed_theta_model(d$y, d$X)

  p <- predictCG(k, d$X, return_stdev = TRUE)
  expect_lt(max(abs(p$mean - d$y)), 0.05 * sd(d$y))
  expect_lt(max(p$stdev), 0.05 * sd(d$y))
})

test_that("predictCG is reachable via k$predictCG(...) too", {
  d <- make_data(20)
  k <- make_fixed_theta_model(d$y, d$X)
  Xt <- matrix(runif(2 * 5), ncol = 2)

  p1 <- predictCG(k, Xt)
  p2 <- k$predictCG(Xt)
  expect_equal(p1$mean, p2$mean)
})

test_that("predictCG rejects a negative max_iter", {
  d <- make_data(20)
  k <- make_fixed_theta_model(d$y, d$X)
  Xt <- matrix(runif(2 * 5), ncol = 2)

  expect_error(predictCG(k, Xt, max_iter = -1))
})
