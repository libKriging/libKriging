library(testthat)
library(rlibkriging)

context("Kriging method \"iterative(...)\" (matrix-free CG / SLQ) and Gpu settings")

f2d <- function(x1, x2) sin(3 * x1) + cos(5 * x2) + x1 * x2

make_data <- function(n, seed = 123) {
  set.seed(seed)
  X <- matrix(runif(2 * n), ncol = 2)
  list(X = X, y = f2d(X[, 1], X[, 2]))
}

# fixed, moderate theta (optim = "none"): a free fit on this noise-free
# function drifts toward a near-singular correlation matrix, unrelated to CG
make_model <- function(y, X, method = "exact", theta_val = 0.1) {
  Kriging(y, X, "matern5_2", optim = "none",
          parameters = list(theta = matrix(theta_val, 1, ncol(X)), is_theta_estim = FALSE),
          method = method)
}

test_that("predict method iterative matches the exact predictor on an exact fit", {
  d <- make_data(60)
  k <- make_model(d$y, d$X)
  Xt <- make_data(20, seed = 456)$X
  p_ex <- predict(k, Xt, return_stdev = TRUE)
  p_cg <- predict(k, Xt, return_stdev = TRUE, method = "iterative")
  expect_lt(max(abs(p_ex$mean - p_cg$mean)), 1e-5 * sd(d$y))
  expect_lt(max(abs(p_ex$stdev - p_cg$stdev)), 1e-4 * sd(d$y))
  p_pc <- predict(k, Xt, return_stdev = TRUE, method = "iterative(30,20)")
  expect_lt(max(abs(p_ex$mean - p_pc$mean)), 1e-5 * sd(d$y))
})

test_that("iterative fit: no exact factorization, explicit method, save / load", {
  d <- make_data(40)
  k <- make_model(d$y, d$X, method = "iterative(8,4)")
  expect_equal(k$method(), "iterative(8,4)")
  Xt <- make_data(5, seed = 789)$X
  expect_error(predict(k, Xt))
  p <- predict(k, Xt, method = "iterative")
  expect_true(all(is.finite(p$mean)))
  expect_error(simulate(k, nsim = 2, seed = 1, x = Xt))
  ll <- logLikelihood(k, method = "iterative")
  expect_true(is.finite(ll))
  f <- tempfile(fileext = ".json")
  save(k, filename = f)
  kl <- load(filename = f)
  expect_equal(kl$method(), k$method())
  expect_equal(logLikelihood(kl, method = "iterative"), ll, tolerance = 1e-10)
})

test_that("iterative restrictions raise", {
  d <- make_data(20)
  expect_error(Kriging(d$y, d$X, "matern5_2", objective = "LOO", method = "iterative"))
  expect_error(Kriging(d$y, d$X, "matern5_2", noise = "nugget", method = "iterative"))
})

test_that("WarpKriging predict method iterative matches the exact predictor", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 8))
  y <- sin(6 * X[, 1])
  xp <- as.matrix(seq(0.01, 0.99, length.out = 20))
  wk <- WarpKriging(y, X, warping = "kumaraswamy", kernel = "gauss", optim = "none",
                    parameters = list(theta = 0.3))
  p_ex <- predict(wk, xp, return_stdev = TRUE)
  p_cg <- predict(wk, xp, return_stdev = TRUE, method = "iterative(30,0,20,250,1e-12)")
  expect_lt(max(abs(p_ex$mean - p_cg$mean)), 1e-4 * sd(y))
  expect_lt(max(abs(p_ex$stdev - p_cg$stdev)), 1e-4 * sd(y))
})

test_that("Gpu and LinearAlgebra iterative settings", {
  expect_true(is.character(gpu_compiled_backends()))
  expect_true(gpu_active_backend() %in% c("none", "cuda", "hip", "sycl", "metal"))
  enabled <- gpu_get_enabled()
  gpu_set_enabled(FALSE)
  expect_false(gpu_get_enabled())
  expect_equal(gpu_active_backend(), "none")
  gpu_set_enabled(enabled)
  mb <- gpu_get_dense_max_mb()
  gpu_set_dense_max_mb(0)
  expect_equal(gpu_get_dense_max_mb(), 0)
  expect_error(gpu_set_dense_max_mb(-1))
  gpu_set_dense_max_mb(mb)
  host_mb <- linalg_get_iterative_dense_max_mb()
  linalg_set_iterative_dense_max_mb(0)
  expect_equal(linalg_get_iterative_dense_max_mb(), 0)
  linalg_set_iterative_dense_max_mb(host_mb)
  linalg_set_cg_warning(TRUE)
  linalg_set_cg_auto_precond(TRUE)
})
