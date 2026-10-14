context("WarpKriging / MLPKriging simulate: with_noise holds noise variances")

library(testthat)
library(rlibkriging)

f1d <- function(x) {
  1 - 0.5 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}

X <- as.matrix(seq(0.01, 0.99, length.out = 10))
y <- f1d(X)
Xn <- as.matrix(c(0.15, 0.45, 0.85))
var_n <- c(0.02, 0.1, 0.3)

# at equal seed the latent draws are identical: the difference with the
# no-noise draws is the added observation noise only
check_with_noise <- function(k) {
  # default call unchanged (no noise); NULL / FALSE are the same
  s0 <- simulate(k, nsim = 5000, seed = 7, x = Xn)
  expect_equal(dim(s0), c(3, 5000))
  expect_equal(simulate(k, nsim = 5000, seed = 7, x = Xn, with_noise = NULL), s0)
  expect_equal(simulate(k, nsim = 5000, seed = 7, x = Xn, with_noise = FALSE), s0)

  # per-point variances
  s1 <- simulate(k, nsim = 5000, seed = 7, x = Xn, with_noise = var_n)
  expect_true(all(abs(apply(s1 - s0, 1, var) / var_n - 1) < 0.15))
  # a scalar is a variance common to every point (k$simulate works too)
  s2 <- k$simulate(nsim = 5000, seed = 8, x = Xn, with_noise = 0.2)
  s20 <- k$simulate(nsim = 5000, seed = 8, x = Xn)
  expect_true(all(abs(apply(s2 - s20, 1, var) / 0.2 - 1) < 0.15))

  # invalid values raise
  expect_error(simulate(k, nsim = 10, seed = 7, x = Xn, with_noise = -var_n), "non-negative noise variances")
  expect_error(simulate(k, nsim = 10, seed = 7, x = Xn, with_noise = c(0.1, NA, 0.1)))
  expect_error(simulate(k, nsim = 10, seed = 7, x = Xn, with_noise = TRUE), "fitted-nugget")
  # unknown arguments are no longer silently ignored
  expect_warning(simulate(k, nsim = 2, seed = 7, x = Xn, with_nugget = TRUE), "ignored")
}

test_that("WarpKriging simulate with_noise holds noise variances", {
  k <- WarpKriging(y, X, warping = "kumaraswamy", kernel = "gauss", optim = "Adam",
                   parameters = list(max_iter_adam = "100"))
  check_with_noise(k)
})

test_that("MLPKriging simulate with_noise holds noise variances", {
  k <- MLPKriging(y, X, hidden_dims = 8, d_out = 2, activation = "tanh", kernel = "gauss",
                  parameters = list(max_iter_adam = "100"))
  check_with_noise(k)
})
