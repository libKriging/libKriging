context("Kriging noise API: with_noise variances, update without noise, approximate LOO / LMP")

set.seed(5)
n <- 30
X <- matrix(runif(2 * n), ncol = 2)
noise <- rep(0.01, n)
y <- sin(3 * X[, 1]) + cos(5 * X[, 2]) + sqrt(noise) * rnorm(n)
Xn <- matrix(runif(6), ncol = 2)
kh <- Kriging(y, X, "matern5_2", noise = noise)

test_that("simulate with_noise holds noise variances", {
  var_n <- c(0.05, 0.2, 0.5)
  for (method in c("exact", "vecchia")) {
    # same seed: the latent draws are identical, the difference is the noise
    s_noisy <- kh$simulate(nsim = 20000, seed = 7, x = Xn, with_noise = var_n, method = method)
    s_latent <- kh$simulate(nsim = 20000, seed = 7, x = Xn, with_noise = rep(0, 3), method = method)
    emp_var <- apply(s_noisy - s_latent, 1, var)
    expect_true(all(abs(emp_var / var_n - 1) < 0.05))
  }
  # a numeric scalar is a variance for every point (not the nugget flag)
  s1 <- kh$simulate(nsim = 20000, seed = 8, x = Xn, with_noise = 0.3)
  s0 <- kh$simulate(nsim = 20000, seed = 8, x = Xn, with_noise = 0)
  expect_true(all(abs(apply(s1 - s0, 1, var) / 0.3 - 1) < 0.05))
  expect_error(kh$simulate(nsim = 10, seed = 7, x = Xn, with_noise = -var_n))
})

test_that("update without noise_u raises on a heterogeneous model", {
  k2 <- copy(kh)
  expect_error(k2$update(y[1:2], X[1:2, , drop = FALSE] + 0.01, refit = FALSE),
               "heterogeneous noise model: use update\\(y_u, noise_u, X_u\\)")
  expect_equal(nrow(k2$X()), n)
  k2$update(y[1:2], noise[1:2], X[1:2, , drop = FALSE] + 0.01, FALSE)
  expect_equal(nrow(k2$X()), n + 2)
})

test_that("approximate LOO / LMP raise on noise models", {
  kv <- Kriging(y, X, "matern5_2", noise = "nugget", method = "vecchia(10)")
  expect_true(is.finite(kv$logLikelihood(method = "vecchia")))
  expect_error(kv$leaveOneOut(method = "vecchia"), "not implemented for Nugget/Heterogeneous noise models")
  expect_error(kv$logMargPost(method = "vecchia"), "not implemented for Nugget/Heterogeneous noise models")
})
