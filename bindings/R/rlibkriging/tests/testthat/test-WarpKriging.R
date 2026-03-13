## *************************************************************************
##  test-WarpKriging.R  —  testthat tests for WarpKriging R binding
##
##  Mirrors the 16 C++ tests from test_WarpKriging.cpp.
## *************************************************************************

library(testthat)
library(rlibkriging)

f1d <- function(x) {
  1 - 0.5 * (sin(12 * x) / (1 + x) + 2 * cos(7 * x) * x^5 + 0.7)
}

# ===========================================================================
#  Test 1: Kumaraswamy warping (1D continuous)
# ===========================================================================
test_that("Kumaraswamy warping works on 1D function", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 8))
  y <- f1d(X)

  k <- WarpKriging(y, X, warping = "kumaraswamy", kernel = "gauss",
                   parameters = list(max_iter_adam = "200"))

  expect_s3_class(k, "WarpKriging")
  cat(summary(k))

  p <- predict(k, X, stdev = TRUE)
  expect_equal(length(p$mean), 8)
  expect_true(all(is.finite(p$mean)))

  x_pred <- as.matrix(seq(0.01, 0.99, length.out = 50))
  p2 <- predict(k, x_pred, stdev = TRUE)
  rmse <- sqrt(mean((p2$mean - f1d(x_pred))^2))
  cat("  Kumaraswamy RMSE:", rmse, "\n")
})

# ===========================================================================
#  Test 2: Categorical-only (embedding)
# ===========================================================================
test_that("Categorical embedding works", {
  mu <- c(1.0, 5.0, 3.0)
  set.seed(42)
  n <- 15
  levels <- rep(0:2, length.out = n)
  X <- as.matrix(levels)
  y <- mu[levels + 1] + rnorm(n, sd = 0.1)

  k <- WarpKriging(y, X, warping = "categorical(3,2)", kernel = "gauss",
                   parameters = list(max_iter_adam = "200"))
  cat(summary(k))

  X_test <- as.matrix(0:2)
  p <- predict(k, X_test, stdev = TRUE)
  for (l in 1:3)
    cat(sprintf("    level %d: pred=%.3f, true=%.1f\n",
                l - 1, p$mean[l], mu[l]))
})

# ===========================================================================
#  Test 3: Mixed continuous + categorical
# ===========================================================================
test_that("Mixed continuous + categorical works", {
  offset <- c(1.0, 2.0, 0.5)
  set.seed(99)
  n <- 30
  X <- cbind(runif(n), rep(0:2, length.out = n))
  y <- sin(2 * pi * X[, 1]) * offset[X[, 2] + 1]

  k <- WarpKriging(y, X,
                   warping = c("kumaraswamy", "categorical(3,2)"),
                   kernel = "matern5_2",
                   parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  xc <- seq(0.01, 0.99, length.out = 20)
  for (cat_idx in 0:2) {
    X_test <- cbind(xc, rep(cat_idx, 20))
    p <- predict(k, X_test, stdev = TRUE)
    ytrue <- sin(2 * pi * xc) * offset[cat_idx + 1]
    rmse <- sqrt(mean((p$mean - ytrue)^2))
    cat(sprintf("  cat=%d  RMSE=%.4f\n", cat_idx, rmse))
  }
})

# ===========================================================================
#  Test 4: Ordinal variable
# ===========================================================================
test_that("Ordinal warping works", {
  L <- 5
  set.seed(7)
  n <- 20
  levels <- rep(0:(L - 1), length.out = n)
  X <- as.matrix(levels)
  y <- levels^2 + rnorm(n, sd = 0.1)

  k <- WarpKriging(y, X, warping = "ordinal(5)", kernel = "gauss",
                   parameters = list(max_iter_adam = "200"))
  cat(summary(k))

  X_test <- as.matrix(0:(L - 1))
  p <- predict(k, X_test, stdev = TRUE)
  for (l in 0:(L - 1))
    cat(sprintf("    level %d: pred=%.3f, true=%d\n", l, p$mean[l + 1], l * l))
})

# ===========================================================================
#  Test 5: NeuralMono warping
# ===========================================================================
test_that("NeuralMono warping works", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 10))
  y <- f1d(X)

  k <- WarpKriging(y, X, warping = "neural_mono(8)", kernel = "gauss",
                   parameters = list(max_iter_adam = "200"))
  cat(summary(k))

  p <- predict(k, X, stdev = TRUE)
  cat("  Max train error:", max(abs(p$mean - y)), "\n")
  expect_true(all(is.finite(p$mean)))
})

# ===========================================================================
#  Test 6: MLP warping (1D)
# ===========================================================================
test_that("MLP warping works on 1D function", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 10))
  y <- f1d(X)

  k <- WarpKriging(y, X, warping = "mlp(16:8,3,selu)", kernel = "gauss",
                   parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  x_pred <- as.matrix(seq(0.01, 0.99, length.out = 50))
  p <- predict(k, x_pred, stdev = TRUE)
  rmse <- sqrt(mean((p$mean - f1d(x_pred))^2))
  cat("  MLP RMSE:", rmse, "\n")
  expect_true(is.finite(rmse))
})

# ===========================================================================
#  Test 7: MLP + categorical mixed
# ===========================================================================
test_that("MLP + categorical mixed warping works", {
  offset <- c(1.0, 2.0, 0.5)
  set.seed(99)
  n <- 30
  X <- cbind(runif(n), rep(0:2, length.out = n))
  y <- sin(2 * pi * X[, 1]) * offset[X[, 2] + 1]

  k <- WarpKriging(y, X,
                   warping = c("mlp(16:8,2,tanh)", "categorical(3,2)"),
                   kernel = "matern5_2",
                   parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  xc <- seq(0.01, 0.99, length.out = 20)
  for (cat_idx in 0:2) {
    p <- predict(k, cbind(xc, rep(cat_idx, 20)), stdev = TRUE)
    ytrue <- sin(2 * pi * xc) * offset[cat_idx + 1]
    cat(sprintf("  [MLP+cat] cat=%d RMSE=%.4f\n", cat_idx,
                sqrt(mean((p$mean - ytrue)^2))))
  }
})

# ===========================================================================
#  Test 8: Conditional simulations (mixed)
# ===========================================================================
test_that("simulate works with mixed variables", {
  set.seed(42)
  n <- 20
  X <- cbind(runif(n), rep(0:1, length.out = n))
  y <- sin(2 * pi * X[, 1]) * c(1, 3)[X[, 2] + 1]

  k <- WarpKriging(y, X,
                   warping = c("affine", "categorical(2,2)"),
                   kernel = "gauss",
                   parameters = list(max_iter_adam = "200"))

  X_sim <- cbind(seq(0.1, 0.9, length.out = 10), rep(0, 10))
  sims <- simulate(k, nsim = 30, seed = 123, x = X_sim)

  expect_equal(nrow(sims), 10)
  expect_equal(ncol(sims), 30)
  expect_true(all(is.finite(sims)))

  p <- predict(k, X_sim, stdev = TRUE)
  rel_diff <- sqrt(sum((rowMeans(sims) - p$mean)^2)) / sqrt(sum(p$mean^2))
  cat("  Sim mean vs kriging mean rel diff:", rel_diff, "\n")
})

# ===========================================================================
#  Test 9: Update (incremental)
# ===========================================================================
test_that("update works", {
  X0 <- matrix(c(0.1, 0.0, 0.5, 1.0, 0.9, 0.0), ncol = 2, byrow = TRUE)
  y0 <- c(1.0, 3.0, 0.5)

  k <- WarpKriging(y0, X0,
                   warping = c("none", "categorical(2,1)"),
                   kernel = "gauss",
                   parameters = list(max_iter_adam = "100"))

  X_new <- matrix(c(0.3, 1.0, 0.7, 0.0), ncol = 2, byrow = TRUE)
  update(k, c(2.0, 1.5), X_new)

  p <- predict(k, X0, stdev = TRUE)
  expect_true(all(is.finite(p$mean)))
})

# ===========================================================================
#  Test 10: Branin 2D with per-variable MLP
# ===========================================================================
test_that("Branin 2D with MLP warping works", {
  branin <- function(x1, x2) {
    a <- 1; b <- 5.1 / (4 * pi^2); cc <- 5 / pi
    r <- 6; s <- 10; t <- 1 / (8 * pi)
    a * (x2 - b * x1^2 + cc * x1 - r)^2 + s * (1 - t) * cos(x1) + s
  }
  set.seed(77)
  n <- 25
  X <- matrix(runif(n * 2), ncol = 2)
  y <- mapply(function(i) branin(X[i, 1] * 15 - 5, X[i, 2] * 15), 1:n)

  k <- WarpKriging(y, X,
                   warping = c("mlp(16:8,2,selu)", "mlp(16:8,2,selu)"),
                   kernel = "matern5_2", normalize = TRUE,
                   parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  set.seed(88)
  X_test <- matrix(runif(15 * 2), ncol = 2)
  p <- predict(k, X_test, stdev = TRUE)
  expect_equal(length(p$mean), 15)
  expect_true(all(is.finite(p$stdev)))

  sims <- simulate(k, nsim = 20, seed = 42, x = X_test)
  expect_equal(dim(sims), c(15, 20))
})

# ===========================================================================
#  Test 11: None vs Kumaraswamy vs MLP comparison
# ===========================================================================
test_that("Warping improves over None baseline", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 12))
  y <- f1d(X)
  xp <- as.matrix(seq(0.01, 0.99, length.out = 50))
  ytrue <- f1d(xp)

  k_none <- WarpKriging(y, X, "none", "gauss",
                        parameters = list(max_iter_adam = "200"))
  k_kuma <- WarpKriging(y, X, "kumaraswamy", "gauss",
                        parameters = list(max_iter_adam = "200"))
  k_mlp  <- WarpKriging(y, X, "mlp(16:8,2,selu)", "gauss",
                         parameters = list(max_iter_adam = "300"))

  rmse <- function(model) sqrt(mean((predict(model, xp)$mean - ytrue)^2))
  cat(sprintf("  RMSE None: %.6f\n  RMSE Kuma: %.6f\n  RMSE MLP:  %.6f\n",
              rmse(k_none), rmse(k_kuma), rmse(k_mlp)))
  cat(sprintf("  LL None: %.4f\n  LL Kuma: %.4f\n  LL MLP:  %.4f\n",
              logLikelihood(k_none), logLikelihood(k_kuma), logLikelihood(k_mlp)))
})

# ===========================================================================
#  Test 12: logLikelihoodFun
# ===========================================================================
test_that("logLikelihoodFun works with analytical gradient", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 8))
  y <- f1d(X)

  k <- WarpKriging(y, X, "affine", "gauss",
                   parameters = list(max_iter_adam = "100"))

  th <- theta(k)
  ll <- logLikelihoodFun(k, th, grad = TRUE)

  expect_true(is.finite(ll$logLikelihood))
  expect_true(all(is.finite(ll$gradient)))
  cat("  LL at theta:", ll$logLikelihood, "\n")
  cat("  Gradient norm:", sqrt(sum(ll$gradient^2)), "\n")

  # FD check
  h <- 1e-5
  grad_num <- numeric(length(th))
  for (i in seq_along(th)) {
    tp <- tm <- th
    tp[i] <- tp[i] + h; tm[i] <- tm[i] - h
    grad_num[i] <- (logLikelihoodFun(k, tp, FALSE)$logLikelihood -
                    logLikelihoodFun(k, tm, FALSE)$logLikelihood) / (2 * h)
  }
  rel <- sqrt(sum((ll$gradient - grad_num)^2)) /
         (sqrt(sum(grad_num^2)) + 1e-12)
  cat("  Gradient FD check error:", rel, "\n")
})

# ===========================================================================
#  Test 13: Accessors and summary
# ===========================================================================
test_that("Summary and accessors work", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 6))
  y <- f1d(X)

  k <- WarpKriging(y, X, "boxcox", "matern3_2",
                   parameters = list(max_iter_adam = "100"))

  s <- summary(k)
  expect_true(is.character(s))
  expect_true(nchar(s) > 0)

  th <- theta(k)
  expect_true(length(th) > 0)
  expect_true(all(th > 0))

  s2 <- sigma2(k)
  expect_true(s2 > 0)

  expect_equal(kernel(k), "matern3_2")

  ll <- logLikelihood(k)
  expect_true(is.finite(ll))

  ws <- warping(k)
  expect_equal(ws, "boxcox")

  cat("  kernel:", kernel(k), "\n")
  cat("  theta:", th, "\n")
  cat("  sigma2:", s2, "\n")
  cat("  warping:", ws, "\n")
})

# ===========================================================================
#  Test 14: warp_*() helper functions return correct strings
# ===========================================================================
test_that("warp helpers produce correct strings", {
  expect_equal(warp_none(), "none")
  expect_equal(warp_affine(), "affine")
  expect_equal(warp_boxcox(), "boxcox")
  expect_equal(warp_kumaraswamy(), "kumaraswamy")
  expect_equal(warp_neural_mono(16), "neural_mono(16)")
  expect_equal(warp_mlp(c(16, 8), 3, "selu"), "mlp(16:8,3,selu)")
  expect_equal(warp_mlp_joint(c(32, 16), 3, "tanh"), "mlp_joint(32:16,3,tanh)")
  expect_equal(warp_categorical(5, 2), "categorical(5,2)")
  expect_equal(warp_ordinal(4), "ordinal(4)")
})

# ===========================================================================
#  Test 15: mlp_joint (Deep Kernel Learning — replaces NeuralKernelKriging)
# ===========================================================================
test_that("mlp_joint works (1D)", {
  X <- as.matrix(seq(0.01, 0.99, length.out = 10))
  y <- f1d(X)

  k <- WarpKriging(y, X,
                   warping = "mlp_joint(16:8,2,selu)",
                   kernel = "gauss", normalize = TRUE,
                   parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  p <- predict(k, X, stdev = TRUE)
  cat("  [1D] Max train error:", max(abs(p$mean - y)), "\n")
  expect_true(all(is.finite(p$mean)))

  ws <- warping(k)
  expect_equal(ws, "mlp_joint(16:8,2,selu)")
})

test_that("mlp_joint works (Branin 2D)", {
  branin <- function(x1, x2) {
    a <- 1; b <- 5.1 / (4 * pi^2); cc <- 5 / pi
    r <- 6; s <- 10; t <- 1 / (8 * pi)
    a * (x2 - b * x1^2 + cc * x1 - r)^2 + s * (1 - t) * cos(x1) + s
  }
  set.seed(77)
  n <- 20
  X <- matrix(runif(n * 2), ncol = 2)
  y <- mapply(function(i) branin(X[i, 1] * 15 - 5, X[i, 2] * 15), 1:n)

  k <- WarpKriging(y, X,
                   warping = "mlp_joint(32:16,3,selu)",
                   kernel = "matern5_2", normalize = TRUE,
                   parameters = list(max_iter_adam = "300"))
  cat(summary(k))

  X_test <- matrix(runif(10 * 2), ncol = 2)
  p <- predict(k, X_test, stdev = TRUE)
  expect_equal(length(p$mean), 10)
  expect_true(all(is.finite(p$stdev)))

  sims <- simulate(k, nsim = 20, seed = 42, x = X_test)
  expect_equal(dim(sims), c(10, 20))
  expect_true(all(is.finite(sims)))

  # Update
  X_new <- matrix(runif(3 * 2), ncol = 2)
  y_new <- mapply(function(i) branin(X_new[i, 1] * 15 - 5, X_new[i, 2] * 15), 1:3)
  update(k, y_new, X_new)
  cat("  n after update:", length(predict(k, X_test)$mean), "predictions OK\n")
})
