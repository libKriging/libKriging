# Test suite for newly added features: covMat, model(), and Optim class
# Run with: testthat::test_file("test-new-features.R")

library(testthat)
library(rlibkriging)

context("New Features: covMat, model, Optim")

test_that("covMat basic functionality works", {
  set.seed(123)
  n <- 20
  X <- matrix(runif(n * 2), ncol = 2)
  y <- sin(X[, 1]) + cos(X[, 2])
  
  # Fit model
  k <- Kriging(y, X, kernel = "matern3_2")
  
  # Test covMat computation
  X1 <- matrix(runif(5 * 2), ncol = 2)
  X2 <- matrix(runif(10 * 2), ncol = 2)
  
  cov <- covMat(k, X1, X2)
  
  # Check dimensions
  expect_equal(dim(cov), c(5, 10))
  
  # Check symmetry when X1 == X2
  cov_sym <- covMat(k, X1, X1)
  expect_equal(dim(cov_sym), c(5, 5))
  expect_true(max(abs(cov_sym - t(cov_sym))) < 1e-10)
  
  # Covariance should be positive semi-definite
  eigenvals <- eigen(cov_sym, only.values = TRUE)$values
  expect_true(all(eigenvals >= -1e-10))
})

test_that("covMat works for all Kriging classes", {
  set.seed(456)
  n <- 15
  X <- matrix(runif(n), ncol = 1)
  y <- sin(3 * X[, 1])
  noise <- rep(0.01, n)
  
  X_test <- matrix(runif(5), ncol = 1)
  
  # Test Kriging
  k1 <- Kriging(y, X, kernel = "gauss")
  cov1 <- covMat(k1, X_test, X_test)
  expect_equal(dim(cov1), c(5, 5))
  
  # Test NoiseKriging
  k2 <- NoiseKriging(y, noise, X, kernel = "gauss")
  cov2 <- covMat(k2, X_test, X_test)
  expect_equal(dim(cov2), c(5, 5))
  
  # Test NuggetKriging
  k3 <- NuggetKriging(y, X, kernel = "gauss")
  cov3 <- covMat(k3, X_test, X_test)
  expect_equal(dim(cov3), c(5, 5))
})

test_that("model/as.list basic functionality works", {
  set.seed(789)
  n <- 10
  X <- matrix(runif(n), ncol = 1)
  y <- exp(X[, 1])
  
  k <- Kriging(y, X, kernel = "matern5_2", regmodel = "linear", 
               normalize = TRUE, optim = "BFGS", objective = "LL")
  
  # Get model parameters using as.list
  params <- as.list(k)
  
  # Check that all expected elements are present
  expected_names <- c('kernel', 'optim', 'objective', 'theta', 'is_theta_estim',
                     'sigma2', 'is_sigma2_estim', 'X', 'centerX', 'scaleX',
                     'y', 'centerY', 'scaleY', 'normalize', 'regmodel',
                     'beta', 'is_beta_estim', 'F', 'T', 'M', 'z')
  
  for (name in expected_names) {
    expect_true(name %in% names(params), 
                info = paste("Missing element:", name))
  }
  
  # Check types and values
  expect_equal(params$kernel, 'matern5_2')
  expect_equal(params$optim, 'BFGS')
  expect_equal(params$objective, 'LL')
  expect_true(params$normalize)
  expect_equal(params$regmodel, 'linear')
  
  # Check array shapes
  expect_equal(dim(params$X), c(n, 1))
  expect_equal(length(params$y), n)
  expect_true(length(params$theta) > 0)
  expect_true(length(params$beta) > 0)
})

test_that("as.list for NoiseKriging includes noise field", {
  set.seed(123)
  n <- 20
  X <- matrix(runif(n * 2), ncol = 2)
  y <- sin(X[, 1] * 3) * cos(X[, 2] * 3) + 1
  noise <- rep(0.1, n)
  
  k <- NoiseKriging(y, noise, X, kernel = "gauss")
  params <- as.list(k)
  
  # NoiseKriging should have 'noise' field
  expect_true('noise' %in% names(params))
  expect_equal(length(params$noise), n)
})

test_that("as.list for NuggetKriging includes nugget fields", {
  set.seed(654)
  n <- 15
  X <- matrix(runif(n), ncol = 1)
  y <- X[, 1]^2
  
  k <- NuggetKriging(y, X, kernel = "matern3_2")
  params <- as.list(k)
  
  # NuggetKriging should have 'nugget' and 'is_nugget_estim' fields
  expect_true('nugget' %in% names(params))
  expect_true('is_nugget_estim' %in% names(params))
  expect_true(is.numeric(params$nugget))
  expect_true(is.logical(params$is_nugget_estim))
})

test_that("Optim reparametrization works", {
  # Save original state
  orig_state <- optim_is_reparametrized()
  
  # Test setter and getter
  optim_use_reparametrize(TRUE)
  expect_true(optim_is_reparametrized())
  
  optim_use_reparametrize(FALSE)
  expect_false(optim_is_reparametrized())
  
  # Restore original state
  optim_use_reparametrize(orig_state)
})

test_that("Optim theta bounds work", {
  # Save original values
  orig_lower <- optim_get_theta_lower_factor()
  orig_upper <- optim_get_theta_upper_factor()
  
  # Test lower factor
  optim_set_theta_lower_factor(0.05)
  expect_equal(optim_get_theta_lower_factor(), 0.05, tolerance = 1e-10)
  
  # Test upper factor
  optim_set_theta_upper_factor(15.0)
  expect_equal(optim_get_theta_upper_factor(), 15.0, tolerance = 1e-10)
  
  # Restore original values
  optim_set_theta_lower_factor(orig_lower)
  optim_set_theta_upper_factor(orig_upper)
})

test_that("Optim variogram bounds work", {
  orig_state <- optim_variogram_bounds_heuristic_used()
  
  optim_use_variogram_bounds_heuristic(TRUE)
  expect_true(optim_variogram_bounds_heuristic_used())
  
  optim_use_variogram_bounds_heuristic(FALSE)
  expect_false(optim_variogram_bounds_heuristic_used())
  
  optim_use_variogram_bounds_heuristic(orig_state)
})

test_that("Optim log level works", {
  orig_level <- optim_get_log_level()
  
  for (level in c(0, 1, 2, 3)) {
    optim_set_log_level(level)
    expect_equal(optim_get_log_level(), level)
  }
  
  optim_set_log_level(orig_level)
})

test_that("Optim max iteration works", {
  orig_max <- optim_get_max_iteration()
  
  optim_set_max_iteration(500)
  expect_equal(optim_get_max_iteration(), 500)
  
  optim_set_max_iteration(1000)
  expect_equal(optim_get_max_iteration(), 1000)
  
  optim_set_max_iteration(orig_max)
})

test_that("Optim tolerances work", {
  orig_grad <- optim_get_gradient_tolerance()
  orig_obj <- optim_get_objective_rel_tolerance()
  
  optim_set_gradient_tolerance(1e-6)
  expect_equal(optim_get_gradient_tolerance(), 1e-6, tolerance = 1e-15)
  
  optim_set_objective_rel_tolerance(1e-8)
  expect_equal(optim_get_objective_rel_tolerance(), 1e-8, tolerance = 1e-15)
  
  optim_set_gradient_tolerance(orig_grad)
  optim_set_objective_rel_tolerance(orig_obj)
})

test_that("Optim thread settings work", {
  orig_delay <- optim_get_thread_start_delay_ms()
  orig_pool <- optim_get_thread_pool_size()
  
  optim_set_thread_start_delay_ms(20)
  expect_equal(optim_get_thread_start_delay_ms(), 20)
  
  optim_set_thread_pool_size(4)
  expect_equal(optim_get_thread_pool_size(), 4)
  
  optim_set_thread_start_delay_ms(orig_delay)
  optim_set_thread_pool_size(orig_pool)
})

test_that("All classes have covMat", {
  set.seed(111)
  n <- 10
  X <- matrix(runif(n), ncol = 1)
  y <- X[, 1]
  noise <- rep(0.01, n)
  
  k1 <- Kriging(y, X, kernel = "gauss")
  k2 <- NoiseKriging(y, noise, X, kernel = "gauss")
  k3 <- NuggetKriging(y, X, kernel = "gauss")
  
  X_test <- matrix(runif(3), ncol = 1)
  
  # All should work
  cov1 <- covMat(k1, X_test, X_test)
  cov2 <- covMat(k2, X_test, X_test)
  cov3 <- covMat(k3, X_test, X_test)
  
  expect_equal(dim(cov1), c(3, 3))
  expect_equal(dim(cov2), c(3, 3))
  expect_equal(dim(cov3), c(3, 3))
})

test_that("All classes have as.list/model", {
  set.seed(222)
  n <- 10
  X <- matrix(runif(n), ncol = 1)
  y <- X[, 1]
  noise <- rep(0.01, n)
  
  k1 <- Kriging(y, X, kernel = "gauss")
  k2 <- NoiseKriging(y, noise, X, kernel = "gauss")
  k3 <- NuggetKriging(y, X, kernel = "gauss")
  
  # All should return lists
  m1 <- as.list(k1)
  m2 <- as.list(k2)
  m3 <- as.list(k3)
  
  expect_true(is.list(m1))
  expect_true(is.list(m2))
  expect_true(is.list(m3))
  
  # Check class-specific fields
  expect_true('noise' %in% names(m2))
  expect_true('nugget' %in% names(m3))
  expect_true('is_nugget_estim' %in% names(m3))
})
