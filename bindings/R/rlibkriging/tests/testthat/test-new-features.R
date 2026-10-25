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
  
  # Test Kriging with noise
  k2 <- Kriging(y, X, kernel = "gauss", noise = noise)
  cov2 <- covMat(k2, X_test, X_test)
  expect_equal(dim(cov2), c(5, 5))
  
  # Test NuggetKriging
  k3 <- Kriging(y, X, kernel = "gauss", noise = "nugget")
  cov3 <- covMat(k3, X_test, X_test)
  expect_equal(dim(cov3), c(5, 5))
})

test_that("model/as.list basic functionality works", {
  set.seed(789)
  n <- 11
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

test_that("as.list for Kriging with noise includes noise field", {
  set.seed(123)
  n <- 20
  X <- matrix(runif(n * 2), ncol = 2)
  y <- sin(X[, 1] * 3) * cos(X[, 2] * 3) + 1
  noise <- rep(0.1, n)
  
  k <- Kriging(y, X, kernel = "gauss", noise = noise)
  params <- as.list(k)
  
  # Kriging with noise should have 'noise' field
  expect_true('noise' %in% names(params))
  expect_equal(length(params$noise), n)
})

test_that("as.list for NuggetKriging includes nugget fields", {
  set.seed(654)
  n <- 15
  X <- matrix(runif(n), ncol = 1)
  y <- X[, 1]^2
  
  k <- Kriging(y, X, kernel = "matern3_2", noise = "nugget")
  params <- as.list(k)
  
  # NuggetKriging should have 'nugget' and 'is_nugget_estim' fields
  expect_true('nugget' %in% names(params))
  expect_true('is_nugget_estim' %in% names(params))
  expect_true(is.numeric(params$nugget))
  expect_true(is.logical(params$is_nugget_estim))
})

test_that("Optim reparametrization works", {
  # Save original state
  orig_state <- rlibkriging:::optim_is_reparametrized()
  
  # Test setter and getter
  rlibkriging:::optim_use_reparametrize(TRUE)
  expect_true(rlibkriging:::optim_is_reparametrized())
  
  rlibkriging:::optim_use_reparametrize(FALSE)
  expect_false(rlibkriging:::optim_is_reparametrized())
  
  # Restore original state
  rlibkriging:::optim_use_reparametrize(orig_state)
})

test_that("Optim theta bounds work", {
  # Save original values
  orig_lower <- rlibkriging:::optim_get_theta_lower_factor()
  orig_upper <- rlibkriging:::optim_get_theta_upper_factor()
  
  # Test lower factor
  rlibkriging:::optim_set_theta_lower_factor(0.05)
  expect_equal(rlibkriging:::optim_get_theta_lower_factor(), 0.05, tolerance = 1e-10)
  
  # Test upper factor
  rlibkriging:::optim_set_theta_upper_factor(15.0)
  expect_equal(rlibkriging:::optim_get_theta_upper_factor(), 15.0, tolerance = 1e-10)

  # Restore original values
  rlibkriging:::optim_set_theta_lower_factor(orig_lower)
  rlibkriging:::optim_set_theta_upper_factor(orig_upper)
})

test_that("Optim variogram bounds work", {
  orig_state <- rlibkriging:::optim_variogram_bounds_heuristic_used()

  rlibkriging:::optim_use_variogram_bounds_heuristic(TRUE)
  expect_true(rlibkriging:::optim_variogram_bounds_heuristic_used())

  rlibkriging:::optim_use_variogram_bounds_heuristic(FALSE)
  expect_false(rlibkriging:::optim_variogram_bounds_heuristic_used())

  rlibkriging:::optim_use_variogram_bounds_heuristic(orig_state)
})

test_that("Optim log level works", {
  orig_level <- rlibkriging:::optim_get_log_level()

  for (level in c(0, 1, 2, 3)) {
    rlibkriging:::optim_set_log_level(level)
    expect_equal(rlibkriging:::optim_get_log_level(), level)
  }

  rlibkriging:::optim_set_log_level(orig_level)
})

test_that("Optim max iteration works", {
  orig_max <- rlibkriging:::optim_get_max_iteration()

  rlibkriging:::optim_set_max_iteration(500)
  expect_equal(rlibkriging:::optim_get_max_iteration(), 500)

  rlibkriging:::optim_set_max_iteration(1000)
  expect_equal(rlibkriging:::optim_get_max_iteration(), 1000)

  rlibkriging:::optim_set_max_iteration(orig_max)
})

test_that("Optim tolerances work", {
  orig_grad <- rlibkriging:::optim_get_gradient_tolerance()
  orig_obj <- rlibkriging:::optim_get_objective_rel_tolerance()

  rlibkriging:::optim_set_gradient_tolerance(1e-6)
  expect_equal(rlibkriging:::optim_get_gradient_tolerance(), 1e-6, tolerance = 1e-15)

  rlibkriging:::optim_set_objective_rel_tolerance(1e-8)
  expect_equal(rlibkriging:::optim_get_objective_rel_tolerance(), 1e-8, tolerance = 1e-15)

  rlibkriging:::optim_set_gradient_tolerance(orig_grad)
  rlibkriging:::optim_set_objective_rel_tolerance(orig_obj)
})

test_that("Optim thread settings work", {
  orig_delay <- rlibkriging:::optim_get_thread_start_delay_ms()
  orig_pool <- rlibkriging:::optim_get_thread_pool_size()

  rlibkriging:::optim_set_thread_start_delay_ms(20)
  expect_equal(rlibkriging:::optim_get_thread_start_delay_ms(), 20)

  rlibkriging:::optim_set_thread_pool_size(4)
  expect_equal(rlibkriging:::optim_get_thread_pool_size(), 4)

  rlibkriging:::optim_set_thread_start_delay_ms(orig_delay)
  rlibkriging:::optim_set_thread_pool_size(orig_pool)
})

test_that("Optim thread budget works", {
  orig <- rlibkriging:::optim_get_num_threads_percent()

  rlibkriging:::optim_set_num_threads_percent(100)
  full <- rlibkriging:::optim_get_num_threads()
  expect_gte(full, 1)
  rlibkriging:::optim_set_num_threads_percent(50)
  expect_equal(rlibkriging:::optim_get_num_threads_percent(), 50)
  expect_gte(rlibkriging:::optim_get_num_threads(), 1)
  expect_lte(rlibkriging:::optim_get_num_threads(), full)
  rlibkriging:::optim_set_num_threads_percent(0.5) # ratio: same as 50 %
  expect_lte(rlibkriging:::optim_get_num_threads(), full)
  # a fit runs with a reduced budget
  set.seed(1)
  X <- matrix(runif(20), ncol = 1)
  k <- Kriging(sin(6 * X[, 1]), X, "matern3_2")
  expect_true(is.finite(k$logLikelihood()))

  rlibkriging:::optim_set_num_threads_percent(orig)
})

test_that("All classes have covMat", {
  set.seed(111)
  n <- 11
  X <- matrix(runif(n), ncol = 1)
  y <- X[, 1]
  noise <- rep(0.01, n)
  
  k1 <- Kriging(y, X, kernel = "gauss")
  k2 <- Kriging(y, X, kernel = "gauss", noise = noise)
  k3 <- Kriging(y, X, kernel = "gauss", noise = "nugget")
  
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
  n <- 11
  X <- matrix(runif(n), ncol = 1)
  y <- X[, 1]
  noise <- rep(0.01, n)
  
  k1 <- Kriging(y, X, kernel = "gauss")
  k2 <- Kriging(y, X, kernel = "gauss", noise = noise)
  k3 <- Kriging(y, X, kernel = "gauss", noise = "nugget")
  
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

test_that("summary() of Kriging and NestedKriging is their description", {
  set.seed(123)
  X <- matrix(runif(20), ncol = 1)
  y <- sin(6 * X[, 1])
  k <- Kriging(y, X, kernel = "matern3_2")
  expect_true(is.character(summary(k)) && nchar(summary(k)) > 0)
  invisible(capture.output(p <- print(k)))  # print() returns the description
  expect_equal(summary(k), p)
  nk <- NestedKriging(y, X, "matern3_2", nb_groups = 2)
  expect_true(is.character(summary(nk)) && nchar(summary(nk)) > 0)
})

test_that("jitters of a fitted model are kept by save / load", {
  X <- matrix(seq(0, 1, length.out = 10), ncol = 1)
  y <- sin(6 * X[, 1])
  models <- list(
    Kriging(y, X, kernel = "gauss"),
    WarpKriging(y, X, warping = "kumaraswamy", kernel = "gauss"),
    MLPKriging(y, X, hidden_dims = 4, d_out = 1, kernel = "gauss",
               parameters = list(max_iter_adam = "20"))
  )
  for (k in models) {
    expect_gt(k$chol_jitter(), 0)  # gauss, noise-free
    expect_true(k$model_chol_jitter() >= 0 && k$model_chol_jitter() <= k$chol_jitter())
    f <- tempfile(fileext = ".json")
    save(k, f)
    k2 <- load(f)
    expect_equal(k2$chol_jitter(), k$chol_jitter())
    expect_equal(k2$model_chol_jitter(), k$model_chol_jitter())
    unlink(f)
  }
  expect_equal(models[[1]]$model_chol_jitter(), 0)  # Kriging: exact committed model
})


test_that("update / update_simulate: X_u required, by name or position", {
  set.seed(1)
  X <- matrix(runif(10), ncol = 1)
  y <- sin(6 * X[, 1])
  Xu <- matrix(c(0.33, 0.66), ncol = 1)
  yu <- sin(6 * Xu[, 1])
  k1 <- Kriging(y, X, "gauss")
  k2 <- Kriging(y, X, "gauss")
  update(k1, yu, Xu, refit = FALSE)
  update(k2, yu, X_u = Xu, refit = FALSE)
  expect_equal(k1$X(), k2$X())
  expect_equal(nrow(k1$X()), 12)
  expect_error(update(k1, yu), "X_u")
  expect_false("X_u" %in% names(Filter(function(d) !identical(d, quote(expr = )),
                                       formals(rlibkriging:::update.Kriging))))
})
