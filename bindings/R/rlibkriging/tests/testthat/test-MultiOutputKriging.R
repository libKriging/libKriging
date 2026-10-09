library(testthat)
library(rlibkriging)

# Damped oscillation sampled at q time steps, two inputs (frequency, damping)
t_out <- seq(0.5, 10, length.out = 30)
code <- function(x) exp(-(0.1 + 0.5 * x[2]) * t_out) * cos(2 * pi * (0.5 + 1.5 * x[1]) * t_out / 5)

set.seed(1)
X <- matrix(runif(2 * 40), ncol = 2)
Y <- t(apply(X, 1, code))
Xt <- matrix(runif(2 * 10), ncol = 2)
Yt <- t(apply(Xt, 1, code))

test_that("pca: shapes, accuracy and accessors", {
  mo <- MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "pca(0.999)")
  expect_s3_class(mo, "MultiOutputKriging")
  expect_equal(mo$nb_outputs(), ncol(Y))
  K <- mo$nb_components()
  expect_true(K >= 1)
  expect_equal(dim(mo$pca_basis()), c(ncol(Y), K))
  expect_gte(mo$pca_explained()[K], 0.999)
  expect_equal(kernel(mo), "matern5_2")
  expect_equal(mo$output_model(), "pca(0.999)")

  p <- predict(mo, Xt, return_stdev = TRUE, return_cov = TRUE, return_deriv = TRUE)
  expect_equal(dim(p$mean), dim(Yt))
  expect_equal(dim(p$stdev), dim(Yt))
  expect_equal(dim(p$cov), c(length(Yt), length(Yt)))
  expect_equal(dim(p$mean_deriv), c(nrow(Xt), ncol(X), ncol(Y)))
  expect_equal(sqrt(diag(p$cov)), as.vector(p$stdev), tolerance = 1e-8)
  expect_lt(sqrt(mean((p$mean - Yt)^2)), 0.25 * sd(Y))

  # a latent model is a plain Kriging (copy)
  k1 <- mo$component(1)
  expect_s3_class(k1, "Kriging")
  expect_equal(nrow(k1$X()), nrow(X))
})

test_that("shared with one output is Kriging", {
  y <- Y[, 5]
  k <- Kriging(y, X, kernel = "matern5_2")
  mo <- MultiOutputKriging(matrix(y, ncol = 1), X, kernel = "matern5_2", output_model = "shared")
  expect_equal(as.numeric(theta(mo)), as.numeric(k$theta()), tolerance = 1e-6)
  expect_equal(logLikelihood(mo), logLikelihood(k), tolerance = 1e-8)
  pk <- predict(k, Xt)
  pm <- predict(mo, Xt)
  expect_equal(as.numeric(pm$mean), as.numeric(pk$mean), tolerance = 1e-6)
  expect_equal(as.numeric(pm$stdev), as.numeric(pk$stdev), tolerance = 1e-6)
})

test_that("shared: logLikelihoodFun gradient matches finite differences", {
  mo <- MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "shared")
  th <- as.numeric(theta(mo)) * 1.3
  r <- logLikelihoodFun(mo, th, return_grad = TRUE)
  h <- 1e-4  # smaller steps are dominated by the rounding noise of the LL
  for (i in seq_along(th)) {
    e <- replace(numeric(length(th)), i, h)
    fd <- (logLikelihoodFun(mo, th + e)$logLikelihood - logLikelihoodFun(mo, th - e)$logLikelihood) / (2 * h)
    expect_equal(r$logLikelihoodGrad[1, i], as.numeric(fd), tolerance = 1e-4)
  }
})

test_that("shared matches RobustGaSP::ppgasp(method = 'mle')", {
  skip_if_not_installed("RobustGaSP")
  mo <- MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "shared", optim = "BFGS10")
  set.seed(1)
  invisible(capture.output(  # ppgasp prints its iterations with cat()
    rg <- suppressMessages(RobustGaSP::ppgasp(design = X, response = Y, kernel_type = "matern_5_2",
                                              method = "mle", nugget.est = FALSE, lower_bound = FALSE,
                                              num_initial_values = 5, max_eval = 200))))
  # same likelihood, so the libKriging optimum is at least as good
  expect_gte(logLikelihood(mo), logLikelihoodFun(mo, 1 / rg@beta_hat)$logLikelihood[1] - 1e-6)
  expect_equal(as.numeric(theta(mo)), 1 / rg@beta_hat, tolerance = 1e-3)
  pr <- RobustGaSP::predict(rg, Xt)
  expect_lt(max(abs(predict(mo, Xt)$mean - pr$mean)), 1e-4 * sd(Y))
})

test_that("separable: joint covariance factors and simulations", {
  Y3 <- Y[, c(3, 10, 20)]
  mo <- MultiOutputKriging(Y3, X, kernel = "matern5_2", output_model = "separable")
  S <- mo$output_cov()
  expect_equal(dim(S), c(3, 3))
  expect_equal(S, t(S))
  # same mean as "shared" at the same theta
  sh <- MultiOutputKriging(Y3, X, kernel = "matern5_2", output_model = "shared",
                           optim = "none", parameters = list(theta = matrix(theta(mo), nrow = 1)))
  expect_equal(predict(mo, Xt)$mean, predict(sh, Xt)$mean, tolerance = 1e-8)

  f <- mo$predictCovFactors(Xt)
  p <- predict(mo, Xt, return_cov = TRUE)
  expect_equal(kronecker(f$Sigma, f$Cx), p$cov, tolerance = 1e-8)

  s <- simulate(mo, nsim = 2000, seed = 3, x = Xt)
  expect_equal(dim(s), c(nrow(Xt), 3, 2000))
  emp <- apply(s, c(1, 2), mean)
  expect_lt(max(abs(emp - p$mean) / pmax(p$stdev, 1e-12)), 5 / sqrt(2000) * 3)
})

test_that("update and update_simulate", {
  for (om in c("pca(0.999)", "shared")) {
    mo <- MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = om)
    Xu <- Xt[1:3, , drop = FALSE]
    Yu <- Yt[1:3, , drop = FALSE]
    xs <- Xt[4:10, , drop = FALSE]
    expect_error(update_simulate(mo, Yu, Xu))
    s0 <- simulate(mo, nsim = 1000, seed = 5, x = xs, will_update = TRUE)
    s1 <- update_simulate(mo, Yu, Xu)
    expect_equal(dim(s1), dim(s0))
    update(mo, Yu, Xu, refit = FALSE)
    expect_equal(nrow(mo$X()), nrow(X) + 3)
    p <- predict(mo, xs)
    emp <- apply(s1, c(1, 2), mean)
    expect_lt(max(abs(emp - p$mean)), 5 * max(p$stdev) / sqrt(1000) + 1e-8)
  }
})

test_that("leaveOneOut", {
  mo <- MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "shared")
  l <- mo$leaveOneOutMat()
  expect_equal(dim(l$mean), dim(Y))
  expect_equal(leaveOneOut(mo), mean((Y - l$mean)^2), tolerance = 1e-10)
  r <- leaveOneOutFun(mo, as.numeric(theta(mo)), return_grad = TRUE)
  expect_equal(dim(r$leaveOneOutGrad), c(1, ncol(X)))
})

test_that("bad usage fails clearly", {
  expect_error(MultiOutputKriging(Y[-1, ], X, kernel = "matern5_2"))
  expect_error(MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "foo"))
  expect_error(MultiOutputKriging(Y, X, kernel = "matern5_2", output_model = "shared", objective = "LMP"))
  expect_error(MultiOutputKriging(Y, X, kernel = "matern5_2", parameters = list(sigma2 = 1)))
})
